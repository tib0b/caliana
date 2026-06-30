"""Analyses on ROI traces. SPEC.md §3 Stage III.

Built-ins: ΔF/F, peak detection, cross-ROI propagation. Custom analyses are
plain Python callables ``f(traces, data) -> result`` (full trust, no sandbox).

Onset / change-point detection helpers already exist in the repo
(calcium_onset_detection.py: PELT via `ruptures`; Pivat/src/core/utils.py:
threshold-on-baseline). Those are the intended building blocks for response
timing and cross-ROI propagation, to be folded in here.
"""
from __future__ import annotations

import numpy as np

from .models import BaselineMethod, ROI, Traces


def compute_dff(
    traces: Traces,
    method: BaselineMethod = BaselineMethod.FIRST_N,
    n: int | None = None,
    region: tuple[int, int] | None = None,
) -> Traces:
    """Compute ΔF/F = (F - F0)/F0 per ROI, storing it on ``traces.dff``. SPEC §3.

    F0 is either the mean of the first ``n`` frames (FIRST_N) or the mean over a
    user-selected ``region`` ``[start, end)`` (REGION).
    """
    F = traces.raw
    if method == BaselineMethod.FIRST_N:
        if n is None:
            raise ValueError("FIRST_N baseline requires n (number of frames)")
        F0 = F[:, :n].mean(axis=1, keepdims=True)
    elif method == BaselineMethod.REGION:
        if region is None:
            raise ValueError("REGION baseline requires (start, end)")
        s, e = region
        F0 = F[:, s:e].mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown baseline method {method!r}")

    traces.dff = (F - F0) / F0
    return traces


def detect_peaks(trace: np.ndarray, threshold: float | None = None,
                 prominence: float | None = None) -> dict:
    """Per-trace peaks: indices, amplitudes, time-to-peak, count. SPEC §3.

    Thin wrapper over ``scipy.signal.find_peaks`` (imported lazily).
    """
    from scipy.signal import find_peaks

    kwargs: dict = {}
    if threshold is not None:
        kwargs["height"] = threshold
    if prominence is not None:
        kwargs["prominence"] = prominence
    idx, _props = find_peaks(trace, **kwargs)
    return {
        "indices": idx,
        "amplitudes": trace[idx] if len(idx) else np.array([]),
        "time_to_peak": int(idx[np.argmax(trace[idx])]) if len(idx) else None,
        "count": int(len(idx)),
    }


def onset_time(
    sig: np.ndarray,
    method: str = "half_max",
    baseline_frames: int | None = None,
    frac: float = 0.5,
    k: float = 3.0,
) -> float:
    """Frame at which a trace first rises from baseline. SPEC §3 (response timing).

    Robust for step-like sustained responses (unlike peak finding). Returns a
    sub-frame value via linear interpolation, or NaN if no rise is detected.

    - ``half_max``: threshold = baseline + ``frac`` * (max - baseline).
    - ``std``: threshold = baseline_mean + ``k`` * baseline_std
      (mirrors the detector in Pivat/src/core/utils.py).
    """
    sig = np.asarray(sig, dtype=float)
    base = sig[:baseline_frames].mean() if baseline_frames else float(sig.min())

    if method == "half_max":
        amp = float(sig.max()) - base
        if amp <= 0:
            return float("nan")
        thresh = base + frac * amp
    elif method == "std":
        n = baseline_frames or max(1, len(sig) // 10)
        thresh = sig[:n].mean() + k * sig[:n].std()
    else:
        raise ValueError(f"Unknown onset method {method!r} (expected 'half_max' | 'std')")

    above = np.flatnonzero(sig >= thresh)
    if above.size == 0:
        return float("nan")
    j = int(above[0])
    if j == 0:
        return 0.0
    y0, y1 = sig[j - 1], sig[j]
    return float(j) if y1 == y0 else (j - 1) + (thresh - y0) / (y1 - y0)


def cross_roi_propagation(
    traces: Traces,
    rois: list[ROI],
    signal: str = "dff",
    method: str = "half_max",
    baseline_frames: int | None = None,
    frac: float = 0.5,
    k: float = 3.0,
) -> dict:
    """Estimate signal propagation across ROIs from response timing. SPEC §3.

    Detects per-ROI onset times and fits onset = a*x + b*y + c over ROI pixel
    coordinates. The onset-time gradient is the slowness vector (frames/px), so
    speed = 1/|gradient| (px/frame) and its unit vector points in the direction
    of propagation (toward later onset). Units are px/frame (SPEC §3).
    """
    data = traces.dff if (signal == "dff" and traces.dff is not None) else traces.raw
    n = data.shape[0]
    if n != len(rois):
        raise ValueError(f"traces ({n}) and rois ({len(rois)}) count mismatch")

    onsets = np.array(
        [onset_time(data[i], method, baseline_frames, frac, k) for i in range(n)]
    )
    coords = np.array([roi.center for roi in rois], dtype=float)  # (y, x)
    valid = ~np.isnan(onsets)
    nv = int(valid.sum())

    result: dict = {
        "onsets": onsets,
        "method": method,
        "source_roi": int(np.nanargmin(onsets)) if nv else None,
        "speed_px_per_frame": None,
        "direction": None,
        "pairwise": [],
    }

    # Pairwise speeds along ROI-to-ROI lines.
    idxs = np.flatnonzero(valid)
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            i, j = int(idxs[a]), int(idxs[b])
            dt = float(onsets[j] - onsets[i])
            dist = float(np.hypot(*(coords[j] - coords[i])))
            result["pairwise"].append({
                "i": i, "j": j, "distance": dist, "delta_t": dt,
                "speed_px_per_frame": dist / abs(dt) if dt != 0 else float("inf"),
            })

    if nv >= 3:
        X = coords[valid]
        A = np.column_stack([X[:, 1], X[:, 0], np.ones(nv)])  # [x, y, 1]
        (a_x, b_y, _c), *_ = np.linalg.lstsq(A, onsets[valid], rcond=None)
        smag = float(np.hypot(a_x, b_y))
        if smag > 1e-9:
            result["speed_px_per_frame"] = 1.0 / smag
            result["direction"] = (float(b_y / smag), float(a_x / smag))  # (dy, dx)
        else:
            result["speed_px_per_frame"] = float("inf")
    elif nv == 2:
        p = result["pairwise"][0]
        result["speed_px_per_frame"] = p["speed_px_per_frame"]
        d = coords[p["j"]] - coords[p["i"]]
        if p["delta_t"] < 0:
            d = -d
        norm = float(np.hypot(*d))
        if norm > 0:
            result["direction"] = (float(d[0] / norm), float(d[1] / norm))

    return result


def apply_custom(func, traces: Traces, data: np.ndarray):
    """Run a user-supplied callable ``f(traces, data) -> result``. SPEC §3.

    Full trust, no sandboxing — intended for notebook use.
    """
    return func(traces, data)
