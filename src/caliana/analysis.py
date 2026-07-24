"""Analyses on ROI traces.

Built-ins: ΔF/F (``compute_dff``), Gaussian smoothing (``smooth_traces``),
response-onset timing (``onset_time``, ``onset_time_map``), and cross-ROI
propagation (``cross_roi_propagation``). Custom analyses are plain callables
``f(traces, data) -> result`` (full trust, no sandbox), run via ``apply_custom``.
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
    """Compute ΔF/F = (F - F0)/F0 per ROI, storing it on ``traces.dff``.

    method: ``BaselineMethod.FIRST_N`` — F0 is the mean of the first ``n`` frames
        (``n`` required); ``BaselineMethod.REGION`` — F0 is the mean over
        ``region`` ``[start, end)`` (``region`` required).
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


def smooth_traces(traces: Traces, sigma: float) -> Traces:
    """Gaussian-smooth ``traces.dff`` along time, storing the result on
    ``traces.smoothed``.

    ``sigma`` is the Gaussian kernel's standard deviation, in frames (its variance
    is ``sigma**2``); larger values smooth more. Always smooths ΔF/F (``dff``
    defaults to a first-10-frame baseline — see ``Traces``) — never the raw F.

    ``traces.dff`` is left untouched — the smoothed copy lives only in
    ``traces.smoothed``, alongside ``smoothed_sigma`` recording the σ used.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma!r}")
    if traces.dff is None:
        raise ValueError("dff not available (no traces); call compute_dff() first")
    source = traces.dff

    if sigma == 0:
        traces.smoothed = source.copy()
    else:
        from scipy.ndimage import gaussian_filter1d

        traces.smoothed = gaussian_filter1d(source, sigma=sigma, axis=1)
    traces.smoothed_sigma = sigma
    return traces


def onset_time(
    sig: np.ndarray,
    method: str = "fraction_of_max",
    frac: float = 0.5,
    k: float = 3.0,
    d: float = 0.0,
    baseline_region: tuple[int, int] | None = None,
) -> float:
    """Sub-frame index at which a trace first rises from baseline (NaN if none).

    Robust for step-like sustained responses (unlike peak finding); the crossing is
    linearly interpolated between frames.

    method:
      - ``"fraction_of_max"``: threshold = baseline + ``frac`` * (max - baseline),
        with ``frac`` in ``(0, 1]``. ``frac=1`` gives time-to-max.
      - ``"std"``: threshold = baseline_mean + ``k`` * baseline_std.
      - ``"derivative"``: differentiate the trace (``np.gradient``) and cross where
        the rate of change exceeds baseline_derivative_mean + ``k`` *
        baseline_derivative_std + ``d`` — catches the steepest-rise moment
        irrespective of absolute level. Needs >= 2 frames.

    Baseline is measured over ``baseline_region`` ``[start, end)`` if given, else the
    per-method default (trace min for ``fraction_of_max``, first 10% of frames for
    ``std`` and ``derivative``). With ``baseline_region`` the rise is searched only
    from its end onward.
    """

    sig = np.asarray(sig, dtype=float)
    if baseline_region is not None:
        s, e = baseline_region
        base_slice = sig[s:e]
    else:
        base_slice = None
    have_base = base_slice is not None and base_slice.size > 0

    if method == "fraction_of_max":
        base = float(base_slice.mean()) if have_base else float(sig.min())
        peak = float(sig.max())
        amp = peak - base
        if amp <= 0:
            return float("nan")
        # frac == 1 targets the peak; clamp so float error in base + amp can't push
        # the threshold above the attained maximum (which would miss the crossing).
        thresh = min(base + frac * amp, peak)
    elif method == "std":
        base_slice = base_slice if have_base else sig[: max(1, len(sig) // 10)]
        thresh = base_slice.mean() + k * base_slice.std()
    elif method == "derivative":
        # Differentiate, then threshold the rate of change. np.gradient needs >= 2
        # samples; baseline stats read off the same derivative the search sees.
        if sig.size < 2:
            return float("nan")
        sig = np.gradient(sig)
        if have_base and baseline_region is not None:
            base = sig[baseline_region[0]:baseline_region[1]]
        else:
            base = sig[: max(1, len(sig) // 10)]
        thresh = float(base.mean()) + k * float(base.std()) + d
    else:
        raise ValueError(
            f"Unknown onset method {method!r} "
            "(expected 'fraction_of_max' | 'std' | 'derivative')"
        )

    # An onset can only occur after the baseline window: restrict the crossing
    # search to frames from the region's end onward, so a rise within or before the
    # baseline (e.g. a pre-stimulus artifact) can't be picked up. Indices stay in
    # the original trace's frame coordinates.
    start = baseline_region[1] if baseline_region is not None else 0
    above = np.flatnonzero(sig[start:] >= thresh)
    if above.size == 0:
        return float("nan")
    j = start + int(above[0])
    if j == start:
        return float(start)
    y0, y1 = sig[j - 1], sig[j]
    return float(j) if y1 == y0 else (j - 1) + (thresh - y0) / (y1 - y0)


def onset_time_map(
    stack: np.ndarray,
    method: str = "fraction_of_max",
    frac: float = 0.5,
    k: float = 3.0,
    d: float = 0.0,
    baseline_region: tuple[int, int] | None = None,
    bin_size: int = 1,
) -> np.ndarray:
    """Per-pixel ``onset_time`` over a ``[T, Y, X]`` stack → 2D ``[Y, X]`` map.

    Applies the same detector to every pixel's temporal trace, returning onset
    frames (NaN where no rise). ``method``, ``frac``, ``k``, ``d``,
    ``baseline_region`` mean exactly what they do in ``onset_time``.

    bin_size: mean-pool into non-overlapping ``bin_size × bin_size`` blocks first
        (2 ⇒ 2×2 binning), trading resolution for SNR and speed; the map is then
        ``[Y // bin_size, X // bin_size]`` and partial edge blocks are dropped.
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError(f"stack must be [T, Y, X]; got shape {stack.shape}")
    T, Y, X = stack.shape
    b = max(1, int(bin_size))
    if b > 1:
        Yb, Xb = Y // b, X // b
        if Yb == 0 or Xb == 0:
            raise ValueError(f"bin_size {b} larger than the {Y}×{X} frame")
        stack = stack[:, : Yb * b, : Xb * b].reshape(T, Yb, b, Xb, b).mean(axis=(2, 4))
    else:
        Yb, Xb = Y, X
    sig = stack.reshape(T, -1)                         # [T, P] one column per pixel
    P = sig.shape[1]

    # Per-pixel baseline, mirroring onset_time's precedence: explicit region, else
    # the per-method default.
    if baseline_region is not None:
        s, e = baseline_region
        base_slice = sig[s:e]
    else:
        base_slice = None
    have_base = base_slice is not None and base_slice.shape[0] > 0

    if method == "fraction_of_max":
        base = base_slice.mean(axis=0) if have_base else sig.min(axis=0)
        peak = sig.max(axis=0)
        amp = peak - base
        # frac == 1 targets the peak; clamp so float error can't lift the threshold
        # above the attained maximum (which would miss the crossing).
        thresh = np.minimum(base + frac * amp, peak)
        undefined = amp <= 0                            # flat pixel -> no onset
    elif method == "std":
        bs = base_slice if have_base else sig[: max(1, T // 10)]
        thresh = bs.mean(axis=0) + k * bs.std(axis=0)
        undefined = np.zeros(P, dtype=bool)
    elif method == "derivative":
        # Per-pixel rate of change (np.gradient along time); onset = derivative
        # crossing. Mirrors onset_time's derivative branch so map and per-ROI agree.
        if T < 2:
            raise ValueError("derivative onset needs at least 2 frames")
        sig = np.gradient(sig, axis=0)
        if have_base and baseline_region is not None:
            base = sig[baseline_region[0]:baseline_region[1]]
        else:
            base = sig[: max(1, T // 10)]
        thresh = base.mean(axis=0) + k * base.std(axis=0) + d
        undefined = np.zeros(P, dtype=bool)
    else:
        raise ValueError(
            f"Unknown onset method {method!r} "
            "(expected 'fraction_of_max' | 'std' | 'derivative')"
        )

    # First frame at/after the baseline window whose value reaches threshold.
    start = baseline_region[1] if baseline_region is not None else 0
    above = sig[start:] >= thresh[None, :]
    crossed = above.any(axis=0)
    j = start + np.argmax(above, axis=0)                # argmax=0 where never crossed

    onset = np.full(P, np.nan)
    valid = crossed & ~undefined
    cols = np.flatnonzero(valid)
    jj = j[cols]
    thr = thresh[cols]
    y1 = sig[jj, cols]
    y0 = sig[np.clip(jj - 1, 0, T - 1), cols]
    denom = y1 - y0
    # Sub-frame crossing by linear interpolation, except a crossing already at the
    # search start (no earlier sample) or a flat step (denom==0) sits on the frame.
    interp = np.where(denom == 0, jj.astype(float),
                      (jj - 1) + (thr - y0) / np.where(denom == 0, 1.0, denom))
    onset[cols] = np.where(jj == start, float(start), interp)
    return onset.reshape(Yb, Xb)


def cross_roi_propagation(
    traces: Traces,
    rois: list[ROI],
    signal: str = "dff",
    method: str = "fraction_of_max",
    frac: float = 0.5,
    k: float = 3.0,
    d: float = 0.0,
    baseline_region: tuple[int, int] | None = None,
) -> dict:
    """Estimate signal propagation across ROIs from per-ROI onset timing.

    Detects each ROI's onset (``onset_time``) and fits ``onset = a*x + b*y + c``
    over ROI centres. Returns a dict with per-ROI ``onsets``, the earliest
    ``source_roi``, ``speed_px_per_frame``, a ``direction`` unit vector ``(dy, dx)``
    toward later onset, and ``pairwise`` speeds.

    signal: ``"smoothed"`` (``traces.smoothed``), ``"dff"`` (``traces.dff``), or
        ``"raw"`` — each falls back to ``raw`` if the requested array isn't computed.
    method / frac / k / d / baseline_region: passed to ``onset_time``.
    """
    if signal == "smoothed" and traces.smoothed is not None:
        data = traces.smoothed
    elif signal == "dff" and traces.dff is not None:
        data = traces.dff
    else:
        data = traces.raw
    n = data.shape[0]
    if n != len(rois):
        raise ValueError(f"traces ({n}) and rois ({len(rois)}) count mismatch")

    onsets = np.array(
        [onset_time(data[i], method=method, frac=frac, k=k, d=d,
                    baseline_region=baseline_region)
         for i in range(n)]
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
    """Run a user-supplied callable ``f(traces, data) -> result``.

    Full trust, no sandboxing — intended for notebook use.
    """
    return func(traces, data)
