"""Cross-ROI propagation. SPEC.md §3 Stage III.

Builds step-like ΔF/F traces whose onsets follow a known linear arrival-time
field over ROI coordinates, then checks the recovered speed, direction, and
source. Runnable with pytest or as a script.
"""
from __future__ import annotations

import numpy as np

from caliana.analysis import cross_roi_propagation, onset_time
from caliana.models import ROI, ROIShape, Traces


def _logistic_step(T, onset, tau=1.0):
    t = np.arange(T)
    return 1.0 / (1.0 + np.exp(-(t - onset) / tau))


def test_onset_time_recovers_crossing():
    sig = _logistic_step(60, onset=18.0)
    assert abs(onset_time(sig, frac=0.5) - 18.0) < 0.5


def test_fraction_of_max_frac_one_returns_time_to_peak():
    # frac=1 targets the maximum itself, so the onset is the frame the trace peaks.
    sig = np.array([0.0, 1.0, 2.0, 5.0, 3.0, 2.0])   # peak at frame 3
    assert onset_time(sig, method="fraction_of_max", frac=1.0) == 3.0
    # A non-integer baseline still resolves the peak frame exactly (no float miss).
    ramp = np.linspace(0.1, 0.3, 50)                 # peak at the last frame
    assert onset_time(ramp, method="fraction_of_max", frac=1.0) == float(len(ramp) - 1)


def test_propagation_speed_and_direction():
    # Known arrival-time field: onset = 5 + 0.2*x + 0.1*y  -> slowness (dy,dx)=(0.1,0.2)
    coords = [(10, 10), (10, 30), (30, 10), (30, 30)]  # (y, x)
    onsets = [5 + 0.2 * x + 0.1 * y for (y, x) in coords]
    T = 80
    dff = np.stack([_logistic_step(T, o, tau=0.6) for o in onsets])
    traces = Traces(raw=dff.copy(), dff=dff, labels=[f"r{i}" for i in range(4)])
    rois = [ROI(center=c, size=3, shape=ROIShape.CIRCLE) for c in coords]

    res = cross_roi_propagation(traces, rois, signal="dff", frac=0.5)

    expected_speed = 1.0 / np.hypot(0.2, 0.1)        # ~4.47 px/frame
    assert abs(res["speed_px_per_frame"] - expected_speed) / expected_speed < 0.1

    exp_dir = np.array([0.1, 0.2]) / np.hypot(0.2, 0.1)   # (dy, dx) unit
    got = np.array(res["direction"])
    assert np.dot(got, exp_dir) > 0.99               # near-parallel

    assert res["source_roi"] == 0                    # earliest onset at (10,10)
    assert len(res["pairwise"]) == 6                 # 4 choose 2


def test_propagation_two_rois_speed_and_direction():
    # Two ROIs 20 px apart in x; the second responds 10 frames later, so the
    # signal propagates in +x at 20/10 = 2 px/frame. Exercises the nv==2 branch
    # (a single pair -> direction from the ROI-to-ROI line, not a plane fit).
    coords = [(10, 10), (10, 30)]           # (y, x): same row, dx = 20
    onsets = [10.0, 20.0]
    T = 60
    dff = np.stack([_logistic_step(T, o, tau=0.6) for o in onsets])
    traces = Traces(raw=dff.copy(), dff=dff, labels=["a", "b"])
    rois = [ROI(center=c, size=3, shape=ROIShape.CIRCLE) for c in coords]

    res = cross_roi_propagation(traces, rois, signal="dff", frac=0.5)

    assert res["source_roi"] == 0            # earliest onset at (10, 10)
    assert len(res["pairwise"]) == 1         # 2 choose 2
    assert abs(res["speed_px_per_frame"] - 2.0) / 2.0 < 0.1
    # Direction points from the earlier to the later ROI: +x, no y component.
    got = np.array(res["direction"])
    assert np.dot(got, [0.0, 1.0]) > 0.99


def test_propagation_two_rois_direction_flips_with_order():
    # Same geometry, but now the *first* ROI responds later. The direction must
    # flip to point toward the later onset (−x), covering the delta_t < 0 branch.
    coords = [(10, 10), (10, 30)]
    onsets = [20.0, 10.0]                    # roi 0 later than roi 1
    T = 60
    dff = np.stack([_logistic_step(T, o, tau=0.6) for o in onsets])
    traces = Traces(raw=dff.copy(), dff=dff, labels=["a", "b"])
    rois = [ROI(center=c, size=3, shape=ROIShape.CIRCLE) for c in coords]

    res = cross_roi_propagation(traces, rois, signal="dff", frac=0.5)

    assert res["source_roi"] == 1            # earliest onset is now roi 1
    got = np.array(res["direction"])
    assert np.dot(got, [0.0, -1.0]) > 0.99   # points back toward −x (roi 0, later)


def test_no_response_gives_nan_onset():
    flat = np.zeros(40)
    assert np.isnan(onset_time(flat, frac=0.5))


def test_onset_baseline_region_sets_threshold():
    # On a steady ramp the onset is purely threshold-driven, so a higher baseline
    # window raises the threshold and pushes the crossing later.
    sig = np.linspace(0.0, 10.0, 100)
    early = onset_time(sig, method="fraction_of_max", frac=0.5, baseline_region=(0, 10))
    late = onset_time(sig, method="fraction_of_max", frac=0.5, baseline_region=(80, 90))
    assert late > early
    # The recovered onset sits where the ramp crosses base + frac*(max - base).
    base = sig[0:10].mean()
    thresh = base + 0.5 * (sig.max() - base)
    assert abs(sig[int(round(early))] - thresh) < 0.2


def test_onset_only_after_baseline_region():
    # A pre-baseline artifact would trip an early crossing if the whole trace were
    # searched; restricting to after the baseline window finds the real rise.
    sig = np.zeros(60)
    sig[:5] = 10.0          # artifact before the baseline
    sig[40:] = 10.0         # the real response
    t = onset_time(sig, method="fraction_of_max", frac=0.5, baseline_region=(10, 30))
    assert t >= 30                     # onset cannot fall within/before the baseline
    assert abs(t - 39.5) < 1.0         # picks up the rise at frame 40
    # Without the region the early artifact is (wrongly) detected as the onset.
    assert onset_time(sig, method="fraction_of_max", frac=0.5) < 5


if __name__ == "__main__":
    test_onset_time_recovers_crossing()
    test_fraction_of_max_frac_one_returns_time_to_peak()
    test_propagation_speed_and_direction()
    test_propagation_two_rois_speed_and_direction()
    test_propagation_two_rois_direction_flips_with_order()
    test_no_response_gives_nan_onset()
    test_onset_baseline_region_sets_threshold()
    test_onset_only_after_baseline_region()
    print("propagation tests OK")
