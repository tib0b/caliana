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


def test_no_response_gives_nan_onset():
    flat = np.zeros(40)
    assert np.isnan(onset_time(flat, frac=0.5))


if __name__ == "__main__":
    test_onset_time_recovers_crossing()
    test_propagation_speed_and_direction()
    test_no_response_gives_nan_onset()
    print("propagation tests OK")
