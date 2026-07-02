"""Smoke test for the headless Caliana spine. SPEC.md §3.

Exercises load -> downsample -> ROI -> trace -> ΔF/F -> peaks -> provenance
against the repo's synthetic_calcium_imaging.tif. Runnable with pytest or as a
plain script (`python tests/test_smoke.py`).
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

import caliana

TIF = Path(__file__).resolve().parent / "data" / "synthetic_calcium_imaging.tif"


def test_spine():
    s = caliana.Session.from_file(TIF, temporal_step=2)
    assert s.data is not None and s.data.ndim == 3
    T = len(s.data)

    # Max-projection heatmap is normalized to [0, 1] (SPEC §3 Stage I).
    mip = s.max_projection()
    assert mip.shape == s.data.shape[1:]
    assert 0.0 <= float(mip.min()) and float(mip.max()) <= 1.0 + 1e-6

    # Place a couple of ROIs and extract traces.
    h, w = s.data.shape[1:]
    s.add_roi(center=(h / 2, w / 2), size=4, label="centre")
    s.add_roi(center=(h / 4, w / 4), size=4, label="corner")
    traces = s.extract_traces()
    assert traces.raw.shape == (2, T)

    # ΔF/F with a first-N-frames baseline (SPEC §3 Stage III).
    s.compute_dff(n=min(10, T))
    assert s.traces.dff.shape == (2, T)

    # Peak detection runs and returns a result per ROI.
    peaks = s.detect_peaks()
    assert len(peaks) == 2

    # Provenance is JSON-serializable and records the import params.
    prov = s.provenance()
    assert prov["source"]["import_params"]["temporal_step"] == 2
    assert len(prov["rois"]) == 2

    # Round-trip the CSV + provenance exports.
    with tempfile.TemporaryDirectory() as d:
        s.export_traces(Path(d) / "traces.csv")
        s.export_provenance(Path(d) / "prov.json")
        assert (Path(d) / "traces.csv").stat().st_size > 0
        assert (Path(d) / "prov.json").stat().st_size > 0


def test_crop_time_axis_is_offset():
    """A crop window shifts the reported frame/second axis by its start, and the
    time axis length always matches the (cropped) trace length — so calibrated
    figures don't raise and the CSV labels the true recording frames."""
    s = caliana.Session()
    s.data = np.random.default_rng(0).random((100, 12, 12))
    s.timeline = caliana.Timeline(n_frames=100)
    s.add_roi(center=(6, 6), size=3, label="c")
    s.set_frame_interval(2.0)      # seconds axis
    s.set_crop(20, 60)             # 40-frame window
    s.compute_dff(n=5)

    frames = s.trace_frames()
    assert (frames[0], frames[-1], len(frames)) == (20, 59, 40)
    assert len(frames) == s.traces.raw.shape[1]   # axis matches trace length

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "traces.csv"
        s.export_traces(path)
        rows = list(csv.reader(path.open()))
    assert rows[0][:2] == ["frame", "seconds"]
    assert rows[1][:2] == ["20", "40.0"]          # first cropped frame -> 40 s
    assert rows[-1][:2] == ["59", "118.0"]        # last cropped frame -> 118 s


if __name__ == "__main__":
    test_spine()
    test_crop_time_axis_is_offset()
    print("smoke test OK")
