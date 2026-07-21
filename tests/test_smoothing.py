"""Default ΔF/F baseline and Gaussian trace smoothing. SPEC.md §3 Stage III.

`Traces.dff` defaults to a first-10-frame-baseline ΔF/F as soon as `raw` is set
(`Traces.__post_init__`), so trace analysis is available without an explicit
`compute_dff()` call. `analysis.smooth_traces` / `Session.smooth_traces` always
low-pass `traces.dff` (never `raw`) into `traces.smoothed`, a separate array that
never overwrites the original. Runnable with pytest or as a plain script.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

import caliana
from caliana.analysis import compute_dff, smooth_traces
from caliana.models import BaselineMethod, DEFAULT_DFF_BASELINE_FRAMES, Traces


def _noisy_step(T, onset, baseline=10.0, amp=5.0, noise=0.5, seed=0):
    """A positive-baseline step response, safely away from zero so (F-F0)/F0
    doesn't blow up when F0 is the pre-step baseline."""
    rng = np.random.default_rng(seed)
    sig = np.where(np.arange(T) >= onset, baseline + amp, baseline)
    return sig + rng.normal(0, noise, T)


# --------------------------------------------------------------------------- #
# Default ΔF/F baseline
# --------------------------------------------------------------------------- #
def test_traces_dff_defaults_to_first_10_frame_baseline():
    T = 50
    raw = np.stack([_noisy_step(T, 30, seed=i) for i in range(2)])
    traces = Traces(raw=raw.copy(), labels=["a", "b"])

    f0 = raw[:, :DEFAULT_DFF_BASELINE_FRAMES].mean(axis=1, keepdims=True)
    expected = (raw - f0) / f0
    assert traces.dff is not None
    assert np.allclose(traces.dff, expected)


def test_traces_dff_short_recording_uses_all_frames_as_baseline():
    # Fewer than 10 frames: the whole trace is the baseline window.
    raw = np.array([[1.0, 2.0, 3.0, 4.0]])
    traces = Traces(raw=raw.copy())
    f0 = raw.mean(axis=1, keepdims=True)
    assert np.allclose(traces.dff, (raw - f0) / f0)


def test_traces_empty_raw_leaves_dff_none():
    # No ROIs -> raw is 0-row; there's nothing to baseline against.
    traces = Traces(raw=np.empty((0, 20)), labels=[])
    assert traces.dff is None


def test_traces_explicit_dff_is_not_overwritten():
    raw = np.array([[1.0, 2.0, 3.0]])
    custom_dff = np.array([[0.1, 0.2, 0.3]])
    traces = Traces(raw=raw, dff=custom_dff)
    assert traces.dff is custom_dff


def test_compute_dff_overrides_the_default():
    T = 50
    raw = np.stack([_noisy_step(T, 30, seed=i) for i in range(2)])
    traces = Traces(raw=raw.copy())
    default_dff = traces.dff.copy()

    compute_dff(traces, method=BaselineMethod.FIRST_N, n=5)

    assert not np.allclose(traces.dff, default_dff)     # different baseline window
    f0 = raw[:, :5].mean(axis=1, keepdims=True)
    assert np.allclose(traces.dff, (raw - f0) / f0)


# --------------------------------------------------------------------------- #
# Gaussian smoothing (always on ΔF/F)
# --------------------------------------------------------------------------- #
def test_smooth_traces_reduces_noise_without_touching_dff():
    T = 200
    raw = np.stack([_noisy_step(T, 80, seed=i) for i in range(3)])
    traces = Traces(raw=raw.copy(), labels=["a", "b", "c"])
    dff_before = traces.dff.copy()

    smooth_traces(traces, sigma=3.0)

    assert np.array_equal(traces.dff, dff_before)        # dff untouched
    assert traces.smoothed is not None
    assert traces.smoothed.shape == traces.dff.shape
    assert traces.smoothed_sigma == 3.0

    # Smoothing should reduce frame-to-frame jitter (variance of the first
    # difference) relative to the unsmoothed ΔF/F.
    dff_jitter = np.diff(traces.dff, axis=1).var()
    smooth_jitter = np.diff(traces.smoothed, axis=1).var()
    assert smooth_jitter < dff_jitter


def test_smooth_traces_sigma_zero_is_identity():
    raw = np.array([[1.0, 5.0, 2.0, 8.0]])
    traces = Traces(raw=raw.copy())
    smooth_traces(traces, sigma=0.0)
    assert np.array_equal(traces.smoothed, traces.dff)


def test_smooth_traces_negative_sigma_rejected():
    traces = Traces(raw=np.stack([_noisy_step(20, 10, seed=0)]))
    with pytest.raises(ValueError):
        smooth_traces(traces, sigma=-1.0)


def test_smooth_traces_requires_dff_present():
    # No ROIs -> raw is empty -> no default dff -> nothing to smooth.
    traces = Traces(raw=np.empty((0, 20)), labels=[])
    assert traces.dff is None
    with pytest.raises(ValueError):
        smooth_traces(traces, sigma=1.0)


def test_smooth_traces_never_reads_raw():
    """Smoothing ΔF/F that diverges wildly from raw (e.g. after compute_dff with a
    different baseline) proves the filter runs on dff, not raw."""
    T = 100
    raw = np.stack([_noisy_step(T, 40, seed=0)])
    traces = Traces(raw=raw.copy())
    compute_dff(traces, method=BaselineMethod.FIRST_N, n=5)
    smooth_traces(traces, sigma=2.0)

    from scipy.ndimage import gaussian_filter1d
    expected = gaussian_filter1d(traces.dff, sigma=2.0, axis=1)
    assert np.allclose(traces.smoothed, expected)
    assert not np.allclose(traces.smoothed, raw, atol=1e-3)


# --------------------------------------------------------------------------- #
# Session (headless)
# --------------------------------------------------------------------------- #
def test_session_extract_traces_populates_default_dff():
    s = caliana.Session()
    s.data = np.random.default_rng(1).random((60, 20, 20)).astype(np.float32) + 1.0
    s.timeline = caliana.Timeline(n_frames=60)
    s.add_roi(center=(10, 10), size=4, label="c")

    traces = s.extract_traces()
    assert traces.dff is not None                        # available with no compute_dff() call


def test_session_smooth_traces_headless():
    """`Session.smooth_traces` is callable without any widget, extracts traces on
    demand, and always smooths ΔF/F into its own array."""
    s = caliana.Session()
    s.data = np.random.default_rng(1).random((60, 20, 20)).astype(np.float32) + 1.0
    s.timeline = caliana.Timeline(n_frames=60)
    s.add_roi(center=(10, 10), size=4, label="c")

    assert s.traces is None                              # not extracted yet
    result = s.smooth_traces(sigma=2.0)                   # auto-extracts
    assert result is s.traces
    assert s.traces.dff is not None
    assert s.traces.smoothed is not None
    assert s.traces.smoothed.shape == s.traces.dff.shape
    assert s.traces.smoothed_sigma == 2.0
    assert not np.array_equal(s.traces.smoothed, s.traces.dff)  # actually smoothed


# --------------------------------------------------------------------------- #
# CSV export
# --------------------------------------------------------------------------- #
def test_export_includes_dff_and_smoothed_columns_when_present():
    s = caliana.Session()
    s.data = np.random.default_rng(3).random((30, 12, 12)).astype(np.float32) + 1.0
    s.timeline = caliana.Timeline(n_frames=30)
    s.add_roi(center=(6, 6), size=3, label="roi0")
    s.extract_traces()
    s.smooth_traces(sigma=1.0)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "traces.csv"
        s.export_traces(path)
        header = next(csv.reader(path.open()))
    assert "roi0_F" in header
    assert "roi0_dFF" in header          # present by default, no compute_dff() call
    assert "roi0_smoothed" in header


def test_export_omits_smoothed_column_when_absent():
    s = caliana.Session()
    s.data = np.random.default_rng(4).random((10, 8, 8)).astype(np.float32) + 1.0
    s.timeline = caliana.Timeline(n_frames=10)
    s.add_roi(center=(4, 4), size=2, label="roi0")
    s.extract_traces()

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "traces.csv"
        s.export_traces(path)
        header = next(csv.reader(path.open()))
    assert "roi0_dFF" in header          # default dff still exported
    assert "roi0_smoothed" not in header  # never smoothed -> no column


if __name__ == "__main__":
    test_traces_dff_defaults_to_first_10_frame_baseline()
    test_traces_dff_short_recording_uses_all_frames_as_baseline()
    test_traces_empty_raw_leaves_dff_none()
    test_traces_explicit_dff_is_not_overwritten()
    test_compute_dff_overrides_the_default()
    test_smooth_traces_reduces_noise_without_touching_dff()
    test_smooth_traces_sigma_zero_is_identity()
    test_smooth_traces_negative_sigma_rejected()
    test_smooth_traces_requires_dff_present()
    test_smooth_traces_never_reads_raw()
    test_session_extract_traces_populates_default_dff()
    test_session_smooth_traces_headless()
    test_export_includes_dff_and_smoothed_columns_when_present()
    test_export_omits_smoothed_column_when_absent()
    print("smoothing test OK")
