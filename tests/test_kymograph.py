"""Kymographs along a hand-drawn path. SPEC.md §3 Stage III.

`analysis.resample_path` walks a polyline at a fixed pixel spacing;
`analysis.kymograph` samples the stack at those points to build a
distance × time image, optionally as per-position ΔF/F.
`Session.kymograph` runs it over the working stack, honoring the crop window.
Runnable with pytest or as a plain script.
"""
from __future__ import annotations

import numpy as np
import pytest

import caliana
from caliana.analysis import kymograph, resample_path


def _travelling_wave(T=40, Y=20, X=30, speed=2.0, start=5, baseline=10.0, amp=20.0):
    """A step response sweeping left to right across the frame.

    Column ``x`` steps up at frame ``start + x / speed``, so a kymograph along a
    row must show a straight band of exactly that slope. ``start`` keeps every
    column flat over the leading frames, so a first-N baseline is clean whichever
    position it is taken at.
    """
    stack = np.full((T, Y, X), baseline, dtype=float)
    for x in range(X):
        onset = start + int(round(x / speed))
        if onset < T:
            stack[onset:, :, x] += amp
    return stack


# --------------------------------------------------------------------------- #
# Path resampling
# --------------------------------------------------------------------------- #
def test_resample_path_walks_a_straight_line_at_the_step():
    coords, distance = resample_path([(5, 0), (5, 10)], step=1.0)

    assert len(coords) == 11                        # both ends included
    assert np.allclose(coords[:, 0], 5.0)           # the row never moves
    assert np.allclose(coords[:, 1], np.arange(11))
    assert np.allclose(distance, np.arange(11))


def test_resample_path_follows_corners_by_arc_length():
    # An L: 4 px down, then 3 px right -> 7 px of path, 5 px end to end.
    coords, distance = resample_path([(0, 0), (4, 0), (4, 3)], step=1.0)

    assert distance[-1] == pytest.approx(7.0)
    assert tuple(coords[0]) == (0.0, 0.0)
    assert coords[-1] == pytest.approx((4.0, 3.0))
    assert coords[4] == pytest.approx((4.0, 0.0))   # the corner, 4 px along
    assert coords[6] == pytest.approx((4.0, 2.0))
    # Consecutive samples stay one step apart, corner included.
    assert np.allclose(np.hypot(*np.diff(coords, axis=0).T), 1.0)


def test_resample_path_always_lands_on_the_far_end():
    # 10 px of path at step 3 doesn't divide evenly; the last sample is still the end.
    coords, distance = resample_path([(0, 0), (0, 10)], step=3.0)
    assert distance[-1] == pytest.approx(10.0)
    assert coords[-1] == pytest.approx((0.0, 10.0))
    assert np.allclose(distance[:-1], [0, 3, 6, 9])


def test_resample_path_ignores_repeated_points():
    # A double-clicked vertex is a zero-length segment, not a division by zero.
    coords, distance = resample_path([(0, 0), (0, 4), (0, 4), (0, 8)], step=2.0)
    assert distance[-1] == pytest.approx(8.0)
    assert np.allclose(coords[:, 1], [0, 2, 4, 6, 8])


def test_resample_path_rejects_degenerate_input():
    with pytest.raises(ValueError):
        resample_path([(1, 1)])                       # a single point is not a path
    with pytest.raises(ValueError):
        resample_path([(1, 1), (1, 1)])               # zero length
    with pytest.raises(ValueError):
        resample_path([(0, 0), (0, 5)], step=0.0)


# --------------------------------------------------------------------------- #
# Kymograph
# --------------------------------------------------------------------------- #
def test_kymograph_rows_are_the_pixel_traces_along_the_path():
    stack = _travelling_wave()
    result = kymograph(stack, [(10, 0), (10, 29)], step=1.0)

    values = result["values"]
    assert values.shape == (30, len(stack))          # 30 positions × T frames
    assert np.allclose(result["distance"], np.arange(30))
    # Each row is exactly that pixel's temporal trace (the samples sit on centres).
    for x in (0, 7, 29):
        assert np.allclose(values[x], stack[:, 10, x])


def test_kymograph_band_slope_is_the_wave_speed():
    # 2 px/frame: position x rises at frame 5 + x/2, a straight diagonal band whose
    # slope read off the image is the propagation speed.
    stack = _travelling_wave(speed=2.0)
    values = kymograph(stack, [(10, 0), (10, 29)], step=1.0)["values"]

    onsets = np.argmax(values > 20.0, axis=1)        # first frame above baseline
    # Onsets land on whole frames, so the fit carries that rounding and no more.
    slope, intercept = np.polyfit(np.arange(len(onsets)), onsets, 1)
    assert 1.0 / slope == pytest.approx(2.0, abs=0.01)   # px per frame
    assert intercept == pytest.approx(5.0, abs=0.5)      # the wave's start frame


def test_kymograph_samples_between_pixels():
    # A path along a half-row reads the average of the two rows it lies between.
    stack = np.zeros((3, 8, 8))
    stack[:, 2, :] = 10.0
    stack[:, 3, :] = 20.0
    values = kymograph(stack, [(2.5, 1), (2.5, 6)], step=1.0)["values"]
    assert np.allclose(values, 15.0)


def test_kymograph_width_averages_across_the_path():
    # Rows 4/5/6 at 0/30/60; a 3-wide path down the middle reads their mean.
    stack = np.zeros((2, 10, 10))
    for row, value in ((4, 0.0), (5, 30.0), (6, 60.0)):
        stack[:, row, :] = value
    values = kymograph(stack, [(5, 1), (5, 8)], width=3, step=1.0)["values"]
    assert np.allclose(values, 30.0)


def test_kymograph_baseline_gives_per_position_dff():
    stack = _travelling_wave()
    path = [(10, 0), (10, 29)]
    raw = kymograph(stack, path, step=1.0)["values"]
    dff = kymograph(stack, path, step=1.0, baseline=(0, 5))["values"]

    f0 = raw[:, :5].mean(axis=1, keepdims=True)
    assert np.allclose(dff, (raw - f0) / f0)
    # Every position responds by the same fraction, whatever its raw brightness.
    assert np.allclose(dff.max(axis=1)[:10], 2.0)


def test_kymograph_zero_baseline_row_stays_finite():
    stack = np.zeros((6, 8, 8))
    stack[3:, 4, :] = 5.0                            # F0 = 0 on this row
    values = kymograph(stack, [(4, 1), (4, 6)], step=1.0, baseline=(0, 2))["values"]
    assert np.all(np.isfinite(values))
    assert np.all(values == 0.0)


def test_kymograph_records_what_it_measured():
    stack = _travelling_wave()
    result = kymograph(stack, [(10, 0), (10, 20)], width=3, step=2.0, baseline=(0, 4))
    assert result["path"] == [(10.0, 0.0), (10.0, 20.0)]
    assert (result["width"], result["step"], result["baseline"]) == (3, 2.0, (0, 4))
    assert result["coords"].shape == (len(result["distance"]), 2)


def test_kymograph_rejects_a_bad_stack_or_baseline():
    stack = _travelling_wave()
    with pytest.raises(ValueError):
        kymograph(stack[0], [(10, 0), (10, 5)])                  # 2D, not [T, Y, X]
    with pytest.raises(ValueError):
        kymograph(stack, [(10, 0), (10, 5)], baseline=(4, 4))    # selects no frames


def test_kymograph_clamps_a_path_running_off_the_frame():
    # The width around an edge-hugging path would sample outside the frame.
    stack = _travelling_wave()
    values = kymograph(stack, [(0, 0), (0, 29)], width=5, step=1.0)["values"]
    assert values.shape[0] == 30 and np.all(np.isfinite(values))


# --------------------------------------------------------------------------- #
# Session (headless)
# --------------------------------------------------------------------------- #
def test_session_kymograph_stores_the_result():
    s = caliana.Session()
    s.data = _travelling_wave().astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=len(s.data))

    result = s.kymograph([(10, 0), (10, 29)], baseline=(0, 5))
    assert s.analyses["kymograph"] is result
    assert result["values"].shape == (30, len(s.data))
    assert "kymograph" in s.provenance()["analyses"]


def test_session_kymograph_honors_the_crop_window():
    """The kymograph covers the same frame interval as the traces, and its
    baseline window is in those (post-crop) columns."""
    s = caliana.Session()
    s.data = _travelling_wave(T=40).astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=40)
    s.add_roi(center=(10, 15), size=3)
    s.set_crop(10, 30)

    result = s.kymograph([(10, 0), (10, 29)])
    assert result["values"].shape[1] == 20 == s.traces.raw.shape[1]
    assert np.allclose(result["values"][:, 0], s.data[10, 10, :])


def test_session_kymograph_uses_the_stabilized_stack():
    """Like traces and heatmaps, the kymograph reads the working stack — the
    warped one once registration has been applied."""
    s = caliana.Session()
    s.data = _travelling_wave().astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=len(s.data))
    s.registered_data = np.zeros_like(s.data) + 7.0

    result = s.kymograph([(10, 0), (10, 29)])
    assert np.allclose(result["values"], 7.0)


if __name__ == "__main__":
    test_resample_path_walks_a_straight_line_at_the_step()
    test_resample_path_follows_corners_by_arc_length()
    test_resample_path_always_lands_on_the_far_end()
    test_resample_path_ignores_repeated_points()
    test_resample_path_rejects_degenerate_input()
    test_kymograph_rows_are_the_pixel_traces_along_the_path()
    test_kymograph_band_slope_is_the_wave_speed()
    test_kymograph_samples_between_pixels()
    test_kymograph_width_averages_across_the_path()
    test_kymograph_baseline_gives_per_position_dff()
    test_kymograph_zero_baseline_row_stays_finite()
    test_kymograph_records_what_it_measured()
    test_kymograph_rejects_a_bad_stack_or_baseline()
    test_kymograph_clamps_a_path_running_off_the_frame()
    test_session_kymograph_stores_the_result()
    test_session_kymograph_honors_the_crop_window()
    test_session_kymograph_uses_the_stabilized_stack()
    print("kymograph test OK")
