"""Stage I preview widget smoke test (offscreen). SPEC.md §3 Stage I.

Skipped if the GUI stack (qtpy + a Qt binding + pyqtgraph) is not installed.
Run headless with: QT_QPA_PLATFORM=offscreen python tests/test_widgets.py
"""
from __future__ import annotations

import os

import numpy as np

import caliana

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from caliana.widgets._qt import ensure_app
    from caliana.widgets.import_widget import ImportPreviewWidget
    from caliana.widgets.leaf_widget import LeafSelectionWidget
    from caliana.widgets.roi_widget import RoiSelectionWidget
    from caliana.widgets.crop_widget import CropTracesWidget
    from caliana.widgets.analysis_widget import AnalysisWidget
    HAVE_GUI = True
except Exception:  # pragma: no cover - depends on optional deps
    HAVE_GUI = False


def _session():
    s = caliana.Session()
    s.data = (np.random.default_rng(0).random((8, 32, 24)) * 255).astype(np.uint16)
    s.timeline = caliana.Timeline(n_frames=8)
    return s


def test_import_preview_widget():
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    w = ImportPreviewWidget(s)

    # The widget mirrors the session into both views.
    assert w.result is s
    assert w.movie.image is not None
    assert w.heatmap.image.shape == s.data.shape[1:]

    # Playback controls drive without error.
    w.play_btn.setChecked(True)
    w.play_btn.setChecked(False)
    w._on_time_changed(3, 0.0)
    assert "frame 3" in w.frame_label.text()

    w.close()
    print("import preview widget OK")


def test_roi_selection_widget():
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    w = RoiSelectionWidget(s)

    # Placing ROIs updates the session and draws a graphic + a live trace each.
    w.add_roi_at(10, 12)
    w.size_box.setValue(3)
    w.shape_box.setCurrentText("square")
    sq = w.add_roi_at(20, 8)
    assert len(s.rois) == 2
    assert sq.shape.value == "square" and sq.size == 3
    assert len(w._roi_records) == 2
    assert s.traces.raw.shape[0] == 2  # live preview recomputed traces

    # Moving an ROI's graphic updates its model centre (read back from the item).
    rec = w._roi_records[0]
    rec["item"].setPos(2, 3)
    pos, size = rec["item"].pos(), rec["item"].size()
    cy, cx = rec["model"].center
    assert abs(cx - (pos.x() + size.x() / 2)) < 1e-6
    assert abs(cy - (pos.y() + size.y() / 2)) < 1e-6

    # Size/shape are shared: changing them updates every existing ROI.
    w.size_box.setValue(6)
    assert all(r.size == 6 for r in s.rois)
    assert all(abs(rec["item"].size().x() - 12) < 1e-6 for rec in w._roi_records)
    w.shape_box.setCurrentText("circle")
    assert all(r.shape.value == "circle" for r in s.rois)

    # Deleting removes graphic + model.
    w.delete_last_roi()
    assert len(s.rois) == 1 and len(w._roi_records) == 1

    w.close()
    print("roi selection widget OK")


def test_leaf_selection_widget():
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    w = LeafSelectionWidget(s)

    # Adding leaf boxes flows into the session and draws a graphic each.
    w.add_leaf_box((0, 16, 0, 12))
    leaf = w.add_leaf_box((10, 31, 8, 23))
    assert len(s.leaf_regions) == 2
    assert len(w._leaf_records) == 2
    assert leaf.bbox == (10, 31, 8, 23)

    # Moving a leaf box's graphic updates its model bbox (read back from item).
    rec = w._leaf_records[0]
    rec["item"].setPos(2, 3)
    pos, size = rec["item"].pos(), rec["item"].size()
    y0, y1, x0, x1 = rec["model"].bbox
    assert (y0, x0) == (int(pos.y()), int(pos.x()))
    assert (y1, x1) == (int(pos.y() + size.y()), int(pos.x() + size.x()))

    # Deleting removes graphic + model.
    w.delete_last_leaf()
    assert len(s.leaf_regions) == 1 and len(w._leaf_records) == 1

    w.close()
    print("leaf selection widget OK")


def test_roi_widget_shows_leaf_reference():
    """Leaf boxes drawn elsewhere appear in the ROI widget as non-interactive
    reference, and clicks inside them still place ROIs (no click stealing)."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_leaf_region((0, 31, 0, 23))
    w = RoiSelectionWidget(s)

    # The leaf box is shown for reference but is not a movable pg.ROI.
    assert len(w._leaf_records) == 1
    assert not isinstance(w._leaf_records[0]["item"], pg.ROI)

    # A click inside the leaf box still places an ROI.
    w.add_roi_at(15, 11)
    assert len(s.rois) == 1

    w.close()
    print("roi widget leaf-reference OK")


def test_roi_widget_freehand():
    """Free-hand mode traces a polygon outline and commits it as a polygon ROI,
    editable afterwards and left out of the shared size/shape controls."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana.models import ROIShape

    s = _session()
    w = RoiSelectionWidget(s)

    # Enter free-hand mode, trace an outline, finish.
    w.poly_btn.setChecked(True)
    for row, col in [(4, 4), (4, 18), (20, 18), (20, 4)]:
        w.add_polygon_point(row, col)
    assert w._poly_preview is not None
    w.finish_polygon()                                   # unchecks -> commits

    assert not w.poly_btn.isChecked() and w._poly_preview is None
    assert len(s.rois) == 1
    roi = s.rois[0]
    assert roi.shape == ROIShape.POLYGON and len(roi.vertices) == 4
    rec = w._roi_records[0]
    assert isinstance(rec["item"], pg.PolyLineROI)
    assert s.traces.raw.shape[0] == 1                    # live preview recomputed

    # The graphic reports its vertices back in image (y, x) coordinates.
    verts = w._polygon_item_vertices(rec["item"])
    assert {(round(y), round(x)) for y, x in verts} == {(4, 4), (4, 18), (20, 18), (20, 4)}

    # Shared size/shape controls leave polygon ROIs untouched.
    w.size_box.setValue(9)
    w.shape_box.setCurrentText("square")
    assert roi.shape == ROIShape.POLYGON

    # A finish with too few points discards rather than committing.
    w.poly_btn.setChecked(True)
    w.add_polygon_point(2, 2)
    w.finish_polygon()
    assert len(s.rois) == 1 and w._poly_preview is None

    w.close()
    print("roi widget freehand OK")


def test_roi_widget_track_motion():
    """Track-motion mode shows raw footage and moves ROI markers to follow the
    tissue per frame, without disturbing the stored ROI centres."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana.models import RegistrationMode, RegistrationResult, RigidTransform

    s = _session()
    n = len(s.data)
    # Whole-frame transforms: frame k shifts the tissue by dx = k (dy = 0).
    s.registration = RegistrationResult(
        mode=RegistrationMode.WHOLE_FRAME, reference="mean",
        transforms=[RigidTransform(dy=0.0, dx=float(k)) for k in range(n)],
    )
    roi = s.add_roi(center=(16, 12), size=4)
    poly = s.add_polygon_roi([(10, 6), (10, 10), (14, 10), (14, 6)])  # centroid (12, 8)
    w = RoiSelectionWidget(s)

    # Transforms exist -> the toggle is available; off by default.
    assert w.track_box.isEnabled() and not w._tracking
    rec = w._roi_records[0]
    r = rec["item"].size().x() / 2
    assert abs((rec["item"].pos().x() + r) - 12) < 1e-6   # at model centre

    # Turn tracking on: editing is paused and the geometry resolves correctly.
    w.track_box.setChecked(True)
    assert w._tracking and not w.size_box.isEnabled() and not w.del_btn.isEnabled()
    cy, cx = w._roi_raw_center(roi, 3)
    assert abs(cy - 16) < 1e-6 and abs(cx - (12 + 3)) < 1e-6

    # Scrubbing to frame 3 moves the marker; the stored centre is untouched.
    w._on_frame_changed(3, 0.0)
    assert abs((rec["item"].pos().x() + r) - 15) < 1e-6
    assert roi.center == (16, 12)

    # A polygon ROI tracks too: every vertex shifts by dx = 3 at frame 3.
    prec = w._roi_records[1]
    raw_verts = w._roi_raw_vertices(poly, 3)
    assert {(round(y), round(x)) for y, x in raw_verts} == {(10, 9), (10, 13), (14, 13), (14, 9)}
    disp = {(round(y), round(x)) for y, x in w._polygon_item_vertices(prec["item"])}
    assert disp == {(10, 9), (10, 13), (14, 13), (14, 9)}
    assert poly.vertices == [(10, 6), (10, 10), (14, 10), (14, 6)]   # model untouched

    # Turn tracking off: markers return to model geometry, editing re-enabled.
    w.track_box.setChecked(False)
    assert not w._tracking and w.size_box.isEnabled()
    assert abs((rec["item"].pos().x() + r) - 12) < 1e-6
    assert {(round(y), round(x)) for y, x in w._polygon_item_vertices(prec["item"])} \
        == {(10, 6), (10, 10), (14, 10), (14, 6)}

    w.close()
    print("roi widget track-motion OK")


def test_crop_traces_widget():
    """Selecting a window crops every trace to that interval and feeds the same
    window to downstream extraction / the analysis widget."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_roi(center=(16, 12), size=4)
    s.add_roi(center=(8, 6), size=3)
    w = CropTracesWidget(s)

    # Full recording is previewed; the window spans everything by default.
    T = len(s.data)
    assert w._preview.raw.shape == (2, T)
    assert w.start_box.value() == 0 and w.end_box.value() == T

    # Region and spinboxes stay in sync (both directions).
    w.set_interval(2, 6)
    assert tuple(round(v) for v in w.region.getRegion()) == (2, 6)
    w.region.setRegion((3, 7))
    assert (w.start_box.value(), w.end_box.value()) == (3, 7)

    # Validating crops the traces and returns them; the session agrees.
    w.set_interval(2, 6)
    cropped = w.apply_crop()
    assert s.crop_window == (2, 6)
    assert cropped.raw.shape == (2, 4)
    assert cropped is s.traces

    # The crop is honored on re-extraction (so `analyze` sees the same window).
    assert s.extract_traces().raw.shape == (2, 4)
    a = AnalysisWidget(s)
    assert a.session.traces.raw.shape[1] == 4
    # The analysis widget plots in original frame coordinates, so its windows and
    # event range start at the crop start (2), not 0.
    assert round(a.region.getRegion()[0]) == 2
    assert a.event_box.minimum() == 2
    xs = a._curves[0].getData()[0]
    assert (int(xs[0]), int(xs[-1])) == (2, 5)   # frames 2..5 for a [2, 6) crop
    a.close()

    # Reset clears the crop back to the whole recording.
    w2 = CropTracesWidget(s)
    assert w2.start_box.value() == 2 and w2.end_box.value() == 6  # reflects session
    full = w2.reset_crop()
    assert s.crop_window is None and full.raw.shape == (2, T)

    w.close()
    w2.close()
    print("crop traces widget OK")


def test_analysis_widget():
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_roi(center=(16, 12), size=4)
    s.add_roi(center=(8, 6), size=3)
    w = AnalysisWidget(s)

    # ΔF/F (first-N) computes and flips the display to ΔF/F.
    w.n_box.setValue(4)
    w.compute_dff()
    assert s.traces.dff is not None
    assert w.show_dff.isChecked()
    data, _ = w._display_data()
    assert data is s.traces.dff

    # REGION baseline uses the draggable window bounds.
    w.baseline_box.setCurrentText("region")
    w.region.setRegion((1, 5))
    w.compute_dff()
    assert s.traces.dff is not None

    # Event markers land on the timeline and draw a line.
    w.add_event(3)
    assert len(s.timeline.events) == 1 and s.timeline.events[0].frame == 3
    assert len(w._event_lines) == 1

    # Propagation now returns a real result and overlays per-ROI onset markers.
    prop = w.compute_propagation()
    assert prop is not None and "onsets" in prop
    assert "propagation" in s.analyses
    assert "Propagation" in w.results.toPlainText()

    w.close()
    print("analysis widget OK")


def test_analysis_widget_smoothing():
    """Gaussian smoothing always acts on ΔF/F (available by default, no
    "Compute ΔF/F" click needed) and is stored separately, toggled via its own
    checkbox; it never overwrites raw or ΔF/F."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_roi(center=(16, 12), size=4)
    w = AnalysisWidget(s)

    assert s.traces.dff is not None                     # default first-10-frame baseline
    dff_before = s.traces.dff.copy()
    raw_before = s.traces.raw.copy()
    w.smooth_sigma_box.setValue(1.5)
    w.smooth_traces()

    assert s.traces.smoothed is not None
    assert s.traces.smoothed.shape == s.traces.dff.shape
    assert s.traces.smoothed_sigma == 1.5
    assert np.array_equal(s.traces.raw, raw_before)     # raw untouched
    assert np.array_equal(s.traces.dff, dff_before)     # dff untouched
    assert w.show_smoothed.isChecked()                  # auto-enabled after smoothing
    data, _ = w._display_data()
    assert data is s.traces.smoothed

    # Toggling off falls back to raw (the default display).
    w.show_smoothed.setChecked(False)
    data, _ = w._display_data()
    assert data is s.traces.raw

    w.close()
    print("analysis widget smoothing OK")


def test_analysis_widget_analysis_selection():
    """Picking an analysis type shows only that analysis' controls; the onset
    method toggles which propagation parameter is active; and propagation draws
    the onset-vs-distance graph."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    for center, size in [((16, 12), 4), ((8, 6), 3), ((24, 18), 3)]:
        s.add_roi(center=center, size=size)
    w = AnalysisWidget(s)

    # Nothing chosen yet: the empty stack page is shown and the graph + baseline
    # band are hidden.
    assert w.param_stack.currentIndex() == 0
    assert w.prop_plot.isHidden()
    assert w.prop_region.scene() is None

    # Selecting propagation reveals its panel, the graph, and the onset-baseline band.
    w.analysis_box.setCurrentText("Cross-ROI propagation")
    assert w.param_stack.currentIndex() == 1 and not w.prop_plot.isHidden()
    assert w.prop_region.scene() is not None  # baseline band on the trace plot

    # Onset method gates frac (fraction_of_max) vs k (std).
    w.onset_method_box.setCurrentText("fraction_of_max")
    assert w.frac_box.isEnabled() and not w.k_box.isEnabled()
    w.onset_method_box.setCurrentText("std")
    assert not w.frac_box.isEnabled() and w.k_box.isEnabled()

    # Running propagation (baseline = a dragged leading window) overlays onsets and
    # populates the onset-vs-distance graph.
    w.onset_method_box.setCurrentText("fraction_of_max")
    w.prop_region.setRegion((0, 2))
    res = w.compute_propagation()
    assert res is not None and len(w._onset_lines) >= 1
    assert len(w.prop_plot.getPlotItem().listDataItems()) >= 1

    w.close()
    print("analysis widget selection OK")


def test_analysis_widget_onset_heatmap():
    """The Heatmaps page runs the per-pixel onset detector and shows a map that
    agrees, pixel for pixel, with the per-ROI onset_time."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana import analysis

    # Data with a vertical onset gradient: lower rows respond later.
    rng = np.random.default_rng(2)
    T, Y, X = 30, 16, 12
    stack = rng.normal(10, 0.5, (T, Y, X))
    for y in range(Y):
        stack[5 + y:, y, :] += 20.0
    s = caliana.Session()
    s.data = stack.astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=T)
    s.add_roi(center=(4, 6), size=3)

    w = AnalysisWidget(s)
    assert w.tabs.count() == 2 and w.tabs.tabText(1) == "Heatmaps"

    # fraction_of_max over the whole trace (no baseline window) matches onset_time per pixel.
    w.hm_base_start.setValue(0)
    w.hm_base_end.setValue(0)             # empty window ⇒ default (trace-min) baseline
    mp = w.compute_onset_heatmap()
    assert mp.shape == (Y, X)
    ref = analysis.onset_time(s.data[:, 4, 6].astype(float), method="fraction_of_max", frac=0.5)
    assert abs(float(mp[4, 6]) - float(ref)) < 1e-6
    # Onset increases down the frame (the injected gradient).
    per_row = np.nanmean(mp, axis=1)
    assert per_row[0] < per_row[-1]

    # n×n binning lowers the map resolution.
    w.hm_bin_box.setValue(2)
    mp2 = w.compute_onset_heatmap()
    assert mp2.shape == (Y // 2, X // 2)

    # Method toggle gates frac (fraction_of_max) vs k (std), as on the propagation panel.
    w.hm_method_box.setCurrentText("std")
    assert w.hm_k_box.isEnabled() and not w.hm_frac_box.isEnabled()

    # Calibration flips the colour-scale label to seconds.
    s.set_frame_interval(0.5)
    w.interval_box.setValue(0.5)
    w.hm_method_box.setCurrentText("fraction_of_max")
    w.compute_onset_heatmap()
    assert w.hm_cbar.getAxis("left").labelText == "onset (s)"

    w.close()
    print("analysis widget onset heatmap OK")


def test_analysis_widget_propagation_uses_displayed_signal():
    """Propagation detects onsets on whatever the trace plot shows: with 'Show
    smoothed ΔF/F' checked it reads traces.smoothed, otherwise traces.dff."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana import analysis

    # Three ROIs, each stepping up at a different frame -> real ΔF/F onsets.
    rng = np.random.default_rng(5)
    T, Y, X = 40, 12, 10
    stack = rng.normal(10, 0.3, (T, Y, X))
    centers = [(3, 3), (3, 6), (6, 3)]
    for i, (cy, cx) in enumerate(centers):
        stack[15 + 3 * i:, cy - 1:cy + 2, cx - 1:cx + 2] += 20.0
    s = caliana.Session()
    s.data = stack.astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=T)
    for c in centers:
        s.add_roi(center=c, size=3)

    w = AnalysisWidget(s)
    n = len(s.rois)
    Tt = s.traces.raw.shape[1]

    # A hand-built smoothed array with clearly different per-ROI onsets (steps at
    # 25/28/31) so we can tell which signal drove detection.
    crafted = np.zeros((n, Tt))
    for i in range(n):
        crafted[i, 25 + 3 * i:] = 5.0
    s.traces.smoothed = crafted
    s.traces.smoothed_sigma = 1.0

    w.prop_region.setRegion((0, 10))            # baseline window [0, 10)
    lo, hi = sorted(int(round(v)) for v in w.prop_region.getRegion())
    region = (lo - w._crop_start(), hi - w._crop_start())
    kw = dict(method=w.onset_method_box.currentText(), frac=w.frac_box.value(),
              k=w.k_box.value(), d=w.d_box.value(), baseline_region=region)

    # 'Show smoothed' checked -> onsets come from the crafted smoothed rows.
    w.show_smoothed.setChecked(True)
    res_sm = w.compute_propagation()
    for i in range(n):
        ref = analysis.onset_time(crafted[i], **kw)
        got = res_sm["onsets"][i]
        assert (np.isnan(got) and np.isnan(ref)) or abs(got - ref) < 1e-6

    # Unchecked (ΔF/F shown) -> onsets come from traces.dff instead.
    w.show_smoothed.setChecked(False)
    w.show_dff.setChecked(True)
    res_dff = w.compute_propagation()
    for i in range(n):
        ref = analysis.onset_time(s.traces.dff[i], **kw)
        got = res_dff["onsets"][i]
        assert (np.isnan(got) and np.isnan(ref)) or abs(got - ref) < 1e-6

    # The toggle genuinely changed which signal was used.
    assert not np.allclose(res_sm["onsets"], res_dff["onsets"], equal_nan=True)

    w.close()
    print("analysis widget propagation displayed-signal OK")


def test_analysis_widget_derivative_onset():
    """The 'derivative' onset method is selectable on both panels: it gates k+d
    (not frac), and the heatmap it produces matches onset_time per pixel."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana import analysis

    # Flat baseline then a per-row step, so the rate of change spikes at the rise.
    rng = np.random.default_rng(4)
    T, Y, X = 30, 12, 10
    stack = rng.normal(10, 0.3, (T, Y, X))
    for y in range(Y):
        stack[8 + y:, y, :] += 20.0
    s = caliana.Session()
    s.data = stack.astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=T)
    s.add_roi(center=(4, 6), size=3)

    w = AnalysisWidget(s)

    # Heatmap panel: derivative enables k and d, disables frac.
    w.hm_method_box.setCurrentText("derivative")
    assert w.hm_k_box.isEnabled() and w.hm_d_box.isEnabled()
    assert not w.hm_frac_box.isEnabled()

    # Map agrees, pixel for pixel, with the scalar detector at the same params.
    w.hm_k_box.setValue(2.0)
    w.hm_d_box.setValue(1.0)
    w.hm_base_start.setValue(0)
    w.hm_base_end.setValue(6)             # baseline window [0, 6)
    mp = w.compute_onset_heatmap()
    assert mp.shape == (Y, X)
    ref = analysis.onset_time(s.data[:, 4, 6].astype(float), method="derivative",
                              k=2.0, d=1.0, baseline_region=(0, 6))
    assert (np.isnan(mp[4, 6]) and np.isnan(ref)) or abs(float(mp[4, 6]) - float(ref)) < 1e-6

    # Propagation panel gates the same way.
    w.onset_method_box.setCurrentText("derivative")
    assert w.k_box.isEnabled() and w.d_box.isEnabled() and not w.frac_box.isEnabled()

    w.close()
    print("analysis widget derivative onset OK")


if __name__ == "__main__":
    test_import_preview_widget()
    test_roi_selection_widget()
    test_leaf_selection_widget()
    test_roi_widget_shows_leaf_reference()
    test_roi_widget_freehand()
    test_roi_widget_track_motion()
    test_crop_traces_widget()
    test_analysis_widget()
    test_analysis_widget_smoothing()
    test_analysis_widget_analysis_selection()
    test_analysis_widget_onset_heatmap()
    test_analysis_widget_propagation_uses_displayed_signal()
    test_analysis_widget_derivative_onset()
