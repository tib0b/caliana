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
    from qtpy import QtCore
    from caliana.widgets._qt import ensure_app
    from caliana.widgets.import_widget import ImportPreviewWidget
    from caliana.widgets.leaf_widget import LeafSelectionWidget
    from caliana.widgets.roi_widget import RoiSelectionWidget
    from caliana.widgets.crop_widget import CropTracesWidget
    from caliana.widgets.analysis_widget import AnalysisWidget
    from caliana.widgets.registration_widget import RegistrationWidget
    from caliana.widgets.source_widget import SourceWidget
    HAVE_GUI = True
except Exception:  # pragma: no cover - depends on optional deps
    HAVE_GUI = False

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                    "synthetic_calcium_imaging.tif")

# Message boxes are modal: shown for real they would block these tests forever
# with nobody to dismiss them. Record them instead, so a test can also assert
# that the failure *was* reported to the user.
DIALOGS: list[tuple[str, str]] = []
if HAVE_GUI:
    from qtpy import QtWidgets as _QtWidgets

    def _record(kind):
        def shim(_parent, title, text, *args, **kwargs):
            DIALOGS.append((title, text))
            return _QtWidgets.QMessageBox.StandardButton.Ok
        return staticmethod(shim)

    for _kind in ("critical", "warning", "information", "about"):
        setattr(_QtWidgets.QMessageBox, _kind, _record(_kind))

# Every panel of the standalone app, in workflow order. They share one contract:
# construct on any Session, `reload()` to re-read it.
PANELS = (SourceWidget, ImportPreviewWidget, RegistrationWidget,
          RoiSelectionWidget, CropTracesWidget, AnalysisWidget) if HAVE_GUI else ()


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


def test_leaf_widget_mask_polygon():
    """Tracing a polygon inside a leaf box masks that box's registration."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    w = LeafSelectionWidget(s)
    masked = w.add_leaf_box((0, 20, 0, 16))
    other = w.add_leaf_box((20, 31, 16, 23))

    # Trace an outline inside the first box: it commits to that box alone.
    w.mask_btn.setChecked(True)
    assert not w.leaf_btn.isEnabled()             # boxes frozen while tracing
    for point in [(4, 3), (4, 12), (15, 12), (15, 3)]:
        w.add_mask_point(*point)
    w.finish_mask()
    assert masked.mask_polygon == [(4, 3), (4, 12), (15, 12), (15, 3)]
    assert other.mask_polygon is None
    assert w._leaf_records[0]["mask"] is not None
    assert w.leaf_btn.isEnabled()

    # Dragging the box carries its outline along.
    w._leaf_records[0]["item"].setPos(2, 3)       # (x, y) => dy=3, dx=2
    assert masked.mask_polygon[0] == (7, 5)

    # Editing the polygon graphic writes back to the model.
    w._leaf_records[0]["mask"].setPos(1, 0)
    assert masked.mask_polygon[0] == (7, 6)

    # An outline traced over no box is dropped rather than stored anywhere.
    w.mask_btn.setChecked(True)
    for point in [(2, 20), (2, 22), (5, 22)]:
        w.add_mask_point(*point)
    w.finish_mask()
    assert other.mask_polygon is None
    assert "discarded" in w.hint.text()

    # Reload rebuilds the outline from the model, exactly one graphic per mask.
    w.reload()
    assert [r["mask"] is not None for r in w._leaf_records] == [True, False]

    w.clear_masks()
    assert masked.mask_polygon is None and w._leaf_records[0]["mask"] is None

    w.close()
    print("leaf mask polygon OK")


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
    data = w._displayed().data
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
    data = w._displayed().data
    assert data is s.traces.smoothed

    # Toggling off falls back to raw (the default display).
    w.show_smoothed.setChecked(False)
    data = w._displayed().data
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
    assert w.tabs.tabText(1) == "Heatmaps"

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


def test_analysis_widget_kymograph():
    """The Kymograph page: click out a path — no mode to enter — and read the
    intensity along it over time as a distance × time image."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()

    # A step response sweeping left to right at 1 px per frame from frame 12 — a
    # diagonal band for the kymograph to show, and clear of the 10-frame baseline.
    T, Y, X = 40, 12, 20
    stack = np.full((T, Y, X), 10.0, dtype=np.float32)
    for x in range(X):
        stack[12 + x:, :, x] += 20.0
    s = caliana.Session()
    s.data = stack
    s.timeline = caliana.Timeline(n_frames=T)
    s.add_roi(center=(6, 10), size=3)

    w = AnalysisWidget(s)
    assert w.tabs.count() == 3 and w.tabs.tabText(2) == "Kymograph"

    # Nothing drawn yet: computing says what to do rather than raising.
    assert w.compute_kymograph() is None
    assert "at least two points" in w.status.text()

    # Points go straight onto the path. One is only a marker; from two on it is a
    # polyline whose handles are individually draggable.
    w.add_path_point(6, 2)
    assert w.path_points() == [(6.0, 2.0)]
    assert not isinstance(w._path_item, pg.PolyLineROI)
    for col in (10, 17):
        w.add_path_point(6, col)
    assert isinstance(w._path_item, pg.PolyLineROI)
    assert [(round(y), round(x)) for y, x in w.path_points()] == [(6, 2), (6, 10), (6, 17)]

    # ΔF/F is the default; the kymograph is one row per pixel along the path.
    assert w.kymo_signal_box.currentText() == "ΔF/F"
    result = w.compute_kymograph()
    assert result is not None and s.analyses["kymograph"] is result
    values = result["values"]
    assert values.shape == (16, T)                   # 15 px of path, both ends
    assert result["baseline"] == (0, w.kymo_base_box.value())
    # Each position steps from 0 to 20/10 = 2 ΔF/F, one frame later than the last.
    onsets = np.argmax(values > 1.0, axis=1)
    assert np.allclose(np.diff(onsets), 1.0)
    assert np.allclose(values.max(axis=1), 2.0)

    # Raw values instead: same shape, the untouched intensities, baseline disabled.
    w.kymo_signal_box.setCurrentText("raw intensity")
    assert not w.kymo_base_box.isEnabled()
    raw = w.compute_kymograph()["values"]
    assert raw.shape == values.shape
    assert np.allclose(raw[0], stack[:, 6, 2])
    assert raw.min() == 10.0 and raw.max() == 30.0

    # The image is placed in data coordinates: frames across, path length up.
    rect = w.kymo_image.mapRectToView(w.kymo_image.boundingRect())
    assert (rect.width(), rect.height()) == (T, 15.0)
    assert w.kymo_plot.getAxis("left").labelText.endswith("(px)")

    # Calibrating relabels and rescales both axes of the kymograph already on
    # screen — the sampled values never move, only the units they're reported in.
    w.interval_box.setValue(0.5)
    w.pixel_box.setValue(2.0)
    assert w.kymo_plot.getAxis("bottom").labelText == "time (s)"
    assert w.kymo_plot.getAxis("left").labelText.endswith("(µm)")
    rect = w.kymo_image.mapRectToView(w.kymo_image.boundingRect())
    assert (rect.width(), rect.height()) == (T * 0.5, 15.0 * 2.0)

    # ‘Clear last point’ walks the path back a point at a time, graphic included:
    # three points, then two (still a polyline), then one (a bare marker).
    w.clear_last_point()
    assert [(round(y), round(x)) for y, x in w.path_points()] == [(6, 2), (6, 10)]
    assert isinstance(w._path_item, pg.PolyLineROI)
    w.clear_last_point()
    assert len(w.path_points()) == 1 and not isinstance(w._path_item, pg.PolyLineROI)

    # ‘Clear path’ drops the rest; the next compute asks for a new one.
    w.clear_path()
    assert w._path_item is None and w.path_points() == []
    assert w.compute_kymograph() is None
    w.clear_last_point()                             # nothing to undo, no error
    assert "No path" in w.status.text()

    # The second click of a double-click is dropped, so it can't stack a
    # duplicate point on the first one's.
    w.add_path_point(6, 2)
    w._on_kymo_click(_Click(QtCore.QPointF(5.0, 5.0), double=True))
    assert len(w.path_points()) == 1

    w.close()
    print("analysis widget kymograph OK")


def test_analysis_widget_kymograph_points_are_movable():
    """Each point of the path is draggable, and the kymograph follows it — the
    graphic is the path, not a picture of where it was clicked."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()

    # Rows differ so a dragged point lands on visibly different pixels.
    T, Y, X = 12, 16, 20
    stack = np.zeros((T, Y, X), dtype=np.float32)
    for y in range(Y):
        stack[:, y, :] = 10.0 * (y + 1)
    s = caliana.Session()
    s.data = stack
    s.timeline = caliana.Timeline(n_frames=T)

    w = AnalysisWidget(s)
    w.kymo_signal_box.setCurrentText("raw intensity")
    w.add_path_point(4, 2)
    w.add_path_point(4, 17)
    before = w.compute_kymograph()["values"]
    assert np.allclose(before, 50.0)                 # row 4 -> 10 * 5

    # Drag the whole polyline down 6 rows: the points, and so the kymograph,
    # follow it onto row 10 without anything being re-clicked.
    w._path_item.setPos(0, 6)
    assert [round(y) for y, _x in w.path_points()] == [10, 10]
    after = w.compute_kymograph()["values"]
    assert np.allclose(after, 110.0)                 # row 10 -> 10 * 11

    # Moving one handle moves only its end of the path.
    handles = w._path_item.getHandles()
    w._path_item.movePoint(handles[0], pg.Point(2.0, 0.0))
    moved = w.path_points()
    assert round(moved[0][0]) == 0 and round(moved[1][0]) == 10

    w.close()
    print("analysis widget kymograph movable points OK")


def test_analysis_widget_trace_page_gates_itself_on_rois():
    """The Analysis tab opens on a stack alone (heatmaps and kymographs need no
    ROIs); only the trace page is closed until ROIs exist."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = caliana.Session()
    s.data = (np.random.default_rng(8).random((10, 24, 20)) * 255).astype(np.uint16)
    s.timeline = caliana.Timeline(n_frames=10)

    w = AnalysisWidget(s)
    assert not w.tabs.isTabEnabled(0)                       # Trace analysis
    assert w.tabs.isTabEnabled(1) and w.tabs.isTabEnabled(2)
    assert "ROI" in w.tabs.tabToolTip(0)
    assert "No ROIs yet" in w.status.text()

    # Both dataset-wide pages run with no ROIs at all.
    assert w.compute_onset_heatmap() is not None
    w.add_path_point(4, 4)
    w.add_path_point(18, 16)
    assert w.compute_kymograph() is not None

    # Placing an ROI opens the trace page on the next reload.
    s.add_roi(center=(12, 10), size=3)
    w.reload()
    assert w.tabs.isTabEnabled(0) and w.status.text() == ""

    w.close()
    print("analysis widget trace-page gating OK")


def test_analysis_widget_kymograph_path_is_dropped_on_reload():
    """A path is pixel coordinates on the stack it was drawn over, so a reload —
    a new stack, a re-registration — clears it, as it does the ROI graphics."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_roi(center=(16, 12), size=4)
    w = AnalysisWidget(s)

    w.add_path_point(4, 4)
    w.add_path_point(20, 18)
    assert w.compute_kymograph() is not None

    w.reload()
    assert w._path_item is None and w.path_points() == []
    assert w._kymo_result is None
    assert w._kymo_shape_yx == s.data.shape[1:]

    # A one-point path (no polyline yet) is dropped just the same.
    w.add_path_point(3, 3)
    w.reload()
    assert w._path_item is None and w.path_points() == []

    w.close()
    print("analysis widget kymograph reload OK")


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


def test_analysis_widget_propagation_direction_mode():
    """The Direction combo picks how the propagation vector is found. It defaults
    to the ROI line, which for ROIs strung along one line keeps the direction on
    that line — where the free 2D fit can swing off it."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()

    # ROIs in a straight horizontal line, responding left to right (+x).
    rng = np.random.default_rng(7)
    T, Y, X = 40, 14, 24
    stack = rng.normal(10, 0.3, (T, Y, X))
    centers = [(7, 4), (7, 9), (7, 14), (7, 19)]
    for i, (cy, cx) in enumerate(centers):
        stack[12 + 4 * i:, cy - 1:cy + 2, cx - 1:cx + 2] += 20.0
    s = caliana.Session()
    s.data = stack.astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=T)
    for c in centers:
        s.add_roi(center=c, size=3)

    w = AnalysisWidget(s)
    w.analysis_box.setCurrentText("Cross-ROI propagation")
    w.prop_region.setRegion((0, 10))

    # Default: along the ROI line -> direction is the +x line the ROIs sit on.
    assert w.prop_dir_box.currentText() == "along ROI line"
    res_line = w.compute_propagation()
    assert res_line["direction_mode"] == "roi_line"
    dy, dx = res_line["direction"]
    assert abs(dy) < 1e-6 and dx > 0.999      # exactly along +x, toward later onset
    assert "along ROI line" in w.results.toPlainText()

    # Switching to the automatic 2D fit reaches the plane-fit path.
    w.prop_dir_box.setCurrentText("automatic (2D fit)")
    res_auto = w.compute_propagation()
    assert res_auto["direction_mode"] == "auto"
    assert np.allclose(res_auto["onsets"], res_line["onsets"], equal_nan=True)

    w.close()
    print("analysis widget propagation direction mode OK")


def test_analysis_widget_spatial_scale():
    """The pixel-size box is the spatial twin of the frame-interval box: it
    calibrates the session, relabels the propagation graph in µm, and combines
    with the time axis into a µm/s speed — while the analysis stays px/frame."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()

    # ROIs along a line, responding left to right, 5 px and 4 frames apart.
    rng = np.random.default_rng(11)
    T, Y, X = 40, 14, 24
    stack = rng.normal(10, 0.3, (T, Y, X))
    centers = [(7, 4), (7, 9), (7, 14), (7, 19)]
    for i, (cy, cx) in enumerate(centers):
        stack[12 + 4 * i:, cy - 1:cy + 2, cx - 1:cx + 2] += 20.0
    s = caliana.Session()
    s.data = stack.astype(np.float32)
    s.timeline = caliana.Timeline(n_frames=T)
    for c in centers:
        s.add_roi(center=c, size=2)

    w = AnalysisWidget(s)
    w.show_dff.setChecked(True)
    result = w.compute_propagation()
    speed = result["speed_px_per_frame"]        # ~5 px / 4 frames
    assert np.isfinite(speed)

    # Uncalibrated: pixels and frames throughout.
    assert w.prop_plot.getAxis("left").labelText.endswith("(px)")
    assert w._speed_str(speed).endswith("px/frame")

    # Pixel size alone -> µm distances, still per frame.
    w.pixel_box.setValue(2.0)
    assert s.space.pixel_size == 2.0            # the box calibrates the session
    assert w.prop_plot.getAxis("left").labelText.endswith("(µm)")
    assert w._speed_str(speed).endswith("µm/frame")

    # Both axes -> µm/s, and the graph's y values are the px distances in µm.
    w.interval_box.setValue(0.5)
    assert w._speed_str(speed).endswith("µm/s")
    assert abs(float(w._speed_str(speed).split()[0]) - speed * 2.0 / 0.5) < 1e-2
    assert np.allclose(np.sort(np.abs(w._prop_fit["dist"])), [0.0, 10.0, 20.0, 30.0])

    # The stored result never left pixels/frames.
    assert s.analyses["propagation"]["speed_px_per_frame"] == speed

    # Clearing the box goes back to pixels.
    w.pixel_box.setValue(0.0)
    assert s.space.pixel_size is None
    assert w.prop_plot.getAxis("left").labelText.endswith("(px)")

    w.close()
    print("analysis widget spatial scale OK")


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


def test_stack_contrast_scales_on_first_frame():
    """A stack's display range comes from frame 0 alone, in every image view.

    A bright transient mid-recording must not darken the frames before it, and
    the preview / ROI / leaf views must all agree on the range.
    """
    from caliana import figures

    stack = np.full((8, 16, 12), 100.0, dtype=np.float32)
    stack[4] += 5000.0                              # transient, frame 4 only
    expected = figures.intensity_levels(stack[0])
    assert figures.intensity_levels(stack) == expected

    if not HAVE_GUI:
        print("GUI stack not available; skipping widget half")
        return
    ensure_app()
    s = caliana.Session()
    s.data = stack
    s.timeline = caliana.Timeline(n_frames=8)

    for cls in (ImportPreviewWidget, RoiSelectionWidget, LeafSelectionWidget):
        w = cls(s)
        view = w.movie if cls is ImportPreviewWidget else w.image
        assert np.allclose(view.getLevels(), expected), cls.__name__
        w.close()

    # Toggling tracking keeps whatever contrast is on screen (raw and stabilized
    # views are the same tissue), including a manual histogram drag.
    w = RoiSelectionWidget(s)
    w.image.setLevels(3.0, 77.0)
    w.track_box.setChecked(True)
    assert np.allclose(w.image.getLevels(), (3.0, 77.0))
    w.close()
    print("first-frame stack contrast OK")


def test_panels_build_on_an_empty_session():
    """Every panel constructs against a Session with no data and stays usable.

    The app builds all of them up front, before any file is chosen, so nothing
    may assume a stack exists — the crop widget previewing traces, the ROI widget
    bounds-checking a click, the analysis widget reading a frame count.
    """
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    for cls in PANELS:
        s = caliana.Session()
        w = cls(s)
        w.reload()                       # idempotent, and a no-op without data
        w.close()

    # A click with no stack loaded is bounds-checked, not an AttributeError.
    s = caliana.Session()
    w = RoiSelectionWidget(s)
    assert w._shape_yx == (1, 1)
    w._on_scene_click(_Click(QtCore.QPointF(5.0, 5.0)))
    assert s.rois == []
    w.close()
    print("panels build on empty session OK")


class _Click:
    """The bits of a pyqtgraph scene click the ROI widget actually reads."""

    def __init__(self, pos, double=False):
        self._pos, self._double = pos, double

    def button(self):
        return QtCore.Qt.MouseButton.LeftButton

    def scenePos(self):
        return self._pos

    def double(self):
        return self._double


def test_panels_reload_into_a_changed_session():
    """`reload()` re-reads the Session, and repeating it never duplicates anything.

    This is what lets the app keep one set of panels across a whole run: the
    session changes underneath (a file loaded, ROIs placed, a crop applied) and
    each tab catches up when it is next opened.
    """
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = caliana.Session()
    panels = [cls(s) for cls in PANELS]

    # The session gains a stack, two ROIs and a leaf box after every panel exists.
    s.data = (np.random.default_rng(1).random((10, 32, 24)) * 255).astype(np.uint16)
    s.timeline = caliana.Timeline(n_frames=10)
    s.add_leaf_region((0, 16, 0, 12))
    s.add_roi(center=(16, 12), size=4)
    s.add_roi(center=(8, 6), size=3)
    for w in panels:
        w.reload()
        w.reload()                       # twice: graphics must not accumulate

    by_type = {type(w): w for w in panels}
    roi_w = by_type[RoiSelectionWidget]
    assert len(roi_w._roi_records) == 2 and len(roi_w._leaf_records) == 1
    assert roi_w._shape_yx == (32, 24)
    leaf_w = by_type[RegistrationWidget].leaf_panel
    assert len(leaf_w._leaf_records) == 1
    crop_w = by_type[CropTracesWidget]
    assert crop_w._preview.raw.shape == (2, 10)
    assert crop_w.end_box.value() == 10 and crop_w.crop_btn.isEnabled()
    analysis_w = by_type[AnalysisWidget]
    assert len(analysis_w._curves) == 2

    # A crop applied elsewhere shows up on the next reload of each panel.
    s.set_crop(2, 8)
    for w in panels:
        w.reload()
    assert crop_w.start_box.value() == 2 and crop_w.end_box.value() == 8
    assert analysis_w._curves[0].getData()[0][0] == 2      # original frame coords

    for w in panels:
        w.close()
    print("panels reload into changed session OK")


def test_session_revision_tracks_changes():
    """`Session._revision` moves on every change a panel would need to redraw for.

    It is the app's whole change-propagation mechanism (no widget knows about any
    other), so anything that changes what a panel shows has to bump it.
    """
    s = caliana.Session()
    s.data = (np.random.default_rng(3).random((6, 16, 12)) * 255).astype(np.uint16)
    s.timeline = caliana.Timeline(n_frames=6)

    revisions = []
    for change in (lambda: s.add_leaf_region((0, 8, 0, 6)),
                   lambda: s.add_roi(center=(8, 6), size=3),
                   lambda: s.set_crop(1, 5),
                   lambda: s.set_frame_interval(0.5),
                   lambda: s.set_pixel_size(2.0),
                   lambda: s.extract_traces() and s._invalidate_traces()):
        before = s._revision
        change()
        revisions.append(s._revision - before)
    assert all(delta > 0 for delta in revisions), revisions
    print("session revision OK")


def test_load_resets_derived_state():
    """Loading drops ROIs/leaf boxes/registration from the previous stack.

    They are expressed in that stack's pixel coordinates, so carrying them into
    another file (or the same file re-imported at a coarser step, which the app's
    Import tab makes a one-click operation) would silently measure the wrong
    pixels.
    """
    s = caliana.Session.from_file(DATA)
    s.add_leaf_region((0, 16, 0, 16))
    s.add_roi(center=(20, 20), size=4)
    s.extract_traces()
    s.set_crop(1, 10)
    before = s._revision

    s.load(DATA, spatial_step=2)
    assert s.rois == [] and s.leaf_regions == []
    assert s.crop_window is None and s.traces is None
    assert s.registered_data is None and not s.track_motion
    assert s.registration.mode == caliana.RegistrationMode.NONE
    assert s._revision > before
    print("load resets derived state OK")


def test_crop_widget_signals_instead_of_closing():
    """Validating a crop emits `applied` and leaves the widget open.

    The app keeps the tab; the notebook wrapper is what closes the window on that
    signal (`_qt.run_widget_blocking(close_on=...)`), so one widget serves both.
    """
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = _session()
    s.add_roi(center=(16, 12), size=4)
    w = CropTracesWidget(s)

    seen, closed = [], []
    w.applied.connect(lambda: seen.append(True))
    w.closed.connect(lambda: closed.append(True))
    w.show()
    w.set_interval(1, 5)
    w.apply_crop()

    assert seen == [True] and closed == []        # announced, not closed
    assert w.isVisible()
    assert s.crop_window == (1, 5)

    # …and the notebook wrapper's wiring is what turns that into a close.
    w.applied.connect(w.close)
    w.apply_crop()
    assert closed == [True]
    w.close()
    print("crop widget applied signal OK")


def test_source_widget_loads_a_file():
    """The import panel turns its controls into ImportParams and loads through them."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = caliana.Session()
    w = SourceWidget(s)
    w.set_path(DATA)
    w.tstep_box.setValue(2)
    w.sstep_box.setValue(2)
    w.window_check.setChecked(True)
    for spin, value in zip(w.window_boxes, (0, 32, 0, 48)):
        spin.setValue(value)

    params = w.import_params()
    assert params.temporal_step == 2 and params.spatial_step == 2
    assert params.spatial_window == (0, 32, 0, 48) and params.end is None

    loaded = []
    w.loaded.connect(lambda sess: loaded.append(sess))
    task = w.load()
    assert task.wait() and task.error is None

    assert loaded == [s]                       # the same Session, loaded in place
    assert s.data.shape == (50, 16, 24)        # 100 frames /2, (32, 48) /2
    assert s.source.import_params.spatial_window == (0, 32, 0, 48)

    # Re-importing at another step is one click, and starts from a clean slate.
    w.sstep_box.setValue(1)
    task = w.load()
    assert task.wait() and task.error is None
    assert s.data.shape == (50, 32, 48)
    w.close()
    print("source widget load OK")


def test_source_widget_reports_a_bad_file():
    """An unreadable file is reported, not raised, and leaves the session empty."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    s = caliana.Session()
    w = SourceWidget(s)
    w.set_path(DATA + ".nope.xyz")
    DIALOGS.clear()
    task = w.load()
    assert task.wait()
    assert isinstance(task.error, (ValueError, FileNotFoundError))
    assert s.data is None
    assert "Could not load" in w.status.text()
    # Reported to the user, in terms of what to do about it — not a traceback.
    assert DIALOGS and DIALOGS[-1][0] == "Load failed"
    assert ".tif" in DIALOGS[-1][1]
    w.close()
    print("source widget bad file OK")


def test_registration_widget_runs():
    """The registration panel drives Session.register and reports what it did."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping widget test")
        return
    ensure_app()
    from caliana.models import RegistrationMode

    s = caliana.Session.from_file(DATA, temporal_step=10)   # 10 frames
    w = RegistrationWidget(s)

    # None mode: no boxes, no estimator controls, nothing to warp.
    assert w.mode == RegistrationMode.NONE
    assert w.left_stack.currentIndex() == 0 and not w.ref_box.isEnabled()

    # Per-leaf without boxes refuses rather than raising out of the worker.
    w.mode_box.setCurrentText("Per leaf")
    assert w.left_stack.currentIndex() == 1 and w.ref_box.isEnabled()
    assert w.run() is None and "leaf box" in w.status.text()

    # With a box drawn on the embedded leaf pane it runs, per leaf.
    w.leaf_panel.add_leaf_box((4, 60, 4, 60))
    w.transform_box.setCurrentText("rigid_body")
    task = w.run()
    assert task.wait() and task.error is None
    assert s.registration.mode == RegistrationMode.PER_LEAF
    assert len(s.leaf_regions[0].transforms) == len(s.data)
    assert s.registered_data is not None                 # "Warp the stack"
    assert "per-leaf" in w.summary() and "leaf 0" in w.summary()

    # Whole frame, correcting by moving the ROIs instead of warping.
    w.mode_box.setCurrentText("Whole frame")
    w.apply_box.setCurrentIndex(1)
    task = w.run()
    assert task.wait() and task.error is None
    assert s.registration.mode == RegistrationMode.WHOLE_FRAME
    assert s.registered_data is None and s.track_motion
    assert len(s.registration.transforms) == len(s.data)
    assert "ROIs follow the tissue" in w.summary()
    w.close()
    print("registration widget OK")


if __name__ == "__main__":
    test_stack_contrast_scales_on_first_frame()
    test_import_preview_widget()
    test_roi_selection_widget()
    test_leaf_selection_widget()
    test_leaf_widget_mask_polygon()
    test_roi_widget_shows_leaf_reference()
    test_roi_widget_freehand()
    test_roi_widget_track_motion()
    test_crop_traces_widget()
    test_analysis_widget()
    test_analysis_widget_smoothing()
    test_analysis_widget_analysis_selection()
    test_analysis_widget_onset_heatmap()
    test_analysis_widget_kymograph()
    test_analysis_widget_kymograph_points_are_movable()
    test_analysis_widget_trace_page_gates_itself_on_rois()
    test_analysis_widget_kymograph_path_is_dropped_on_reload()
    test_analysis_widget_propagation_uses_displayed_signal()
    test_analysis_widget_propagation_direction_mode()
    test_analysis_widget_derivative_onset()
    test_analysis_widget_spatial_scale()
    test_panels_build_on_an_empty_session()
    test_panels_reload_into_a_changed_session()
    test_session_revision_tracks_changes()
    test_load_resets_derived_state()
    test_crop_widget_signals_instead_of_closing()
    test_source_widget_loads_a_file()
    test_source_widget_reports_a_bad_file()
    test_registration_widget_runs()
