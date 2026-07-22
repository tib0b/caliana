"""Stage II — ROI selection widget. SPEC.md §3 Stage II.

- Click the image to place fixed circle/square ROIs of a chosen size (overlap OK).
- "Freehand ROI": click around a feature (e.g. a whole leaf) to trace a polygon
  outline; finish to commit it. Polygon ROIs are movable/editable afterwards.
- Each ROI is movable; its label/index is shown next to it.
- Live trace preview: the right panel updates as ROIs are added/moved.
- Leaf boxes (drawn in the separate leaf-selection widget, `select_leaves`) are
  shown here as non-interactive reference so ROIs land in the right place and
  auto-assign to the box containing them; they cannot be moved from here.
- "Track motion": when the stack is registered, switch the scrollable preview to
  the raw (unstabilized) footage and move each ROI marker frame-by-frame so it
  follows the tissue it sits on — a visual check that the ROI tracks correctly as
  the leaf moves. Editing is paused while tracking; stored ROIs are unchanged.

The interaction logic lives in plain methods (`add_roi_at`, `delete_last_roi`,
`_refresh_traces`) so it can be driven from tests without a real mouse; mouse
clicks are wired to `add_roi_at`.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..models import RegistrationMode, ROIShape
from ..registration import map_point
from ..roi import polygon_centroid
from ._plot import FrameTimeAxis
from ._qt import get_qt, save_figure_dialog

QtCore, QtGui, QtWidgets = get_qt()

pg.setConfigOption("imageAxisOrder", "row-major")

_LEFT = QtCore.Qt.MouseButton.LeftButton
_ROI_PEN = pg.mkPen("#00ff7f", width=2)
_LEAF_PEN = pg.mkPen("#ffd000", width=2, style=QtCore.Qt.PenStyle.DashLine)


class RoiSelectionWidget(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session.rois
        self.setWindowTitle("Caliana — ROI Selection")
        self.resize(1040, 580)

        # Bookkeeping: parallel records linking model ROIs to their graphics.
        self._roi_records: list[dict] = []
        self._leaf_records: list[dict] = []
        # When True the preview shows raw footage and ROI markers track tissue
        # per-frame; placement/editing is paused so the two don't conflict.
        self._tracking = False
        # In-progress free-hand polygon: accumulated (y, x) points + preview item.
        self._poly_points: list[tuple[float, float]] = []
        self._poly_preview = None

        self._build_ui()
        self._load_session()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar.
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Shape:"))
        self.shape_box = QtWidgets.QComboBox()
        self.shape_box.addItems([ROIShape.CIRCLE.value, ROIShape.SQUARE.value])
        # Shape is shared across all ROIs (applies to existing ones too).
        self.shape_box.currentTextChanged.connect(self._on_shape_changed)
        bar.addWidget(self.shape_box)

        bar.addWidget(QtWidgets.QLabel("Size (px):"))
        self.size_box = QtWidgets.QSpinBox()
        self.size_box.setRange(1, 200)
        self.size_box.setValue(5)
        # Size is shared across all ROIs (applies to existing ones too).
        self.size_box.valueChanged.connect(self._on_size_changed)
        bar.addWidget(self.size_box)

        self.poly_btn = QtWidgets.QPushButton("Freehand ROI")
        self.poly_btn.setCheckable(True)
        self.poly_btn.setToolTip(
            "Click around a feature to trace an outline; click again to finish"
        )
        self.poly_btn.toggled.connect(self._on_poly_toggled)
        bar.addWidget(self.poly_btn)

        self.del_btn = QtWidgets.QPushButton("Delete last ROI")
        self.del_btn.clicked.connect(self.delete_last_roi)
        bar.addWidget(self.del_btn)

        # Enabled only when the stack carries per-frame transforms to track.
        self.track_box = QtWidgets.QCheckBox("Track motion")
        self.track_box.setToolTip(
            "Show raw footage and move ROI markers to follow tissue as you scrub"
        )
        self.track_box.toggled.connect(self._on_track_toggled)
        bar.addWidget(self.track_box)

        # Export the ROI overlay (first frame) and the mean-intensity traces as
        # paper-grade static figures via figures.py.
        self.save_overlay_btn = QtWidgets.QPushButton("Save overlay…")
        self.save_overlay_btn.setToolTip("Save the first frame with ROI outlines as a figure")
        self.save_overlay_btn.clicked.connect(self._save_overlay)
        bar.addWidget(self.save_overlay_btn)

        self.save_traces_btn = QtWidgets.QPushButton("Save traces…")
        self.save_traces_btn.setToolTip("Save the ROI mean-intensity traces as a figure")
        self.save_traces_btn.clicked.connect(self._save_traces)
        bar.addWidget(self.save_traces_btn)

        bar.addStretch(1)
        self.hint = QtWidgets.QLabel("Click the image to place an ROI")
        bar.addWidget(self.hint)
        layout.addLayout(bar)

        # Image (left) + live traces (right).
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        self.image = pg.ImageView(name="roi_image")
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        # Simplified contrast: keep the level region, drop the colormap editor.
        self.image.ui.histogram.gradient.hide()
        split.addWidget(self.image)

        self._time_axis = FrameTimeAxis(orientation="bottom")
        self.trace_plot = pg.PlotWidget(title="ROI traces (live)",
                                        axisItems={"bottom": self._time_axis})
        self.trace_plot.setLabel("bottom", "frame")
        self.trace_plot.setLabel("left", "ΔF/F₀")
        self.trace_plot.addLegend()
        split.addWidget(self.trace_plot)
        split.setSizes([620, 420])

        self.image.view.scene().sigMouseClicked.connect(self._on_scene_click)
        self.image.sigTimeChanged.connect(self._on_frame_changed)

    def _load_session(self):
        if self.session.data is None:
            self.image.setImage(np.zeros((1, 1, 1)))
            return
        stack = np.asarray(self.session._working_stack())
        self._shape_yx = stack.shape[1:]
        self.image.setImage(stack, axes={"t": 0, "y": 1, "x": 2})
        # Re-draw any ROIs/leaf boxes already on the session.
        for roi in list(self.session.rois):
            self._add_roi_graphic(roi)
        for i, leaf in enumerate(list(self.session.leaf_regions)):
            self._add_leaf_reference(i, leaf)
        self.track_box.setEnabled(self._has_transforms())
        self._refresh_traces()

    # ------------------------------------------------------- current controls
    @property
    def shape(self) -> ROIShape:
        return ROIShape(self.shape_box.currentText())

    @property
    def size(self) -> int:
        return self.size_box.value()

    # --------------------------------------------------------------- ROIs
    def add_roi_at(self, row: float, col: float):
        """Place an ROI of the current shape/size centred at (row, col)."""
        roi = self.session.add_roi(center=(row, col), size=self.size, shape=self.shape)
        self._add_roi_graphic(roi)
        self._refresh_traces()
        return roi

    def _make_roi_item(self, roi):
        """Build a translate-only circle/square graphic for an ROI model."""
        r = roi.size
        cy, cx = roi.center
        pos = (cx - r, cy - r)
        size = (2 * r, 2 * r)
        cls = pg.CircleROI if roi.shape == ROIShape.CIRCLE else pg.RectROI
        item = cls(pos, size, pen=_ROI_PEN, movable=True)
        # Fixed size (set via the shared spinbox): remove scale handles.
        for h in list(item.handles):
            item.removeHandle(h["item"])
        return item

    def _make_polygon_item(self, roi):
        """Build a closed, movable/editable polyline graphic for a polygon ROI."""
        pts = [(x, y) for (y, x) in roi.vertices]      # pyqtgraph points are (x, y)
        return pg.PolyLineROI(pts, closed=True, pen=_ROI_PEN, movable=True)

    def _add_roi_graphic(self, roi):
        if roi.shape == ROIShape.POLYGON:
            item = self._make_polygon_item(roi)
        else:
            item = self._make_roi_item(roi)
        cy, cx = roi.center
        label = roi.label or f"{len(self._roi_records)}"
        text = pg.TextItem(label, color="#00ff7f", anchor=(0.5, 1.2))
        text.setPos(cx, cy)

        self.image.view.addItem(item)
        self.image.view.addItem(text)
        record = {"model": roi, "item": item, "text": text}
        self._roi_records.append(record)
        item.sigRegionChanged.connect(lambda it, rec=record: self._on_roi_moved(rec))

    def _on_size_changed(self, value):
        """Shared size: resize every ROI about its centre. SPEC §3 (fixed size).

        Polygon ROIs have their own outline and are left untouched.
        """
        for rec in self._roi_records:
            if rec["model"].shape == ROIShape.POLYGON:
                continue
            rec["model"].size = value
            rec["item"].setSize([2 * value, 2 * value], center=[0.5, 0.5])
        self._refresh_traces()

    def _on_shape_changed(self, text):
        """Shared shape: convert circle/square ROIs in place (polygons untouched)."""
        shape = ROIShape(text)
        for rec in self._roi_records:
            if rec["model"].shape == ROIShape.POLYGON:
                continue
            rec["model"].shape = shape
            old = rec["item"]
            old.sigRegionChanged.disconnect()
            self.image.view.removeItem(old)
            new = self._make_roi_item(rec["model"])
            self.image.view.addItem(new)
            new.sigRegionChanged.connect(lambda it, r=rec: self._on_roi_moved(r))
            rec["item"] = new
        self._refresh_traces()

    def _on_roi_moved(self, record):
        # While tracking we reposition items programmatically per frame; ignore
        # those moves so the stored ROI geometry is never overwritten.
        if self._tracking:
            return
        roi = record["model"]
        if roi.shape == ROIShape.POLYGON:
            verts = self._polygon_item_vertices(record["item"])
            if len(verts) < 3:           # mid-edit / teardown — keep last valid outline
                return
            roi.vertices = verts
            cy, cx = polygon_centroid(verts)
        else:
            pos, size = record["item"].pos(), record["item"].size()
            cx = pos.x() + size.x() / 2
            cy = pos.y() + size.y() / 2
        roi.center = (cy, cx)
        if self.session.registration.mode.value == "per-leaf":
            from ..roi import assign_roi_to_leaf
            roi.leaf_region = assign_roi_to_leaf(roi, self.session.leaf_regions)
        record["text"].setPos(cx, cy)
        self._refresh_traces()

    @staticmethod
    def _polygon_item_vertices(item) -> list[tuple[float, float]]:
        """Current polygon vertices as (y, x) in image coordinates."""
        out = []
        for h in item.getHandles():
            p = item.mapToParent(h.pos())
            out.append((p.y(), p.x()))
        return out

    def delete_last_roi(self):
        if not self._roi_records:
            return
        record = self._roi_records.pop()
        self.image.view.removeItem(record["item"])
        self.image.view.removeItem(record["text"])
        self.session.rois.remove(record["model"])
        self.session._invalidate_traces()
        self._refresh_traces()

    # ------------------------------------------------------- free-hand polygon
    def _on_poly_toggled(self, checked: bool):
        """Enter free-hand mode on check; commit (or discard) the outline on uncheck."""
        if checked:
            self.start_polygon()
        elif len(self._poly_points) >= 3:
            self._commit_polygon()
        else:
            self._clear_polygon_preview()
        if not checked:
            self.hint.setText("Click the image to place an ROI")

    def start_polygon(self):
        """Begin a new free-hand outline (clears any in-progress one)."""
        self._clear_polygon_preview()
        self._poly_preview = pg.PlotDataItem(
            pen=_ROI_PEN, symbol="o", symbolSize=6, symbolBrush="#00ff7f"
        )
        self.image.view.addItem(self._poly_preview)
        self.hint.setText("Click to add outline points; click ‘Freehand ROI’ to finish")

    def add_polygon_point(self, row: float, col: float):
        """Append a vertex to the in-progress outline and redraw the preview."""
        self._poly_points.append((row, col))
        if self._poly_preview is not None:
            ys = [p[0] for p in self._poly_points]
            xs = [p[1] for p in self._poly_points]
            self._poly_preview.setData(xs, ys)

    def finish_polygon(self):
        """Commit the in-progress outline as a polygon ROI (needs >= 3 points)."""
        # Unchecking the button routes through _on_poly_toggled -> _commit_polygon.
        self.poly_btn.setChecked(False)

    def _commit_polygon(self):
        roi = self.session.add_polygon_roi(self._poly_points)
        self._clear_polygon_preview()
        self._add_roi_graphic(roi)
        self._refresh_traces()
        return roi

    def _clear_polygon_preview(self):
        if self._poly_preview is not None:
            self.image.view.removeItem(self._poly_preview)
            self._poly_preview = None
        self._poly_points = []

    # --------------------------------------------------------- motion tracking
    def _has_transforms(self) -> bool:
        """Whether registration produced per-frame transforms to track."""
        reg = self.session.registration
        if reg.mode == RegistrationMode.WHOLE_FRAME:
            return bool(reg.transforms)
        if reg.mode == RegistrationMode.PER_LEAF:
            return any(leaf.transforms for leaf in self.session.leaf_regions)
        return False

    def _frame_transform(self, roi, frame: int):
        """The (transform, y-origin, x-origin) acting on ``roi`` at ``frame``.

        Per-leaf transforms are estimated in box-local coordinates, so the box's
        top-left corner is returned as the origin to offset by; whole-frame
        transforms use the full-image origin. None if nothing applies.
        """
        reg = self.session.registration
        if reg.mode == RegistrationMode.WHOLE_FRAME:
            if 0 <= frame < len(reg.transforms):
                return reg.transforms[frame], 0.0, 0.0
        elif reg.mode == RegistrationMode.PER_LEAF:
            idx = roi.leaf_region
            if idx is not None and 0 <= idx < len(self.session.leaf_regions):
                leaf = self.session.leaf_regions[idx]
                if 0 <= frame < len(leaf.transforms):
                    y0, _y1, x0, _x1 = leaf.bbox
                    return leaf.transforms[frame], float(y0), float(x0)
        return None

    def _roi_raw_center(self, roi, frame: int):
        """Where ``roi``'s tissue sits in the raw frame ``frame`` -> (cy, cx).

        The rigid transform maps raw -> stabilized; applied to the (stabilized)
        ROI centre it gives back the raw position, so the marker follows the
        tissue as the leaf moves. Falls back to the stored centre when no
        transform applies (e.g. an ROI in no leaf box).
        """
        info = self._frame_transform(roi, frame)
        if info is None:
            return roi.center
        return self._raw_point(roi.center, *info)

    def _roi_raw_vertices(self, roi, frame: int):
        """Polygon vertices mapped to their raw-frame positions (else the stored ones)."""
        info = self._frame_transform(roi, frame)
        if info is None:
            return roi.vertices
        return [self._raw_point(v, *info) for v in roi.vertices]

    @staticmethod
    def _raw_point(point, tf, oy, ox):
        """Map a (cy, cx) point through transform ``tf`` about box origin (oy, ox).

        Delegates to ``registration.map_point`` so the on-screen tracked marker and
        the headless tracked trace (``roi.extract_trace_tracked``) share one map.
        """
        return map_point(point, tf, (oy, ox))

    def _place_roi_item(self, record, cy, cx):
        """Move a circle/square ROI graphic so its centre lands at (cy, cx)."""
        item = record["item"]
        r = item.size().x() / 2
        item.setPos(cx - r, cy - r)
        record["text"].setPos(cx, cy)

    def _place_polygon_item(self, record, vertices):
        """Redraw a polygon ROI graphic at the given (y, x) vertices."""
        item = record["item"]
        item.setPos(0, 0)
        item.setPoints([(x, y) for (y, x) in vertices])
        cy, cx = polygon_centroid(vertices)
        record["text"].setPos(cx, cy)

    def _sync_roi_positions(self, frame=None):
        """Place every ROI graphic: tracked raw position if tracking, else model."""
        if frame is None:
            frame = self.image.currentIndex
        for rec in self._roi_records:
            roi = rec["model"]
            if roi.shape == ROIShape.POLYGON:
                verts = self._roi_raw_vertices(roi, int(frame)) if self._tracking else roi.vertices
                self._place_polygon_item(rec, verts)
            else:
                cy, cx = self._roi_raw_center(roi, int(frame)) if self._tracking else roi.center
                self._place_roi_item(rec, cy, cx)

    def _on_track_toggled(self, checked: bool):
        self._tracking = bool(checked)
        # A half-drawn outline can't survive the switch to the raw view; drop it.
        if self._tracking and self.poly_btn.isChecked():
            self.poly_btn.setChecked(False)
        # Editing only makes sense on the stabilized view; pause it while tracking.
        for w in (self.shape_box, self.size_box, self.del_btn, self.poly_btn):
            w.setEnabled(not self._tracking)
        for rec in self._roi_records:
            rec["item"].translatable = not self._tracking
        if self._tracking:
            self.hint.setText("Tracking motion — scrub the preview; uncheck to edit")
            stack = np.asarray(self.session.data)            # raw, unstabilized
        else:
            self.hint.setText("Click the image to place an ROI")
            stack = np.asarray(self.session._working_stack())
        frame = self.image.currentIndex                       # preserve the slider
        self.image.setImage(stack, axes={"t": 0, "y": 1, "x": 2})
        self.image.setCurrentIndex(frame)
        self._sync_roi_positions(frame)

    def _on_frame_changed(self, ind, _time):
        if self._tracking:
            self._sync_roi_positions(int(ind))

    # ---------------------------------------------------------- leaf boxes
    def _add_leaf_reference(self, index, leaf):
        """Draw a leaf box as a static, non-interactive reference overlay.

        Editing leaf boxes lives in the separate leaf-selection widget; here they
        are read-only context (a plain rectangle, not a pg.ROI) so they neither
        move nor intercept the clicks/drags used to place and move ROIs.
        """
        y0, y1, x0, x1 = leaf.bbox
        item = QtWidgets.QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        item.setPen(_LEAF_PEN)
        item.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        text = pg.TextItem(leaf.label or f"leaf {index}", color="#ffd000", anchor=(0, 1.1))
        text.setPos(x0, y0)
        self.image.view.addItem(item)
        self.image.view.addItem(text)
        self._leaf_records.append({"model": leaf, "item": item, "text": text})

    # ------------------------------------------------------------- traces
    def _sync_time_axis(self):
        """Label the live-trace x-axis in seconds when the Timeline is calibrated.

        Read-only here: ROIs are placed in Stage II, so the frame interval is set
        elsewhere (notebook or analysis widget); this just reflects it. SPEC §3.
        """
        tl = self.session.timeline
        interval = tl.frame_interval if tl is not None else None
        self._time_axis.set_frame_interval(interval)
        self.trace_plot.setLabel("bottom", "time (s)" if interval else "frame")

    def _refresh_traces(self):
        """Live preview: redraw every ROI as ΔF/F0 with F0 = the first frame.

        Display-only normalization ((F - F[0]) / F[0]) so the response is
        comparable across ROIs regardless of brightness; the stored/exported
        traces remain raw mean intensity (SPEC §3).
        """
        self.trace_plot.clear()
        self._sync_time_axis()
        if not self.session.rois:
            return
        traces = self.session.extract_traces()
        n = traces.raw.shape[0]
        for i in range(n):
            raw = traces.raw[i]
            f0 = float(raw[0])
            dff = (raw - f0) / f0 if f0 else np.zeros_like(raw)
            self.trace_plot.plot(dff, pen=pg.intColor(i, hues=max(6, n)),
                                 name=traces.labels[i])

    # -------------------------------------------------------------- saving
    def _overlay_specs(self, frame):
        """ROI + leaf overlay specs at ``frame`` (tracked positions if tracking).

        Colours use the Okabe-Ito palette so an ROI matches its trace colour in
        the exported traces figure.
        """
        from .. import figures

        labels = self.session.traces.labels if self.session.traces else None
        specs = []
        for i, roi in enumerate(self.session.rois):
            color = figures.roi_color(i)
            label = (labels[i] if labels and i < len(labels)
                     else roi.label or str(i))
            if roi.shape == ROIShape.POLYGON:
                verts = self._roi_raw_vertices(roi, frame) if self._tracking else roi.vertices
                specs.append({"kind": "polygon", "vertices": verts,
                              "center": polygon_centroid(verts), "size": 0,
                              "color": color, "label": label})
            else:
                cy, cx = self._roi_raw_center(roi, frame) if self._tracking else roi.center
                specs.append({"kind": "circle" if roi.shape == ROIShape.CIRCLE else "rect",
                              "center": (cy, cx), "size": roi.size,
                              "color": color, "label": label})
        for i, leaf in enumerate(self.session.leaf_regions):
            y0, _y1, x0, _x1 = leaf.bbox
            specs.append({"kind": "bbox", "bbox": leaf.bbox, "center": (y0, x0),
                          "size": 0, "color": "#E69F00", "lw": 0.6, "ls": "--",
                          "label": leaf.label or f"leaf {i}"})
        return specs

    def _save_overlay(self):
        """Export the ROI overlay on the current frame as shown (WYSIWYG, clean).

        Uses the frame you're viewing and the view's current contrast; ROI
        shapes/labels are drawn in the Okabe-Ito palette.
        """
        if self.session.data is None:
            self.hint.setText("No data loaded.")
            return
        frame = int(self.image.currentIndex)
        stack = np.asarray(self.session.data if self._tracking
                           else self.session._working_stack())
        image = stack[frame]
        levels = self.image.getLevels()
        overlays = self._overlay_specs(frame)

        def render(path):
            from .. import figures

            fig = figures.export_image(image, levels=levels, cmap="gray",
                                       overlays=overlays, save=path)
            import matplotlib.pyplot as plt

            plt.close(fig)

        save_figure_dialog(self, render, title="Save ROI overlay", status=self.hint)

    def _save_traces(self):
        """Export the live ΔF/F₀ trace panel as shown (WYSIWYG, clean palette).

        Reproduces the panel's display-only normalization ((F − F[0]) / F[0]);
        the stored/exported traces themselves remain raw mean intensity (SPEC §3).
        """
        if not self.session.rois:
            self.hint.setText("Place an ROI first.")
            return
        traces = self.session.extract_traces()
        dff0 = []
        for raw in traces.raw:
            f0 = float(raw[0])
            dff0.append((raw - f0) / f0 if f0 else np.zeros_like(raw))
        tl = self.session.timeline
        iv = tl.frame_interval if (tl is not None and tl.frame_interval) else None
        x = np.arange(traces.raw.shape[1]) * (iv or 1)
        xlabel = "time (s)" if iv else "frame"

        def render(path):
            from .. import figures

            fig = figures.export_traces(dff0, x=x, xlabel=xlabel, ylabel="ΔF/F₀",
                                        labels=list(traces.labels), save=path)
            import matplotlib.pyplot as plt

            plt.close(fig)

        save_figure_dialog(self, render, title="Save ROI traces", status=self.hint)

    # -------------------------------------------------------------- events
    def _on_scene_click(self, ev):
        if ev.button() != _LEFT or self._tracking:
            return
        vb = self.image.view
        p = vb.mapSceneToView(ev.scenePos())
        col, row = p.x(), p.y()
        h, w = self._shape_yx
        inside = 0 <= row < h and 0 <= col < w

        # Free-hand mode: clicks trace an outline; a double-click finishes it.
        if self.poly_btn.isChecked():
            if ev.double():
                self.finish_polygon()
            elif inside:
                self.add_polygon_point(row, col)
            return

        # Don't spawn a new ROI when the click lands on an existing ROI (let it
        # drag). Leaf reference rectangles are not pg.ROI items, so clicks inside
        # a leaf box still place an ROI.
        roi_items = {rec["item"] for rec in self._roi_records}
        if any(it in roi_items for it in vb.scene().items(ev.scenePos())):
            return
        if inside:
            self.add_roi_at(row, col)

    def closeEvent(self, event):
        self.result = self.session.rois
        self.closed.emit()
        super().closeEvent(event)
