"""Stage II — leaf-box selection widget. SPEC.md §3 Stage II (per-leaf mode).

Draw one box per leaf; each box is registered independently (per-leaf motion
correction) and ROIs later auto-assign to the box that contains them. Leaf boxes
live on their own widget — separate from ROI placement — so the movable leaf
rectangles and the ROI click-to-place interaction don't fight over the mouse.

A box may also carry a **mask**: a polygon traced inside it, the way the ROI
widget traces a free-hand ROI, naming the tissue whose motion should drive that
box's registration. Only pixels inside the outline take part in the fit (the box
is still warped whole), which is what lets a leaf be tracked when its box
unavoidably also contains a neighbour or a bright static background. Masks are
optional; without one the whole box is used.

The interaction logic lives in plain methods (`add_leaf_box`, `delete_last_leaf`,
`add_mask_point`, `finish_mask`) so it can be driven from tests without a real
mouse.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from .. import figures
from ..roi import polygon_centroid
from ._plot import polyline_vertices
from ._qt import get_qt

QtCore, QtGui, QtWidgets = get_qt()

pg.setConfigOption("imageAxisOrder", "row-major")

_LEFT = QtCore.Qt.MouseButton.LeftButton
_LEAF_PEN = pg.mkPen("#ffd000", width=2, style=QtCore.Qt.PenStyle.DashLine)
# Masks are drawn solid and in a different hue from the boxes: the two are
# nested on screen and mean different things.
_MASK_COLOR = "#00e5ff"
_MASK_PEN = pg.mkPen(_MASK_COLOR, width=2)

_DEFAULT_HINT = "Add a box per leaf, then drag/resize it to cover the leaf generously"
_MASK_HINT = "Click inside a leaf box to trace the tissue to track; double-click to finish"


class LeafSelectionWidget(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session.leaf_regions
        self.setWindowTitle("Caliana — Leaf Selection")
        self.resize(720, 580)

        # Bookkeeping: parallel records linking model leaf regions to graphics.
        self._leaf_records: list[dict] = []
        self._shape_yx = (1, 1)
        # In-progress mask outline: accumulated (y, x) points + preview item.
        self._mask_points: list[tuple[float, float]] = []
        self._mask_preview = None

        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.leaf_btn = QtWidgets.QPushButton("Add leaf box")
        self.leaf_btn.clicked.connect(lambda: self.add_leaf_box())
        bar.addWidget(self.leaf_btn)

        self.del_btn = QtWidgets.QPushButton("Delete last leaf")
        self.del_btn.clicked.connect(self.delete_last_leaf)
        bar.addWidget(self.del_btn)

        self.mask_btn = QtWidgets.QPushButton("Draw mask")
        self.mask_btn.setCheckable(True)
        self.mask_btn.setToolTip(
            "Trace the tissue to track inside a leaf box: only pixels inside the "
            "outline drive that box's registration, so a neighbouring leaf or a "
            "static background sharing the box cannot anchor the estimate.\n"
            "Click to add points, double-click (or press the button again) to "
            "finish. Draw it a couple of pixels outside the tissue edge."
        )
        self.mask_btn.toggled.connect(self._on_mask_toggled)
        bar.addWidget(self.mask_btn)

        self.clear_mask_btn = QtWidgets.QPushButton("Clear masks")
        self.clear_mask_btn.setToolTip(
            "Drop every mask outline, registering each box on all its pixels again "
            "(a single outline can also be removed by right-clicking it)"
        )
        self.clear_mask_btn.clicked.connect(self.clear_masks)
        bar.addWidget(self.clear_mask_btn)

        bar.addStretch(1)
        self.hint = QtWidgets.QLabel(_DEFAULT_HINT)
        bar.addWidget(self.hint)
        layout.addLayout(bar)

        self.image = pg.ImageView(name="leaf_image")
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        # Simplified contrast: keep the level region, drop the colormap editor.
        self.image.ui.histogram.gradient.hide()
        layout.addWidget(self.image, stretch=1)

        self.image.view.scene().sigMouseClicked.connect(self._on_scene_click)

    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        Existing box graphics are dropped first, so a session whose leaf regions
        changed elsewhere (or whose stack was replaced) comes back with exactly
        one graphic per model box rather than duplicates.
        """
        self._clear_leaf_graphics()
        if self.session.data is None:
            self._shape_yx = (1, 1)
            self.image.setImage(np.zeros((1, 1, 1)))
            self.leaf_btn.setEnabled(False)
            self.mask_btn.setEnabled(False)
            self.hint.setText("No data loaded.")
            return
        self.leaf_btn.setEnabled(True)
        self.mask_btn.setEnabled(True)
        stack = np.asarray(self.session._working_stack())
        self._shape_yx = stack.shape[1:]
        # Same first-frame [min, 99th pct] contrast as the import preview, so a
        # leaf looks identical whichever tab you're on.
        self.image.setImage(stack, axes={"t": 0, "y": 1, "x": 2},
                            autoLevels=False, levels=figures.intensity_levels(stack))
        # Re-draw any leaf boxes already on the session.
        for i, leaf in enumerate(list(self.session.leaf_regions)):
            self._add_leaf_graphic(i, leaf)

    def _clear_leaf_graphics(self):
        """Remove every box/mask graphic, leaving the session's models untouched.

        Any half-traced outline goes too: it belongs to the boxes on screen, not
        to whatever ``reload`` is about to draw.
        """
        self._clear_mask_preview()          # before unchecking: discard, don't commit
        if self.mask_btn.isChecked():
            self.mask_btn.setChecked(False)
        for record in self._leaf_records:
            self.image.view.removeItem(record["item"])
            self.image.view.removeItem(record["text"])
            self._remove_mask_graphic(record)
        self._leaf_records.clear()

    # ---------------------------------------------------------- leaf boxes
    def add_leaf_box(self, bbox=None):
        """Add a leaf region. SPEC §3 (per-leaf mode). bbox = (y0, y1, x0, x1).

        Unlike ROIs, leaf boxes stay resizable — drag the handles to cover the
        whole leaf (tissue that drifts outside its box can't be stabilized).
        """
        if bbox is None:
            h, w = self._shape_yx
            y0, y1 = int(h * 0.3), int(h * 0.7)
            x0, x1 = int(w * 0.3), int(w * 0.7)
            bbox = (y0, y1, x0, x1)
        leaf = self.session.add_leaf_region(bbox)
        self._add_leaf_graphic(len(self.session.leaf_regions) - 1, leaf)
        return leaf

    def _add_leaf_graphic(self, index, leaf):
        y0, y1, x0, x1 = leaf.bbox
        item = pg.RectROI((x0, y0), (x1 - x0, y1 - y0), pen=_LEAF_PEN, movable=True)
        text = pg.TextItem(leaf.label or f"leaf {index}", color="#ffd000", anchor=(0, 1.1))
        text.setPos(x0, y0)
        self.image.view.addItem(item)
        self.image.view.addItem(text)
        record = {"model": leaf, "item": item, "text": text, "mask": None}
        self._leaf_records.append(record)
        item.sigRegionChanged.connect(lambda it, rec=record: self._on_leaf_moved(rec))
        if leaf.mask_polygon:
            self._add_mask_graphic(record)

    def _on_leaf_moved(self, record):
        item = record["item"]
        pos, size = item.pos(), item.size()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + size.x(), y0 + size.y()
        leaf = record["model"]
        oy0, _oy1, ox0, _ox1 = leaf.bbox
        leaf.bbox = (int(y0), int(y1), int(x0), int(x1))
        record["text"].setPos(x0, y0)
        # A mask names tissue *within* its box, so dragging the box carries the
        # outline along; a resize (the handle moves the far corner, not the
        # origin) leaves it where it is.
        dy, dx = leaf.bbox[0] - oy0, leaf.bbox[2] - ox0
        if leaf.mask_polygon and (dy or dx):
            leaf.mask_polygon = [(y + dy, x + dx) for (y, x) in leaf.mask_polygon]
            self._place_mask_graphic(record)

    def delete_last_leaf(self):
        if not self._leaf_records:
            return
        record = self._leaf_records.pop()
        self.image.view.removeItem(record["item"])
        self.image.view.removeItem(record["text"])
        self._remove_mask_graphic(record)
        self.session.leaf_regions.remove(record["model"])
        self.session._bump()       # other panels draw these boxes too

    # -------------------------------------------------------------- masks
    def _on_mask_toggled(self, checked: bool):
        """Enter mask-tracing mode on check; commit (or discard) the outline on
        uncheck. Boxes are frozen while tracing so their drag/resize handles
        don't swallow the clicks meant for the outline."""
        if checked:
            self.start_mask()
        elif len(self._mask_points) >= 3:
            self._commit_mask()
        else:
            self._clear_mask_preview()
            self.hint.setText(_DEFAULT_HINT)
        self.leaf_btn.setEnabled(not checked)
        self.del_btn.setEnabled(not checked)
        for rec in self._leaf_records:
            rec["item"].setEnabled(not checked)
            if rec["mask"] is not None:
                rec["mask"].setEnabled(not checked)

    def start_mask(self):
        """Begin a new mask outline (clears any in-progress one)."""
        self._clear_mask_preview()
        self._mask_preview = pg.PlotDataItem(
            pen=_MASK_PEN, symbol="o", symbolSize=6, symbolBrush=_MASK_COLOR
        )
        self.image.view.addItem(self._mask_preview)
        self.hint.setText(_MASK_HINT)

    def add_mask_point(self, row: float, col: float):
        """Append a vertex to the in-progress outline and redraw the preview."""
        self._mask_points.append((row, col))
        if self._mask_preview is not None:
            ys = [p[0] for p in self._mask_points]
            xs = [p[1] for p in self._mask_points]
            self._mask_preview.setData(xs, ys)

    def finish_mask(self):
        """Commit the in-progress outline to its leaf box (needs >= 3 points)."""
        # Unchecking the button routes through _on_mask_toggled -> _commit_mask.
        self.mask_btn.setChecked(False)

    def _commit_mask(self):
        """Attach the traced outline to the box containing its centroid.

        An outline drawn over no box is dropped rather than stored: masks are
        per-box, and one with no box would silently do nothing at registration.
        """
        vertices = self._mask_points
        self._clear_mask_preview()
        record = self._record_at(*polygon_centroid(vertices))
        if record is None:
            self.hint.setText("Mask discarded — trace it inside a leaf box.")
            return None
        self.session.set_leaf_mask(record["model"], vertices)
        self._add_mask_graphic(record)
        self.hint.setText(f"{self._leaf_name(record)} masked to {len(vertices)} points")
        return record["model"]

    def _clear_mask_preview(self):
        if self._mask_preview is not None:
            self.image.view.removeItem(self._mask_preview)
            self._mask_preview = None
        self._mask_points = []

    def _record_at(self, row: float, col: float):
        """The record whose box contains ``(row, col)``, or None."""
        for record in self._leaf_records:
            y0, y1, x0, x1 = record["model"].bbox
            if y0 <= row < y1 and x0 <= col < x1:
                return record
        return None

    def _leaf_name(self, record) -> str:
        """The box's label, else ``leaf <index>`` — matching its on-image text."""
        index = next(i for i, r in enumerate(self._leaf_records) if r is record)
        return record["model"].label or f"leaf {index}"

    def _add_mask_graphic(self, record):
        """Draw (or redraw) a leaf's stored outline as an editable polygon."""
        self._remove_mask_graphic(record)
        pts = [(x, y) for (y, x) in record["model"].mask_polygon]  # pg points are (x, y)
        item = pg.PolyLineROI(pts, closed=True, pen=_MASK_PEN, movable=True,
                              removable=True)
        self.image.view.addItem(item)
        record["mask"] = item
        item.sigRegionChanged.connect(lambda it, rec=record: self._on_mask_moved(rec))
        item.sigRemoveRequested.connect(lambda it, rec=record: self.clear_leaf_mask(rec["model"]))

    def _place_mask_graphic(self, record):
        """Move a mask graphic onto its model's current vertices."""
        item = record["mask"]
        if item is None:
            return
        item.setPos(0, 0)
        item.setPoints([(x, y) for (y, x) in record["model"].mask_polygon])

    def _remove_mask_graphic(self, record):
        if record["mask"] is not None:
            self.image.view.removeItem(record["mask"])
            record["mask"] = None

    def _on_mask_moved(self, record):
        """Read an edited outline back onto its leaf (handles are dragged live)."""
        vertices = polyline_vertices(record["mask"])
        if len(vertices) < 3:          # mid-edit / teardown — keep the last valid one
            return
        record["model"].mask_polygon = vertices

    def clear_leaf_mask(self, leaf):
        """Drop one leaf's mask outline — model and graphic."""
        record = next((r for r in self._leaf_records if r["model"] is leaf), None)
        if record is None:
            return
        self._remove_mask_graphic(record)
        self.session.clear_leaf_mask(leaf)
        self.hint.setText(f"{self._leaf_name(record)} mask cleared")

    def clear_masks(self):
        """Drop every mask outline: each box registers on all its pixels again."""
        masked = [r for r in self._leaf_records if r["model"].mask_polygon]
        for record in masked:
            self._remove_mask_graphic(record)
            self.session.clear_leaf_mask(record["model"])
        self.hint.setText(f"{len(masked)} mask(s) cleared" if masked
                          else "No masks to clear.")

    # -------------------------------------------------------------- events
    def _on_scene_click(self, ev):
        """Left clicks trace a mask outline; a double-click finishes it.

        Outside mask mode the click is left alone — leaf boxes are placed with
        the button and positioned by dragging, so nothing here should react to a
        stray click on the image.
        """
        if ev.button() != _LEFT or not self.mask_btn.isChecked():
            return
        if ev.double():
            self.finish_mask()
            return
        p = self.image.view.mapSceneToView(ev.scenePos())
        row, col = p.y(), p.x()
        h, w = self._shape_yx
        if 0 <= row < h and 0 <= col < w:
            self.add_mask_point(row, col)

    def closeEvent(self, event):
        self.result = self.session.leaf_regions
        self.closed.emit()
        super().closeEvent(event)
