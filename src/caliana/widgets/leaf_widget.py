"""Stage II — leaf-box selection widget. SPEC.md §3 Stage II (per-leaf mode).

Draw one box per leaf; each box is registered independently (per-leaf motion
correction) and ROIs later auto-assign to the box that contains them. Leaf boxes
live on their own widget — separate from ROI placement — so the movable leaf
rectangles and the ROI click-to-place interaction don't fight over the mouse.

The interaction logic lives in plain methods (`add_leaf_box`, `delete_last_leaf`)
so it can be driven from tests without a real mouse.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ._qt import get_qt

QtCore, QtGui, QtWidgets = get_qt()

pg.setConfigOption("imageAxisOrder", "row-major")

_LEAF_PEN = pg.mkPen("#ffd000", width=2, style=QtCore.Qt.PenStyle.DashLine)


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

        self._build_ui()
        self._load_session()

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

        bar.addStretch(1)
        self.hint = QtWidgets.QLabel(
            "Add a box per leaf, then drag/resize it to cover the leaf generously"
        )
        bar.addWidget(self.hint)
        layout.addLayout(bar)

        self.image = pg.ImageView(name="leaf_image")
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        # Simplified contrast: keep the level region, drop the colormap editor.
        self.image.ui.histogram.gradient.hide()
        layout.addWidget(self.image, stretch=1)

    def _load_session(self):
        if self.session.data is None:
            self.image.setImage(np.zeros((1, 1, 1)))
            return
        stack = np.asarray(self.session._working_stack())
        self._shape_yx = stack.shape[1:]
        self.image.setImage(stack, axes={"t": 0, "y": 1, "x": 2})
        # Re-draw any leaf boxes already on the session.
        for i, leaf in enumerate(list(self.session.leaf_regions)):
            self._add_leaf_graphic(i, leaf)

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
        record = {"model": leaf, "item": item, "text": text}
        self._leaf_records.append(record)
        item.sigRegionChanged.connect(lambda it, rec=record: self._on_leaf_moved(rec))

    def _on_leaf_moved(self, record):
        item = record["item"]
        pos, size = item.pos(), item.size()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + size.x(), y0 + size.y()
        record["model"].bbox = (int(y0), int(y1), int(x0), int(x1))
        record["text"].setPos(x0, y0)

    def delete_last_leaf(self):
        if not self._leaf_records:
            return
        record = self._leaf_records.pop()
        self.image.view.removeItem(record["item"])
        self.image.view.removeItem(record["text"])
        self.session.leaf_regions.remove(record["model"])

    # -------------------------------------------------------------- events
    def closeEvent(self, event):
        self.result = self.session.leaf_regions
        self.closed.emit()
        super().closeEvent(event)
