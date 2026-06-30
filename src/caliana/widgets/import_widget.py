"""Stage I — import & preview widget. SPEC.md §3 Stage I.

Provides:
- frame scrubbing + playback of the (stabilized, if registered) movie,
- contrast / colormap controls (pyqtgraph's histogram LUT),
- a side-by-side normalized max-intensity heatmap.

Built on pyqtgraph's ImageView, which supplies the time slider, play(), and the
contrast/colormap histogram out of the box.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ._qt import get_qt

QtCore, QtGui, QtWidgets = get_qt()

# Display images as [row=y, col=x] (numpy convention) rather than pyqtgraph's
# legacy [x, y]. Set once at import; harmless if set repeatedly.
pg.setConfigOption("imageAxisOrder", "row-major")


class ImportPreviewWidget(QtWidgets.QWidget):
    """Scrub/playback + contrast + max-projection preview of a Session's stack."""

    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session
        self.setWindowTitle("Caliana — Import & Preview")
        self.resize(1000, 560)
        self._build_ui()
        self._load_session()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        # Movie: ImageView gives the time slider, scrubbing and contrast.
        self.movie = pg.ImageView(name="movie")
        split.addWidget(self._titled("Movie (scrub / play)", self.movie))

        # Heatmap: 2D ImageView (no time axis), fixed inferno colormap.
        self.heatmap = pg.ImageView(name="heatmap")
        self.heatmap.ui.histogram.gradient.loadPreset("inferno")
        split.addWidget(self._titled("Normalized max-intensity", self.heatmap))
        split.setSizes([550, 450])

        # Simplify contrast: keep the level region for brightness/contrast, but
        # drop the colormap gradient editor and the ROI/menu buttons (clutter).
        for iv in (self.movie, self.heatmap):
            iv.ui.roiBtn.hide()
            iv.ui.menuBtn.hide()
            iv.ui.histogram.gradient.hide()

        # Playback controls.
        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggled)
        controls.addWidget(self.play_btn)

        controls.addWidget(QtWidgets.QLabel("fps:"))
        self.fps = QtWidgets.QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(10)
        self.fps.valueChanged.connect(self._on_fps_changed)
        controls.addWidget(self.fps)

        self.frame_label = QtWidgets.QLabel("frame 0")
        controls.addWidget(self.frame_label)
        controls.addStretch(1)
        layout.addLayout(controls)

    def _titled(self, title, widget):
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QtWidgets.QLabel(title))
        v.addWidget(widget)
        return box

    # --------------------------------------------------------------- data
    def _load_session(self):
        if self.session.data is None:
            self.movie.setImage(np.zeros((1, 1, 1)))
            return
        stack = self.session._working_stack()
        # axes maps stack dims -> ImageView roles (row-major: 1=y, 2=x).
        self.movie.setImage(np.asarray(stack), axes={"t": 0, "y": 1, "x": 2})
        self.movie.sigTimeChanged.connect(self._on_time_changed)
        self.heatmap.setImage(self.session.max_projection())

    # ------------------------------------------------------------ signals
    def _on_play_toggled(self, playing):
        self.play_btn.setText("Pause" if playing else "Play")
        self.movie.play(self.fps.value() if playing else 0)

    def _on_fps_changed(self, value):
        if self.play_btn.isChecked():
            self.movie.play(value)

    def _on_time_changed(self, index, _time):
        self.frame_label.setText(f"frame {int(index)}")

    def closeEvent(self, event):
        self.movie.play(0)
        self.closed.emit()
        super().closeEvent(event)
