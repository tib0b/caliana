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

from ._qt import get_qt, save_figure_dialog

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
        hm_box = self._titled("Normalized max-intensity", self.heatmap)
        # Save the max-intensity image as a paper-grade figure (the scrubbing
        # movie has no static counterpart, so only the heatmap gets a button).
        self.save_heatmap_btn = QtWidgets.QPushButton("Save image…")
        self.save_heatmap_btn.setToolTip("Save the max-intensity image as a figure (PNG/PDF/SVG)")
        self.save_heatmap_btn.clicked.connect(self._save_heatmap)
        hm_box.layout().addWidget(self.save_heatmap_btn)
        split.addWidget(hm_box)
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
        self.status = QtWidgets.QLabel("")
        controls.addWidget(self.status)
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
        stack = np.asarray(self.session._working_stack())
        # axes maps stack dims -> ImageView roles (row-major: 1=y, 2=x). Default
        # the contrast range to [min, 99th pct] so a few very bright pixels don't
        # wash out the display (see _default_levels); the histogram still spans the
        # full data range, so the region can be dragged wider if needed.
        self.movie.setImage(stack, axes={"t": 0, "y": 1, "x": 2},
                            autoLevels=False, levels=self._default_levels(stack))
        self.movie.sigTimeChanged.connect(self._on_time_changed)
        mip = self.session.max_projection()
        self.heatmap.setImage(mip, autoLevels=False,
                              levels=self._default_levels(mip))

    @staticmethod
    def _default_levels(image):
        """Default display range ``(min, 99th percentile)`` of the intensities.

        Anchoring the low end at the data minimum and clipping the high end at the
        99th percentile keeps a handful of outlier-bright pixels from flattening
        the contrast across the rest of the image. Shared with the static figures
        (``figures.intensity_levels``) so a saved image matches the preview.
        """
        from ..figures import intensity_levels

        return intensity_levels(image)

    # ------------------------------------------------------------- saving
    def _save_heatmap(self):
        """Export the max-intensity heatmap as shown (WYSIWYG, cleaned up).

        Mirrors the on-screen view — same inferno colormap and current contrast
        levels (the histogram region) — as a clean matplotlib figure with a
        colourbar.
        """
        if self.session.data is None:
            self.status.setText("No data loaded.")
            return
        image = self.heatmap.image
        levels = self.heatmap.getLevels()

        def render(path):
            from .. import figures

            fig = figures.export_image(
                image, levels=levels, cmap="inferno",
                cbar_label="normalized max intensity", save=path,
            )
            import matplotlib.pyplot as plt

            plt.close(fig)

        save_figure_dialog(self, render, title="Save max-intensity image",
                           status=self.status)

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
