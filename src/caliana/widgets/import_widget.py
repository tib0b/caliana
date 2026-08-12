"""Stage I — import & preview widget. SPEC.md §3 Stage I.

Provides:
- frame scrubbing of the (stabilized, if registered) movie,
- contrast / colormap controls (pyqtgraph's histogram LUT),
- a normalized max-intensity heatmap beside the movie.

Built on pyqtgraph's ImageView, which supplies the time slider and the
contrast/colormap histogram out of the box. Playback is deliberately not
offered: the time slider is a scrubber only.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from .. import figures
from ._plot import pixel_size
from ._qt import get_qt, save_figure_dialog

QtCore, QtGui, QtWidgets = get_qt()

# Display images as [row=y, col=x] (numpy convention) rather than pyqtgraph's
# legacy [x, y]. Set once at import; harmless if set repeatedly.
pg.setConfigOption("imageAxisOrder", "row-major")

# Width of the contrast (histogram LUT) strip beside each image, in px.
_HISTOGRAM_WIDTH = 58


class _ScrubImageView(pg.ImageView):
    """ImageView with playback disabled — the time slider only scrubs.

    pyqtgraph binds the space bar to play/pause; swallowing it (and its release)
    leaves scrubbing and arrow-key frame stepping intact.
    """

    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            ev.accept()
            return
        super().keyPressEvent(ev)

    def keyReleaseEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space:
            ev.accept()
            return
        super().keyReleaseEvent(ev)


class ImportPreviewWidget(QtWidgets.QWidget):
    """Scrub + contrast + max-projection preview of a Session's stack."""

    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session
        self.setWindowTitle("Caliana — Import & Preview")
        self.resize(1000, 560)
        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        # Movie: ImageView gives the time slider, scrubbing and contrast.
        self.movie = _ScrubImageView(name="movie")
        split.addWidget(self._titled("Movie (scrub)", self.movie))

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
        # With the gradient gone the histogram only needs room for its curve and
        # tick labels, so it is narrowed to leave the images the width they earn
        # (pyqtgraph's defaults — 95 px widget, 45 px plot — are sized for the
        # gradient editor we just hid).
        for iv in (self.movie, self.heatmap):
            iv.ui.roiBtn.hide()
            iv.ui.menuBtn.hide()
            iv.ui.histogram.gradient.hide()
            hist = iv.ui.histogram
            hist.setMinimumWidth(0)
            hist.setFixedWidth(_HISTOGRAM_WIDTH)
            hist.item.vb.setMinimumWidth(0)
            hist.item.vb.setMaximumWidth(16)

        controls = QtWidgets.QHBoxLayout()
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
    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        The app calls this when the Import tab is activated after the stack
        changed (a file opened, or re-imported at another downsampling); in a
        notebook it runs once, from ``__init__``.
        """
        if self.session.data is None:
            self.movie.setImage(np.zeros((1, 1, 1)))
            self.heatmap.setImage(np.zeros((1, 1)))
            self.status.setText("No data loaded.")
            return
        stack = np.asarray(self.session._working_stack())
        # axes maps stack dims -> ImageView roles (row-major: 1=y, 2=x). Contrast
        # defaults to figures.intensity_levels' [min, 99th pct] of the first frame
        # — the same range the saved figure uses, so the export matches the
        # preview — while the histogram still spans the full data range and can be
        # dragged wider.
        self.movie.setImage(stack, axes={"t": 0, "y": 1, "x": 2},
                            autoLevels=False, levels=figures.intensity_levels(stack))
        mip = self.session.max_projection()
        self.heatmap.setImage(mip, autoLevels=False,
                              levels=figures.intensity_levels(mip))
        self.status.setText("")

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
            return figures.export_image(
                image, levels=levels, cmap="inferno",
                cbar_label="normalized max intensity",
                pixel_size=pixel_size(self.session), save=path,
            )

        save_figure_dialog(self, render, title="Save max-intensity image",
                           status=self.status)

    # ------------------------------------------------------------ signals
    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
