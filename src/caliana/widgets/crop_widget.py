"""Stage II→III — trace cropping widget. SPEC.md §3.

After ROIs are placed, restrict every trace to a time interval before analysis:

- Preview the full-length ROI traces (raw or a ΔF/F preview).
- Drag the shaded window (or type start/end frames) to the interval of interest.
- "Crop to window" validates: the interval is stored on the Session so every
  later `extract_traces`/`analyze` sees only that window, and the cropped
  `Traces` is returned to the notebook.

The interaction logic lives in plain methods (`set_interval`, `apply_crop`) so it
can be driven from tests without a real mouse; the button is wired to `apply_crop`.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from .. import roi as roi_mod
from ._plot import FrameTimeAxis
from ._qt import get_qt, save_figure_dialog

QtCore, QtGui, QtWidgets = get_qt()

_WINDOW_BRUSH = pg.mkBrush(0, 128, 255, 40)


class CropTracesWidget(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Caliana — Crop Traces")
        self.resize(940, 560)

        # Preview the full (uncropped) traces so the whole recording is visible
        # and the window is chosen in original frame coordinates.
        self._preview = roi_mod.extract_all_traces(
            session._working_stack(), session.rois
        )
        self._n_frames = self._preview.raw.shape[1]
        # Result defaults to the session's current traces so closing without
        # cropping is a no-op for the caller.
        self.result = session.traces

        self._syncing = False
        self._build_ui()
        self._load()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QHBoxLayout()
        self.show_dff = QtWidgets.QCheckBox("ΔF/F preview")
        self.show_dff.setChecked(True)
        self.show_dff.setToolTip("Preview each trace as (F − F[0]) / F[0]")
        self.show_dff.toggled.connect(self._redraw_traces)
        bar.addWidget(self.show_dff)
        bar.addSpacing(24)

        bar.addWidget(QtWidgets.QLabel("Start:"))
        self.start_box = QtWidgets.QSpinBox()
        self.start_box.setRange(0, max(0, self._n_frames - 1))
        self.start_box.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.start_box)

        bar.addWidget(QtWidgets.QLabel("End:"))
        self.end_box = QtWidgets.QSpinBox()
        self.end_box.setRange(1, self._n_frames)
        self.end_box.setValue(self._n_frames)
        self.end_box.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.end_box)

        self.crop_btn = QtWidgets.QPushButton("Crop to window")
        self.crop_btn.clicked.connect(self.apply_crop)
        bar.addWidget(self.crop_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.setToolTip("Clear the crop and use the whole recording")
        self.reset_btn.clicked.connect(self.reset_crop)
        bar.addWidget(self.reset_btn)

        self.save_btn = QtWidgets.QPushButton("Save traces…")
        self.save_btn.setToolTip("Save the ROI mean-intensity traces as a figure")
        self.save_btn.clicked.connect(self._save_traces)
        bar.addWidget(self.save_btn)

        bar.addStretch(1)
        self.status = QtWidgets.QLabel("")
        bar.addWidget(self.status)
        layout.addLayout(bar)

        self._time_axis = FrameTimeAxis(orientation="bottom")
        self.plot = pg.PlotWidget(title="ROI traces — drag to select the crop window",
                                  axisItems={"bottom": self._time_axis})
        self.plot.setLabel("bottom", "frame")
        self.plot.addLegend()
        layout.addWidget(self.plot, stretch=1)

        # Draggable interval; edges snap to whole frames as it moves.
        self.region = pg.LinearRegionItem(
            values=(0, self._n_frames), brush=_WINDOW_BRUSH, swapMode="sort"
        )
        self.region.setBounds((0, self._n_frames))
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.plot.addItem(self.region)

    def _load(self):
        # Reflect any crop already on the session; else the whole recording.
        if self.session.crop_window is not None:
            lo, hi = self.session.crop_window
        else:
            lo, hi = 0, self._n_frames
        self.set_interval(lo, hi)
        self._redraw_traces()

    # ------------------------------------------------------------- helpers
    def _frame_interval(self):
        tl = self.session.timeline
        return tl.frame_interval if (tl is not None and tl.frame_interval) else None

    def _preview_data(self):
        """The array plotted: a ΔF/F preview if requested, else the raw traces."""
        raw = self._preview.raw
        if not self.show_dff.isChecked():
            return raw, "mean intensity"
        f0 = raw[:, :1]
        with np.errstate(divide="ignore", invalid="ignore"):
            dff = np.where(f0 != 0, (raw - f0) / f0, 0.0)
        return dff, "ΔF/F₀"

    # ------------------------------------------------------------- drawing
    def _redraw_traces(self):
        self.plot.clear()
        self.plot.addItem(self.region)

        interval = self._frame_interval()
        self._time_axis.set_frame_interval(interval)
        self.plot.setLabel("bottom", "time (s)" if interval else "frame")

        data, ylabel = self._preview_data()
        self.plot.setLabel("left", ylabel)
        n = data.shape[0]
        for i in range(n):
            self.plot.plot(data[i], pen=pg.intColor(i, hues=max(6, n)),
                           name=self._preview.labels[i])

    # --------------------------------------------------------- interval sync
    def set_interval(self, start: int, end: int):
        """Set the crop window [start, end) across the region and both spinboxes."""
        start = int(np.clip(start, 0, self._n_frames - 1))
        end = int(np.clip(end, start + 1, self._n_frames))
        self._syncing = True
        try:
            self.start_box.setValue(start)
            self.end_box.setValue(end)
            self.region.setRegion((start, end))
        finally:
            self._syncing = False

    def _on_region_changed(self):
        if self._syncing:
            return
        lo, hi = self.region.getRegion()
        self.set_interval(round(lo), round(hi))

    def _on_spin_changed(self, _value):
        if self._syncing:
            return
        self.set_interval(self.start_box.value(), self.end_box.value())

    # -------------------------------------------------------------- actions
    def apply_crop(self):
        """Validate: store the window on the session and re-extract cropped traces."""
        start, end = self.start_box.value(), self.end_box.value()
        self.result = self.session.set_crop(start, end)
        span = "whole recording" if self.session.crop_window is None \
            else f"frames [{start}, {end})"
        self.status.setText(f"Cropped to {span}.")
        self.close()
        return self.result

    def reset_crop(self):
        """Clear the crop (use the whole recording) and reset the window."""
        self.set_interval(0, self._n_frames)
        self.result = self.session.set_crop(None, None)
        self.status.setText("Crop cleared (whole recording).")
        return self.result

    # -------------------------------------------------------------- saving
    def _save_traces(self):
        """Export the preview traces as shown (WYSIWYG), with the crop window.

        Mirrors the panel: raw or ΔF/F preview per the checkbox, the shaded crop
        window, and the frames/seconds x-axis — restyled with a cleaner palette.
        """
        data, ylabel = self._preview_data()
        if data.shape[0] == 0:
            self.status.setText("No ROIs to plot.")
            return
        iv = self._frame_interval()
        scale = iv or 1
        x = np.arange(data.shape[1]) * scale
        xlabel = "time (s)" if iv else "frame"
        lo, hi = self.region.getRegion()
        regions = [(lo * scale, hi * scale, "#0072B2")]
        labels = list(self._preview.labels)

        def render(path):
            from .. import figures

            fig = figures.export_traces(
                [data[i] for i in range(data.shape[0])], x=x, xlabel=xlabel,
                ylabel=ylabel, labels=labels, regions=regions, save=path,
            )
            import matplotlib.pyplot as plt

            plt.close(fig)

        save_figure_dialog(self, render, title="Save traces", status=self.status)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
