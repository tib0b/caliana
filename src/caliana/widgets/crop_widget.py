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

from .. import figures
from ._plot import FrameTimeAxis, dff0, frame_interval
from ._qt import get_qt, save_figure_dialog

QtCore, QtGui, QtWidgets = get_qt()

_WINDOW_BRUSH = pg.mkBrush(0, 128, 255, 40)


class CropTracesWidget(QtWidgets.QWidget):
    closed = QtCore.Signal()
    # "The crop was validated" — the notebook wrapper closes the window on it,
    # the app's Crop tab just refreshes (see _qt.run_widget_blocking).
    applied = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Caliana — Crop Traces")
        self.resize(940, 560)

        self._preview = None       # full-length Traces; None until data + ROIs exist
        self._n_frames = 0
        # Result defaults to the session's current traces so closing without
        # cropping is a no-op for the caller.
        self.result = session.traces

        self._syncing = False
        self._build_ui()
        self.reload()

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
        self.start_box.valueChanged.connect(self._on_spin_changed)
        bar.addWidget(self.start_box)

        bar.addWidget(QtWidgets.QLabel("End:"))
        self.end_box = QtWidgets.QSpinBox()
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

        # Draggable interval; edges snap to whole frames as it moves. Its bounds
        # follow the recording length, set in reload().
        self.region = pg.LinearRegionItem(brush=_WINDOW_BRUSH, swapMode="sort")
        self.region.sigRegionChanged.connect(self._on_region_changed)
        self.plot.addItem(self.region)

    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        Re-previews the *uncropped* traces through the session's own extraction
        path, so a stack that has since been registered (or ROIs added on another
        tab) is previewed as the crop will actually measure it. A session with no
        data or no ROIs disables the controls rather than raising — the app can
        show this tab before either exists.
        """
        ready = self.session.data is not None and bool(self.session.rois)
        for w in (self.start_box, self.end_box, self.crop_btn, self.reset_btn,
                  self.save_btn, self.show_dff):
            w.setEnabled(ready)
        if not ready:
            self._preview, self._n_frames = None, 0
            self._set_bounds(0)
            self.plot.clear()
            self.plot.addItem(self.region)
            self.status.setText("Load a stack and place ROIs first.")
            return

        # Preview the full (uncropped) traces so the whole recording is visible
        # and the window is chosen in original frame coordinates. Extracted through
        # the session's own path, so a motion-tracked session previews the same
        # signal the crop will actually produce.
        self._preview = self.session._extract_window()
        self._n_frames = self._preview.raw.shape[1]
        self._set_bounds(self._n_frames)
        # Reflect any crop already on the session; else the whole recording.
        if self.session.crop_window is not None:
            lo, hi = self.session.crop_window
        else:
            lo, hi = 0, self._n_frames
        self.set_interval(lo, hi)
        self.status.setText("")
        self._redraw_traces()

    def _set_bounds(self, n_frames: int):
        """Point the spinboxes and the draggable window at an ``n_frames`` recording."""
        self._syncing = True
        try:
            self.start_box.setRange(0, max(0, n_frames - 1))
            self.end_box.setRange(min(1, n_frames), n_frames)
            self.end_box.setValue(n_frames)
            self.region.setBounds((0, n_frames))
            self.region.setRegion((0, n_frames))
        finally:
            self._syncing = False

    # ------------------------------------------------------------- helpers
    def _frame_interval(self):
        return frame_interval(self.session)

    def _preview_data(self):
        """The array plotted: a ΔF/F preview if requested, else the raw traces."""
        if self._preview is None:
            return np.empty((0, 0)), "mean intensity"
        if not self.show_dff.isChecked():
            return self._preview.raw, "mean intensity"
        return dff0(self._preview.raw), "ΔF/F₀"

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
        if self._n_frames <= 0:          # nothing loaded; there is no window to set
            return
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
        """Validate: store the window on the session and re-extract cropped traces.

        Emits ``applied`` rather than closing: that is the notebook wrapper's job
        (one cell, one window), while the app keeps the tab open.
        """
        start, end = self.start_box.value(), self.end_box.value()
        self.result = self.session.set_crop(start, end)
        span = "whole recording" if self.session.crop_window is None \
            else f"frames [{start}, {end})"
        self.status.setText(f"Cropped to {span}.")
        self.applied.emit()
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
            return figures.export_traces(
                list(data), x=x, xlabel=xlabel, ylabel=ylabel, labels=labels,
                regions=regions, save=path,
            )

        save_figure_dialog(self, render, title="Save traces", status=self.status)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
