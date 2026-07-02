"""Stage III — analysis widget. SPEC.md §3 Stage III.

- Choose the ΔF/F baseline: first-N frames, or drag a window on the trace.
- Compute ΔF/F and toggle raw / ΔF/F display.
- Pick one analysis to run; only that analysis' controls are shown:
  - Peak detection (height / prominence): overlay markers, summarise per ROI.
  - Cross-ROI propagation: choose the onset-time method (half_max / std) and its
    parameters (frac / k), and drag a baseline window (the green band) that sets
    the level onsets are measured from; overlay per-ROI onset times, summarise
    speed / direction / source ROI, and plot distance-along-propagation vs onset
    delay with the line implied by that speed and its R². Time readouts switch to
    seconds when a frame interval is set.
- Mark optional stimulus events as draggable vertical lines.

Interaction logic lives in plain methods (`compute_dff`, `detect_peaks`,
`compute_propagation`, `add_event`, `_redraw_traces`) so tests can drive it
without a mouse.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..models import BaselineMethod
from ._plot import FrameTimeAxis
from ._qt import get_qt

QtCore, QtGui, QtWidgets = get_qt()


class AnalysisWidget(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session.analyses
        self.setWindowTitle("Caliana — Analysis")
        self.resize(1060, 600)

        self._curves: list = []
        self._scatters: list = []
        self._event_lines: list = []
        self._onset_lines: list = []

        self._build_ui()
        self._load_session()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Row 1 — baseline / ΔF/F.
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Baseline:"))
        self.baseline_box = QtWidgets.QComboBox()
        self.baseline_box.addItems([BaselineMethod.FIRST_N.value, BaselineMethod.REGION.value])
        self.baseline_box.currentTextChanged.connect(self._on_baseline_changed)
        row1.addWidget(self.baseline_box)

        row1.addWidget(QtWidgets.QLabel("N:"))
        self.n_box = QtWidgets.QSpinBox()
        self.n_box.setRange(1, 100000)
        self.n_box.setValue(10)
        row1.addWidget(self.n_box)

        self.dff_btn = QtWidgets.QPushButton("Compute ΔF/F")
        self.dff_btn.clicked.connect(self.compute_dff)
        row1.addWidget(self.dff_btn)

        self.show_dff = QtWidgets.QCheckBox("Show ΔF/F")
        self.show_dff.toggled.connect(self._redraw_traces)
        row1.addWidget(self.show_dff)

        row1.addWidget(QtWidgets.QLabel("Frame interval (s):"))
        self.interval_box = QtWidgets.QDoubleSpinBox()
        self.interval_box.setRange(0.0, 1e6)
        self.interval_box.setDecimals(4)
        self.interval_box.setSpecialValueText("frames")  # 0 ⇒ frames-only axis
        self.interval_box.valueChanged.connect(self._on_interval_changed)
        row1.addWidget(self.interval_box)
        row1.addStretch(1)
        layout.addLayout(row1)

        # Row 2 — pick the analysis to run + shared stimulus events.
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Analysis:"))
        self.analysis_box = QtWidgets.QComboBox()
        self.analysis_box.addItems(
            ["(select analysis)", "Peak detection", "Cross-ROI propagation"]
        )
        self.analysis_box.currentIndexChanged.connect(self._on_analysis_changed)
        row2.addWidget(self.analysis_box)
        row2.addSpacing(24)

        row2.addWidget(QtWidgets.QLabel("Event @"))
        self.event_box = QtWidgets.QSpinBox()
        self.event_box.setRange(0, 100000)
        row2.addWidget(self.event_box)
        self.event_btn = QtWidgets.QPushButton("Add event")
        self.event_btn.clicked.connect(lambda: self.add_event(self.event_box.value()))
        row2.addWidget(self.event_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        # Row 3 — controls specific to the chosen analysis (empty until picked).
        # Stack pages line up 1:1 with the analysis_box items above.
        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(QtWidgets.QWidget())        # 0: nothing selected
        self.param_stack.addWidget(self._build_peak_panel())   # 1: peak detection
        self.param_stack.addWidget(self._build_prop_panel())   # 2: propagation
        layout.addWidget(self.param_stack)

        # Plot (left) + results / propagation graph (right).
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        self._time_axis = FrameTimeAxis(orientation="bottom")
        self.plot = pg.PlotWidget(title="ROI traces", axisItems={"bottom": self._time_axis})
        self.plot.setLabel("bottom", "frame")
        self.plot.addLegend()
        # Draggable baseline window (used in REGION mode).
        self.region = pg.LinearRegionItem(values=(0, self.n_box.value()))
        self.region.setZValue(-10)
        # Draggable window for the onset-detection baseline (cross-ROI propagation);
        # a distinct green tint keeps it apart from the ΔF/F baseline band above.
        self.prop_region = pg.LinearRegionItem(
            values=(0, self.n_box.value()), brush=pg.mkBrush(0, 200, 120, 40)
        )
        self.prop_region.setZValue(-10)
        split.addWidget(self.plot)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        right.addWidget(self.results)
        # Onset-vs-distance graph, shown only for cross-ROI propagation.
        # Autoranges to fit every point; extra padding keeps the point labels
        # (and the fit line ends) inside the view rather than clipped at the edge.
        self.prop_plot = pg.PlotWidget(title="Distance vs onset delay")
        self.prop_plot.setLabel("bottom", "onset delay (frame)")
        self.prop_plot.setLabel("left", "distance from source (px)")
        self.prop_plot.addLegend()
        self.prop_plot.getViewBox().setDefaultPadding(0.12)
        self.prop_plot.setVisible(False)
        right.addWidget(self.prop_plot)
        right.setSizes([240, 360])
        split.addWidget(right)
        split.setSizes([700, 360])

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

    # ------------------------------------------------------- analysis panels
    def _build_peak_panel(self) -> "QtWidgets.QWidget":
        panel = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("Height:"))
        self.height_box = QtWidgets.QDoubleSpinBox()
        self.height_box.setRange(-1e9, 1e9)
        self.height_box.setSpecialValueText("auto")
        self.height_box.setValue(self.height_box.minimum())  # "auto"
        row.addWidget(self.height_box)

        row.addWidget(QtWidgets.QLabel("Prominence:"))
        self.prom_box = QtWidgets.QDoubleSpinBox()
        self.prom_box.setRange(0.0, 1e9)
        self.prom_box.setValue(0.0)
        row.addWidget(self.prom_box)

        self.peaks_btn = QtWidgets.QPushButton("Detect peaks")
        self.peaks_btn.clicked.connect(self.detect_peaks)
        row.addWidget(self.peaks_btn)
        row.addStretch(1)
        return panel

    def _build_prop_panel(self) -> "QtWidgets.QWidget":
        panel = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel("Baseline: drag the green band")
        hint.setToolTip("The green band on the trace plot sets the onset-detection baseline")
        row.addWidget(hint)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Onset method:"))
        self.onset_method_box = QtWidgets.QComboBox()
        self.onset_method_box.addItems(["half_max", "std"])
        self.onset_method_box.currentTextChanged.connect(self._on_onset_method_changed)
        row.addWidget(self.onset_method_box)

        self.frac_label = QtWidgets.QLabel("frac:")
        row.addWidget(self.frac_label)
        self.frac_box = QtWidgets.QDoubleSpinBox()
        self.frac_box.setRange(0.01, 0.99)
        self.frac_box.setSingleStep(0.05)
        self.frac_box.setValue(0.5)
        self.frac_box.setToolTip("half_max threshold = baseline + frac·(max − baseline)")
        row.addWidget(self.frac_box)

        self.k_label = QtWidgets.QLabel("k:")
        row.addWidget(self.k_label)
        self.k_box = QtWidgets.QDoubleSpinBox()
        self.k_box.setRange(0.0, 100.0)
        self.k_box.setSingleStep(0.5)
        self.k_box.setValue(3.0)
        self.k_box.setToolTip("std threshold = baseline_mean + k·baseline_std")
        row.addWidget(self.k_box)

        self.prop_btn = QtWidgets.QPushButton("Propagation")
        self.prop_btn.clicked.connect(self.compute_propagation)
        row.addWidget(self.prop_btn)
        row.addStretch(1)
        self._on_onset_method_changed(self.onset_method_box.currentText())
        return panel

    def _load_session(self):
        if self.session.data is not None and self.session.rois:
            self.session.extract_traces()
            T = self.session.traces.raw.shape[1]
            start = self._crop_start()
            self.n_box.setMaximum(T)                          # n is a frame count
            # Coordinates (events, baseline windows) are in original frames.
            self.event_box.setRange(start, start + max(0, T - 1))
            self.region.setBounds((start, start + T))
            self.region.setRegion((start, start + min(self.n_box.value(), T)))
            # Onset baseline defaults to the same leading window; clamp it in-bounds.
            self.prop_region.setBounds((start, start + T))
            self.prop_region.setRegion((start, start + min(self.n_box.value(), T)))
        tl = self.session.timeline
        if tl is not None and tl.frame_interval:
            self.interval_box.setValue(tl.frame_interval)  # reflect notebook calibration
        self._redraw_traces()

    # ------------------------------------------------------------- helpers
    def _display_data(self):
        """The array currently plotted: ΔF/F if requested & available, else raw."""
        traces = self.session.traces
        if traces is None:
            return None, []
        if self.show_dff.isChecked() and traces.dff is not None:
            return traces.dff, traces.labels
        return traces.raw, traces.labels

    def _frame_interval(self):
        """Seconds/frame if the Timeline is calibrated, else None (frames-only)."""
        tl = self.session.timeline
        return tl.frame_interval if (tl is not None and tl.frame_interval) else None

    def _time_unit(self) -> str:
        """Unit label for time readouts: 's' when calibrated, else 'frame'."""
        return "s" if self._frame_interval() else "frame"

    def _to_time(self, frames: float) -> float:
        """A frame index/count in seconds when calibrated, else left in frames."""
        iv = self._frame_interval()
        return frames * iv if iv else frames

    def _speed_str(self, speed) -> str:
        """Propagation speed as px/s when calibrated, else px/frame ('n/a' if unset)."""
        if not isinstance(speed, float) or not np.isfinite(speed):
            return "n/a"
        iv = self._frame_interval()
        return f"{speed / iv:.3g} px/s" if iv else f"{speed:.3g} px/frame"

    def _crop_start(self):
        """First original frame index of the current traces (0 if uncropped).

        The plot works in original frame coordinates so its x-axis, events, onset
        and peak markers stay consistent with a crop window and with CSV/figure
        export; trace *columns* are offset by this when indexing the arrays.
        """
        cw = self.session.crop_window
        return cw[0] if cw is not None else 0

    # ------------------------------------------------------------- actions
    def compute_dff(self):
        if self.session.traces is None:
            self.session.extract_traces()
        method = BaselineMethod(self.baseline_box.currentText())
        if method == BaselineMethod.FIRST_N:
            self.session.compute_dff(method=method, n=self.n_box.value())
        else:
            lo, hi = self.region.getRegion()
            start = self._crop_start()  # region is in frames; traces index from 0
            self.session.compute_dff(method=method, region=(int(lo) - start, int(hi) - start))
        self.show_dff.setChecked(True)
        self._redraw_traces()
        self.status.setText(f"ΔF/F computed ({method.value}).")

    def detect_peaks(self):
        data, _labels = self._display_data()
        if data is None or data.shape[0] == 0:
            self.status.setText("No traces; place ROIs first.")
            return
        height = None if self.height_box.value() == self.height_box.minimum() else self.height_box.value()
        prom = self.prom_box.value() or None
        results = self.session.detect_peaks(
            use_dff=self.show_dff.isChecked(), threshold=height, prominence=prom
        )
        self._overlay_peaks(data, results)
        self._write_peak_summary(results)
        self.status.setText("Peaks detected.")
        return results

    def add_event(self, frame: int):
        ev = self.session.timeline.add_event(int(frame))
        line = pg.InfiniteLine(pos=ev.frame, angle=90, movable=True,
                               pen=pg.mkPen("#ff5050", width=2))
        line.sigPositionChanged.connect(lambda ln, e=ev: setattr(e, "frame", int(ln.value())))
        self.plot.addItem(line)
        self._event_lines.append((ev, line))
        if self._frame_interval():
            self.status.setText(f"Event added at {self._to_time(ev.frame):.4g} s.")
        else:
            self.status.setText(f"Event added at frame {ev.frame}.")
        return ev

    def compute_propagation(self):
        if not self.session.rois:
            self.status.setText("No traces; place ROIs first.")
            return None
        signal = "dff" if (self.show_dff.isChecked() and self.session.traces is not None
                           and self.session.traces.dff is not None) else "raw"
        start = self._crop_start()  # region is in frames; traces index from 0
        lo, hi = sorted(int(round(v)) for v in self.prop_region.getRegion())
        result = self.session.cross_roi_propagation(
            signal=signal, method=self.onset_method_box.currentText(),
            frac=self.frac_box.value(), k=self.k_box.value(),
            baseline_region=(lo - start, hi - start),
        )
        self._overlay_onsets(result["onsets"])
        self._plot_propagation_fit(result)
        self._write_propagation_summary(result)
        n_unit = result["direction"]
        self.status.setText(
            "Propagation: " + self._speed_str(result["speed_px_per_frame"])
            + (f", dir(dy,dx)=({n_unit[0]:.2f},{n_unit[1]:.2f})" if n_unit else "")
        )
        return result

    def _overlay_onsets(self, onsets):
        self._clear_onsets()
        start = self._crop_start()  # onsets are trace-column indices -> frames
        for i, t in enumerate(onsets):
            if np.isnan(t):
                continue
            line = pg.InfiniteLine(
                pos=start + float(t), angle=90,
                pen=pg.mkPen(pg.intColor(i, hues=max(6, len(onsets))),
                             width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line)
            self._onset_lines.append(line)

    def _clear_onsets(self):
        for line in self._onset_lines:
            self.plot.removeItem(line)
        self._onset_lines.clear()

    def _clear_peaks(self):
        for item in self._scatters:
            self.plot.removeItem(item)
        self._scatters.clear()

    def _plot_propagation_fit(self, result):
        """Scatter each ROI's distance from the source (y) against its onset *delay*
        (x, onset − source onset), with the line implied by the reported speed.

        Distances are projected onto the propagation direction, so the planar-wave
        model predicts distance = speed · delay exactly: a straight line through the
        origin whose slope encodes the *same* ``speed_px_per_frame`` shown in the
        summary — the graph and the summary stay coherent by construction. R²
        reports how well that line explains the measured onsets. With no estimable
        direction/speed (fewer than two responding ROIs) the points are shown
        against Euclidean distance and no line is drawn. The delay axis follows the
        Timeline calibration so it reads in seconds when the frame interval is set
        (SPEC §3 time axis). The view autoranges to fit all points.
        """
        self.prop_plot.clear()
        src = result["source_roi"]
        if src is None:
            return
        onsets = np.asarray(result["onsets"], dtype=float)
        coords = np.array([r.center for r in self.session.rois], dtype=float)  # (y, x)
        direction = result["direction"]
        speed = result["speed_px_per_frame"]
        coherent = (direction is not None and isinstance(speed, float)
                    and np.isfinite(speed) and speed > 0)

        delta = coords - coords[src]                    # (dy, dx) from the source
        if coherent:
            # Signed distance along the propagation direction (px).
            dist = delta @ np.asarray(direction, dtype=float)
            self.prop_plot.setLabel("left", "distance along propagation (px)")
        else:
            dist = np.hypot(*delta.T)
            self.prop_plot.setLabel("left", "distance from source (px)")

        iv = self._frame_interval()
        scale = iv or 1.0
        self.prop_plot.setLabel("bottom", "onset delay (s)" if iv else "onset delay (frame)")

        valid = ~np.isnan(onsets)
        r = dist[valid]                               # distance (px) -> y-axis
        delay = (onsets[valid] - onsets[src]) * scale  # onset delay -> x-axis
        labels = self.session.traces.labels
        self.prop_plot.addItem(pg.ScatterPlotItem(
            x=delay, y=r, symbol="o", size=9, pen=None, brush=pg.mkBrush("#3388ff"),
        ))
        for di, ri, i in zip(delay, r, np.flatnonzero(valid)):
            lab = labels[i] if i < len(labels) else f"roi_{i}"
            text = pg.TextItem(lab, color="#999999", anchor=(0, 1))
            text.setPos(float(di), float(ri))
            self.prop_plot.addItem(text)

        # Line for the reported speed: distance = speed · delay, through the origin
        # (the source), so its slope reads back as exactly the summary's speed. R²
        # measures how well that line matches the (noisy) onset delays.
        if coherent and delay.size >= 1:
            delay_hat = (r / speed) * scale             # delay a perfect wave predicts
            ss_res = float(np.sum((delay - delay_hat) ** 2))
            ss_tot = float(np.sum((delay - delay.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            dline = np.array([min(0.0, float(delay.min())), max(0.0, float(delay.max()))])
            self.prop_plot.plot(dline, (speed / scale) * dline,
                                pen=pg.mkPen("#ff5050", width=2),
                                name=f"{self._speed_str(speed)} (R²={r2:.3f})")

        # Autorange so every point (and its label) stays in frame.
        self.prop_plot.getViewBox().autoRange(padding=0.12)

    def _write_propagation_summary(self, result):
        labels = self.session.traces.labels
        lines = ["Propagation", "==========="]
        lines.append(f"speed: {self._speed_str(result['speed_px_per_frame'])}")
        if result["direction"]:
            lines.append("direction (dy, dx): "
                         f"({result['direction'][0]:.3f}, {result['direction'][1]:.3f})")
        if result["source_roi"] is not None:
            src = labels[result["source_roi"]]
            lines.append(f"source (earliest): {src}")
        lines.append("")
        start = self._crop_start()  # onsets are trace columns -> original frames
        lines.append(f"onset times ({self._time_unit()}):")
        for i, t in enumerate(result["onsets"]):
            lab = labels[i] if i < len(labels) else f"roi_{i}"
            val = "n/a" if np.isnan(t) else f"{self._to_time(start + t):.2f}"
            lines.append(f"  {lab}: {val}")
        self.results.setPlainText("\n".join(lines))

    # -------------------------------------------------------------- drawing
    def _redraw_traces(self):
        for item in self._curves + self._scatters:
            self.plot.removeItem(item)
        self._curves.clear()
        self._scatters.clear()

        # Baseline region visible only in REGION mode.
        in_plot = self.region.scene() is not None
        if BaselineMethod(self.baseline_box.currentText()) == BaselineMethod.REGION:
            if not in_plot:
                self.plot.addItem(self.region)
        elif in_plot:
            self.plot.removeItem(self.region)

        data, labels = self._display_data()
        if data is None or data.shape[0] == 0:
            return
        n = data.shape[0]
        x = self._crop_start() + np.arange(data.shape[1])  # frames, crop-aware
        for i in range(n):
            curve = self.plot.plot(x, data[i], pen=pg.intColor(i, hues=max(6, n)),
                                   name=labels[i] if i < len(labels) else f"roi_{i}")
            self._curves.append(curve)
        ylabel = "ΔF/F" if (self.show_dff.isChecked() and self.session.traces.dff is not None) else "mean intensity"
        self.plot.setLabel("left", ylabel)

    def _overlay_peaks(self, data, results):
        self._clear_peaks()
        start = self._crop_start()  # peak indices are trace columns -> frames
        for i, res in enumerate(results):
            idx = res["indices"]
            if len(idx) == 0:
                continue
            scatter = pg.ScatterPlotItem(
                x=start + np.asarray(idx), y=data[i, idx], symbol="t",
                brush=pg.intColor(i, hues=max(6, len(results))), size=10, pen=None,
            )
            self.plot.addItem(scatter)
            self._scatters.append(scatter)

    def _write_peak_summary(self, results):
        labels = self.session.traces.labels
        lines = ["Peak summary", "============"]
        iv = self._frame_interval()
        for i, res in enumerate(results):
            lab = labels[i] if i < len(labels) else f"roi_{i}"
            ttp = res["time_to_peak"]  # frames from the trace start
            if ttp is None:
                ttp_str = "n/a"
            else:
                ttp_str = f"{ttp * iv:.4g} s" if iv else f"{ttp} frames"
            lines.append(
                f"{lab}: count={res['count']}, time-to-peak={ttp_str}, "
                f"max amp={float(np.max(res['amplitudes'])) if res['count'] else float('nan'):.4g}"
            )
        self.results.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.result = self.session.analyses
        self.closed.emit()
        super().closeEvent(event)

    # ------------------------------------------------------------- signals
    def _on_interval_changed(self, value: float):
        """Toggle the trace x-axis between frames and seconds. SPEC §3 time axis.

        Data coords stay in frames; only tick labels are converted, so the value
        also propagates to the Timeline (and thus CSV export / static figures).
        """
        interval = value or None
        if self.session.timeline is not None:
            self.session.timeline.frame_interval = interval
        self._time_axis.set_frame_interval(interval)
        self.plot.setLabel("bottom", "time (s)" if interval else "frame")

    def _on_baseline_changed(self, _text):
        self.n_box.setEnabled(BaselineMethod(self.baseline_box.currentText()) == BaselineMethod.FIRST_N)
        self._redraw_traces()

    def _on_analysis_changed(self, index: int):
        """Show only the picked analysis' controls (stack pages match combo items).

        Overlays from the other analysis are cleared so the trace plot only ever
        shows markers for the analysis currently selected.
        """
        self.param_stack.setCurrentIndex(index)
        is_prop = index == 2  # "Cross-ROI propagation"
        self.prop_plot.setVisible(is_prop)
        # The onset-baseline band lives on the trace plot only while propagation
        # is the active analysis.
        in_plot = self.prop_region.scene() is not None
        if is_prop and not in_plot:
            self.plot.addItem(self.prop_region)
        elif not is_prop and in_plot:
            self.plot.removeItem(self.prop_region)
        if index != 1:        # leaving peak detection
            self._clear_peaks()
        if not is_prop:       # leaving propagation
            self._clear_onsets()

    def _on_onset_method_changed(self, method: str):
        """frac applies to half_max, k to std — enable only the relevant one."""
        is_half = method == "half_max"
        self.frac_label.setEnabled(is_half)
        self.frac_box.setEnabled(is_half)
        self.k_label.setEnabled(not is_half)
        self.k_box.setEnabled(not is_half)
