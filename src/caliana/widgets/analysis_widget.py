"""Stage III — analysis widget. SPEC.md §3 Stage III.

- Choose the ΔF/F baseline: first-N frames, or drag a window on the trace.
- Compute ΔF/F and toggle raw / ΔF/F display.
- Detect peaks (height / prominence) and overlay markers; summarise per ROI.
- Mark optional stimulus events as draggable vertical lines.
- Cross-ROI propagation is wired but still a backend stub (shows a notice).

Interaction logic lives in plain methods (`compute_dff`, `detect_peaks`,
`add_event`, `_redraw_traces`) so tests can drive it without a mouse.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..models import BaselineMethod
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
        row1.addStretch(1)
        layout.addLayout(row1)

        # Row 2 — peaks / events / propagation.
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Height:"))
        self.height_box = QtWidgets.QDoubleSpinBox()
        self.height_box.setRange(-1e9, 1e9)
        self.height_box.setSpecialValueText("auto")
        self.height_box.setValue(self.height_box.minimum())  # "auto"
        row2.addWidget(self.height_box)

        row2.addWidget(QtWidgets.QLabel("Prominence:"))
        self.prom_box = QtWidgets.QDoubleSpinBox()
        self.prom_box.setRange(0.0, 1e9)
        self.prom_box.setValue(0.0)
        row2.addWidget(self.prom_box)

        self.peaks_btn = QtWidgets.QPushButton("Detect peaks")
        self.peaks_btn.clicked.connect(self.detect_peaks)
        row2.addWidget(self.peaks_btn)

        row2.addWidget(QtWidgets.QLabel("Event @"))
        self.event_box = QtWidgets.QSpinBox()
        self.event_box.setRange(0, 100000)
        row2.addWidget(self.event_box)
        self.event_btn = QtWidgets.QPushButton("Add event")
        self.event_btn.clicked.connect(lambda: self.add_event(self.event_box.value()))
        row2.addWidget(self.event_btn)

        self.prop_btn = QtWidgets.QPushButton("Propagation")
        self.prop_btn.clicked.connect(self.compute_propagation)
        row2.addWidget(self.prop_btn)
        row2.addStretch(1)
        layout.addLayout(row2)

        # Plot (left) + results (right).
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        self.plot = pg.PlotWidget(title="ROI traces")
        self.plot.setLabel("bottom", "frame")
        self.plot.addLegend()
        # Draggable baseline window (used in REGION mode).
        self.region = pg.LinearRegionItem(values=(0, self.n_box.value()))
        self.region.setZValue(-10)
        split.addWidget(self.plot)

        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        split.addWidget(self.results)
        split.setSizes([700, 360])

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

    def _load_session(self):
        if self.session.data is not None and self.session.rois:
            self.session.extract_traces()
            T = self.session.traces.raw.shape[1]
            self.n_box.setMaximum(T)
            self.event_box.setMaximum(max(0, T - 1))
            self.region.setRegion((0, min(self.n_box.value(), T)))
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

    # ------------------------------------------------------------- actions
    def compute_dff(self):
        if self.session.traces is None:
            self.session.extract_traces()
        method = BaselineMethod(self.baseline_box.currentText())
        if method == BaselineMethod.FIRST_N:
            self.session.compute_dff(method=method, n=self.n_box.value())
        else:
            lo, hi = self.region.getRegion()
            self.session.compute_dff(method=method, region=(int(lo), int(hi)))
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
        self.status.setText(f"Event added at frame {ev.frame}.")
        return ev

    def compute_propagation(self):
        if not self.session.rois:
            self.status.setText("No traces; place ROIs first.")
            return None
        signal = "dff" if (self.show_dff.isChecked() and self.session.traces is not None
                           and self.session.traces.dff is not None) else "raw"
        result = self.session.cross_roi_propagation(signal=signal)
        self._overlay_onsets(result["onsets"])
        self._write_propagation_summary(result)
        n_unit = result["direction"]
        spd = result["speed_px_per_frame"]
        self.status.setText(
            "Propagation: "
            + (f"{spd:.3g} px/frame" if isinstance(spd, float) else "n/a")
            + (f", dir(dy,dx)=({n_unit[0]:.2f},{n_unit[1]:.2f})" if n_unit else "")
        )
        return result

    def _overlay_onsets(self, onsets):
        for line in self._onset_lines:
            self.plot.removeItem(line)
        self._onset_lines.clear()
        for i, t in enumerate(onsets):
            if np.isnan(t):
                continue
            line = pg.InfiniteLine(
                pos=float(t), angle=90,
                pen=pg.mkPen(pg.intColor(i, hues=max(6, len(onsets))),
                             width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.plot.addItem(line)
            self._onset_lines.append(line)

    def _write_propagation_summary(self, result):
        labels = self.session.traces.labels
        lines = ["Propagation", "==========="]
        spd = result["speed_px_per_frame"]
        lines.append(f"speed: {spd:.4g} px/frame" if isinstance(spd, float) else "speed: n/a")
        if result["direction"]:
            lines.append("direction (dy, dx): "
                         f"({result['direction'][0]:.3f}, {result['direction'][1]:.3f})")
        if result["source_roi"] is not None:
            src = labels[result["source_roi"]]
            lines.append(f"source (earliest): {src}")
        lines.append("")
        lines.append("onset times (frame):")
        for i, t in enumerate(result["onsets"]):
            lab = labels[i] if i < len(labels) else f"roi_{i}"
            lines.append(f"  {lab}: {'n/a' if np.isnan(t) else f'{t:.2f}'}")
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
        for i in range(n):
            curve = self.plot.plot(data[i], pen=pg.intColor(i, hues=max(6, n)),
                                   name=labels[i] if i < len(labels) else f"roi_{i}")
            self._curves.append(curve)
        ylabel = "ΔF/F" if (self.show_dff.isChecked() and self.session.traces.dff is not None) else "mean intensity"
        self.plot.setLabel("left", ylabel)

    def _overlay_peaks(self, data, results):
        for item in self._scatters:
            self.plot.removeItem(item)
        self._scatters.clear()
        for i, res in enumerate(results):
            idx = res["indices"]
            if len(idx) == 0:
                continue
            scatter = pg.ScatterPlotItem(
                x=np.asarray(idx), y=data[i, idx], symbol="t",
                brush=pg.intColor(i, hues=max(6, len(results))), size=10, pen=None,
            )
            self.plot.addItem(scatter)
            self._scatters.append(scatter)

    def _write_peak_summary(self, results):
        labels = self.session.traces.labels
        lines = ["Peak summary", "============"]
        for i, res in enumerate(results):
            lab = labels[i] if i < len(labels) else f"roi_{i}"
            lines.append(
                f"{lab}: count={res['count']}, time-to-peak={res['time_to_peak']}, "
                f"max amp={float(np.max(res['amplitudes'])) if res['count'] else float('nan'):.4g}"
            )
        self.results.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.result = self.session.analyses
        self.closed.emit()
        super().closeEvent(event)

    # ------------------------------------------------------------- signals
    def _on_baseline_changed(self, _text):
        self.n_box.setEnabled(BaselineMethod(self.baseline_box.currentText()) == BaselineMethod.FIRST_N)
        self._redraw_traces()
