"""Stage III — analysis widget. SPEC.md §3 Stage III.

Two tabbed pages:

**Trace analysis** — analyses of the ROI traces:
- ΔF/F defaults to a first-10-frame baseline as soon as traces exist (`Traces`);
  choose a different baseline (first-N frames, or drag a window on the trace) and
  recompute, and toggle raw / ΔF/F display.
- Smooth ΔF/F with a Gaussian kernel of user-chosen σ (frames); the result is
  kept in its own `traces.smoothed` array — `dff` is never overwritten — and can
  be toggled on/off independently (`smooth_traces`, `session.smooth_traces` for
  headless use).
- Pick one analysis to run; only that analysis' controls are shown:
  - Cross-ROI propagation: choose the onset-time method (fraction_of_max / std) and its
    parameters (frac / k), and drag a baseline window (the green band) that sets
    the level onsets are measured from; overlay per-ROI onset times, summarise
    speed / direction / source ROI, and plot distance-along-propagation vs onset
    delay with the line implied by that speed and its R². Time readouts switch to
    seconds when a frame interval is set.
- Mark optional stimulus events as draggable vertical lines.

**Heatmaps** — dataset-wide (not per-ROI) maps. Currently a per-pixel onset-time
heatmap: the same onset detector used for propagation (`analysis.onset_time`) is
run on every pixel's temporal trace (optionally after n×n binning), colouring
each pixel by when it first responds. Method / frac / k / baseline mirror the
propagation controls, so a heatmap pixel and a same-parameter ROI onset agree.

Interaction logic lives in plain methods (`compute_dff`, `smooth_traces`,
`compute_propagation`, `compute_onset_heatmap`, `add_event`, `_redraw_traces`) so
tests can drive it without a mouse.
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
        self._event_lines: list = []
        self._onset_lines: list = []

        self._build_ui()
        self._load_session()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, stretch=1)
        self.tabs.addTab(self._build_traces_page(), "Trace analysis")
        self.tabs.addTab(self._build_heatmap_page(), "Heatmaps")

        # A single status line shared by both pages.
        self.status = QtWidgets.QLabel("")
        outer.addWidget(self.status)

    def _build_traces_page(self) -> "QtWidgets.QWidget":
        """The ROI-trace analysis page (baseline/ΔF/F, propagation)."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

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

        # Row 1b — Gaussian smoothing. Always smooths ΔF/F (traces.dff, which
        # defaults to a first-10-frame baseline as soon as traces exist — see
        # Traces) into traces.smoothed; never touches raw or dff.
        row1b = QtWidgets.QHBoxLayout()
        row1b.addWidget(QtWidgets.QLabel("Smoothing σ (frames):"))
        self.smooth_sigma_box = QtWidgets.QDoubleSpinBox()
        self.smooth_sigma_box.setRange(0.0, 1000.0)
        self.smooth_sigma_box.setSingleStep(0.5)
        self.smooth_sigma_box.setValue(1.0)
        self.smooth_sigma_box.setToolTip(
            "Standard deviation of the Gaussian kernel (frames) applied to ΔF/F"
        )
        row1b.addWidget(self.smooth_sigma_box)

        self.smooth_btn = QtWidgets.QPushButton("Smooth ΔF/F")
        self.smooth_btn.clicked.connect(self.smooth_traces)
        row1b.addWidget(self.smooth_btn)

        self.show_smoothed = QtWidgets.QCheckBox("Show smoothed ΔF/F")
        self.show_smoothed.toggled.connect(self._redraw_traces)
        row1b.addWidget(self.show_smoothed)
        row1b.addStretch(1)
        layout.addLayout(row1b)

        # Row 2 — pick the analysis to run + shared stimulus events.
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Analysis:"))
        self.analysis_box = QtWidgets.QComboBox()
        self.analysis_box.addItems(
            ["(select analysis)", "Cross-ROI propagation"]
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
        self.param_stack.addWidget(self._build_prop_panel())   # 1: propagation
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
        return page

    # ---------------------------------------------------------- heatmap page
    def _build_heatmap_page(self) -> "QtWidgets.QWidget":
        """Dataset onset-time heatmap: the per-ROI onset detector run per pixel.

        Reuses ``session.onset_heatmap`` (which wraps the same ``onset_time``
        detector as the propagation analysis) so the map and a same-parameter ROI
        onset agree. Controls mirror the propagation panel — method + frac/k and a
        baseline window — plus spatial binning to trade resolution for SNR/speed.
        """
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Onset method:"))
        self.hm_method_box = QtWidgets.QComboBox()
        self.hm_method_box.addItems(["fraction_of_max", "std"])
        self.hm_method_box.currentTextChanged.connect(self._on_hm_method_changed)
        row.addWidget(self.hm_method_box)

        self.hm_frac_label = QtWidgets.QLabel("frac:")
        row.addWidget(self.hm_frac_label)
        self.hm_frac_box = QtWidgets.QDoubleSpinBox()
        self.hm_frac_box.setRange(0.01, 1.0)   # frac=1 targets the peak (time-to-max)
        self.hm_frac_box.setSingleStep(0.05)
        self.hm_frac_box.setValue(0.5)
        self.hm_frac_box.setToolTip("fraction_of_max threshold = baseline + frac·(max − baseline)")
        row.addWidget(self.hm_frac_box)

        self.hm_k_label = QtWidgets.QLabel("k:")
        row.addWidget(self.hm_k_label)
        self.hm_k_box = QtWidgets.QDoubleSpinBox()
        self.hm_k_box.setRange(0.0, 100.0)
        self.hm_k_box.setSingleStep(0.5)
        self.hm_k_box.setValue(3.0)
        self.hm_k_box.setToolTip("std threshold = baseline_mean + k·baseline_std")
        row.addWidget(self.hm_k_box)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Baseline [start, end):"))
        self.hm_base_start = QtWidgets.QSpinBox()
        self.hm_base_start.setRange(0, 100000)
        row.addWidget(self.hm_base_start)
        self.hm_base_end = QtWidgets.QSpinBox()
        self.hm_base_end.setRange(0, 100000)
        row.addWidget(self.hm_base_end)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Bin (n×n):"))
        self.hm_bin_box = QtWidgets.QSpinBox()
        self.hm_bin_box.setRange(1, 64)
        self.hm_bin_box.setValue(1)
        self.hm_bin_box.setToolTip("Mean-pool n×n pixel blocks before onset detection")
        row.addWidget(self.hm_bin_box)

        self.hm_btn = QtWidgets.QPushButton("Compute heatmap")
        self.hm_btn.clicked.connect(self.compute_onset_heatmap)
        row.addWidget(self.hm_btn)
        row.addStretch(1)
        v.addLayout(row)

        # Image + linked colorbar. NaN pixels (no detected onset) render transparent.
        self._hm_cmap = pg.colormap.get("inferno")
        self.hm_view = pg.GraphicsLayoutWidget()
        self.hm_plot = self.hm_view.addPlot()
        self.hm_plot.setAspectLocked(True)
        self.hm_plot.invertY(True)              # image row 0 at the top
        self.hm_plot.getViewBox().setDefaultPadding(0.02)
        self.hm_image = pg.ImageItem()
        self.hm_image.setOpts(axisOrder="row-major")  # data is [Y, X]
        self.hm_plot.addItem(self.hm_image)
        # interactive=True gives draggable level handles on the colour bar — the
        # intensity/contrast control, echoing the preview widget's level region.
        self.hm_cbar = pg.ColorBarItem(colorMap=self._hm_cmap, interactive=True,
                                       label="onset (frame)")
        self.hm_cbar.setImageItem(self.hm_image)
        self.hm_view.addItem(self.hm_cbar)
        v.addWidget(self.hm_view, stretch=1)

        self._on_hm_method_changed(self.hm_method_box.currentText())
        return page

    # ------------------------------------------------------- analysis panels
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
        self.onset_method_box.addItems(["fraction_of_max", "std"])
        self.onset_method_box.currentTextChanged.connect(self._on_onset_method_changed)
        row.addWidget(self.onset_method_box)

        self.frac_label = QtWidgets.QLabel("frac:")
        row.addWidget(self.frac_label)
        self.frac_box = QtWidgets.QDoubleSpinBox()
        self.frac_box.setRange(0.01, 1.0)      # frac=1 targets the peak (time-to-max)
        self.frac_box.setSingleStep(0.05)
        self.frac_box.setValue(0.5)
        self.frac_box.setToolTip("fraction_of_max threshold = baseline + frac·(max − baseline)")
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
        if self.session.data is not None:
            # Heatmap baseline window: same leading frames as the trace baselines,
            # in original-frame coordinates, clamped to the (possibly cropped) span.
            start = self._crop_start()
            if self.session.crop_window is not None:
                nframes = self.session.crop_window[1] - self.session.crop_window[0]
            else:
                nframes = len(self.session._working_stack())
            self.hm_base_start.setRange(start, start + max(0, nframes - 1))
            self.hm_base_end.setRange(start, start + nframes)
            self.hm_base_start.setValue(start)
            self.hm_base_end.setValue(start + min(self.n_box.value(), nframes))
        tl = self.session.timeline
        if tl is not None and tl.frame_interval:
            self.interval_box.setValue(tl.frame_interval)  # reflect notebook calibration
        self._redraw_traces()

    # ------------------------------------------------------------- helpers
    def _display_data(self):
        """The array currently plotted: smoothed, else ΔF/F, else raw — in that
        priority, each gated on its checkbox and on having been computed."""
        traces = self.session.traces
        if traces is None:
            return None, []
        if self.show_smoothed.isChecked() and traces.smoothed is not None:
            return traces.smoothed, traces.labels
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

        The plot works in original frame coordinates so its x-axis, events, and
        onset markers stay consistent with a crop window and with CSV/figure
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

    def smooth_traces(self):
        """Gaussian-smooth ΔF/F, storing the result on ``traces.smoothed`` without
        touching ``dff``. ``traces.dff`` defaults to a first-10-frame baseline as
        soon as traces are extracted, so this works without first clicking
        "Compute ΔF/F"."""
        if self.session.traces is None:
            self.session.extract_traces()
        sigma = self.smooth_sigma_box.value()
        self.session.smooth_traces(sigma)
        self.show_smoothed.setChecked(True)
        self._redraw_traces()
        self.status.setText(f"ΔF/F smoothed (σ={sigma:g} frames).")

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

    def compute_onset_heatmap(self):
        """Run the per-pixel onset detector over the dataset and show the heatmap."""
        if self.session.data is None:
            self.status.setText("No data loaded.")
            return None
        start = self._crop_start()  # baseline boxes are in frames; map indexes from 0
        lo, hi = self.hm_base_start.value() - start, self.hm_base_end.value() - start
        region = (lo, hi) if hi > lo else None
        mp = self.session.onset_heatmap(
            method=self.hm_method_box.currentText(),
            frac=self.hm_frac_box.value(), k=self.hm_k_box.value(),
            baseline_region=region, bin_size=self.hm_bin_box.value(),
        )
        self._show_heatmap(mp)
        return mp

    def _show_heatmap(self, mp):
        """Display an onset map, converting frames→seconds when calibrated.

        Onset columns are shifted by the crop start so the colour scale reads in
        original-recording frames (or seconds), matching the trace page. Pixels
        with no detected onset are NaN and render transparent.
        """
        start = self._crop_start()
        iv = self._frame_interval()
        disp = (start + np.asarray(mp, dtype=float)) * (iv or 1.0)
        finite = np.isfinite(disp)
        if not finite.any():
            self.hm_image.clear()
            self.status.setText("No onsets detected with these parameters.")
            return
        lo = float(np.nanmin(disp))
        hi = float(np.nanmax(disp))
        if hi <= lo:
            hi = lo + 1.0
        self.hm_image.setImage(disp, autoLevels=False)
        # Fine, unit-agnostic drag steps for the interactive level handles: ~200
        # steps across the data span, so contrast is adjustable in frames or seconds.
        self.hm_cbar.rounding = max((hi - lo) / 200.0, 1e-9)
        self.hm_cbar.setLevels((lo, hi))
        self.hm_cbar.setLabel("left", f"onset ({self._time_unit()})")
        self.hm_plot.getViewBox().autoRange(padding=0.02)
        self.status.setText(
            f"Onset heatmap: {int(finite.sum())}/{disp.size} pixels responded."
        )

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
        for item in self._curves:
            self.plot.removeItem(item)
        self._curves.clear()

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
        traces = self.session.traces
        if self.show_smoothed.isChecked() and traces.smoothed is not None:
            ylabel = f"ΔF/F (smoothed, σ={traces.smoothed_sigma:g})"
        elif self.show_dff.isChecked() and traces.dff is not None:
            ylabel = "ΔF/F"
        else:
            ylabel = "mean intensity"
        self.plot.setLabel("left", ylabel)

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
        is_prop = index == 1  # "Cross-ROI propagation"
        self.prop_plot.setVisible(is_prop)
        # The onset-baseline band lives on the trace plot only while propagation
        # is the active analysis.
        in_plot = self.prop_region.scene() is not None
        if is_prop and not in_plot:
            self.plot.addItem(self.prop_region)
        elif not is_prop and in_plot:
            self.plot.removeItem(self.prop_region)
        if not is_prop:       # leaving propagation
            self._clear_onsets()

    def _on_onset_method_changed(self, method: str):
        """frac applies to fraction_of_max, k to std — enable only the relevant one."""
        is_frac = method == "fraction_of_max"
        self.frac_label.setEnabled(is_frac)
        self.frac_box.setEnabled(is_frac)
        self.k_label.setEnabled(not is_frac)
        self.k_box.setEnabled(not is_frac)

    def _on_hm_method_changed(self, method: str):
        """Heatmap counterpart of ``_on_onset_method_changed``."""
        is_frac = method == "fraction_of_max"
        self.hm_frac_label.setEnabled(is_frac)
        self.hm_frac_box.setEnabled(is_frac)
        self.hm_k_label.setEnabled(not is_frac)
        self.hm_k_box.setEnabled(not is_frac)
