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
  - Cross-ROI propagation: choose the onset-time method (fraction_of_max / std /
    derivative) and its parameters (frac / k / d), and drag a baseline window (the green band) that sets
    the level onsets are measured from; overlay per-ROI onset times, summarise
    speed / direction / source ROI, and plot distance-along-propagation vs onset
    delay with the line implied by that speed and its R². Onsets are detected on the
    signal currently displayed (raw / ΔF/F / smoothed). "Direction" picks how the
    propagation vector is found: along the line the ROIs sit on (default — only the
    speed along it is fitted, so ROIs placed along the propagation path stay
    well-posed) or automatically from a 2D plane fit over the ROI centres (which
    needs ROIs spread in two dimensions). Time readouts switch to seconds when a
    frame interval is set.
- Mark optional stimulus events as draggable vertical lines.

**Heatmaps** — dataset-wide (not per-ROI) maps. Currently a per-pixel onset-time
heatmap: the same onset detector used for propagation (`analysis.onset_time`) is
run on every pixel's temporal trace (optionally after n×n binning), colouring
each pixel by when it first responds. Method / frac / k / d / baseline mirror the
propagation controls, so a heatmap pixel and a same-parameter ROI onset agree.

**Kymograph** — click along a feature (a vein, a petiole) to trace a path, then
read the intensity along it over time as a distance × time image
(`analysis.kymograph`). No ROIs are involved: a response travelling along the
path draws a diagonal band whose slope is its speed, which is the continuous
counterpart of the per-ROI propagation fit. Every point stays draggable, and
`clear_last_point` / `clear_path` walk it back. Values default to per-position
ΔF/F, since raw brightness varies far more along a leaf than the response does.

Only the trace page needs ROIs, and it gates itself on them (`reload`) — the
other two are dataset-wide, which is why the app opens this tab on a stack alone.

Interaction logic lives in plain methods (`compute_dff`, `smooth_traces`,
`compute_propagation`, `compute_onset_heatmap`, `add_path_point`,
`clear_last_point`, `compute_kymograph`, `add_event`, `_redraw_traces`) so tests
can drive it without a mouse.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pyqtgraph as pg

from .. import figures
from ..models import BaselineMethod
from ..space import distance_units, speed_units
from ._plot import FrameTimeAxis, frame_interval, pixel_size, polyline_vertices
from ._qt import get_qt, save_figure_dialog

QtCore, QtGui, QtWidgets = get_qt()

pg.setConfigOption("imageAxisOrder", "row-major")

_LEFT = QtCore.Qt.MouseButton.LeftButton
_PATH_PEN = pg.mkPen("#00ff7f", width=2)

# Propagation direction modes: combo label -> analysis.cross_roi_propagation name.
# Insertion order is the combo order, so the first entry is the default.
_DIRECTION_MODES = {
    "along ROI line": "roi_line",
    "automatic (2D fit)": "auto",
}

# Kymograph value modes. ΔF/F leads (and is the default) because raw brightness
# varies far more from one end of a leaf to the other than a response does, so a
# raw kymograph mostly draws the tissue rather than the signal.
_KYMO_DFF = "ΔF/F"
_KYMO_RAW = "raw intensity"


class _Displayed(NamedTuple):
    """Which trace array the plot is showing, and how to label/analyse it."""
    signal: str                        # "smoothed" | "dff" | "raw" (analysis.py name)
    data: Optional[np.ndarray]         # [n_roi, T]; None when no traces exist
    labels: list
    ylabel: str


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
        self._prop_fit: dict | None = None   # snapshot of the last propagation scatter

        # Kymograph page: the path's points, the graphic drawn from them, and the
        # last computed result (kept for the figure export).
        self._path_points: list[tuple[float, float]] = []
        self._path_item = None
        self._kymo_result: dict | None = None
        # Frame size path clicks are bounds-checked against; a placeholder until a
        # stack is loaded, so clicking an empty widget adds nothing.
        self._kymo_shape_yx = (1, 1)

        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, stretch=1)
        self.tabs.addTab(self._build_traces_page(), "Trace analysis")
        self.tabs.addTab(self._build_heatmap_page(), "Heatmaps")
        self.tabs.addTab(self._build_kymograph_page(), "Kymograph")

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
        self.interval_box.setToolTip(
            "Seconds per frame of the loaded stack; 0 keeps the time axis in frames"
        )
        self.interval_box.valueChanged.connect(self._on_interval_changed)
        row1.addWidget(self.interval_box)

        row1.addWidget(QtWidgets.QLabel("Pixel size (µm):"))
        self.pixel_box = QtWidgets.QDoubleSpinBox()
        self.pixel_box.setRange(0.0, 1e6)
        self.pixel_box.setDecimals(4)
        self.pixel_box.setSpecialValueText("pixels")  # 0 ⇒ pixels-only distances
        self.pixel_box.setToolTip(
            "µm per pixel of the loaded stack; 0 keeps distances in pixels"
        )
        self.pixel_box.valueChanged.connect(self._on_pixel_size_changed)
        row1.addWidget(self.pixel_box)
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
        self.save_traces_btn = QtWidgets.QPushButton("Save traces…")
        self.save_traces_btn.setToolTip("Save the ROI traces (raw or ΔF/F, matching the display) as a figure")
        self.save_traces_btn.clicked.connect(self._save_traces)
        row2.addWidget(self.save_traces_btn)
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
        self.hm_method_box.addItems(["fraction_of_max", "std", "derivative"])
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
        self.hm_k_box.setToolTip("std / derivative threshold both scale baseline_std by k")
        row.addWidget(self.hm_k_box)

        self.hm_d_label = QtWidgets.QLabel("d:")
        row.addWidget(self.hm_d_label)
        self.hm_d_box = QtWidgets.QDoubleSpinBox()
        self.hm_d_box.setRange(-1e6, 1e6)
        self.hm_d_box.setDecimals(3)
        self.hm_d_box.setSingleStep(0.01)
        self.hm_d_box.setValue(0.0)
        self.hm_d_box.setToolTip("derivative threshold = baseline_deriv_mean + k·baseline_deriv_std + d")
        row.addWidget(self.hm_d_box)

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

    # ----------------------------------------------------- kymograph page
    def _build_kymograph_page(self) -> "QtWidgets.QWidget":
        """Trace a path on the stack, read the intensity along it over time.

        The movie (left) is where the path is drawn — scrub to a frame where the
        response is visible, then click along the feature to follow. There is no
        drawing mode to enter or leave: every click on the image extends the path,
        and each point stays draggable afterwards (clicking a segment inserts one
        between its ends), so a path is adjusted rather than redrawn. The
        kymograph (right) is ``session.kymograph`` rendered as a distance × time
        image with a draggable-level colour bar, mirroring the heatmap page.
        """
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        row = QtWidgets.QHBoxLayout()
        # `clicked` carries a checked flag Qt would pass on as a positional
        # argument, so both go through a lambda rather than connecting directly.
        self.path_undo_btn = QtWidgets.QPushButton("Clear last point")
        self.path_undo_btn.setToolTip("Remove the last point of the path")
        self.path_undo_btn.clicked.connect(lambda: self.clear_last_point())
        row.addWidget(self.path_undo_btn)

        self.path_clear_btn = QtWidgets.QPushButton("Clear path")
        self.path_clear_btn.setToolTip("Remove every point and start over")
        self.path_clear_btn.clicked.connect(lambda: self.clear_path())
        row.addWidget(self.path_clear_btn)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Width (px):"))
        self.kymo_width_box = QtWidgets.QSpinBox()
        self.kymo_width_box.setRange(1, 199)
        self.kymo_width_box.setValue(1)
        self.kymo_width_box.setToolTip(
            "Average this many samples across the path (perpendicular to it) into "
            "each row — more width, less noise, coarser detail"
        )
        row.addWidget(self.kymo_width_box)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Values:"))
        self.kymo_signal_box = QtWidgets.QComboBox()
        self.kymo_signal_box.addItems([_KYMO_DFF, _KYMO_RAW])   # ΔF/F first = default
        self.kymo_signal_box.setToolTip(
            "ΔF/F normalises each position against its own baseline, so a dim "
            "region's response reads as strongly as a bright one's"
        )
        self.kymo_signal_box.currentTextChanged.connect(self._on_kymo_signal_changed)
        row.addWidget(self.kymo_signal_box)

        self.kymo_base_label = QtWidgets.QLabel("baseline N:")
        row.addWidget(self.kymo_base_label)
        self.kymo_base_box = QtWidgets.QSpinBox()
        self.kymo_base_box.setRange(1, 100000)
        self.kymo_base_box.setValue(10)
        self.kymo_base_box.setToolTip("F0 = mean of the first N frames, per path position")
        row.addWidget(self.kymo_base_box)

        self.kymo_btn = QtWidgets.QPushButton("Compute kymograph")
        self.kymo_btn.clicked.connect(self.compute_kymograph)
        row.addWidget(self.kymo_btn)

        self.save_kymo_btn = QtWidgets.QPushButton("Save kymograph…")
        self.save_kymo_btn.setToolTip("Save the distance-vs-time image as shown")
        self.save_kymo_btn.clicked.connect(self._save_kymograph)
        row.addWidget(self.save_kymo_btn)
        row.addStretch(1)
        # Standing instruction: with no mode button there is nothing else on
        # screen saying that the image is what you click on.
        self.kymo_hint = QtWidgets.QLabel("Click the image to add path points")
        row.addWidget(self.kymo_hint)
        v.addLayout(row)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        v.addWidget(split, stretch=1)

        # Left: the movie the path is drawn on (scrubbable, same contrast as the
        # ROI/import views so the tissue looks identical whichever tab you're on).
        self.kymo_movie = pg.ImageView(name="kymograph_image")
        self.kymo_movie.ui.roiBtn.hide()
        self.kymo_movie.ui.menuBtn.hide()
        self.kymo_movie.ui.histogram.gradient.hide()
        split.addWidget(self.kymo_movie)

        # Right: the kymograph. Distance runs up the y-axis from the path's start,
        # time along x — both in the session's units, set in _show_kymograph.
        self._kymo_cmap = pg.colormap.get("inferno")
        self.kymo_view = pg.GraphicsLayoutWidget()
        self.kymo_plot = self.kymo_view.addPlot()
        self.kymo_plot.setLabel("bottom", "frame")
        self.kymo_plot.setLabel("left", "distance along path (px)")
        self.kymo_plot.getViewBox().setDefaultPadding(0.02)
        self.kymo_image = pg.ImageItem()
        self.kymo_image.setOpts(axisOrder="row-major")   # data is [position, frame]
        self.kymo_plot.addItem(self.kymo_image)
        self.kymo_cbar = pg.ColorBarItem(colorMap=self._kymo_cmap, interactive=True,
                                         label="ΔF/F")
        self.kymo_cbar.setImageItem(self.kymo_image)
        self.kymo_view.addItem(self.kymo_cbar)
        split.addWidget(self.kymo_view)
        split.setSizes([520, 540])

        self.kymo_movie.view.scene().sigMouseClicked.connect(self._on_kymo_click)
        self._on_kymo_signal_changed(self.kymo_signal_box.currentText())
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
        self.onset_method_box.addItems(["fraction_of_max", "std", "derivative"])
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
        self.k_box.setToolTip("std / derivative threshold both scale baseline_std by k")
        row.addWidget(self.k_box)

        self.d_label = QtWidgets.QLabel("d:")
        row.addWidget(self.d_label)
        self.d_box = QtWidgets.QDoubleSpinBox()
        self.d_box.setRange(-1e6, 1e6)
        self.d_box.setDecimals(3)
        self.d_box.setSingleStep(0.01)
        self.d_box.setValue(0.0)
        self.d_box.setToolTip("derivative threshold = baseline_deriv_mean + k·baseline_deriv_std + d")
        row.addWidget(self.d_box)

        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Direction:"))
        self.prop_dir_box = QtWidgets.QComboBox()
        self.prop_dir_box.addItems(list(_DIRECTION_MODES))   # ROI line first = default
        self.prop_dir_box.setToolTip(
            "Along ROI line: the direction is fixed to the line the ROIs sit on and "
            "only the speed along it is fitted — right when ROIs are placed along the "
            "propagation path.\nAutomatic: fit a 2D plane over the ROI centres and "
            "take its gradient; needs ROIs spread in two dimensions, as collinear "
            "ROIs leave the direction underdetermined."
        )
        row.addWidget(self.prop_dir_box)

        self.prop_btn = QtWidgets.QPushButton("Propagation")
        self.prop_btn.clicked.connect(self.compute_propagation)
        row.addWidget(self.prop_btn)

        self.save_prop_btn = QtWidgets.QPushButton("Save propagation…")
        self.save_prop_btn.setToolTip("Save the distance-vs-onset-delay graph as shown")
        self.save_prop_btn.clicked.connect(self._save_propagation)
        row.addWidget(self.save_prop_btn)

        row.addStretch(1)
        self._on_onset_method_changed(self.onset_method_box.currentText())
        return panel

    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        The app calls this when the Analysis tab is activated after anything
        upstream changed — ROIs added, registration re-run, a crop applied — all
        of which invalidate the traces this page analyses. Overlays from a
        previous session state (onset markers, event lines, the propagation
        graph, the kymograph path) are cleared, then rebuilt from what the
        session actually holds.
        """
        self._clear_onsets()
        self._clear_event_lines()
        self.prop_plot.clear()
        self._prop_fit = None
        self.results.setPlainText("")
        # Only the trace page needs ROIs; the heatmap and kymograph pages are
        # dataset-wide, which is why the app opens this tab on a stack alone.
        has_rois = bool(self.session.rois)
        self.tabs.setTabEnabled(0, has_rois)
        self.tabs.setTabToolTip(0, "ΔF/F, smoothing and propagation over the ROI traces"
                                if has_rois else "Place at least one ROI first (ROIs tab)")
        if self.session.data is not None and has_rois:
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
            self.status.setText("")
        elif self.session.data is None:
            self.status.setText("Load a stack first.")
        else:
            self.status.setText(
                "No ROIs yet — trace analysis needs them; the heatmap and "
                "kymograph pages work on the stack alone."
            )
        if self.session.data is not None:
            # Heatmap baseline window: same leading frames as the trace baselines,
            # in original-frame coordinates, clamped to the (possibly cropped) span.
            start = self._crop_start()
            nframes = self._n_frames()
            self.hm_base_start.setRange(start, start + max(0, nframes - 1))
            self.hm_base_end.setRange(start, start + nframes)
            self.hm_base_start.setValue(start)
            self.hm_base_end.setValue(start + min(self.n_box.value(), nframes))
        # Mirror the calibration the session carries (file metadata, the notebook,
        # or the Import tab). Signals are blocked so reflecting it here can't write
        # a stale box value back onto the session; the axes are refreshed below.
        self._sync_calibration_boxes()
        self._reload_kymograph()
        # Event markers already on the timeline get their draggable line back.
        for ev in (self.session.timeline.events if self.session.timeline else []):
            self._draw_event_line(ev)
        self._redraw_traces()

    def _reload_kymograph(self):
        """Re-point the kymograph page at the current stack, dropping the old path.

        A path is a list of pixel coordinates on the stack it was drawn over, so a
        new (or re-imported, or re-registered) stack invalidates it exactly as it
        invalidates ROIs — the same reason ``Session._reset_derived`` drops those.
        """
        self._path_points = []          # silently: reload owns the status line
        self._rebuild_path_graphic()
        self.kymo_image.clear()
        self._kymo_result = None
        if self.session.data is None:
            self._kymo_shape_yx = (1, 1)
            self.kymo_movie.setImage(np.zeros((1, 1, 1)))
            return
        stack = np.asarray(self.session._working_stack())
        self._kymo_shape_yx = stack.shape[1:]
        self.kymo_movie.setImage(stack, axes={"t": 0, "y": 1, "x": 2},
                                 autoLevels=False,
                                 levels=figures.intensity_levels(stack))
        self.kymo_base_box.setMaximum(self._n_frames())

    def _sync_calibration_boxes(self):
        """Point the frame-interval / pixel-size boxes at the session's values.

        0 is each box's "uncalibrated" special value, so a session with no scale
        reads back as ``frames`` / ``pixels`` rather than keeping whatever the
        previous session was calibrated at.
        """
        tl = self.session.timeline
        interval = tl.frame_interval if tl is not None else None
        for box, value in ((self.interval_box, interval),
                           (self.pixel_box, self.session.space.pixel_size)):
            blocker = QtCore.QSignalBlocker(box)
            box.setValue(value or 0.0)
            del blocker
        self._time_axis.set_frame_interval(interval or None)
        self.plot.setLabel("bottom", "time (s)" if interval else "frame")

    # ------------------------------------------------------------- helpers
    def _displayed(self) -> _Displayed:
        """What the trace plot currently shows: smoothed, else ΔF/F, else raw — in
        that priority, each gated on its checkbox and on having been computed.

        The single place that ladder lives, so the plot, the exported figure and
        the onset detector in ``compute_propagation`` can never disagree about
        which signal is on screen.
        """
        traces = self.session.traces
        if traces is None:
            return _Displayed("raw", None, [], "mean intensity")
        if self.show_smoothed.isChecked() and traces.smoothed is not None:
            return _Displayed("smoothed", traces.smoothed, traces.labels,
                              f"ΔF/F (smoothed, σ={traces.smoothed_sigma:g})")
        if self.show_dff.isChecked() and traces.dff is not None:
            return _Displayed("dff", traces.dff, traces.labels, "ΔF/F")
        return _Displayed("raw", traces.raw, traces.labels, "mean intensity")

    def _frame_interval(self):
        """Seconds/frame if the Timeline is calibrated, else None (frames-only)."""
        return frame_interval(self.session)

    def _pixel_size(self):
        """µm/pixel if the SpatialScale is calibrated, else None (pixels-only)."""
        return pixel_size(self.session)

    def _time_unit(self) -> str:
        """Unit label for time readouts: 's' when calibrated, else 'frame'."""
        return "s" if self._frame_interval() else "frame"

    def _to_time(self, frames: float) -> float:
        """A frame index/count in seconds when calibrated, else left in frames."""
        iv = self._frame_interval()
        return frames * iv if iv else frames

    def _distance_units(self):
        """``(factor, unit)`` taking pixel distances to µm when calibrated.

        The space-axis twin of ``_to_time``/``_time_unit``, in one call because
        the distance scale and its label are always needed together (axis labels
        on the propagation graph, and its figure export).
        """
        return distance_units(self._pixel_size())

    def _speed_str(self, speed) -> str:
        """A px/frame propagation speed in the best available units.

        Reads µm/s with both axes calibrated, and degrades one axis at a time
        (µm/frame, px/s, px/frame). ``'n/a'`` when no speed could be estimated.
        """
        if not isinstance(speed, float) or not np.isfinite(speed):
            return "n/a"
        factor, unit = speed_units(self._pixel_size(), self._frame_interval())
        return f"{speed * factor:.3g} {unit}"

    def _crop_start(self):
        """First original frame index of the current traces (0 if uncropped).

        The plot works in original frame coordinates so its x-axis, events, and
        onset markers stay consistent with a crop window and with CSV/figure
        export; trace *columns* are offset by this when indexing the arrays.
        """
        cw = self.session.crop_window
        return cw[0] if cw is not None else 0

    def _n_frames(self) -> int:
        """How many frames the analyses see: the crop window's, else the stack's.

        The companion of ``_crop_start`` — together they are the ``[start, end)``
        span ``Session._crop_bounds`` hands every analysis, which is what the
        baseline-window controls have to stay inside.
        """
        cw = self.session.crop_window
        if cw is not None:
            return cw[1] - cw[0]
        return len(self.session._working_stack()) if self.session.data is not None else 0

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
        self._draw_event_line(ev)
        if self._frame_interval():
            self.status.setText(f"Event added at {self._to_time(ev.frame):.4g} s.")
        else:
            self.status.setText(f"Event added at frame {ev.frame}.")
        return ev

    def _draw_event_line(self, ev):
        """Draggable marker for a timeline event; dragging writes its new frame back."""
        line = pg.InfiniteLine(pos=ev.frame, angle=90, movable=True,
                               pen=pg.mkPen("#ff5050", width=2))
        line.sigPositionChanged.connect(lambda ln, e=ev: setattr(e, "frame", int(ln.value())))
        self.plot.addItem(line)
        self._event_lines.append((ev, line))
        return line

    def _clear_event_lines(self):
        """Remove every event marker from the plot; the Timeline keeps the events."""
        for _ev, line in self._event_lines:
            self.plot.removeItem(line)
        self._event_lines.clear()

    def compute_propagation(self):
        if not self.session.rois:
            self.status.setText("No traces; place ROIs first.")
            return None
        # Detect onsets on whatever the trace plot shows, so the overlaid onset
        # lines line up with the displayed curves.
        start = self._crop_start()  # region is in frames; traces index from 0
        lo, hi = sorted(int(round(v)) for v in self.prop_region.getRegion())
        result = self.session.cross_roi_propagation(
            signal=self._displayed().signal, method=self.onset_method_box.currentText(),
            frac=self.frac_box.value(), k=self.k_box.value(), d=self.d_box.value(),
            baseline_region=(lo - start, hi - start),
            direction_mode=_DIRECTION_MODES[self.prop_dir_box.currentText()],
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
            frac=self.hm_frac_box.value(), k=self.hm_k_box.value(), d=self.hm_d_box.value(),
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

    # ------------------------------------------------------ kymograph path
    # ``_path_points`` is the path; the graphic is rebuilt from it whenever it
    # changes, and dragging the graphic writes straight back into it. One
    # direction each way, so a point can be clicked in, dragged, and undone
    # without the two representations ever disagreeing.
    def add_path_point(self, row: float, col: float):
        """Extend the path with a point at ``(row, col)``."""
        self._path_points.append((float(row), float(col)))
        self._rebuild_path_graphic()
        self._report_path()
        return self._path_points[-1]

    def clear_last_point(self):
        """Drop the path's last point, wherever it has since been dragged to."""
        if not self._path_points:
            self.status.setText("No path to shorten.")
            return
        self._path_points.pop()
        self._rebuild_path_graphic()
        self._report_path()

    def clear_path(self):
        """Remove the whole path from the image."""
        self._path_points = []
        self._rebuild_path_graphic()
        self.status.setText("Path cleared.")

    def path_points(self) -> list:
        """The path as ``(y, x)`` image points, in the order they were placed."""
        return list(self._path_points)

    def _report_path(self):
        n = len(self._path_points)
        if n == 0:
            self.status.setText("Click the image to start a path.")
        elif n == 1:
            self.status.setText("Path started — click again to extend it.")
        else:
            self.status.setText(
                f"Path of {n} points — drag any point to adjust, then compute."
            )

    def _rebuild_path_graphic(self):
        """Redraw the path graphic from ``_path_points``.

        Two points make it an open ``PolyLineROI``, whose handles are individually
        draggable (and whose segments accept a click to insert a point); a single
        point has no line yet, so it is shown as a plain marker until it does.
        """
        self._remove_path_graphic()
        points = self._path_points
        if len(points) >= 2:
            # pyqtgraph works in (x, y); the session and every path API here are
            # in image (y, x) order.
            self._path_item = pg.PolyLineROI([(x, y) for (y, x) in points],
                                             closed=False, pen=_PATH_PEN, movable=True)
            self._path_item.sigRegionChanged.connect(self._on_path_moved)
        elif len(points) == 1:
            y, x = points[0]
            self._path_item = pg.PlotDataItem([x], [y], pen=None, symbol="o",
                                              symbolSize=8, symbolBrush="#00ff7f")
        else:
            return
        self.kymo_movie.view.addItem(self._path_item)

    def _remove_path_graphic(self):
        """Take the path graphic off the image, muted so its own teardown moves
        (handles being dropped) can't be mistaken for the user editing it."""
        if self._path_item is None:
            return
        if isinstance(self._path_item, pg.PolyLineROI):
            self._path_item.sigRegionChanged.disconnect(self._on_path_moved)
        self.kymo_movie.view.removeItem(self._path_item)
        self._path_item = None

    def _on_path_moved(self):
        """A dragged point (or one inserted on a segment) writes the path back."""
        points = polyline_vertices(self._path_item)
        if len(points) < 2:     # mid-edit / teardown — keep the last valid path
            return
        self._path_points = points

    def _click_hits_path(self, scene_pos) -> bool:
        """Whether a click landed on the path graphic or one of its handles.

        Those clicks belong to the polyline — dragging a point, or inserting one
        on a segment — so they must not *also* drop a new point at the far end.
        """
        if self._path_item is None:
            return False
        for item in self.kymo_movie.view.scene().items(scene_pos):
            node = item
            while node is not None:
                if node is self._path_item:
                    return True
                node = node.parentItem()
        return False

    def _on_kymo_click(self, ev):
        """A left click on the image extends the path; clicks on it are its own."""
        # The second click of a double-click would otherwise stack a duplicate
        # point on top of the first one's.
        if ev.button() != _LEFT or ev.double() or self._click_hits_path(ev.scenePos()):
            return
        p = self.kymo_movie.view.mapSceneToView(ev.scenePos())
        row, col = p.y(), p.x()
        height, width = self._kymo_shape_yx
        if 0 <= row < height and 0 <= col < width:
            self.add_path_point(row, col)

    # -------------------------------------------------------- kymograph
    def compute_kymograph(self):
        """Sample the stack along the drawn path and show the distance × time image."""
        if self.session.data is None:
            self.status.setText("No data loaded.")
            return None
        points = self.path_points()
        if len(points) < 2:
            self.status.setText(
                "A path needs at least two points — click along the feature on the image."
            )
            return None
        dff = self.kymo_signal_box.currentText() == _KYMO_DFF
        # Baseline frames are trace-column (post-crop) indices, like every other
        # first-N baseline on this widget.
        baseline = (0, self.kymo_base_box.value()) if dff else None
        try:
            result = self.session.kymograph(
                points, width=self.kymo_width_box.value(), baseline=baseline,
            )
        except ValueError as exc:          # degenerate path / empty baseline window
            self.status.setText(str(exc))
            return None
        self._kymo_result = result
        self._show_kymograph(result)
        return result

    def _show_kymograph(self, result):
        """Draw a kymograph in the session's own units.

        The image is placed in data coordinates — x from the crop start in frames
        or seconds, y the arc length along the path in pixels or µm — so reading a
        band's slope off the axes gives the propagation speed in the same units
        the propagation summary reports.
        """
        values = np.asarray(result["values"], dtype=float)
        iv = self._frame_interval()
        scale = iv or 1.0
        dfactor, dunit = self._distance_units()
        length = float(result["distance"][-1]) * dfactor
        self.kymo_image.setImage(values, autoLevels=False)
        self.kymo_image.setRect(QtCore.QRectF(
            self._crop_start() * scale, 0.0, values.shape[1] * scale, length or 1.0
        ))

        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        if hi <= lo:
            hi = lo + 1.0
        # Fine, unit-agnostic drag steps for the interactive level handles, as on
        # the heatmap page: ~200 steps across the data span.
        self.kymo_cbar.rounding = max((hi - lo) / 200.0, 1e-9)
        self.kymo_cbar.setLevels((lo, hi))
        self.kymo_cbar.setLabel("left", self.kymo_signal_box.currentText())
        self.kymo_plot.setLabel("bottom", "time (s)" if iv else "frame")
        self.kymo_plot.setLabel("left", f"distance along path ({dunit})")
        self.kymo_plot.getViewBox().autoRange(padding=0.02)
        self.status.setText(
            f"Kymograph: {values.shape[0]} positions over {length:.4g} {dunit}"
            f" × {values.shape[1]} frames."
        )

    def _on_kymo_signal_changed(self, text: str):
        """The baseline count only means anything for the ΔF/F kymograph."""
        for w in (self.kymo_base_label, self.kymo_base_box):
            w.setEnabled(text == _KYMO_DFF)

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
        against Euclidean distance and no line is drawn. Both axes follow their
        calibration — seconds when the frame interval is set, µm when the pixel
        size is (SPEC §3 axes) — so the fitted slope is exactly the speed the
        summary reports, in whatever units that is. The view autoranges to fit all
        points.
        """
        self.prop_plot.clear()
        src = result["source_roi"]
        if src is None:
            self._prop_fit = None
            return
        onsets = np.asarray(result["onsets"], dtype=float)
        coords = np.array([r.center for r in self.session.rois], dtype=float)  # (y, x)
        direction = result["direction"]
        speed = result["speed_px_per_frame"]
        coherent = (direction is not None and isinstance(speed, float)
                    and np.isfinite(speed) and speed > 0)

        delta = coords - coords[src]                    # (dy, dx) from the source
        dfactor, dunit = self._distance_units()         # px -> µm when calibrated
        if coherent:
            # Signed distance along the propagation direction.
            dist = (delta @ np.asarray(direction, dtype=float)) * dfactor
            ylabel = f"distance along propagation ({dunit})"
        else:
            dist = np.hypot(*delta.T) * dfactor
            ylabel = f"distance from source ({dunit})"
        self.prop_plot.setLabel("left", ylabel)

        iv = self._frame_interval()
        scale = iv or 1.0
        xlabel = "onset delay (s)" if iv else "onset delay (frame)"
        self.prop_plot.setLabel("bottom", xlabel)

        valid = ~np.isnan(onsets)
        r = dist[valid]                               # distance -> y-axis
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
        fit = None
        if coherent and delay.size >= 1:
            # Speed in the units the axes are drawn in, so the line's slope is
            # literally the number in the legend and in the summary.
            shown_speed = speed * speed_units(self._pixel_size(), iv)[0]
            delay_hat = r / shown_speed                 # delay a perfect wave predicts
            ss_res = float(np.sum((delay - delay_hat) ** 2))
            ss_tot = float(np.sum((delay - delay.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            dline = np.array([min(0.0, float(delay.min())), max(0.0, float(delay.max()))])
            fit = (dline, shown_speed * dline,
                   f"{self._speed_str(speed)} (R²={r2:.3f})")
            self.prop_plot.plot(dline, shown_speed * dline,
                                pen=pg.mkPen("#ff5050", width=2), name=fit[2])

        # Snapshot the scatter for a clean WYSIWYG export (see _save_propagation).
        self._prop_fit = {
            "delay": np.asarray(delay, dtype=float),
            "dist": np.asarray(r, dtype=float),
            "labels": [labels[i] if i < len(labels) else f"roi_{i}"
                       for i in np.flatnonzero(valid)],
            "xlabel": xlabel,
            "ylabel": ylabel,
            "fit": fit,
            "title": "Distance vs onset delay",
        }

        # Autorange so every point (and its label) stays in frame.
        self.prop_plot.getViewBox().autoRange(padding=0.12)

    def _write_propagation_summary(self, result):
        labels = self.session.traces.labels
        lines = ["Propagation", "==========="]
        lines.append(f"speed: {self._speed_str(result['speed_px_per_frame'])}")
        if result["direction"]:
            lines.append("direction (dy, dx): "
                         f"({result['direction'][0]:.3f}, {result['direction'][1]:.3f})")
        # Name the mode the direction came from — the same onsets give different
        # directions under a ROI-line fit and a free 2D fit.
        mode = result.get("direction_mode")
        label = next((k for k, v in _DIRECTION_MODES.items() if v == mode), mode)
        if label:
            lines.append(f"direction mode: {label}")
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

        shown = self._displayed()
        data, labels = shown.data, shown.labels
        if data is None or data.shape[0] == 0:
            return
        n = data.shape[0]
        x = self._crop_start() + np.arange(data.shape[1])  # frames, crop-aware
        for i in range(n):
            curve = self.plot.plot(x, data[i], pen=pg.intColor(i, hues=max(6, n)),
                                   name=labels[i] if i < len(labels) else f"roi_{i}")
            self._curves.append(curve)
        self.plot.setLabel("left", shown.ylabel)

    # -------------------------------------------------------------- saving
    def _save_traces(self):
        """Export the trace plot as shown (WYSIWYG): the displayed signal plus the
        visible events, baseline band(s), and onset markers, cleanly restyled.

        Mirrors the currently displayed signal (raw / ΔF/F / smoothed), the
        stimulus event lines, whichever baseline window is on the plot, and the
        per-ROI onset markers when propagation onsets are overlaid.
        """
        shown = self._displayed()
        data = shown.data
        if data is None or data.shape[0] == 0:
            self.status.setText("No traces to save.")
            return
        start = self._crop_start()
        iv = self._frame_interval()
        scale = iv or 1
        x = (start + np.arange(data.shape[1])) * scale
        xlabel = "time (s)" if iv else "frame"

        events = [(ev.frame * scale, ev.label) for ev, _ in self._event_lines]
        regions = []
        if (BaselineMethod(self.baseline_box.currentText()) == BaselineMethod.REGION
                and self.region.scene() is not None):
            lo, hi = self.region.getRegion()
            regions.append((lo * scale, hi * scale, "#0072B2"))
        if self.prop_region.scene() is not None:
            lo, hi = self.prop_region.getRegion()
            regions.append((lo * scale, hi * scale, "#009E73"))
        onsets = None
        prop = self.session.analyses.get("propagation")
        if self._onset_lines and prop is not None:
            onsets = [((start + t) * scale) if np.isfinite(t) else None
                      for t in np.asarray(prop["onsets"], dtype=float)]

        def render(path):
            return figures.export_traces(
                list(data), x=x, xlabel=xlabel, ylabel=shown.ylabel,
                labels=list(shown.labels), events=events, regions=regions,
                onsets=onsets, save=path,
            )

        save_figure_dialog(self, render, title="Save traces", status=self.status)

    def _save_propagation(self):
        """Export the distance-vs-onset-delay scatter as shown (WYSIWYG, clean)."""
        fit = self._prop_fit
        if not fit:
            self.status.setText("Run propagation first.")
            return

        def render(path):
            return figures.export_scatter(
                fit["delay"], fit["dist"], xlabel=fit["xlabel"],
                ylabel=fit["ylabel"], point_labels=fit["labels"],
                fit=fit["fit"], title=fit["title"], save=path,
            )

        save_figure_dialog(self, render, title="Save propagation figure",
                           status=self.status)

    def _save_kymograph(self):
        """Export the kymograph as shown (WYSIWYG): same axes, units and contrast."""
        result = self._kymo_result
        if result is None:
            self.status.setText("Compute a kymograph first.")
            return
        values = np.asarray(result["values"], dtype=float)
        scale = self._frame_interval() or 1.0
        dfactor, dunit = self._distance_units()
        x0 = self._crop_start() * scale
        extent = (x0, x0 + values.shape[1] * scale,
                  0.0, float(result["distance"][-1]) * dfactor)
        signal = self.kymo_signal_box.currentText()

        def render(path):
            return figures.export_kymograph(
                values, extent=extent,
                xlabel="time (s)" if self._frame_interval() else "frame",
                ylabel=f"distance along path ({dunit})", cbar_label=signal,
                levels=self.kymo_cbar.levels(), save=path,
            )

        save_figure_dialog(self, render, title="Save kymograph", status=self.status)

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
            # Through the setter, so the change is announced to the other panels
            # showing this calibration (the app's Import tab).
            self.session.set_frame_interval(interval)
        self._time_axis.set_frame_interval(interval)
        self.plot.setLabel("bottom", "time (s)" if interval else "frame")
        self._refresh_units()

    def _on_pixel_size_changed(self, value: float):
        """Switch distance readouts between pixels and µm. SPEC §3 space axis.

        The spatial twin of ``_on_interval_changed``: ROI coordinates stay in
        pixels (nothing on screen moves), and the value goes to the session's
        SpatialScale so the notebook, provenance and saved figures agree.
        """
        if self.session.data is not None:
            self.session.set_pixel_size(value or None)   # announces to other panels
        else:
            self.session.space.pixel_size = value or None
        self._refresh_units()

    def _refresh_units(self):
        """Redraw existing results after a calibration change.

        The propagation speed, its graph axes and onset times — and the
        kymograph's two axes — are all *reported* in calibrated units, so they
        would otherwise keep the units they were computed under. The results
        themselves (px/frame, frame onsets, sampled intensities) are unchanged;
        only their presentation is.
        """
        prop = self.session.analyses.get("propagation")
        if prop is not None and self.session.traces is not None:
            self._plot_propagation_fit(prop)
            self._write_propagation_summary(prop)
        if self._kymo_result is not None:
            self._show_kymograph(self._kymo_result)

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

    @staticmethod
    def _enable_onset_params(method: str, frac, k, d) -> None:
        """Enable only the params a method uses: frac→fraction_of_max, k→std &
        derivative, d→derivative. Each of ``frac``/``k``/``d`` is that param's
        (label, spinbox) pair, so the trace page and the heatmap page — which own
        separate widgets for the same three params — share one rule.
        """
        for widgets, used in ((frac, method == "fraction_of_max"),
                              (k, method in ("std", "derivative")),
                              (d, method == "derivative")):
            for w in widgets:
                w.setEnabled(used)

    def _on_onset_method_changed(self, method: str):
        self._enable_onset_params(
            method, (self.frac_label, self.frac_box), (self.k_label, self.k_box),
            (self.d_label, self.d_box),
        )

    def _on_hm_method_changed(self, method: str):
        self._enable_onset_params(
            method, (self.hm_frac_label, self.hm_frac_box),
            (self.hm_k_label, self.hm_k_box), (self.hm_d_label, self.hm_d_box),
        )
