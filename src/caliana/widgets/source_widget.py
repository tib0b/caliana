"""Stage I — dataset selection & import widget. SPEC.md §3 Stage I.

Picks the file and the downsample-on-load parameters, then loads them into the
Session. Its counterpart, ``ImportPreviewWidget``, shows the *result*; keeping
them apart is what makes re-importing cheap to judge — change ``spatial_step``,
press Load, watch the preview next door change.

Also owns the two optional scales (seconds per frame, µm per pixel), pre-filled
from the file's metadata when it declares them. They are editable here because
this is where a user who knows their microscope will look; the analysis widget
shows the same two values and either may be used (SPEC §3 Stage I, "Units").

The interaction logic lives in plain methods (``set_path``, ``import_params``,
``load``) so it can be driven from tests without a file dialog.
"""
from __future__ import annotations

from pathlib import Path

from ..models import ImportParams
from ._qt import get_qt
from ._task import run_in_background

QtCore, QtGui, QtWidgets = get_qt()

_FILE_FILTER = "Recordings (*.tif *.tiff *.nd2);;TIFF (*.tif *.tiff);;Nikon ND2 (*.nd2);;All files (*)"


def _compact_form(box) -> "QtWidgets.QFormLayout":
    """A form layout inside ``box``, tightened to keep the panel narrow."""
    form = QtWidgets.QFormLayout(box)
    form.setContentsMargins(8, 6, 8, 6)
    form.setSpacing(4)
    return form


class SourceWidget(QtWidgets.QWidget):
    """File picker + import parameters + calibration, feeding ``Session.load``."""

    closed = QtCore.Signal()
    # Emitted with the Session after a successful load. The same object is
    # mutated in place (never replaced), so panels holding it stay valid.
    loaded = QtCore.Signal(object)

    def __init__(self, session=None, parent=None):
        super().__init__(parent)
        if session is None:
            from ..session import Session

            session = Session()
        self.session = session
        self.result = session
        self.setWindowTitle("Caliana — Open Recording")
        self.resize(460, 520)

        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        # Compact: this panel sits next to the preview in the app, and every
        # pixel it does not take is one the movie/heatmap get.
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # File row.
        file_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Choose a .tif / .tiff / .nd2 recording…")
        file_row.addWidget(self.path_edit, stretch=1)
        self.browse_btn = QtWidgets.QPushButton("Browse…")
        self.browse_btn.clicked.connect(self.browse)
        file_row.addWidget(self.browse_btn)
        layout.addLayout(file_row)

        layout.addWidget(self._build_import_box())
        layout.addWidget(self._build_scale_box())

        # Load + status.
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.setToolTip("Read the file with these parameters into the session")
        self.load_btn.clicked.connect(self.load)
        layout.addWidget(self.load_btn)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def _build_import_box(self) -> "QtWidgets.QWidget":
        """The six ImportParams fields (SPEC §3 Stage I, "downsample on load")."""
        box = QtWidgets.QGroupBox("Import (applied on load)")
        form = _compact_form(box)

        frames = QtWidgets.QHBoxLayout()
        frames.setSpacing(4)
        self.start_box = QtWidgets.QSpinBox()
        self.start_box.setRange(0, 10_000_000)
        self.start_box.setToolTip("First frame kept (the window is [start, end))")
        frames.addWidget(self.start_box)
        frames.addWidget(QtWidgets.QLabel("to"))
        self.end_box = QtWidgets.QSpinBox()
        # -1 is the "no end given" sentinel: ImportParams.end is None for
        # "until the last frame", which a plain spinbox cannot express.
        self.end_box.setRange(-1, 10_000_000)
        self.end_box.setValue(-1)
        self.end_box.setSpecialValueText("(end)")
        self.end_box.setToolTip("Last frame kept, exclusive; (end) reads to the last frame")
        frames.addWidget(self.end_box)
        form.addRow("Frames:", frames)

        self.tstep_box = QtWidgets.QSpinBox()
        self.tstep_box.setRange(1, 1000)
        self.tstep_box.setToolTip("Average every N frames into one (1 = off)")
        form.addRow("Temporal step:", self.tstep_box)

        self.sstep_box = QtWidgets.QSpinBox()
        self.sstep_box.setRange(1, 64)
        self.sstep_box.setToolTip("Keep every Nth pixel along Y and X (1 = full resolution)")
        form.addRow("Spatial step:", self.sstep_box)

        self.window_check = QtWidgets.QCheckBox("Crop field of view")
        self.window_check.setToolTip("Keep only a Y/X region, in the file's pixel coordinates")
        self.window_check.toggled.connect(self._on_window_toggled)
        form.addRow(self.window_check)

        window = QtWidgets.QHBoxLayout()
        window.setSpacing(4)
        self.window_boxes = []
        for name in ("y0", "y1", "x0", "x1"):
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 10_000_000)
            spin.setPrefix(f"{name} ")
            spin.setEnabled(False)
            # Four spinboxes on one row are what sets this panel's minimum
            # width; the 7-digit range they accept is never typed in full.
            spin.setMaximumWidth(68)
            self.window_boxes.append(spin)
            window.addWidget(spin)
        form.addRow(window)

        self.channel_box = QtWidgets.QSpinBox()
        self.channel_box.setRange(0, 64)
        self.channel_box.setToolTip(
            "Which channel to keep from a multi-channel file (Caliana is single-channel)"
        )
        form.addRow("Channel:", self.channel_box)
        return box

    def _build_scale_box(self) -> "QtWidgets.QWidget":
        """Time and space calibration of the *loaded* stack (SPEC §3, Units)."""
        box = QtWidgets.QGroupBox("Calibration (from the file when declared)")
        form = _compact_form(box)

        self.interval_box = QtWidgets.QDoubleSpinBox()
        self.interval_box.setRange(0.0, 1e6)
        self.interval_box.setDecimals(4)
        self.interval_box.setSpecialValueText("frames")   # 0 ⇒ frames-only axis
        self.interval_box.setToolTip(
            "Seconds per frame of the loaded stack; 0 keeps the time axis in frames"
        )
        self.interval_box.valueChanged.connect(self._on_interval_changed)
        form.addRow("Frame interval (s):", self.interval_box)

        self.pixel_box = QtWidgets.QDoubleSpinBox()
        self.pixel_box.setRange(0.0, 1e6)
        self.pixel_box.setDecimals(4)
        self.pixel_box.setSpecialValueText("pixels")      # 0 ⇒ pixels-only distances
        self.pixel_box.setToolTip(
            "µm per pixel of the loaded stack; 0 keeps distances in pixels"
        )
        self.pixel_box.valueChanged.connect(self._on_pixel_size_changed)
        form.addRow("Pixel size (µm):", self.pixel_box)
        return box

    # --------------------------------------------------------------- state
    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        Mirrors the session's source path, the parameters it was imported with,
        and its calibration — so the panel keeps telling the truth after a file
        is opened from the app's File menu, or the pixel size is changed on the
        analysis tab.
        """
        src = self.session.source
        if src is not None:
            self.path_edit.setText(str(src.path))
            self._show_import_params(src.import_params)
        self._sync_scale_boxes()
        self.load_btn.setEnabled(bool(self.path_edit.text().strip()))

    def _show_import_params(self, params: ImportParams):
        """Reflect an ImportParams back into the controls (inverse of ``import_params``)."""
        self.start_box.setValue(params.start)
        self.end_box.setValue(-1 if params.end is None else params.end)
        self.tstep_box.setValue(params.temporal_step)
        self.sstep_box.setValue(params.spatial_step)
        self.window_check.setChecked(params.spatial_window is not None)
        if params.spatial_window is not None:
            for spin, value in zip(self.window_boxes, params.spatial_window):
                spin.setValue(int(value))
        self.channel_box.setValue(params.channel)

    def _sync_scale_boxes(self):
        """Point the calibration boxes at the session's scales without writing back."""
        tl = self.session.timeline
        interval = tl.frame_interval if tl is not None else None
        for box, value in ((self.interval_box, interval),
                           (self.pixel_box, self.session.space.pixel_size)):
            blocker = QtCore.QSignalBlocker(box)
            box.setValue(value or 0.0)
            del blocker

    # -------------------------------------------------------------- inputs
    def set_path(self, path):
        """Point the widget at ``path`` (what the file dialog does, minus the dialog)."""
        self.path_edit.setText(str(path))
        self.load_btn.setEnabled(bool(str(path).strip()))
        return self.path_edit.text()

    def browse(self):
        """Ask for a file; a cancelled dialog leaves the current path alone."""
        start = self.path_edit.text().strip() or ""
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", start, _FILE_FILTER
        )
        if path:
            self.set_path(path)
        return path or None

    def import_params(self) -> ImportParams:
        """The controls as an ``ImportParams`` (what ``load`` passes to the session)."""
        window = None
        if self.window_check.isChecked():
            window = tuple(spin.value() for spin in self.window_boxes)
        end = self.end_box.value()
        return ImportParams(
            start=self.start_box.value(),
            end=None if end < 0 else end,
            temporal_step=self.tstep_box.value(),
            spatial_step=self.sstep_box.value(),
            spatial_window=window,
            channel=self.channel_box.value(),
        )

    # ------------------------------------------------------------- actions
    def load(self):
        """Load the chosen file with the current parameters, off the UI thread.

        Returns the ``Task`` doing the work (``None`` if there was nothing to do),
        so a caller that needs the loaded session — a test, or the notebook
        wrapper — can ``wait()`` on it. On success ``loaded`` carries the same
        Session object, now holding the new stack.
        """
        path = self.path_edit.text().strip()
        if not path:
            self.status.setText("Choose a file first.")
            return None
        params = self.import_params()
        self._set_busy(True)
        self.status.setText(f"Loading {Path(path).name}…")
        return run_in_background(
            self, lambda: self.session.load(path, **vars(params)),
            on_done=self._on_loaded, on_error=self._on_load_failed,
            label=f"Loading {Path(path).name}…",
        )

    def _on_loaded(self, session):
        self._set_busy(False)
        self.result = session
        self._sync_scale_boxes()
        shape = "×".join(str(n) for n in session.data.shape)
        self.status.setText(f"Loaded {Path(str(session.source.path)).name} — [T,Y,X] = {shape}")
        self.loaded.emit(session)

    def _on_load_failed(self, exc):
        self._set_busy(False)
        self.status.setText(f"Could not load: {exc}")
        QtWidgets.QMessageBox.critical(self, "Load failed", _load_error_message(exc))

    def _set_busy(self, busy: bool):
        for w in (self.load_btn, self.browse_btn, self.path_edit):
            w.setEnabled(not busy)

    # ------------------------------------------------------------- signals
    def _on_window_toggled(self, checked: bool):
        for spin in self.window_boxes:
            spin.setEnabled(checked)

    def _on_interval_changed(self, value: float):
        """Calibrate the session's time axis (0 ⇒ frames-only). SPEC §3 time axis."""
        if self.session.timeline is not None:
            self.session.set_frame_interval(value or None)

    def _on_pixel_size_changed(self, value: float):
        """Calibrate the session's space axis (0 ⇒ pixels-only). SPEC §3 space axis."""
        if self.session.data is not None:
            self.session.set_pixel_size(value or None)

    def closeEvent(self, event):
        self.result = self.session
        self.closed.emit()
        super().closeEvent(event)


def _load_error_message(exc: Exception) -> str:
    """A load failure phrased for someone who did not write the code.

    The three that actually happen — a format we don't read, an optional reader
    that was never installed, and a file that moved — say nothing useful in their
    raw form, so each gets the next step spelled out.
    """
    if isinstance(exc, ModuleNotFoundError):
        return (f"This file needs the optional '{exc.name}' package, which is not "
                f"installed.\n\nInstall it with:  pip install {exc.name}")
    if isinstance(exc, FileNotFoundError):
        return f"That file no longer exists:\n{exc.filename}"
    if isinstance(exc, ValueError):
        return f"{exc}\n\nCaliana reads .tif, .tiff and .nd2 recordings."
    return f"{type(exc).__name__}: {exc}"
