"""Stage II — registration widget. SPEC.md §3 Stage II (motion correction).

The GUI binding for ``Session.register``: pick the mode, the reference, the
transformation model and how the correction is applied, run it off the UI thread,
and read back what it did — which frames drifted out of their leaf box, and which
extraction path the session is now on.

Per-leaf mode needs boxes drawn on the image, and that widget already exists, so
this one **embeds** ``LeafSelectionWidget`` as its image pane rather than
re-implementing box drawing. The pane is shown only in per-leaf mode: whole-frame
and none register the whole field of view and need no boxes at all.

The interaction logic lives in plain methods (``run``, ``mode``, ``summary``) so
it can be driven from tests without a mouse.
"""
from __future__ import annotations

from ..models import RegistrationMode
from ._qt import get_qt
from ._task import run_in_background
from .leaf_widget import LeafSelectionWidget

QtCore, QtGui, QtWidgets = get_qt()

# Combo label -> Session.register argument. Insertion order is the combo order.
_MODES = {
    "None (raw stack)": RegistrationMode.NONE,
    "Whole frame": RegistrationMode.WHOLE_FRAME,
    "Per leaf": RegistrationMode.PER_LEAF,
}

# TurboReg models, coarsest first (see registration._STACKREG_TRANSFORMS).
_TRANSFORMS = ["translation", "rigid_body", "scaled_rotation", "affine"]

_REFERENCES = ["previous", "first", "mean"]


class RegistrationWidget(QtWidgets.QWidget):
    """Motion-correction controls + results, over an embedded leaf-box pane."""

    closed = QtCore.Signal()
    # Emitted after registration succeeds; the app refreshes the tabs downstream
    # (registration invalidates every trace).
    registered = QtCore.Signal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.result = session.registration
        self.setWindowTitle("Caliana — Registration")
        self.resize(940, 620)

        self._task = None            # the run in flight, if any
        # Settings the last run actually used. The Session records the mode and
        # reference but not the transformation model, so the summary reads it
        # from here rather than from a combo the user may have moved since.
        self._last_run: dict | None = None
        self._build_ui()
        self.reload()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(list(_MODES))
        self.mode_box.setToolTip(
            "None: place ROIs on the raw stack.\n"
            "Whole frame: one transform per frame for the whole field of view.\n"
            "Per leaf: each drawn box registered independently."
        )
        self.mode_box.currentTextChanged.connect(self._on_mode_changed)
        row.addWidget(self.mode_box)

        row.addWidget(QtWidgets.QLabel("Reference:"))
        self.ref_box = QtWidgets.QComboBox()
        self.ref_box.addItems(_REFERENCES)
        self.ref_box.setToolTip("Image every frame is registered to")
        row.addWidget(self.ref_box)

        row.addWidget(QtWidgets.QLabel("Model:"))
        self.transform_box = QtWidgets.QComboBox()
        self.transform_box.addItems(_TRANSFORMS)
        self.transform_box.setCurrentText("affine")
        self.transform_box.setToolTip(
            "Transformation estimated per frame. Coarser models are "
            "more constrained and more robust on low-contrast tissue; affine also "
            "absorbs scale and shear."
        )
        row.addWidget(self.transform_box)

        self.mask_check = QtWidgets.QCheckBox("Mask to tissue")
        self.mask_check.setToolTip(
            "Estimate motion on the tissue silhouette instead of raw intensities, "
            "so registration tracks the dim leaf rather than the static bright "
            "background -- curently unsuported"
        )
        self.mask_check.setChecked(False)
        row.addWidget(self.mask_check)
        row.addStretch(1)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Correction:"))
        self.apply_box = QtWidgets.QComboBox()
        # Order matters: the first entry is the default, and warping is what
        # `Session.register` defaults to.
        self.apply_box.addItems(["Warp the stack", "Move the ROIs (no warping)"])
        self.apply_box.setToolTip(
            "Warp the stack: build a stabilized copy and sample static ROIs on it.\n"
            "Move the ROIs: keep raw pixels and carry each ROI with its tissue at "
            "extraction time — no interpolation bias in ΔF/F, preferable on dim data."
        )
        row2.addWidget(self.apply_box)

        self.run_btn = QtWidgets.QPushButton("Run registration")
        self.run_btn.clicked.connect(self.run)
        row2.addWidget(self.run_btn)
        row2.addStretch(1)
        self.status = QtWidgets.QLabel("")
        row2.addWidget(self.status)
        layout.addLayout(row2)

        # Image pane (per-leaf only) + results readout.
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(split, stretch=1)

        # Left pane: the leaf-box editor in per-leaf mode, a note explaining why
        # there is nothing to draw in the other two.
        self.left_stack = QtWidgets.QStackedWidget()
        self._mode_hint = QtWidgets.QLabel()
        self._mode_hint.setWordWrap(True)
        self._mode_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.left_stack.addWidget(self._mode_hint)              # 0: no boxes needed
        # Full reuse: the same widget `select_leaves` opens, docked as our image
        # pane. It owns box drawing and writes straight to session.leaf_regions.
        self.leaf_panel = LeafSelectionWidget(self.session)
        self.left_stack.addWidget(self.leaf_panel)              # 1: per-leaf mode
        split.addWidget(self.left_stack)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QtWidgets.QLabel("Result"))
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        rv.addWidget(self.results)
        split.addWidget(right)
        split.setSizes([560, 380])
        self._on_mode_changed(self.mode_box.currentText())

    # --------------------------------------------------------------- state
    def reload(self):
        """Re-read the Session and redraw. Safe to call any number of times.

        Shows the mode the session is actually in (a run may have happened in a
        notebook, or before a new file was loaded) and hands the reload on to the
        embedded leaf pane, whose boxes and image come from the same session.
        """
        self.leaf_panel.reload()
        mode = self.session.registration.mode
        label = next((k for k, v in _MODES.items() if v == mode), None)
        if label is not None:
            blocker = QtCore.QSignalBlocker(self.mode_box)
            self.mode_box.setCurrentText(label)
            del blocker
            self._on_mode_changed(label)
        self.ref_box.setCurrentText(self.session.registration.reference)
        if self.session.track_motion:
            self.apply_box.setCurrentIndex(1)
        has_data = self.session.data is not None
        self.run_btn.setEnabled(has_data)
        self.status.setText("" if has_data else "Load a stack first.")
        self.results.setPlainText(self.summary() if self._has_run() else "")

    @property
    def mode(self) -> RegistrationMode:
        """The registration mode currently selected in the combo."""
        return _MODES[self.mode_box.currentText()]

    @property
    def apply_warp(self) -> bool:
        """Whether the run will warp the stack (vs. carrying ROIs with the tissue)."""
        return self.apply_box.currentIndex() == 0

    def _has_run(self) -> bool:
        """Whether the session carries a registration result worth reporting."""
        return (self.session.registration.mode != RegistrationMode.NONE
                or self.session.registered_data is not None
                or self.session.track_motion)

    # ------------------------------------------------------------- actions
    def run(self):
        """Register the stack with the chosen settings, off the UI thread.

        Returns the ``Task`` doing the work (``None`` if it could not start), so a
        caller that needs the result — a test, or a script — can ``wait()`` on it.
        """
        if self.session.data is None:
            self.status.setText("Load a stack first.")
            return None
        mode = self.mode
        if mode == RegistrationMode.PER_LEAF and not self.session.leaf_regions:
            self.status.setText("Draw at least one leaf box first.")
            QtWidgets.QMessageBox.information(
                self, "No leaf boxes",
                "Per-leaf registration needs one box per leaf.\n\n"
                "Use “Add leaf box”, then drag and resize it to cover the leaf "
                "generously — tissue that drifts outside its box cannot be "
                "stabilized.",
            )
            return None

        kwargs = dict(
            mode=mode, reference=self.ref_box.currentText(),
            mask=self.mask_check.isChecked(), apply=self.apply_warp,
            transformation=self.transform_box.currentText(),
        )
        self._set_busy(True)
        self.status.setText("Registering…")
        self._last_run = dict(kwargs)
        self._task = run_in_background(
            self, lambda: self.session.register(**kwargs),
            on_done=self._on_registered, on_error=self._on_failed,
            label="Registering — this can take a while on a long recording…",
        )
        return self._task

    def _on_registered(self, session):
        self._set_busy(False)
        self.result = session.registration
        self.results.setPlainText(self.summary())
        self.status.setText(f"Registered ({session.registration.mode.value}).")
        # Leaf boxes are unchanged, but the pane draws the *stabilized* stack once
        # a warp exists.
        self.leaf_panel.reload()
        self.registered.emit(session)

    def _on_failed(self, exc):
        self._set_busy(False)
        self.status.setText(f"Registration failed: {exc}")
        QtWidgets.QMessageBox.critical(
            self, "Registration failed", _register_error_message(exc)
        )

    def _set_busy(self, busy: bool):
        for w in (self.run_btn, self.mode_box, self.ref_box, self.transform_box,
                  self.mask_check, self.apply_box, self.leaf_panel):
            w.setEnabled(not busy)

    # ------------------------------------------------------------- results
    def summary(self) -> str:
        """What the last run did, in the terms the user chose it in.

        The one number worth reading afterwards is per-leaf drift: frames whose
        estimated shift approaches the box margin are flagged low-confidence
        (SPEC §3, drift-out-of-box) — they are where traces silently corrupt, and
        the fix is a more generous box.
        """
        reg = self.session.registration
        lines = ["Registration", "============",
                 f"mode: {reg.mode.value}",
                 f"reference: {reg.reference}"]
        if reg.mode != RegistrationMode.NONE:
            # Omitted rather than guessed when the run came from elsewhere (a
            # notebook), since the Session does not carry the model used.
            if self._last_run is not None:
                lines.append(f"model: {self._last_run['transformation']}")
                lines.append(f"masked to tissue: {'yes' if self._last_run['mask'] else 'no'}")
            lines.append("correction: " + ("stabilized stack (ROIs static)"
                                           if self.session.registered_data is not None
                                           else "ROIs follow the tissue (raw pixels)"))
        if reg.mode == RegistrationMode.WHOLE_FRAME:
            lines.append(f"transforms: {len(reg.transforms)} frames")
        elif reg.mode == RegistrationMode.PER_LEAF:
            lines.append("")
            for i, leaf in enumerate(self.session.leaf_regions):
                name = leaf.label or f"leaf {i}"
                flagged = leaf.low_confidence_frames
                lines.append(f"{name}: bbox={tuple(leaf.bbox)}, "
                             f"{len(leaf.transforms)} frames")
                if flagged:
                    shown = ", ".join(str(f) for f in flagged[:12])
                    more = "…" if len(flagged) > 12 else ""
                    lines.append(f"  ⚠ {len(flagged)} low-confidence frame(s): {shown}{more}")
                    lines.append("     the leaf drifts near its box edge — redraw it larger")
                else:
                    lines.append("  no drift-out-of-box frames")
        return "\n".join(lines)

    # ------------------------------------------------------------- signals
    def _on_mode_changed(self, label: str):
        """Show the leaf pane only where boxes are used, and gate the estimator
        controls: ``none`` estimates nothing at all."""
        mode = _MODES[label]
        per_leaf = mode == RegistrationMode.PER_LEAF
        self.left_stack.setCurrentIndex(1 if per_leaf else 0)
        self._mode_hint.setText(
            "No motion correction — ROIs are placed on the raw stack."
            if mode == RegistrationMode.NONE else
            "Whole-frame mode registers the entire field of view as one unit.\n"
            "No boxes to draw; press “Run registration”."
        )
        estimating = mode != RegistrationMode.NONE
        for w in (self.ref_box, self.transform_box, self.mask_check, self.apply_box):
            w.setEnabled(estimating)

    def closeEvent(self, event):
        self.result = self.session.registration
        self.closed.emit()
        super().closeEvent(event)


def _register_error_message(exc: Exception) -> str:
    """A registration failure phrased for someone who did not write the code."""
    if isinstance(exc, ModuleNotFoundError):
        return (f"Registration needs the '{exc.name}' package, which is not "
                f"installed.\n\nInstall it with:  pip install {exc.name}")
    if isinstance(exc, MemoryError):
        return ("Not enough memory to build the stabilized stack.\n\n"
                "Re-import the recording with a larger temporal or spatial step, "
                "or choose “Move the ROIs (no warping)”, which keeps the raw stack.")
    return f"{type(exc).__name__}: {exc}"
