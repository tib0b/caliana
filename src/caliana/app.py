"""Standalone PyQt application entry point. SPEC.md §1 (Phase 2).

One window around the same widgets the notebook drives (SPEC §2.2), so the app
adds no analysis logic of its own — only what a self-sufficient tool needs and a
notebook does not:

- the workflow laid out as tabs, each gated on its prerequisite existing, so the
  order (import → register → ROIs → crop → analysis) is visible rather than
  remembered;
- data export and file management as menu commands (``File ▸ Export``);
- error handling for people who do not read tracebacks: every long call runs
  behind a progress dialog and reports failures in a dialog, and an excepthook
  catches whatever still slips through instead of killing the window.

Panels are not rebuilt as the session changes; each exposes ``reload()`` and the
window calls it when a tab is activated with a stale ``Session._revision``. That
keeps "registration invalidated the traces" working without any widget knowing
about any other widget.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from .session import Session
from .widgets._qt import ensure_app, get_qt

QtCore, QtGui, QtWidgets = get_qt()

_ORG = "caliana"
_APP = "Caliana"
_RECENT_KEY = "recent_files"
_MAX_RECENT = 8


class MainWindow(QtWidgets.QMainWindow):
    """The workflow, tab by tab, over a single Session owned by this window."""

    def __init__(self, session: Session | None = None, parent=None):
        super().__init__(parent)
        self.session = session if session is not None else Session()
        self.setWindowTitle(_APP)
        self.resize(1180, 720)

        # Revision each panel was last reloaded at, so activating a tab that
        # nothing touched costs nothing.
        self._seen: dict[QtWidgets.QWidget, int] = {}
        self._revision = -1                     # last revision the chrome reflects

        self._build_tabs()
        self._build_menus()
        self._build_statusbar()

        # The one poll in the app: the ROI/leaf widgets edit the session directly
        # and have no "I changed" signal (by design — see the module docstring),
        # so the chrome watches the revision counter instead. Reloads stay lazy;
        # this only re-gates the tabs and redraws the status line.
        self._watch = QtCore.QTimer(self)
        self._watch.setInterval(400)
        self._watch.timeout.connect(self._refresh_chrome)
        self._watch.start()
        self._refresh_chrome(force=True)

    # ------------------------------------------------------------------ UI
    def _build_tabs(self):
        from .widgets.analysis_widget import AnalysisWidget
        from .widgets.crop_widget import CropTracesWidget
        from .widgets.import_widget import ImportPreviewWidget
        from .widgets.registration_widget import RegistrationWidget
        from .widgets.roi_widget import RoiSelectionWidget
        from .widgets.source_widget import SourceWidget

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # Stage I: parameters on the left, the resulting stack on the right, so
        # re-importing at another downsampling is one click and one glance.
        self.source_panel = SourceWidget(self.session)
        self.preview_panel = ImportPreviewWidget(self.session)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split.addWidget(self.source_panel)
        split.addWidget(self.preview_panel)
        split.setSizes([420, 760])

        self.registration_panel = RegistrationWidget(self.session)
        self.roi_panel = RoiSelectionWidget(self.session)
        self.crop_panel = CropTracesWidget(self.session)
        self.analysis_panel = AnalysisWidget(self.session)

        # tab index -> the panels on it that need reloading, and what must exist
        # before the tab can be opened at all.
        self._panels = {
            0: (self.source_panel, self.preview_panel),
            1: (self.registration_panel,),
            2: (self.roi_panel,),
            3: (self.crop_panel,),
            4: (self.analysis_panel,),
        }
        for widget, title in ((split, "1 · Import"),
                              (self.registration_panel, "2 · Registration"),
                              (self.roi_panel, "3 · ROIs"),
                              (self.crop_panel, "4 · Crop"),
                              (self.analysis_panel, "5 · Analysis")):
            self.tabs.addTab(widget, title)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # A finished step is worth acting on immediately rather than at the next
        # tab switch: the file that just loaded goes into Recent, and a fresh
        # registration/crop re-gates what is reachable.
        self.source_panel.loaded.connect(self._on_loaded)
        self.registration_panel.registered.connect(lambda _s: self._refresh_chrome(force=True))
        self.crop_panel.applied.connect(lambda: self._refresh_chrome(force=True))

    def _build_menus(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        open_action = file_menu.addAction("&Open…")
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self.open_dialog)

        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)

        # Always enabled: an export that has nothing to write says so and what
        # would fix it (see `_require`), which a greyed-out menu entry cannot.
        export_menu = file_menu.addMenu("&Export")
        for title, handler in (("Traces (CSV)…", self.export_traces),
                               ("Stack (TIFF)…", self.export_stack),
                               ("Provenance (JSON)…", self.export_provenance)):
            export_menu.addAction(title).triggered.connect(handler)
        export_menu.addSeparator()
        all_action = export_menu.addAction("Export &all…")
        all_action.setToolTip("Write traces, stack and provenance into one folder")
        all_action.triggered.connect(self.export_all)

        file_menu.addSeparator()
        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut(QtGui.QKeySequence.Quit)
        quit_action.triggered.connect(self.close)

        help_menu = bar.addMenu("&Help")
        help_menu.addAction("&About Caliana").triggered.connect(self.about)

    def _build_statusbar(self):
        # Permanent, so a transient "Exported …" message coexists with it rather
        # than hiding the run summary.
        self._status_label = QtWidgets.QLabel("")
        self.statusBar().addPermanentWidget(self._status_label)

    # --------------------------------------------------------- panel refresh
    def _on_tab_changed(self, index: int):
        """Reload the tab being opened, but only if the session moved under it."""
        self._reload_tab(index)
        self._refresh_chrome(force=True)

    def _reload_tab(self, index: int):
        revision = self.session._revision
        for panel in self._panels.get(index, ()):
            if self._seen.get(panel) != revision:
                panel.reload()
                self._seen[panel] = revision

    def _mark_seen(self, index: int):
        """Record that a tab is up to date without rebuilding it."""
        for panel in self._panels.get(index, ()):
            self._seen[panel] = self.session._revision

    def _refresh_chrome(self, force: bool = False):
        """Re-gate the tabs and redraw the status line when the session changed.

        Called on a timer, so it must stay cheap: an integer compare, then the
        real work only when something actually moved.

        ``force`` marks a *step* having finished (a file loaded, registration run,
        a crop applied) and reloads the visible tab. Without it the change came
        from the panel on screen — placing an ROI, dragging a leaf box — and
        reloading would rip the graphics out from under the mouse, so that panel
        is only recorded as current.
        """
        revision = self.session._revision
        if not force and revision == self._revision:
            return
        self._revision = revision
        has_data = self.session.data is not None
        has_rois = has_data and bool(self.session.rois)
        # Analysis needs only a stack: its heatmap and kymograph pages are
        # dataset-wide, and the one page that does need ROIs gates itself.
        for index, enabled in ((1, has_data), (2, has_data),
                               (3, has_rois), (4, has_data)):
            self.tabs.setTabEnabled(index, enabled)
            self.tabs.setTabToolTip(index, _TAB_HINT[index] if enabled
                                    else _PREREQUISITE[index])
        # Qt moves the selection off a tab it disables while that tab is current;
        # this catches the rest (disabled while another window had focus).
        if not self.tabs.isTabEnabled(self.tabs.currentIndex()):
            self.tabs.setCurrentIndex(0)
        current = self.tabs.currentIndex()
        self._reload_tab(current) if force else self._mark_seen(current)
        self._status_label.setText(self._status_text())

    def _status_text(self) -> str:
        """source · [T,Y,X] · s/frame · µm/px · registration — the run at a glance."""
        session = self.session
        if session.data is None:
            return "No recording loaded — open one on the Import tab."
        parts = [Path(str(session.source.path)).name if session.source else "(in memory)"]
        parts.append("[T,Y,X] = " + "×".join(str(n) for n in session.data.shape))
        tl = session.timeline
        parts.append(f"{tl.frame_interval:g} s/frame" if (tl and tl.frame_interval)
                     else "frames (uncalibrated)")
        parts.append(f"{session.space.pixel_size:g} µm/px" if session.space.pixel_size
                     else "pixels (uncalibrated)")
        mode = session.registration.mode.value
        if session.track_motion:
            mode += " (ROIs follow tissue)"
        elif session.registered_data is not None:
            mode += " (stabilized stack)"
        parts.append(f"registration: {mode}")
        if session.rois:
            parts.append(f"{len(session.rois)} ROI(s)")
        if session.crop_window is not None:
            parts.append("crop [{}, {})".format(*session.crop_window))
        return "  ·  ".join(parts)

    # ------------------------------------------------------------- opening
    def open_dialog(self):
        """File ▸ Open…: pick a recording and load it with the current parameters."""
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", "",
            "Recordings (*.tif *.tiff *.nd2);;All files (*)",
        )
        if path:
            self.open_path(path)
        return path or None

    def open_path(self, path):
        """Load ``path`` through the Import tab, as if it had been chosen there.

        The one loading path in the app: the menu, the Recent list and the
        command-line argument all land here, so import parameters and error
        reporting behave identically whichever way a file arrives.
        """
        self.tabs.setCurrentIndex(0)
        self.source_panel.set_path(path)
        return self.source_panel.load()

    def _on_loaded(self, _session):
        self._remember_recent(self.source_panel.path_edit.text().strip())
        self.setWindowTitle(f"{_APP} — {Path(str(self.session.source.path)).name}")
        self._refresh_chrome(force=True)

    # -------------------------------------------------------- recent files
    def _settings(self):
        return QtCore.QSettings(_ORG, _APP)

    def _recent_files(self) -> list[str]:
        """Recently loaded paths, newest first, dropping any that have since moved."""
        stored = self._settings().value(_RECENT_KEY) or []
        if isinstance(stored, str):            # QSettings collapses 1-item lists
            stored = [stored]
        return [p for p in stored if Path(p).exists()]

    def _remember_recent(self, path: str):
        if not path:
            return
        recent = [p for p in self._recent_files() if p != path]
        self._settings().setValue(_RECENT_KEY, [path] + recent[:_MAX_RECENT - 1])

    def _rebuild_recent_menu(self):
        self.recent_menu.clear()
        recent = self._recent_files()
        if not recent:
            self.recent_menu.addAction("(nothing yet)").setEnabled(False)
            return
        for path in recent:
            action = self.recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.open_path(p))
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear list").triggered.connect(
            lambda: self._settings().remove(_RECENT_KEY)
        )

    # -------------------------------------------------------------- export
    def export_traces(self):
        """Write the per-ROI raw F / ΔF/F table to a CSV."""
        if not self._require(self.session.rois, "Place at least one ROI first."):
            return None
        if self.session.traces is None:
            self.session.extract_traces()
        path = self._save_path("Export traces", "traces.csv", "CSV (*.csv)")
        return self._run_export(path, self.session.export_traces, "traces")

    def export_stack(self):
        """Write the working (stabilized if registered) stack to a TIFF."""
        if not self._require(self.session.data is not None, "Load a recording first."):
            return None
        path = self._save_path("Export stack", "stack.tif", "TIFF (*.tif *.tiff)")
        return self._run_export(path, self.session.export_stack, "stack")

    def export_provenance(self):
        """Write the parameter record (SPEC §4) as a JSON sidecar."""
        if not self._require(self.session.data is not None, "Load a recording first."):
            return None
        path = self._save_path("Export provenance", "provenance.json", "JSON (*.json)")
        return self._run_export(path, self.session.export_provenance, "provenance")

    def export_all(self):
        """Write traces + stack + provenance into one chosen folder.

        The whole record of a run in one place, which is what reproducing it
        later actually needs (SPEC §4).
        """
        if not self._require(self.session.data is not None, "Load a recording first."):
            return None
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export all into…")
        if not folder:
            return None
        folder = Path(folder)
        if self.session.rois and self.session.traces is None:
            self.session.extract_traces()

        def write(_path=None):
            if self.session.rois:
                self.session.export_traces(folder / "traces.csv")
            self.session.export_stack(folder / "stack.tif")
            self.session.export_provenance(folder / "provenance.json")

        return self._run_export(folder, write, "everything")

    def _require(self, condition, message: str) -> bool:
        """Guard an export; explains what is missing rather than raising."""
        if condition:
            return True
        QtWidgets.QMessageBox.information(self, "Nothing to export", message)
        return False

    def _save_path(self, title: str, suggested: str, file_filter: str):
        """Ask where to write, defaulting beside the recording under a clear name."""
        source = self.session.source
        start = str(Path(str(source.path)).with_name(suggested)) if source else suggested
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(self, title, start, file_filter)
        return path or None

    def _run_export(self, path, write, what: str):
        """Write in the background and report the outcome. ``path`` None = cancelled."""
        from .widgets._task import run_in_background

        if not path:
            return None

        def done(_result):
            self.statusBar().showMessage(f"Exported {what} to {path}", 5000)

        def failed(exc):
            QtWidgets.QMessageBox.critical(
                self, "Export failed", _export_error_message(exc, path)
            )

        return run_in_background(self, lambda: write(path), on_done=done,
                                 on_error=failed, label=f"Exporting {what}…")

    # ---------------------------------------------------------------- help
    def about(self):
        from . import __version__

        QtWidgets.QMessageBox.about(
            self, f"About {_APP}",
            f"<h3>{_APP} {__version__}</h3>"
            "<p>Analysis of plant calcium imaging data — import and downsample a "
            "recording, stabilize leaf motion, place ROIs, extract ΔF/F traces and "
            "analyse response propagation.</p>"
            "<p>The same steps are scriptable from a notebook through "
            "<code>caliana.Session</code>.</p>",
        )

    def closeEvent(self, event):
        self._watch.stop()
        super().closeEvent(event)


# Why a tab is closed, said in terms of what to do about it.
_PREREQUISITE = {
    1: "Load a recording first (Import tab)",
    2: "Load a recording first (Import tab)",
    3: "Place at least one ROI first (ROIs tab)",
    4: "Load a recording first (Import tab)",
}

# What a tab is for, once it is reachable.
_TAB_HINT = {
    1: "Optional — stabilize leaf motion before placing ROIs",
    2: "Click the image to place ROIs; traces preview live",
    3: "Optional — restrict every trace to one time window",
    4: "Onset heatmaps and kymographs; ΔF/F, smoothing and propagation once ROIs exist",
}


def _export_error_message(exc: Exception, path) -> str:
    """An export failure phrased for someone who did not write the code."""
    if isinstance(exc, PermissionError):
        return f"No permission to write:\n{path}\n\nChoose another folder."
    if isinstance(exc, OSError):
        return f"Could not write {path}:\n{exc.strerror or exc}"
    if isinstance(exc, ValueError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def error_dialog(parent, exc_type, exc, detail: str) -> "QtWidgets.QMessageBox":
    """The crash report dialog, built but not shown.

    Leads with what failed in one line, says the app is still usable, and keeps
    the traceback behind "Show details" — there for a bug report, out of the way
    of someone who cannot act on it.
    """
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Critical)
    box.setWindowTitle("Caliana — unexpected error")
    box.setText(f"{exc_type.__name__}: {exc}")
    box.setInformativeText(
        "The step that failed was cancelled; the window is still usable. "
        "If this keeps happening, send the details below."
    )
    box.setDetailedText(detail)
    return box


def install_excepthook(parent=None) -> None:
    """Report uncaught exceptions in a dialog instead of dying at the terminal.

    An unhandled error in a Qt slot otherwise prints to a console the user may
    never see and leaves the window in an undefined state. Reporting it at least
    says what happened, and the app keeps running. The traceback still goes to
    stderr for anyone running from a terminal, and ``KeyboardInterrupt`` is left
    to the previous hook so Ctrl-C still quits.
    """
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(detail)
        if QtWidgets.QApplication.instance() is None:      # no GUI to report into
            return
        box = error_dialog(parent, exc_type, exc, detail)
        box.exec() if hasattr(box, "exec") else box.exec_()

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    """``caliana [PATH] [--version]`` — open the window, optionally on a file."""
    import argparse

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="caliana", description="Analysis of plant calcium imaging data.")
    parser.add_argument("path", nargs="?",
                        help="recording to open on start (.tif/.tiff/.nd2)")
    parser.add_argument("--version", action="version", version=f"caliana {__version__}")
    args = parser.parse_args(argv)

    app, _created = ensure_app()
    app.setApplicationName(_APP)
    app.setOrganizationName(_ORG)
    app.setApplicationVersion(__version__)

    window = MainWindow()
    install_excepthook(window)
    window.show()
    if args.path:
        window.open_path(args.path)
    return (app.exec if hasattr(app, "exec") else app.exec_)()


if __name__ == "__main__":
    raise SystemExit(main())
