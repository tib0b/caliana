"""Offscreen smoke test for the standalone app shell. SPEC.md §1 (Phase 2).

Walks a whole run through `MainWindow` the way a user would — open a recording,
register, place ROIs, crop, analyse, export — checking the two things the shell
owns and the widgets do not: tabs gated on their prerequisite, and changes made
on one tab showing up on the next.

Skipped if the GUI stack (qtpy + a Qt binding + pyqtgraph) is not installed.
Run headless with: QT_QPA_PLATFORM=offscreen python tests/test_app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "synthetic_calcium_imaging.tif"

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtCore, QtWidgets

    from caliana.app import MainWindow, install_excepthook, main
    from caliana.widgets._qt import ensure_app
    HAVE_GUI = True
except Exception:  # pragma: no cover - depends on optional deps
    HAVE_GUI = False

# Modal dialogs would block a test run forever; record them instead, so a test
# can still assert the user was told what happened. Settings go to a scratch
# directory so a test run never touches the real recent-files list.
DIALOGS: list[tuple[str, str]] = []
if HAVE_GUI:
    def _record(_parent, title, text, *args, **kwargs):
        DIALOGS.append((title, text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    for _kind in ("critical", "warning", "information", "about"):
        setattr(QtWidgets.QMessageBox, _kind, staticmethod(_record))
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope,
                             tempfile.mkdtemp(prefix="caliana-settings-"))


def _answer_file_dialogs(path):
    """Make the file/folder pickers answer ``path`` without a user."""
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (str(path), ""))
    QtWidgets.QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (str(path), ""))
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(
        lambda *a, **k: str(path))


def test_app_walks_the_workflow():
    if not HAVE_GUI:
        print("GUI stack not available; skipping app test")
        return
    ensure_app()
    window = MainWindow()

    # Nothing is loaded: only the Import tab is reachable, and the status line
    # says what to do about it.
    assert [window.tabs.isTabEnabled(i) for i in range(5)] == \
        [True, False, False, False, False]
    assert "No recording loaded" in window._status_label.text()

    # 1 · Import — the menu, Recent and the CLI all load through open_path.
    task = window.open_path(DATA)
    assert task.wait() and task.error is None
    assert window.session.data.shape == (100, 64, 64)
    assert window.tabs.currentIndex() == 0
    # A stack exists: registration, ROI placement and analysis open up (the
    # heatmap and kymograph pages are dataset-wide); only cropping the traces
    # still waits for an ROI.
    assert [window.tabs.isTabEnabled(i) for i in range(5)] == \
        [True, True, True, False, True]
    assert DATA.name in window._status_label.text()
    assert str(DATA) in window._recent_files()

    # 2 · Registration — whole frame, correcting by moving the ROIs.
    window.tabs.setCurrentIndex(1)
    window.registration_panel.mode_box.setCurrentText("Whole frame")
    window.registration_panel.transform_box.setCurrentText("rigid_body")
    window.registration_panel.apply_box.setCurrentIndex(1)     # no warping
    task = window.registration_panel.run()
    assert task.wait() and task.error is None
    assert window.session.track_motion
    assert "registration: whole-frame" in window._status_label.text()

    # 3 · ROIs — placed here, they open the crop tab and the analysis tab's trace
    # page. The track-motion toggle is live because registration (tab 2) produced
    # transforms: the change propagated across panels.
    window.tabs.setCurrentIndex(2)
    assert window.roi_panel.track_box.isEnabled()
    window.roi_panel.add_roi_at(32, 32)
    window.roi_panel.add_roi_at(20, 44)
    window._refresh_chrome()
    assert [window.tabs.isTabEnabled(i) for i in range(5)] == [True] * 5

    # 4 · Crop — the ROIs placed next door are already previewed here.
    window.tabs.setCurrentIndex(3)
    assert window.crop_panel._preview.raw.shape == (2, 100)
    window.crop_panel.set_interval(5, 60)
    window.crop_panel.apply_crop()
    assert window.session.crop_window == (5, 60)
    assert "crop [5, 60)" in window._status_label.text()

    # 5 · Analysis — sees the cropped traces, in original frame coordinates.
    window.tabs.setCurrentIndex(4)
    assert window.session.traces.raw.shape == (2, 55)
    window.analysis_panel.n_box.setValue(5)
    window.analysis_panel.compute_dff()
    assert window.session.traces.dff is not None
    assert window.analysis_panel._curves[0].getData()[0][0] == 5

    # File ▸ Export — the three exports, then all of them into one folder.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for export, name in ((window.export_traces, "traces.csv"),
                             (window.export_stack, "stack.tif"),
                             (window.export_provenance, "provenance.json")):
            _answer_file_dialogs(tmp / name)
            task = export()
            assert task is not None and task.wait() and task.error is None, name
            assert (tmp / name).exists() and (tmp / name).stat().st_size > 0, name

        folder = tmp / "all"
        folder.mkdir()
        _answer_file_dialogs(folder)
        task = window.export_all()
        assert task.wait() and task.error is None
        assert sorted(p.name for p in folder.iterdir()) == \
            ["provenance.json", "stack.tif", "traces.csv"]

    window.close()
    print("app workflow OK")


def test_app_gates_on_prerequisites():
    """A tab whose prerequisite disappears closes again, and stops being shown."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping app test")
        return
    ensure_app()
    window = MainWindow()
    task = window.open_path(DATA)
    assert task.wait()

    window.tabs.setCurrentIndex(2)
    window.roi_panel.add_roi_at(32, 32)
    window._refresh_chrome()
    assert window.tabs.isTabEnabled(3)

    window.tabs.setCurrentIndex(3)
    window.roi_panel.delete_last_roi()          # the last ROI goes
    window._refresh_chrome()
    assert not window.tabs.isTabEnabled(3)
    # Never left on a dead tab, and the closed one says what would open it.
    assert window.tabs.isTabEnabled(window.tabs.currentIndex())
    assert "ROI" in window.tabs.tabToolTip(3)

    # Analysis needs only the stack, so it stays open; the ROI prerequisite moves
    # inside it, onto the one page that has one.
    assert window.tabs.isTabEnabled(4)
    window.tabs.setCurrentIndex(4)
    assert not window.analysis_panel.tabs.isTabEnabled(0)      # Trace analysis
    assert window.analysis_panel.tabs.isTabEnabled(2)          # Kymograph

    window.close()
    print("app gating OK")


def test_app_reload_is_lazy_and_idempotent():
    """Panels are re-read on activation only, and only when the session moved."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping app test")
        return
    ensure_app()
    window = MainWindow()
    assert window.open_path(DATA).wait()

    # Placing ROIs must not make the shell rebuild the panel under the mouse:
    # the graphics would be ripped out mid-interaction.
    window.tabs.setCurrentIndex(2)
    reloads = []
    original = window.roi_panel.reload
    window.roi_panel.reload = lambda: (reloads.append(1), original())[1]
    window.roi_panel.add_roi_at(32, 32)
    window._refresh_chrome()
    assert reloads == []
    assert len(window.roi_panel._roi_records) == 1

    # Leaving and coming back does not rebuild it either — the panel is already
    # current, because it made the change itself.
    window.tabs.setCurrentIndex(3)
    window.tabs.setCurrentIndex(2)
    assert reloads == []

    # A change made elsewhere does reload it, exactly once, with no duplicates.
    window.tabs.setCurrentIndex(0)
    window.session.add_roi(center=(20, 20), size=3)
    window.tabs.setCurrentIndex(2)
    assert reloads == [1]
    assert len(window.roi_panel._roi_records) == 2
    window.tabs.setCurrentIndex(2)
    assert reloads == [1]

    window.roi_panel.reload = original
    window.close()
    print("app lazy reload OK")


def test_app_reports_uncaught_errors():
    """The excepthook turns a crash into a report, keeping the window alive.

    The dialog is built here rather than shown: modal, it would block this test
    exactly as it blocks the app until acknowledged, which is the point of it.
    """
    if not HAVE_GUI:
        print("GUI stack not available; skipping app test")
        return
    import sys

    from caliana.app import error_dialog

    ensure_app()
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        box = error_dialog(None, type(exc), exc, "Traceback (most recent call last): …")

    assert "RuntimeError: boom" in box.text()
    assert "still usable" in box.informativeText()
    assert "Traceback" in box.detailedText()          # kept, but out of the way

    # Installing it replaces the hook, and Ctrl-C still reaches the previous one
    # (a reported dialog is no use when the user is trying to quit).
    previous, seen = sys.excepthook, []
    try:
        sys.excepthook = lambda exc_type, exc, tb: seen.append(exc_type)
        install_excepthook()
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        assert seen == [KeyboardInterrupt]
    finally:
        sys.excepthook = previous
    print("app excepthook OK")


def test_app_cli_version():
    """`caliana --version` answers without opening a window."""
    if not HAVE_GUI:
        print("GUI stack not available; skipping app test")
        return
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("--version should exit")
    print("app cli version OK")


if __name__ == "__main__":
    test_app_walks_the_workflow()
    test_app_gates_on_prerequisites()
    test_app_reload_is_lazy_and_idempotent()
    test_app_reports_uncaught_errors()
    test_app_cli_version()
