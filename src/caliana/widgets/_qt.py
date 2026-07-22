"""Qt binding shim + notebook blocking helper. SPEC.md §2.2 (dual-mode).

`qtpy` lets the same widgets run under PyQt5/PyQt6/PySide2/PySide6. In a notebook
started with `%gui qt`, the existing QApplication is reused and the window blocks
the cell via a local event loop until closed; in a plain script a QApplication is
created and the loop is run directly. Qt is imported lazily so the core package
imports without a Qt binding installed.
"""
from __future__ import annotations


def get_qt():
    """Return ``(QtCore, QtGui, QtWidgets)`` from whatever binding is installed."""
    from qtpy import QtCore, QtGui, QtWidgets

    return QtCore, QtGui, QtWidgets


def ensure_app():
    """Return ``(app, created)``, creating a QApplication only if none exists."""
    _QtCore, _QtGui, QtWidgets = get_qt()
    app = QtWidgets.QApplication.instance()
    created = app is None
    if created:
        app = QtWidgets.QApplication([])
    return app, created


def save_figure_dialog(parent, render, *, title="Save figure", status=None):
    """Prompt for a path and write a figure there via ``render(path)``.

    ``render`` receives the chosen path and should call the relevant ``figures``
    function with ``save=path`` (and close the returned Figure). Vector
    (PDF/SVG) and raster (PNG/TIFF) formats are offered; the file extension picks
    the format, appended from the chosen filter when the user types none. Any
    render/IO error is reported on ``status`` (a QLabel) when given, else via a
    message box; the widget stays open either way. Returns the path written, or
    ``None`` if cancelled or failed.
    """
    import os

    _QtCore, _QtGui, QtWidgets = get_qt()
    filters = ("PNG image (*.png);;PDF document (*.pdf);;"
               "SVG image (*.svg);;TIFF image (*.tif)")
    path, selected = QtWidgets.QFileDialog.getSaveFileName(parent, title, "", filters)
    if not path:
        return None
    # Append the selected filter's extension when the user typed none.
    if not os.path.splitext(path)[1]:
        for ext in (".png", ".pdf", ".svg", ".tif"):
            if ext[1:] in selected.lower():
                path += ext
                break
    try:
        render(path)
    except Exception as exc:  # noqa: BLE001 — surface any render/IO failure to the UI
        message = f"Could not save figure: {exc}"
        if status is not None:
            status.setText(message)
        else:
            QtWidgets.QMessageBox.warning(parent, "Save failed", message)
        return None
    if status is not None:
        status.setText(f"Saved {path}")
    return path


def run_widget_blocking(factory):
    """Open a widget, block until it closes, and return its ``.result``. SPEC §2.2.

    ``factory`` builds and returns a QWidget that exposes a ``result`` attribute
    and (ideally) a ``closed`` signal. The widget should set ``self.result``
    before closing.
    """
    QtCore, _QtGui, _QtWidgets = get_qt()
    _app, _created = ensure_app()

    widget = factory()
    widget.show()

    # Block this call until the window closes (notebook %gui qt or script alike).
    loop = QtCore.QEventLoop()
    if hasattr(widget, "closed"):
        widget.closed.connect(loop.quit)
    widget.destroyed.connect(loop.quit)
    # exec_ (Qt5) vs exec (Qt6)
    (loop.exec if hasattr(loop, "exec") else loop.exec_)()

    return getattr(widget, "result", None)
