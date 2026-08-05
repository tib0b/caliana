"""Run one long Session call off the UI thread. SPEC.md §1 (Phase 2 UX).

Loading a multi-GB recording, registering it, or computing a per-pixel onset
heatmap takes seconds to minutes with no progress reporting and no way to
interrupt it — run on the UI thread they simply freeze the window, which reads
as a crash. ``run_in_background`` moves the call to a ``QThread`` and puts an
indeterminate, un-cancellable progress dialog over the window meanwhile: honest
about what the compute layer can actually report, and enough to keep the app
responsive and obviously alive.

Notebook use never needs this — a blocking wrapper is already one cell, one
window — so nothing here is wired into ``Session``'s ``[notebook]`` methods.
"""
from __future__ import annotations

from ._qt import get_qt

QtCore, QtGui, QtWidgets = get_qt()


class _Worker(QtCore.QObject):
    """Calls ``fn`` in whatever thread it is moved to and reports the outcome."""

    done = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — reported to the UI, never printed
            self.failed.emit(exc)
        else:
            self.done.emit(result)


class Task(QtCore.QObject):
    """A background call in flight, plus the progress dialog covering it.

    Keep a reference for as long as it runs (``run_in_background`` parents it to
    the calling widget, which is enough). ``result`` / ``error`` hold the outcome
    once ``finished`` has fired.
    """

    finished = QtCore.Signal()

    def __init__(self, fn, on_done=None, on_error=None, *, parent=None,
                 label="Working…"):
        super().__init__(parent)
        self._on_done, self._on_error = on_done, on_error
        self.result = None
        self.error: Exception | None = None
        self.done = False

        self.dialog = QtWidgets.QProgressDialog(label, "", 0, 0, parent)
        self.dialog.setWindowTitle("Caliana")
        # The compute functions report no progress and cannot be interrupted, so
        # an indeterminate bar with no Cancel button is the honest ceiling here.
        self.dialog.setCancelButton(None)
        self.dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)

        self._thread = QtCore.QThread()
        self._worker = _Worker(fn)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        # Queued across threads: both slots run on the UI thread, so the
        # callbacks may touch widgets freely.
        self._worker.done.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._thread.start()
        self.dialog.show()

    # ------------------------------------------------------------- outcome
    def _on_worker_done(self, result):
        self.result = result
        self._shutdown()
        if self._on_done is not None:
            self._on_done(result)

    def _on_worker_failed(self, exc):
        self.error = exc
        self._shutdown()
        if self._on_error is not None:
            self._on_error(exc)

    def _shutdown(self):
        """Stop the thread and drop the dialog, before any callback runs.

        Callbacks routinely open dialogs of their own (an error box, a file
        picker); the progress window must be gone by then, and ``done`` true so a
        ``wait()`` nested in one of them returns immediately.
        """
        self.done = True
        self._thread.quit()
        self._thread.wait()
        self.dialog.close()
        self.finished.emit()

    def wait(self, timeout_ms: int = 120_000) -> bool:
        """Spin the event loop until the call finishes. Returns ``True`` if it did.

        For scripts and tests that need the result before continuing. The UI
        stays live while waiting (this is a nested event loop, not a block), so
        the progress dialog still paints.
        """
        if self.done:
            return True
        loop = QtCore.QEventLoop()
        self.finished.connect(loop.quit)
        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        (loop.exec if hasattr(loop, "exec") else loop.exec_)()
        return self.done


def run_in_background(parent, fn, on_done=None, on_error=None, *,
                      label="Working…") -> Task:
    """Call ``fn()`` on a worker thread behind an indeterminate progress dialog.

    ``on_done(result)`` / ``on_error(exception)`` run on the UI thread once it
    finishes; exactly one of them is called. Returns the live ``Task`` (parented
    to ``parent``, so it outlives this call).
    """
    return Task(fn, on_done, on_error, parent=parent, label=label)
