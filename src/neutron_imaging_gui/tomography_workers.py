"""Single-lane background execution for tomography work."""

from __future__ import annotations

import threading
import traceback

from qtpy import QtCore

from .processing import ProcessingCancelled


class TomographyWorkerSignals(QtCore.QObject):
    progress = QtCore.Signal(str, int, int, str)
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str, str)
    cancelled = QtCore.Signal(str)
    finished = QtCore.Signal(str)


class TomographyRunnable(QtCore.QRunnable):
    def __init__(self, kind, function, cancel_event):
        super().__init__()
        self.kind = str(kind)
        self.function = function
        self.cancel_event = cancel_event
        self.signals = TomographyWorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            result = self.function(
                cancel_event=self.cancel_event,
                progress=lambda stage, current, total, message: self.signals.progress.emit(
                    stage, int(current), int(total), str(message)
                ),
            )
            if self.cancel_event.is_set():
                raise ProcessingCancelled("Tomography operation cancelled.")
        except ProcessingCancelled:
            self.signals.cancelled.emit(self.kind)
        except Exception as exc:
            self.signals.failed.emit(self.kind, str(exc), traceback.format_exc())
        else:
            self.signals.succeeded.emit(self.kind, result)
        finally:
            self.signals.finished.emit(self.kind)


class TomographyJobRunner(QtCore.QObject):
    busyChanged = QtCore.Signal(bool)
    progress = QtCore.Signal(str, int, int, str)
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str, str)
    cancelled = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._cancel_event = None
        self._active_kind = None

    @property
    def busy(self):
        return self._active_kind is not None

    def start(self, kind, function):
        if self.busy:
            raise RuntimeError("A tomography operation is already running.")
        self._active_kind = str(kind)
        self._cancel_event = threading.Event()
        runnable = TomographyRunnable(self._active_kind, function, self._cancel_event)
        runnable.signals.progress.connect(self.progress.emit)
        runnable.signals.succeeded.connect(self.succeeded.emit)
        runnable.signals.failed.connect(self.failed.emit)
        runnable.signals.cancelled.connect(self.cancelled.emit)
        runnable.signals.finished.connect(self._finished)
        self.busyChanged.emit(True)
        self._pool.start(runnable)

    def cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()

    def shutdown(self):
        self.cancel()
        self._pool.waitForDone(3000)

    @QtCore.Slot(str)
    def _finished(self, kind):
        if kind == self._active_kind:
            self._active_kind = None
            self._cancel_event = None
        self.busyChanged.emit(False)
