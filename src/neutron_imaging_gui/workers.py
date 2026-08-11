"""Bounded FIFO worker queue for long-running image reduction."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import traceback
import uuid

from qtpy import QtCore

from .processing import ProcessingCancelled, ReductionConfig, run_reduction


class WorkerSignals(QtCore.QObject):
    progress = QtCore.Signal(str, str, int, int, str)
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str, str)
    cancelled = QtCore.Signal(str)
    finished = QtCore.Signal(str)


class ReductionRunnable(QtCore.QRunnable):
    def __init__(self, job_id: str, config: ReductionConfig, cancel_event: threading.Event):
        super().__init__()
        self.job_id = job_id
        self.config = config
        self.cancel_event = cancel_event
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self):
        try:
            result = run_reduction(
                self.config,
                cancel_event=self.cancel_event,
                progress=lambda stage, current, total, message: self.signals.progress.emit(
                    self.job_id, stage, current, total, message
                ),
            )
        except ProcessingCancelled:
            self.signals.cancelled.emit(self.job_id)
        except Exception as exc:
            self.signals.failed.emit(self.job_id, str(exc), traceback.format_exc())
        else:
            self.signals.succeeded.emit(self.job_id, result)
        finally:
            self.signals.finished.emit(self.job_id)


@dataclass
class _QueuedJob:
    job_id: str
    config: ReductionConfig
    cancel_event: threading.Event


class ReductionQueue(QtCore.QObject):
    """Run reduction jobs sequentially to bound memory and disk pressure."""

    jobQueued = QtCore.Signal(str, int)
    jobStarted = QtCore.Signal(str)
    progress = QtCore.Signal(str, str, int, int, str)
    succeeded = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str, str)
    cancelled = QtCore.Signal(str)
    queueChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._pending: deque[_QueuedJob] = deque()
        self._active: _QueuedJob | None = None

    @property
    def active_job_id(self):
        return self._active.job_id if self._active is not None else None

    def submit(self, config: ReductionConfig) -> str:
        job = _QueuedJob(uuid.uuid4().hex, config, threading.Event())
        self._pending.append(job)
        self.jobQueued.emit(job.job_id, len(self._pending))
        self.queueChanged.emit(len(self._pending) + (1 if self._active else 0))
        self._start_next()
        return job.job_id

    def cancel(self, job_id: str | None = None) -> None:
        target = job_id or self.active_job_id
        if target is None:
            return
        if self._active is not None and self._active.job_id == target:
            self._active.cancel_event.set()
            return
        retained = deque()
        for job in self._pending:
            if job.job_id == target:
                job.cancel_event.set()
                self.cancelled.emit(job.job_id)
            else:
                retained.append(job)
        self._pending = retained
        self.queueChanged.emit(len(self._pending) + (1 if self._active else 0))

    def cancel_all(self) -> None:
        if self._active is not None:
            self._active.cancel_event.set()
        while self._pending:
            job = self._pending.popleft()
            job.cancel_event.set()
            self.cancelled.emit(job.job_id)
        self.queueChanged.emit(1 if self._active else 0)

    def shutdown(self) -> None:
        self.cancel_all()
        self._pool.waitForDone(3000)

    def _start_next(self) -> None:
        if self._active is not None or not self._pending:
            return
        self._active = self._pending.popleft()
        runnable = ReductionRunnable(
            self._active.job_id,
            self._active.config,
            self._active.cancel_event,
        )
        runnable.signals.progress.connect(self.progress.emit)
        runnable.signals.succeeded.connect(self.succeeded.emit)
        runnable.signals.failed.connect(self.failed.emit)
        runnable.signals.cancelled.connect(self.cancelled.emit)
        runnable.signals.finished.connect(self._job_finished)
        self.jobStarted.emit(self._active.job_id)
        self.queueChanged.emit(len(self._pending) + 1)
        self._pool.start(runnable)

    @QtCore.Slot(str)
    def _job_finished(self, job_id: str) -> None:
        if self._active is not None and self._active.job_id == job_id:
            self._active = None
        self.queueChanged.emit(len(self._pending))
        self._start_next()
