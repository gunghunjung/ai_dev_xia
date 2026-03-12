"""
Worker Threads — GUI 블로킹 방지용 QThread 래퍼
"""
from __future__ import annotations
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Callable, Any


class WorkerThread(QThread):
    """범용 백그라운드 작업 스레드"""
    progress = pyqtSignal(int)          # 0-100
    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(object)       # result
    error    = pyqtSignal(str)

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn   = fn
        self._args = args
        self._kwargs = kwargs
        self._stop = False

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            if not self._stop:
                self.finished.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def request_stop(self) -> None:
        self._stop = True

    def emit_log(self, msg: str) -> None:
        self.log_msg.emit(msg)

    def emit_progress(self, pct: int) -> None:
        self.progress.emit(pct)
