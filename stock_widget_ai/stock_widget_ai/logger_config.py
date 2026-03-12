"""
로깅 초기화 — 파일 + 콘솔 동시 출력
"""
import logging
import os
from logging.handlers import RotatingFileHandler

_initialized = False


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    global _initialized
    if _initialized:
        return logging.getLogger("stock_widget_ai")

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(numeric_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(numeric_level)

    root = logging.getLogger("stock_widget_ai")
    root.setLevel(numeric_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root.propagate = False

    _initialized = True
    root.info(f"Logging initialized — level={log_level}, file={log_path}")
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"stock_widget_ai.{name}")
