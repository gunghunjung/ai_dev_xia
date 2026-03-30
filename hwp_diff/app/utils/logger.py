import logging
import logging.handlers
import os
import sys
from pathlib import Path

_logger_initialized = False
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def setup_logger(
    name: str = "hwp_diff",
    level: int = logging.DEBUG,
    log_dir: Path = LOG_DIR,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure root logger with rotating file handler and console handler."""
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger(name)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hwp_diff.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    _logger_initialized = True
    logger.info("Logger initialized. Log file: %s", log_file)
    return logger


def get_logger(module_name: str = "") -> logging.Logger:
    """Get a child logger for a specific module."""
    base = "hwp_diff"
    if module_name:
        return logging.getLogger(f"{base}.{module_name}")
    return logging.getLogger(base)
