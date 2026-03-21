# utils/logger.py — 로깅 설정
import logging
import os
from datetime import datetime


def setup_logger(name: str = "quant", level: str = "INFO",
                 log_dir: str = "outputs") -> logging.Logger:
    """파일 + 콘솔 로거 설정"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 핸들러
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 파일 핸들러
    log_path = os.path.join(log_dir, f"quant_{datetime.now():%Y%m%d}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
