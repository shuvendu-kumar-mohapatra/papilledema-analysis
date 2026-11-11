"""Lightweight console and file logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
import sys


def get_logger(name: str = "papilledema", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        existing = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == resolved
        ]
        if not existing:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger

