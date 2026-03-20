"""
Infrastructure Layer — Logging Setup

Responsibility: Configure and provide the application-wide logger.
Permitted imports: Python stdlib only. No internal project imports.

Provides a single `get_logger()` function that returns a configured
logger instance with both console and file handlers.
"""

import logging
import sys
from pathlib import Path


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "pdf_chatbot.log"
_CONFIGURED = False


def get_logger(name: str = "pdf_chatbot") -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically the module's __name__.

    Returns:
        A configured logging.Logger instance with console and file handlers.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)

    if not _CONFIGURED and not logger.handlers:
        _configure_root_logger()
        _CONFIGURED = True

    return logger


def _configure_root_logger() -> None:
    """Set up the root pdf_chatbot logger with console and file handlers."""
    import os

    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger("pdf_chatbot")
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always present
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — best-effort, skip if directory not writable
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except OSError:
        root_logger.warning(
            "Could not create log file at %s — logging to console only",
            _LOG_FILE,
        )
