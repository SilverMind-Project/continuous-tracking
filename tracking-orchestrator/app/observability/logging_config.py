"""Structlog configuration for the tracking orchestrator.

Call ``configure_logging()`` once at process startup (top of main.py) before
any other log statements execute.

Behaviour:
  - Non-TTY (Docker / CI): JSON lines with ExceptionRenderer (plain text
    traceback, no locals, no ANSI).  Suitable for log aggregators.
  - TTY (local dev): ConsoleRenderer with show_locals=False and max_frames=15.

Both modes run the ``_sanitize_log_values`` processor which replaces bulky
objects in structured log fields with compact summaries:
  - numpy arrays      → "ndarray(shape=..., dtype=...)"
  - bytes/bytearray   → "bytes(N)"
  - protobuf messages → "proto(<classname>)"
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

try:
    import numpy as _np

    _NUMPY_ARRAY = _np.ndarray
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    _NUMPY_ARRAY = type(None)  # type: ignore[assignment,misc]

__all__ = ["configure_logging"]


def _sanitize_log_values(
    _logger: Any,
    _method: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Replace bulky structured-log values with compact summaries.

    Only top-level event dict entries are inspected; nested containers are
    left unchanged (they should not carry raw tensors anyway).
    """
    for key, value in list(event_dict.items()):
        if key in ("event", "_record"):
            continue
        if isinstance(value, _NUMPY_ARRAY):
            event_dict[key] = f"ndarray(shape={value.shape}, dtype={value.dtype})"  # type: ignore[union-attr]
        elif isinstance(value, (bytes, bytearray)) and len(value) > 64:
            event_dict[key] = f"bytes({len(value)})"
        elif hasattr(value, "DESCRIPTOR"):
            event_dict[key] = f"proto({type(value).__name__})"
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for the tracking orchestrator process.

    Must be called before any logger is used.  Calling it more than once
    is safe (structlog.configure is idempotent within a process).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        _sanitize_log_values,
    ]

    if sys.stderr.isatty():
        processors: list[Any] = shared + [
            structlog.dev.ConsoleRenderer(
                exception_formatter=structlog.dev.RichTracebackFormatter(
                    show_locals=False,
                    max_frames=15,
                    word_wrap=False,
                ),
            ),
        ]
    else:
        processors = shared + [
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ]

    # stdlib.LoggerFactory creates logging.Logger objects that carry `.name`,
    # which is required by stdlib.add_logger_name above.
    # basicConfig routes those loggers' output to stdout as plain %(message)s
    # so structlog's rendered string passes through unchanged.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    # Squelch debug logs from noisy third-party libraries.
    for noisy in ("aiobotocore", "botocore", "urllib3", "aiohttp", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
