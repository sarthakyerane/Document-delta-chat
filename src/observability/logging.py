"""
delta-chat · src/observability/logging.py
══════════════════════════════════════════════════════════════════════════════
Structured JSON logging with structlog.

Every log event carries:
  • request_id / trace_id — correlation across a full pipeline run
  • timestamp, level, logger, event
  • Any additional key-value pairs passed at call site

Usage:
    from src.observability.logging import get_logger
    log = get_logger(__name__)
    log.info("ingest.complete", pid=pid, page_count=5, duration_ms=234.1)
    log.error("ocr.failed", pid=pid, error=str(e), page_index=2)

Format (LOG_FORMAT=json):
    {"event":"ingest.complete","pid":"doc_a","page_count":5,"duration_ms":234.1,
     "request_id":"abc123","timestamp":"2024-01-15T10:30:00Z","level":"info",
     "logger":"src.ingest.pdf_native"}

Format (LOG_FORMAT=console):
    2024-01-15 10:30:00 [info     ] ingest.complete    pid=doc_a page_count=5
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

from src.config import get_settings

settings = get_settings()

# ── Context variable for request_id propagation across async call chains ──────
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: str) -> None:
    """Bind a request_id to the current async context."""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    return _request_id_var.get()


# ── Processor: inject request_id from context var into every log event ─────────
def _inject_request_id(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    rid = _request_id_var.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


# ── Configure structlog ───────────────────────────────────────────────────────
def configure_logging() -> None:
    """
    Call once at application startup (in main.py lifespan or CLI entry).
    Idempotent — safe to call multiple times.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Standard library logging (for libraries that use it, e.g. uvicorn)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _inject_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog logger bound to the given module name."""
    return structlog.get_logger(name)
