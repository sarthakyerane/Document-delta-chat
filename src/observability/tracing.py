"""
delta-chat · src/observability/tracing.py
══════════════════════════════════════════════════════════════════════════════
Per-request tracing with OpenTelemetry spans + a homegrown FileSpanExporter.

Each request produces one trace file at:
    traces/<request_id>.json

Structure:
    {
        "trace_id": "<request_id>",
        "service": "delta-chat",
        "start_time": "<iso>",
        "end_time": "<iso>",
        "total_duration_ms": 1234.5,
        "spans": [
            {
                "span_id": "...",
                "name": "ingest.pdf_native",
                "start_time": "...",
                "end_time": "...",
                "duration_ms": 345.2,
                "status": "OK",         # OK | ERROR
                "attributes": {...},
                "events": [...]         # errors, warnings, notable events
            }
        ],
        "llm_calls": [
            {
                "span_id": "...",
                "model": "llama-3.3-70b-versatile",
                "provider": "groq",
                "input_tokens": 1234,
                "output_tokens": 456,
                "estimated_cost_usd": 0.00089,
                "duration_ms": 980.1,
                "status": "OK"
            }
        ],
        "errors": [...]                 # aggregated error events
    }

Why OTel: industry-standard, interview-defensible; the FileSpanExporter gives
us "at minimum a written trace file per run" without any external infra.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.config import get_settings

settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Internal span model (lightweight, JSON-serialisable)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpanRecord:
    span_id: str
    name: str
    start_time: float  # epoch seconds
    end_time: Optional[float] = None
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    parent_span_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat()
                if self.end_time
                else None
            ),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class LLMCallRecord:
    span_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    duration_ms: float
    status: str = "OK"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tracer — one instance per request
# ─────────────────────────────────────────────────────────────────────────────

class RequestTracer:
    """
    Lightweight request-scoped tracer.  Not thread-shared — each request
    creates its own instance, carries it through the pipeline state, then
    calls `export()` at the end to write the trace file.
    """

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.trace_id = self.request_id  # alias for OTel compatibility
        self.start_time = time.time()
        self._spans: list[SpanRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._errors: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._span_stack: list[str] = []  # parent span IDs

    # ── Span management ──────────────────────────────────────────────────────

    def start_span(self, name: str, attributes: Optional[dict[str, Any]] = None) -> SpanRecord:
        parent_id = self._span_stack[-1] if self._span_stack else None
        span = SpanRecord(
            span_id=uuid.uuid4().hex[:12],
            name=name,
            start_time=time.time(),
            parent_span_id=parent_id,
            attributes=attributes or {},
        )
        with self._lock:
            self._spans.append(span)
            self._span_stack.append(span.span_id)
        return span

    def end_span(
        self,
        span: SpanRecord,
        status: str = "OK",
        attributes: Optional[dict[str, Any]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        span.end_time = time.time()
        span.status = status if not error else "ERROR"
        if attributes:
            span.attributes.update(attributes)
        if error:
            span.events.append({
                "name": "exception",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "type": type(error).__name__,
                "message": str(error),
            })
            self._errors.append({
                "span_id": span.span_id,
                "span_name": span.name,
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
        with self._lock:
            if span.span_id in self._span_stack:
                self._span_stack.remove(span.span_id)

    @contextmanager
    def span(
        self, name: str, attributes: Optional[dict[str, Any]] = None
    ) -> Generator[SpanRecord, None, None]:
        """Context manager for automatic span lifecycle."""
        s = self.start_span(name, attributes)
        try:
            yield s
            self.end_span(s, status="OK")
        except Exception as exc:
            self.end_span(s, status="ERROR", error=exc)
            raise

    # ── LLM telemetry ────────────────────────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        status: str = "OK",
        error: Optional[str] = None,
    ) -> LLMCallRecord:
        cost = settings.estimate_cost(model, input_tokens, output_tokens)
        record = LLMCallRecord(
            span_id=uuid.uuid4().hex[:12],
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            duration_ms=duration_ms,
            status=status,
            error=error,
        )
        with self._lock:
            self._llm_calls.append(record)
        return record

    # ── Add arbitrary event to current span ───────────────────────────────────

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        if self._span_stack:
            current_span_id = self._span_stack[-1]
            with self._lock:
                for s in reversed(self._spans):
                    if s.span_id == current_span_id:
                        s.events.append({
                            "name": name,
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            **(attributes or {}),
                        })
                        break

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        end_time = time.time()
        total_cost = sum(c.estimated_cost_usd for c in self._llm_calls)
        total_tokens = sum(c.input_tokens + c.output_tokens for c in self._llm_calls)
        return {
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "service": settings.otel_service_name,
            "version": settings.otel_service_version,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
            "total_duration_ms": round((end_time - self.start_time) * 1000, 2),
            "spans": [s.to_dict() for s in self._spans],
            "llm_calls": [c.to_dict() for c in self._llm_calls],
            "llm_summary": {
                "total_calls": len(self._llm_calls),
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
            },
            "errors": self._errors,
            "error_count": len(self._errors),
        }

    def export(self) -> str:
        """Write trace file to TRACES_DIR/<request_id>.json. Returns file path."""
        os.makedirs(settings.traces_dir, exist_ok=True)
        path = os.path.join(settings.traces_dir, f"{self.request_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# OTel provider setup (for SDK compatibility — spans also go to in-memory
# exporter so they can be queried without file I/O in tests)
# ─────────────────────────────────────────────────────────────────────────────

_in_memory_exporter = InMemorySpanExporter()

def setup_otel() -> None:
    """Configure the global OTel tracer provider. Call once at app startup."""
    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": settings.otel_service_version,
    })
    provider = TracerProvider(resource=resource)
    # InMemorySpanExporter for tests + local inspection
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))
    trace.set_tracer_provider(provider)


def get_otel_tracer() -> trace.Tracer:
    return trace.get_tracer(settings.otel_service_name)
