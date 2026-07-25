"""
delta-chat · src/ingest/base.py
══════════════════════════════════════════════════════════════════════════════
FormatAdapter interface — the only contract every ingestion adapter must satisfy.

Design:
  • One abstract method: `ingest(pid, source_path, tracer) -> Document`
  • A new format (e.g. "xlsx", "ifc") plugs in by subclassing FormatAdapter,
    implementing `ingest`, and registering with `AdapterRegistry`.
  • Downstream code never imports a concrete adapter — it calls
    `AdapterRegistry.get(format_type)` and receives a `FormatAdapter`.

The `detect_format` helper provides PID-resolution + format detection so
callers only need to hand off a file path.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.canonical.model import Document
    from src.observability.tracing import RequestTracer


class FormatAdapter(ABC):
    """
    Abstract base for all ingestion adapters.

    Implementers MUST:
      1. Produce exactly a `Document` — no format-specific objects escape.
      2. Set `document.format` to the correct `FormatType` literal.
      3. Record timing in the tracer (use `tracer.span(...)` context manager).
      4. Capture failures in the tracer — never swallow exceptions silently.
    """

    @property
    @abstractmethod
    def supported_format(self) -> str:
        """Return the FormatType literal this adapter handles."""

    @abstractmethod
    def ingest(
        self,
        pid: str,
        source_path: str,
        tracer: "RequestTracer",
        revision_label: Optional[str] = None,
    ) -> "Document":
        """
        Parse `source_path` and return a canonical Document.

        Args:
            pid:            Caller-assigned stable revision identifier.
            source_path:    Absolute or relative path to the raw file.
            tracer:         Request-scoped tracer — use `tracer.span(name)` to
                            wrap sub-stages (page extraction, OCR, etc.).
            revision_label: Optional human label, e.g. 'Rev B'.

        Returns:
            A fully-populated `Document` in the canonical representation.

        Raises:
            IngestionError: For any unrecoverable parsing / OCR failure.
                            The error is automatically recorded in the tracer
                            by the caller before re-raising.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Raised when a document cannot be ingested. Always has a `pid` attribute."""

    def __init__(self, pid: str, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.pid = pid
        self.cause = cause

    def __str__(self) -> str:
        base = f"[PID={self.pid}] {super().__str__()}"
        if self.cause:
            base += f" — caused by: {type(self.cause).__name__}: {self.cause}"
        return base


class UnsupportedFormatError(IngestionError):
    """Raised when no adapter is registered for the detected format."""


class OCRFailureError(IngestionError):
    """Raised when OCR fails on a scanned page and no fallback is available."""


class DWGConversionError(IngestionError):
    """Raised when DWG→DXF conversion fails."""


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class AdapterRegistry:
    """
    Central registry of format adapters.  Supports the open/closed principle:
    new formats register without touching existing code.
    """

    _registry: dict[str, type[FormatAdapter]] = {}

    @classmethod
    def register(cls, adapter_cls: type[FormatAdapter]) -> type[FormatAdapter]:
        """Decorator / explicit call — register an adapter class."""
        cls._registry[adapter_cls().supported_format] = adapter_cls
        return adapter_cls

    @classmethod
    def get(cls, format_type: str) -> FormatAdapter:
        """Return an adapter instance for `format_type`, or raise."""
        if format_type not in cls._registry:
            raise UnsupportedFormatError(
                pid="unknown",
                message=f"No adapter registered for format '{format_type}'. "
                        f"Registered: {list(cls._registry)}",
            )
        return cls._registry[format_type]()

    @classmethod
    def supported_formats(cls) -> list[str]:
        return list(cls._registry.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

_MAGIC_BYTES: dict[bytes, str] = {
    b"%PDF": "pdf",
    b"AC10": "dwg",   # AutoCAD DWG magic
    b"AC10": "dwg",
    b"AC12": "dwg",
    b"AC14": "dwg",
    b"AC15": "dwg",
    b"AC18": "dwg",
    b"AC21": "dwg",
    b"AC24": "dwg",
    b"AC27": "dwg",
}

_EXT_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".dwg": "dwg",
    ".dxf": "dwg",  # DXF is handled by the DWG adapter
}


def detect_format(path: str) -> str:
    """
    Detect file format by reading magic bytes first, then falling back to
    file extension.  Returns one of: 'pdf', 'dwg'.
    """
    # Magic bytes check (first 4 bytes)
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        for magic, fmt in _MAGIC_BYTES.items():
            if header.startswith(magic):
                return fmt
    except OSError:
        pass

    # Extension fallback
    _, ext = os.path.splitext(path.lower())
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]

    raise UnsupportedFormatError(
        pid=path,
        message=f"Cannot detect format for '{path}'. "
                f"Supported extensions: {list(_EXT_MAP)}",
    )


def is_scanned_pdf(path: str) -> bool:
    """
    Heuristic: if ≥80% of pages have no extractable text layer, treat as scanned.
    Uses pdfplumber for a quick text-layer probe.
    """
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return True
            total = min(len(pdf.pages), 5)  # sample first 5 pages
            empty_count = 0
            for page in pdf.pages[:total]:
                text = page.extract_text() or ""
                if len(text.strip()) < 20:
                    empty_count += 1
            return empty_count / total >= 0.8
    except Exception:
        return False


def resolve_format_type(path: str) -> str:
    """
    Resolve path to one of the canonical FormatType literals:
    'pdf_native' | 'pdf_scanned' | 'dwg'
    """
    fmt = detect_format(path)
    if fmt == "pdf":
        return "pdf_scanned" if is_scanned_pdf(path) else "pdf_native"
    return "dwg"
