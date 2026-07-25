"""delta-chat · src/ingest/__init__.py"""
# Import adapters to trigger @AdapterRegistry.register side-effects
from .pdf_native import NativePDFAdapter  # noqa: F401
from .pdf_scanned import ScannedPDFAdapter  # noqa: F401
from .dwg import DWGAdapter  # noqa: F401

from .base import (
    AdapterRegistry,
    DWGConversionError,
    FormatAdapter,
    IngestionError,
    OCRFailureError,
    UnsupportedFormatError,
    detect_format,
    is_scanned_pdf,
    resolve_format_type,
)

__all__ = [
    "AdapterRegistry", "DWGConversionError", "FormatAdapter",
    "IngestionError", "NativePDFAdapter", "OCRFailureError",
    "ScannedPDFAdapter", "DWGAdapter", "UnsupportedFormatError",
    "detect_format", "is_scanned_pdf", "resolve_format_type",
]
