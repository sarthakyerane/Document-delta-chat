"""
delta-chat · src/canonical/model.py
══════════════════════════════════════════════════════════════════════════════
THE CANONICAL REPRESENTATION — the seam that decouples every format adapter
from the delta engine, chat layer, and eval harness.

Design contract:
  • Every format adapter (pdf_native, pdf_scanned, dwg) MUST produce a
    `Document` and ONLY a `Document` — nothing downstream may special-case
    a source format.
  • The `element_id` is a *content-hash* (NOT pid-scoped) so it is:
      (a) stable per run for the same document, and
      (b) usable as a first-pass alignment signal across revisions.
  • The `metadata` dict on Element and Document is an escape hatch for
    format-specific extras (e.g. DWG handle, PDF font name). It is NEVER
    read by the delta engine or chat layer — only logged in traces.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

ElementType = Literal["text", "dimension", "note", "geometry", "table_cell"]
ChangeType = Literal["added", "removed", "modified"]
FormatType = Literal["pdf_native", "pdf_scanned", "dwg"]


class BoundingBox(BaseModel):
    """
    Axis-aligned bounding box in page-coordinate space (points for PDF,
    model-units for DWG). `page_width` / `page_height` enable normalisation
    to [0, 1] for cross-format geometric comparisons.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: Optional[float] = None
    page_height: Optional[float] = None

    @model_validator(mode="after")
    def _clamp(self) -> "BoundingBox":
        # Ensure x0 < x1 and y0 < y1 (some PDF libs return inverted coords)
        if self.x0 > self.x1:
            self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1:
            self.y0, self.y1 = self.y1, self.y0
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def normalized(self) -> "BoundingBox":
        """Return bbox with coords normalised to [0, 1] using page dimensions."""
        if not (self.page_width and self.page_height):
            return self
        return BoundingBox(
            x0=self.x0 / self.page_width,
            y0=self.y0 / self.page_height,
            x1=self.x1 / self.page_width,
            y1=self.y1 / self.page_height,
            page_width=1.0,
            page_height=1.0,
        )

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union with another bbox."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        intersection = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


# ─────────────────────────────────────────────────────────────────────────────
# Element — atomic unit of content
# ─────────────────────────────────────────────────────────────────────────────

class Element(BaseModel):
    """
    The smallest meaningful unit of document content.

    `element_id` derivation:
        sha256(page_index | type | rounded_bbox | text_prefix)[:16]
        • Rounded to 1 dp to tolerate minor coordinate drift across re-exports
        • NOT pid-scoped so the alignment engine can use it as a cross-revision
          first-pass signal: matching IDs → very likely same element
        • text_prefix capped at 64 chars; whitespace-normalised
    """

    element_id: str = Field(description="Stable content-hash identifier")
    type: ElementType
    bbox: BoundingBox
    text: Optional[str] = Field(None, description="Human-readable text content")
    raw_value: Optional[str] = Field(None, description="Verbatim extracted string, e.g. '42.5 mm'")
    numeric_value: Optional[float] = Field(None, description="Parsed float for dimensions")
    unit: Optional[str] = Field(None, description="Unit string, e.g. 'mm', 'in', '°'")
    layer: Optional[str] = Field(None, description="DWG layer name; None for PDFs")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="OCR/extraction confidence")
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    # Format-specific extras — NEVER read by delta engine or chat layer
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_id(
        cls,
        page_index: int,
        element_type: ElementType,
        bbox: BoundingBox,
        text: Optional[str] = None,
    ) -> str:
        """Deterministic element_id generator — call from adapter code."""
        bbox_str = f"{bbox.x0:.1f},{bbox.y0:.1f},{bbox.x1:.1f},{bbox.y1:.1f}"
        text_part = " ".join((text or "").split())[:64].lower()
        raw = f"{page_index}|{element_type}|{bbox_str}|{text_part}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def display_text(self) -> str:
        """Best human-readable representation of this element."""
        return self.raw_value or self.text or f"[{self.type}]"


# ─────────────────────────────────────────────────────────────────────────────
# Block — a cluster of elements (table, text block, drawing entity, etc.)
# ─────────────────────────────────────────────────────────────────────────────

BlockType = Literal["text_block", "table", "figure", "drawing_entity", "title_block", "unknown"]


class Block(BaseModel):
    block_id: str
    type: BlockType
    bbox: BoundingBox
    elements: list[Element] = Field(default_factory=list)

    @property
    def all_text(self) -> str:
        """Concatenated text from all child elements."""
        return " ".join(e.text for e in self.elements if e.text)


# ─────────────────────────────────────────────────────────────────────────────
# Page / Sheet — a single page or CAD sheet
# ─────────────────────────────────────────────────────────────────────────────

class Page(BaseModel):
    page_index: int = Field(description="0-based page/sheet index")
    page_label: Optional[str] = Field(None, description="Human label, e.g. 'Sheet 3', 'A-101'")
    width: float = Field(description="Page width in native units")
    height: float = Field(description="Page height in native units")
    blocks: list[Block] = Field(default_factory=list)

    @property
    def all_elements(self) -> list[Element]:
        """Flat list of all elements across all blocks on this page."""
        return [e for b in self.blocks for e in b.elements]

    @property
    def element_count(self) -> int:
        return len(self.all_elements)


# ─────────────────────────────────────────────────────────────────────────────
# Document — the top-level canonical object produced by every adapter
# ─────────────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    """
    Format-agnostic canonical representation of a single document revision.

    INVARIANT: Every format adapter produces exactly this type.
               The delta engine, chat layer, and eval harness consume exactly
               this type. No downstream code checks `format` to branch logic.
    """

    pid: str = Field(description="Caller-assigned stable identifier for this revision")
    format: FormatType = Field(description="Source format (informational, never branched on downstream)")
    revision_label: Optional[str] = Field(None, description="e.g. 'Rev B', '2024-01-15'")
    title: Optional[str] = None
    page_count: int
    pages: list[Page]
    source_path: str = Field(description="Resolved filesystem path to raw bytes (logged in trace)")
    # Opaque format-specific extras — logged but never read by downstream layers
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_elements(self) -> list[Element]:
        """All elements in document order (page 0 → page N)."""
        return [e for p in self.pages for e in p.all_elements]

    @property
    def elements_by_page(self) -> dict[int, list[Element]]:
        return {p.page_index: p.all_elements for p in self.pages}

    @property
    def element_count(self) -> int:
        return sum(p.element_count for p in self.pages)

    def get_page(self, page_index: int) -> Optional[Page]:
        for p in self.pages:
            if p.page_index == page_index:
                return p
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Delta items — output of the delta engine, consumed by report + chat
# ─────────────────────────────────────────────────────────────────────────────

class ElementLocation(BaseModel):
    """Source-traceable pointer: PID + page + bbox."""
    pid: str
    page_index: int
    page_label: Optional[str] = None
    bbox: BoundingBox


class DeltaItem(BaseModel):
    """
    A single change between revision A (base) and revision B (revised).

    Fields:
        delta_id        — stable ID, deterministic from content hash
        change_type     — added | removed | modified
        element_type    — the type of the changed element
        location_a      — position in PID A (None for 'added')
        location_b      — position in PID B (None for 'removed')
        description     — human-readable summary of what changed
        old_value       — value in PID A (for modified/removed)
        new_value       — value in PID B (for modified/added)
        confidence      — combined alignment + OCR confidence [0, 1]
        alignment_method— how this pair was matched (deterministic/llm/exact_id)
    """

    delta_id: str
    change_type: ChangeType
    element_type: ElementType
    location_a: Optional[ElementLocation] = None
    location_b: Optional[ElementLocation] = None
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    alignment_method: str = Field(description="deterministic | llm_assisted | exact_id")
    element_a: Optional[Element] = None  # kept for markup overlay
    element_b: Optional[Element] = None

    @classmethod
    def make_id(cls, change_type: ChangeType, loc_a: Optional[ElementLocation],
                loc_b: Optional[ElementLocation], element_type: ElementType) -> str:
        parts = [change_type, element_type,
                 f"{loc_a.page_index}:{loc_a.bbox.x0:.0f}" if loc_a else "none",
                 f"{loc_b.page_index}:{loc_b.bbox.x0:.0f}" if loc_b else "none"]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

    @property
    def primary_page_index(self) -> int:
        """Page index for display/grouping — prefer B location."""
        if self.location_b:
            return self.location_b.page_index
        if self.location_a:
            return self.location_a.page_index
        return 0


class DeltaReport(BaseModel):
    """Structured output of the delta engine — stored, indexed, and cited in chat."""

    report_id: str
    generated_at: str  # ISO 8601
    pid_a: str
    pid_b: str
    run_id: str
    summary: dict[str, Any]  # counts by change type and element type
    items: list[DeltaItem]

    @classmethod
    def build_summary(cls, items: list[DeltaItem]) -> dict[str, Any]:
        from collections import Counter

        by_type = Counter(i.change_type for i in items)
        by_elem = Counter(i.element_type for i in items)
        confidences = [i.confidence for i in items]
        return {
            "total_changes": len(items),
            "added": by_type.get("added", 0),
            "removed": by_type.get("removed", 0),
            "modified": by_type.get("modified", 0),
            "by_element_type": dict(by_elem),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "low_confidence_count": sum(1 for c in confidences if c < 0.70),
        }
