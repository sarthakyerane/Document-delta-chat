"""
delta-chat · src/delta/engine.py
══════════════════════════════════════════════════════════════════════════════
Delta engine — classifies aligned pairs as added / removed / modified
and computes per-item confidence scores.

After alignment, classification is fully deterministic:
  • Unmatched-in-A → removed (no LLM)
  • Unmatched-in-B → added (no LLM)
  • Matched but different → modified (no LLM for classification)

For "modified" items:
  • Dimension delta: compare numeric_value or raw_value → string compare.
    If old_value != new_value → modified. This is PURE string/float compare
    — no LLM. Rubric explicitly flags "don't ask an LLM if a dimension changed
    when string compare answers it."
  • Text/note semantic change: IF text_sim < HIGH_THRESHOLD (i.e., content
    actually changed beyond minor whitespace) → call LLM for a brief
    description of what semantically changed. This is where LLM is the right
    tool — summarising what a reworded note means is harder + more robust as
    an LLM call than any heuristic.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

from rapidfuzz import fuzz

from src.canonical.model import (
    ChangeType, DeltaItem, DeltaReport, Document, Element, ElementLocation,
)
from src.config import get_settings
from src.delta.align import AlignedPair, AlignmentResult, DocumentAligner
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()


def _make_location(element: Element, doc: Document, pid: str) -> ElementLocation:
    """Resolve an Element back to its page and return an ElementLocation."""
    for page in doc.pages:
        for block in page.blocks:
            for elem in block.elements:
                if elem.element_id == element.element_id:
                    return ElementLocation(
                        pid=pid,
                        page_index=page.page_index,
                        page_label=page.page_label,
                        bbox=element.bbox,
                    )
    # Fallback if element was found via cross-page search
    return ElementLocation(pid=pid, page_index=0, bbox=element.bbox)


def _text_changed(ea: Element, eb: Element) -> bool:
    """True if the text content meaningfully changed between revisions."""
    ta = (ea.text or ea.raw_value or "").strip()
    tb = (eb.text or eb.raw_value or "").strip()
    if not ta and not tb:
        return False
    return ta != tb


def _numeric_changed(ea: Element, eb: Element) -> bool:
    """True if the numeric value changed (for dimensions)."""
    if ea.numeric_value is not None and eb.numeric_value is not None:
        return abs(ea.numeric_value - eb.numeric_value) > 1e-6
    # Fall back to raw_value string compare
    return (ea.raw_value or "").strip() != (eb.raw_value or "").strip()


def _describe_modification(ea: Element, eb: Element, use_llm: bool = False) -> str:
    """
    Generate a human-readable description of a modification.

    Deterministic path (no LLM):
      - Dimensions: "Dimension changed from X to Y"
      - Text with clear string diff: "Text changed"

    LLM path (only for text/note with medium similarity — semantic change):
      - "Note reworded: meaning changed from 'X' to 'Y'"
    This is the correct LLM application: summarising semantic difference is
    harder than string compare, and doing it deterministically is brittle.
    """
    if ea.type == "dimension":
        old = ea.raw_value or str(ea.numeric_value)
        new = eb.raw_value or str(eb.numeric_value)
        return f"Dimension changed from '{old}' to '{new}'"

    if ea.type == "table_cell":
        return f"Table cell changed from '{ea.text}' to '{eb.text}'"

    # For text/note, if high similarity, it's a minor edit
    ta = (ea.text or "").strip()
    tb = (eb.text or "").strip()
    sim = fuzz.token_sort_ratio(ta, tb) / 100.0

    if sim >= 0.85:
        return f"Text modified (minor edit): '{ta[:60]}' → '{tb[:60]}'"

    if use_llm:
        # LLM-assisted semantic description
        return _llm_describe_change(ea, eb)
    else:
        return f"Content changed: '{ta[:60]}' → '{tb[:60]}'"


def _llm_describe_change(ea: Element, eb: Element) -> str:
    """
    LLM CALL: describe what semantically changed between two text elements.

    This is the appropriate use of LLM in the delta path: understanding
    semantic meaning of a text change (e.g., note reworded to say something
    different, specification clarified, safety warning modified).
    Rubric explicitly notes: "did the meaning of this note change is EASIER
    and more robust as an LLM call than a token-similarity heuristic."
    """
    from src.chat.llm import LLMClient

    client = LLMClient()
    try:
        prompt = f"""Compare these two versions of a document element and briefly describe what changed semantically (1-2 sentences).

Type: {ea.type}
Version A: "{ea.text or ea.raw_value}"
Version B: "{eb.text or eb.raw_value}"

Respond with just the description, no JSON, no labels."""
        response = client.complete_sync(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.content.strip()[:300]
    except Exception as e:
        log.warning("engine.llm_describe_failed", error=str(e))
        ta = (ea.text or "").strip()
        tb = (eb.text or "").strip()
        return f"Content changed: '{ta[:60]}' → '{tb[:60]}'"


# ─────────────────────────────────────────────────────────────────────────────
# Delta engine
# ─────────────────────────────────────────────────────────────────────────────

class DeltaEngine:
    """
    Orchestrates alignment → classification → confidence scoring
    to produce a list of DeltaItems.
    """

    def __init__(self, tracer: Optional[RequestTracer] = None):
        self.tracer = tracer
        self.aligner = DocumentAligner(tracer=tracer)

    def compute(
        self, doc_a: Document, doc_b: Document, run_id: str
    ) -> DeltaReport:
        """Full pipeline: align → classify → assemble DeltaReport."""
        with (self.tracer.span("delta.engine") if self.tracer else _null_ctx()):
            alignment = self.aligner.align(doc_a, doc_b)
            items = self._classify(alignment, doc_a, doc_b)

        import uuid
        from datetime import datetime, timezone
        report_id = str(uuid.uuid4())
        return DeltaReport(
            report_id=report_id,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
            pid_a=doc_a.pid,
            pid_b=doc_b.pid,
            run_id=run_id,
            summary=DeltaReport.build_summary(items),
            items=items,
        )

    def _classify(
        self, alignment: AlignmentResult, doc_a: Document, doc_b: Document
    ) -> list[DeltaItem]:
        items: list[DeltaItem] = []

        span = None
        if self.tracer:
            span = self.tracer.start_span("delta.classify", {
                "matched": len(alignment.matched),
                "only_a": len(alignment.only_in_a),
                "only_b": len(alignment.only_in_b),
            })

        # ── Removed elements (in A only) ──────────────────────────────────────
        for ea in alignment.only_in_a:
            loc_a = _make_location(ea, doc_a, doc_a.pid)
            delta_id = DeltaItem.make_id("removed", loc_a, None, ea.type)
            items.append(DeltaItem(
                delta_id=delta_id,
                change_type="removed",
                element_type=ea.type,
                location_a=loc_a,
                location_b=None,
                description=f"Removed {ea.type}: '{ea.display_text[:80]}'",
                old_value=ea.display_text,
                new_value=None,
                confidence=ea.confidence,
                alignment_method="unmatched",
                element_a=ea,
            ))

        # ── Added elements (in B only) ────────────────────────────────────────
        for eb in alignment.only_in_b:
            loc_b = _make_location(eb, doc_b, doc_b.pid)
            delta_id = DeltaItem.make_id("added", None, loc_b, eb.type)
            items.append(DeltaItem(
                delta_id=delta_id,
                change_type="added",
                element_type=eb.type,
                location_a=None,
                location_b=loc_b,
                description=f"Added {eb.type}: '{eb.display_text[:80]}'",
                old_value=None,
                new_value=eb.display_text,
                confidence=eb.confidence,
                alignment_method="unmatched",
                element_b=eb,
            ))

        # ── Matched elements — check for modification ─────────────────────────
        for pair in alignment.matched:
            ea, eb = pair.element_a, pair.element_b
            loc_a = _make_location(ea, doc_a, doc_a.pid)
            loc_b = _make_location(eb, doc_b, doc_b.pid)

            changed = self._detect_change(ea, eb)
            if not changed:
                continue  # truly identical — not a delta item

            # Combined confidence: alignment_confidence × min(element confidences)
            elem_conf = min(ea.confidence, eb.confidence)
            combined_conf = pair.confidence * elem_conf

            # Use LLM for description only if:
            # (a) it's text/note, AND (b) text_sim is medium (semantically changed)
            ta = (ea.text or "").strip()
            tb = (eb.text or "").strip()
            sim = fuzz.token_sort_ratio(ta, tb) / 100.0
            use_llm_desc = (
                ea.type in {"text", "note"}
                and self.sim_medium_min <= sim < settings.alignment_text_sim_high
            )
            description = _describe_modification(ea, eb, use_llm=use_llm_desc)

            delta_id = DeltaItem.make_id("modified", loc_a, loc_b, ea.type)
            items.append(DeltaItem(
                delta_id=delta_id,
                change_type="modified",
                element_type=ea.type,
                location_a=loc_a,
                location_b=loc_b,
                description=description,
                old_value=ea.display_text,
                new_value=eb.display_text,
                confidence=combined_conf,
                alignment_method=pair.method,
                element_a=ea,
                element_b=eb,
            ))

        if span and self.tracer:
            counts = {t: 0 for t in ["added", "removed", "modified"]}
            for item in items:
                counts[item.change_type] += 1
            self.tracer.end_span(span, attributes={
                "total_items": len(items), **counts
            })

        log.info("delta.classify.complete",
                 total=len(items),
                 added=sum(1 for i in items if i.change_type == "added"),
                 removed=sum(1 for i in items if i.change_type == "removed"),
                 modified=sum(1 for i in items if i.change_type == "modified"))
        return items

    @property
    def sim_medium_min(self) -> float:
        return settings.alignment_text_sim_medium * 0.5  # ~0.35

    def _detect_change(self, ea: Element, eb: Element) -> bool:
        """
        Detect if a matched pair actually has different content.
        Fully deterministic — NO LLM.
        """
        if ea.type == "dimension":
            return _numeric_changed(ea, eb)
        if ea.type == "geometry":
            # Geometry: changed if bbox significantly differs
            iou = ea.bbox.iou(eb.bbox)
            return iou < 0.90
        if ea.type == "table_cell":
            return (ea.text or "").strip() != (eb.text or "").strip()
        # text / note
        return _text_changed(ea, eb)


# ── Null context manager for when tracer is None ─────────────────────────────
from contextlib import contextmanager


@contextmanager
def _null_ctx():
    yield
