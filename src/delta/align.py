"""
delta-chat · src/delta/align.py
══════════════════════════════════════════════════════════════════════════════
Content alignment between two document revisions.

THE HARD PART IS ALIGNMENT, NOT DIFFING (from spec §04).

Multi-stage strategy:
  Stage 1 — Exact element_id match (same hash → same content + location)
             Confidence: 1.0 | Method: "exact_id"

  Stage 2 — Same page + bbox IoU ≥ high_threshold + text_sim ≥ high_threshold
             Confidence: (iou + text_sim) / 2 | Method: "geometric+text_high"

  Stage 3 — Same page + text_sim ≥ high_threshold (element moved on same page)
             Confidence: text_sim × 0.90 | Method: "text_high"

  Stage 4 — Same page + bbox IoU ≥ medium + text_sim ≥ medium
             Confidence: (iou + text_sim) / 2 × 0.85 | Method: "geometric+text_medium"

  Stage 5 — Cross-page text_sim ≥ high (element moved across pages)
             Confidence: text_sim × 0.70 | Method: "cross_page_text"

  Stage 6 — LLM disambiguation for 0.40 ≤ best_score < MEDIUM
             Called ONLY for genuinely ambiguous pairs where deterministic
             signals are insufficient.  LLM non-determinism is isolated here
             and documented explicitly.
             Confidence: from LLM 0–1 response | Method: "llm_assisted"

  Unmatched in A → "removed"
  Unmatched in B → "added"

DETERMINISM boundary:
  Stages 1–5 are fully deterministic.
  Stage 6 introduces LLM non-determinism.  To reproduce: the same pair with
  the same model and temperature=0 should return the same result, but this
  is NOT guaranteed.  Delta item `alignment_method` field records whether LLM
  was used so callers can audit.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz

from src.canonical.model import Document, Element
from src.config import get_settings
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Output data-class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignedPair:
    element_a: Element
    element_b: Element
    confidence: float
    method: str  # one of: exact_id, geometric+text_high, text_high,
                 # geometric+text_medium, cross_page_text, llm_assisted


@dataclass
class AlignmentResult:
    matched: list[AlignedPair]       # pairs of (A, B) elements
    only_in_a: list[Element]         # elements present only in revision A (removed)
    only_in_b: list[Element]         # elements present only in revision B (added)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _text_similarity(a: Element, b: Element) -> float:
    """
    Normalised edit-distance similarity between two elements' display text.
    Uses rapidfuzz token_sort_ratio for robustness against word reordering.
    Returns [0.0, 1.0].
    """
    ta = (a.text or a.raw_value or "").strip()
    tb = (b.text or b.raw_value or "").strip()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return fuzz.token_sort_ratio(ta, tb) / 100.0


def _bbox_iou(a: Element, b: Element) -> float:
    """
    IoU of bounding boxes. Uses normalised coords if page dimensions are
    available (cross-DPI stability); raw otherwise.
    """
    ba = a.bbox.normalized() if (a.bbox.page_width and a.bbox.page_height) else a.bbox
    bb = b.bbox.normalized() if (b.bbox.page_width and b.bbox.page_height) else b.bbox
    return ba.iou(bb)


def _same_type(a: Element, b: Element) -> bool:
    return a.type == b.type


# ─────────────────────────────────────────────────────────────────────────────
# Main aligner
# ─────────────────────────────────────────────────────────────────────────────

class DocumentAligner:
    """
    Aligns elements between two document revisions using the multi-stage
    strategy described above.
    """

    def __init__(self, tracer: Optional[RequestTracer] = None):
        self.tracer = tracer
        self.sim_high = settings.alignment_text_sim_high
        self.sim_medium = settings.alignment_text_sim_medium
        self.sim_llm_min = settings.alignment_text_sim_llm_min
        self.iou_high = settings.alignment_bbox_iou_high

    def align(self, doc_a: Document, doc_b: Document) -> AlignmentResult:
        """Entry point: align all elements across two documents."""
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                "delta.align",
                {"pid_a": doc_a.pid, "pid_b": doc_b.pid,
                 "elements_a": doc_a.element_count, "elements_b": doc_b.element_count},
            )

        elems_a = doc_a.all_elements
        elems_b = doc_b.all_elements

        matched: list[AlignedPair] = []
        unmatched_a = set(range(len(elems_a)))
        unmatched_b = set(range(len(elems_b)))

        # Build per-page index for efficiency
        pages_a: dict[int, list[tuple[int, Element]]] = {}
        for i, e in enumerate(elems_a):
            pages_a.setdefault(e.bbox.y0 // 1000, []).append((i, e))  # rough page bucket

        pages_b_by_page: dict[int, list[tuple[int, Element]]] = {}
        for i, e in enumerate(elems_b):
            page_idx = self._estimate_page(e, doc_b)
            pages_b_by_page.setdefault(page_idx, []).append((i, e))

        pages_a_by_page: dict[int, list[tuple[int, Element]]] = {}
        for i, e in enumerate(elems_a):
            page_idx = self._estimate_page(e, doc_a)
            pages_a_by_page.setdefault(page_idx, []).append((i, e))

        # ── Stage 1: Exact element_id match ───────────────────────────────────
        id_map_b: dict[str, int] = {e.element_id: i for i, e in enumerate(elems_b)}
        for i, ea in enumerate(elems_a):
            j = id_map_b.get(ea.element_id)
            if j is not None and j in unmatched_b:
                matched.append(AlignedPair(ea, elems_b[j], 1.0, "exact_id"))
                unmatched_a.discard(i)
                unmatched_b.discard(j)

        log.debug("align.stage1", matched=len(matched),
                  remaining_a=len(unmatched_a), remaining_b=len(unmatched_b))

        # ── Stages 2–5: Deterministic geometric + text matching ───────────────
        ambiguous_pairs: list[tuple[Element, Element, float]] = []

        for pa_idx, a_elems in pages_a_by_page.items():
            # Try same page first, then adjacent pages
            for pb_idx in [pa_idx, pa_idx - 1, pa_idx + 1]:
                b_elems = pages_b_by_page.get(pb_idx, [])
                cross_page = pa_idx != pb_idx

                for i, ea in [(i, e) for i, e in a_elems if i in unmatched_a]:
                    best_score = 0.0
                    best_j: Optional[int] = None
                    best_method = ""

                    for j, eb in [(j, e) for j, e in b_elems if j in unmatched_b]:
                        if not _same_type(ea, eb):
                            # Allow type mismatch for dimension↔text (OCR can misclassify)
                            if not (ea.type in {"dimension", "text"} and eb.type in {"dimension", "text"}):
                                continue

                        ts = _text_similarity(ea, eb)
                        iou = _bbox_iou(ea, eb) if not cross_page else 0.0

                        score, method = self._score_pair(ts, iou, cross_page)
                        if score > best_score:
                            best_score = score
                            best_j = j
                            best_method = method
                            best_ts = ts

                    if best_j is not None and best_j in unmatched_b:
                        if best_score >= self.sim_medium:
                            matched.append(AlignedPair(
                                ea, elems_b[best_j], best_score, best_method
                            ))
                            unmatched_a.discard(i)
                            unmatched_b.discard(best_j)
                        elif best_score >= self.sim_llm_min:
                            # Potentially ambiguous — collect for Stage 6
                            ambiguous_pairs.append((ea, elems_b[best_j], best_score))

        log.debug("align.stages2_5", matched=len(matched),
                  ambiguous=len(ambiguous_pairs),
                  remaining_a=len(unmatched_a), remaining_b=len(unmatched_b))

        # ── Stage 6: LLM disambiguation for ambiguous pairs ───────────────────
        if ambiguous_pairs and settings.alignment_use_llm:
            llm_matched = self._llm_disambiguate(ambiguous_pairs, doc_a, doc_b)
            for pair in llm_matched:
                i_match = next(
                    (i for i, e in enumerate(elems_a) if e.element_id == pair.element_a.element_id),
                    None,
                )
                j_match = next(
                    (j for j, e in enumerate(elems_b) if e.element_id == pair.element_b.element_id),
                    None,
                )
                if i_match in unmatched_a and j_match in unmatched_b:
                    matched.append(pair)
                    unmatched_a.discard(i_match)
                    unmatched_b.discard(j_match)

        only_in_a = [elems_a[i] for i in sorted(unmatched_a)]
        only_in_b = [elems_b[j] for j in sorted(unmatched_b)]

        result = AlignmentResult(matched=matched, only_in_a=only_in_a, only_in_b=only_in_b)

        if span and self.tracer:
            self.tracer.end_span(span, attributes={
                "matched": len(matched),
                "removed": len(only_in_a),
                "added": len(only_in_b),
                "llm_disambiguated": sum(1 for p in matched if p.method == "llm_assisted"),
            })

        log.info("align.complete", matched=len(matched),
                 removed=len(only_in_a), added=len(only_in_b))
        return result

    def _score_pair(
        self, text_sim: float, iou: float, cross_page: bool
    ) -> tuple[float, str]:
        """Return (score, method) for a candidate pair."""
        # Stage 2: high geo + high text (same page)
        if not cross_page and iou >= self.iou_high and text_sim >= self.sim_high:
            return (iou + text_sim) / 2, "geometric+text_high"

        # Stage 3: high text only (element moved on page)
        if text_sim >= self.sim_high:
            return text_sim * 0.90, "text_high"

        # Stage 4: medium geo + medium text
        if not cross_page and iou >= 0.40 and text_sim >= self.sim_medium:
            return (iou + text_sim) / 2 * 0.85, "geometric+text_medium"

        # Stage 5: cross-page high text (element moved to different page)
        if cross_page and text_sim >= self.sim_high:
            return text_sim * 0.70, "cross_page_text"

        return text_sim * 0.50, "weak_text"

    def _estimate_page(self, element: Element, doc: Document) -> int:
        """Find which page an element belongs to by scanning pages."""
        for page in doc.pages:
            for block in page.blocks:
                for elem in block.elements:
                    if elem.element_id == element.element_id:
                        return page.page_index
        return 0

    def _llm_disambiguate(
        self,
        candidates: list[tuple[Element, Element, float]],
        doc_a: Document,
        doc_b: Document,
    ) -> list[AlignedPair]:
        """
        LLM-ASSISTED ALIGNMENT (Stage 6).

        Non-determinism boundary: this is the ONLY place LLM is called
        in the alignment pipeline.  Used only when:
          sim_llm_min ≤ best_deterministic_score < sim_medium

        Batches up to ALIGNMENT_LLM_BATCH_SIZE pairs per call.
        Returns matched pairs with confidence from LLM response.

        Why LLM here: determining whether a note that moved AND changed wording
        is the "same" element requires semantic understanding that a similarity
        threshold cannot reliably provide (e.g., "Ref drawing 42" → "See DWG-042"
        is the same note reworded, but text_sim ≈ 0.45).
        """
        from src.chat.llm import LLMClient

        client = LLMClient(tracer=self.tracer)
        results: list[AlignedPair] = []

        batch_size = settings.alignment_llm_batch_size
        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start: batch_start + batch_size]
            pairs_json = json.dumps([
                {
                    "pair_index": idx,
                    "element_a": {
                        "text": ea.display_text,
                        "type": ea.type,
                        "page": self._estimate_page(ea, doc_a),
                    },
                    "element_b": {
                        "text": eb.display_text,
                        "type": eb.type,
                        "page": self._estimate_page(eb, doc_b),
                    },
                    "initial_score": round(score, 3),
                }
                for idx, (ea, eb, score) in enumerate(batch)
            ], indent=2)

            prompt = f"""You are a document element alignment expert for engineering drawings.

For each pair below, determine if element_a (from the base revision) and element_b (from the revised revision) represent the SAME document element (possibly moved or slightly modified).

Return a JSON array with one object per pair:
{{"pair_index": <int>, "is_same": <true|false>, "confidence": <0.0-1.0>, "reason": "<brief>"}}

Rules:
- "is_same": true if they represent the same physical element that was edited/moved
- "confidence": your confidence that is_same is correct
- Consider semantic meaning, not just string similarity
- A note reworded to say the same thing with different words = same element
- A dimension that changed value = same element (it was modified, not replaced)
- Two completely different elements that happen to look similar = NOT same

Pairs to evaluate:
{pairs_json}

Return ONLY valid JSON array. No prose."""

            try:
                t0 = time.time()
                response = client.complete_sync(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.content.strip()
                raw = raw.strip("```json").strip("```").strip()
                judgments: list[dict] = json.loads(raw)
                duration_ms = (time.time() - t0) * 1000

                for j in judgments:
                    idx = j.get("pair_index", -1)
                    if idx < 0 or idx >= len(batch):
                        continue
                    ea, eb, _ = batch[idx]
                    if j.get("is_same") and j.get("confidence", 0) >= 0.60:
                        results.append(AlignedPair(
                            ea, eb,
                            confidence=float(j.get("confidence", 0.60)),
                            method="llm_assisted",
                        ))
            except Exception as e:
                log.warning("align.llm_disambiguation_failed", error=str(e),
                            batch_size=len(batch))

        return results
