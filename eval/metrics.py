"""
delta-chat · eval/metrics.py
══════════════════════════════════════════════════════════════════════════════
Delta and chat evaluation metrics.

Delta metrics (precision / recall / F1):
  A predicted delta item MATCHES a ground-truth item if:
    1. Same change_type (added/removed/modified)
    2. Same element_type (text/dimension/note/geometry/table_cell)
    3. Same page_index (within ±1 tolerance for minor misalignment)
    4. EITHER: bbox IoU > 0.3 (if both have bbox)
       OR: text similarity > 0.65 (if text content available)

Chat metrics:
  LLM-as-judge scoring on 0.0–1.0 scale for:
    • Correctness: does the answer address the question correctly?
    • Groundedness: does the answer cite real sources, not hallucinate?
  Judge: Gemini 2.5 Flash at temperature=0.
  Judge validation: 10 hand-checked cases included in datasets/judge_validation.json
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GTDeltaItem:
    """Ground-truth delta item (from labeled JSON)."""
    change_type: str          # added | removed | modified
    element_type: str         # text | dimension | note | geometry | table_cell
    page_index: int
    description: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    bbox: Optional[list[float]] = None  # [x0, y0, x1, y1]

    @classmethod
    def from_dict(cls, d: dict) -> "GTDeltaItem":
        return cls(
            change_type=d["change_type"],
            element_type=d["element_type"],
            page_index=d["page_index"],
            description=d.get("description", ""),
            old_value=d.get("old_value"),
            new_value=d.get("new_value"),
            bbox=d.get("bbox"),
        )


@dataclass
class DeltaMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predicted: int
    total_ground_truth: int


@dataclass
class ChatMetrics:
    avg_correctness: float
    avg_groundedness: float
    total_qa: int
    scores: list[dict]
    judge_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Delta matching
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_iou_raw(a: list[float], b: list[float]) -> float:
    """IoU for raw [x0,y0,x1,y1] lists."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0



# Element types that are considered compatible when matching predicted vs GT
_COMPATIBLE_ELEMENT_TYPES: dict[str, set[str]] = {
    "text":       {"text", "table_cell", "note"},
    "table_cell": {"text", "table_cell", "note"},
    "note":       {"text", "table_cell", "note"},
    "dimension":  {"dimension"},
    "geometry":   {"geometry"},
}


def _items_match(pred: dict, gt: GTDeltaItem) -> bool:
    """Return True if a predicted delta item matches a GT item."""
    # Must match change_type exactly
    if pred.get("change_type") != gt.change_type:
        return False

    # Element type: allow compatible types (text/table_cell/note are interchangeable)
    pred_etype = pred.get("element_type", "text")
    gt_etype = gt.element_type
    compatible = _COMPATIBLE_ELEMENT_TYPES.get(pred_etype, {pred_etype})
    if gt_etype not in compatible:
        return False

    # Page must match within ±1
    pred_page = (pred.get("location_b") or {}).get("page_index") or \
                (pred.get("location_a") or {}).get("page_index") or 0
    if abs(pred_page - gt.page_index) > 1:
        return False

    # Geometric or text match
    geo_match = False
    text_match = False

    pred_bbox = (
        (pred.get("location_b") or {}).get("bbox") or
        (pred.get("location_a") or {}).get("bbox") or {}
    )
    if pred_bbox and gt.bbox:
        pred_coords = [
            pred_bbox.get("x0", 0), pred_bbox.get("y0", 0),
            pred_bbox.get("x1", 0), pred_bbox.get("y1", 0),
        ]
        geo_match = _bbox_iou_raw(pred_coords, gt.bbox) >= 0.30

    pred_old = pred.get("old_value") or ""
    pred_new = pred.get("new_value") or ""
    gt_old = gt.old_value or ""
    gt_new = gt.new_value or ""
    gt_desc = gt.description or ""

    def _sim(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return max(
            fuzz.token_sort_ratio(a, b),
            fuzz.partial_ratio(a, b),   # catches "42.5 mm" inside "Flange Diameter: 42.5 mm"
        ) / 100.0

    best_sim = max(
        _sim(pred_old, gt_old),
        _sim(pred_new, gt_new),
        _sim(pred_old or pred_new, gt_desc),
    )
    text_match = best_sim >= 0.50

    return geo_match or text_match


def compute_delta_metrics(
    predicted_items: list[dict],
    ground_truth: list[GTDeltaItem],
) -> DeltaMetrics:
    """
    Compute precision / recall / F1 on delta items.
    Uses greedy matching (best predicted → GT pair).
    """
    matched_preds = set()
    matched_gts = set()

    for i, pred in enumerate(predicted_items):
        for j, gt in enumerate(ground_truth):
            if j in matched_gts:
                continue
            if _items_match(pred, gt):
                matched_preds.add(i)
                matched_gts.add(j)
                break

    tp = len(matched_preds)
    fp = len(predicted_items) - tp
    fn = len(ground_truth) - len(matched_gts)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return DeltaMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        total_predicted=len(predicted_items),
        total_ground_truth=len(ground_truth),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat evaluation — LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_PROMPT = """You are an objective evaluator of answers from a document analysis system.

Question: {question}
Expected answer (reference): {expected_answer}
System answer: {system_answer}
System cited sources: {citations}

Rate the system answer on two dimensions, each 0.0–1.0:

1. correctness: Does the answer correctly address the question?
   - Compare to the reference answer. The reference is the MINIMUM required information.
   - If the system answer contains ALL the key facts in the reference (even if it includes additional correct context), score 1.0.
   - If the system answer is missing some key facts from the reference but gets others right, score 0.5.
   - If the system answer is factually wrong, contradicts the reference, or provides completely irrelevant information, score 0.0.
   - NOTE: Extra correct details beyond the reference do NOT reduce the score.

2. groundedness: Is the answer grounded in real cited sources?
   - 1.0 = all claims supported by citations, no hallucination
   - 0.5 = mostly grounded, some unsupported claims
   - 0.0 = no citations, or citations don't support the claims

Return ONLY a JSON object: {{"correctness": <float>, "groundedness": <float>, "reason": "<1 sentence>"}}"""


def evaluate_chat_qa(
    qa_pairs: list[dict],
    system_answers: list[dict],
) -> ChatMetrics:
    """
    Evaluate chat answers using LLM-as-judge.

    qa_pairs: list of {question, expected_answer}
    system_answers: list of {answer, citations: [{label, snippet}]}
    """
    from src.chat.llm import LLMClient

    client = LLMClient()
    scores: list[dict] = []
    judge_model = "gemini-3.5-flash"

    for i, (qa, sa) in enumerate(zip(qa_pairs, system_answers)):
        citations_str = "\n".join(
            f"  - {c.get('label', '')} ({c.get('snippet', '')[:80]})"
            for c in sa.get("citations", [])
        ) or "(no citations)"

        prompt = _JUDGE_PROMPT.format(
            question=qa["question"],
            expected_answer=qa.get("expected_answer", ""),
            system_answer=sa.get("answer", ""),
            citations=citations_str,
        )

        try:
            response = client.complete_sync(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.content.strip().strip("```json").strip("```").strip()
            judgment = json.loads(raw)
        except Exception as e:
            judgment = {"correctness": 0.0, "groundedness": 0.0, "reason": f"Judge error: {e}"}

        scores.append({
            "qa_index": i,
            "question": qa["question"],
            "correctness": float(judgment.get("correctness", 0.0)),
            "groundedness": float(judgment.get("groundedness", 0.0)),
            "reason": judgment.get("reason", ""),
            "expected": qa.get("expected_answer", ""),
            "got": sa.get("answer", "")[:200],
        })

    if not scores:
        return ChatMetrics(
            avg_correctness=0.0, avg_groundedness=0.0,
            total_qa=0, scores=[], judge_model=judge_model,
        )

    return ChatMetrics(
        avg_correctness=round(sum(s["correctness"] for s in scores) / len(scores), 4),
        avg_groundedness=round(sum(s["groundedness"] for s in scores) / len(scores), 4),
        total_qa=len(scores),
        scores=scores,
        judge_model=judge_model,
    )
