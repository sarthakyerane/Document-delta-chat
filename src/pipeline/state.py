"""
delta-chat · src/pipeline/state.py
LangGraph pipeline state — shared across all graph nodes.
"""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.canonical.model import Document, DeltaReport


class PipelineState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    pid_a: str
    pid_b: str
    path_a: str
    path_b: str
    run_id: str
    request_id: str
    revision_label_a: Optional[str]
    revision_label_b: Optional[str]

    # ── Ingestion ──────────────────────────────────────────────────────────────
    doc_a: Optional[Document]
    doc_b: Optional[Document]

    # ── Delta ─────────────────────────────────────────────────────────────────
    delta_report: Optional[DeltaReport]
    delta_json_path: str
    delta_md_path: str
    delta_md_text: str

    # ── Indexing ──────────────────────────────────────────────────────────────
    indexed: bool
    index_stats: dict[str, int]  # {"pid_a": N, "pid_b": M, "delta": K}

    # ── Chat (populated per query, not per run) ────────────────────────────────
    query: Optional[str]
    answer: Optional[dict]  # serialised GroundedAnswer

    # ── Observability ─────────────────────────────────────────────────────────
    tracer: Any  # RequestTracer — Any because TypedDict doesn't allow non-JSON types
    errors: list[dict[str, str]]
    stage_timings: dict[str, float]  # stage_name → duration_ms

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir: str
