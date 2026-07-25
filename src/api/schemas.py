"""
delta-chat · src/api/schemas.py
Request/response Pydantic models for FastAPI routes.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    pid_a: str = Field(description="Stable identifier for revision A")
    pid_b: str = Field(description="Stable identifier for revision B")
    path_a: str = Field(description="File path to revision A document")
    path_b: str = Field(description="File path to revision B document")
    revision_label_a: Optional[str] = None
    revision_label_b: Optional[str] = None


class IngestResponse(BaseModel):
    run_id: str
    request_id: str
    status: str
    pid_a: str
    pid_b: str
    format_a: str
    format_b: str
    elements_a: int
    elements_b: int
    total_changes: int
    added: int
    removed: int
    modified: int
    avg_confidence: float
    delta_json_path: str
    delta_md_path: str
    indexed: bool
    stage_timings: dict[str, float]
    errors: list[dict[str, str]]
    trace_path: str


class ChatRequest(BaseModel):
    run_id: str = Field(description="Run ID from a previous /ingest call")
    query: str = Field(description="Natural language question about the documents")


class CitationResponse(BaseModel):
    label: str
    source_type: str
    source_pid: str
    page_index: int
    similarity: float
    snippet: str


class ChatResponse(BaseModel):
    run_id: str
    query: str
    answer: str
    citations: list[CitationResponse]
    from_cache: bool
    cache_similarity: Optional[float]
    insufficient_grounding: bool
    cache_stats: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    db: bool
    redis: bool
    chroma: bool
    llm_providers: list[str]


class DeltaSummaryResponse(BaseModel):
    run_id: str
    pid_a: str
    pid_b: str
    total_changes: int
    added: int
    removed: int
    modified: int
    avg_confidence: float
    delta_json_path: str
    delta_md_path: str


class MarkupResponse(BaseModel):
    run_id: str
    pid_a_annotated: str
    pid_b_annotated: str
    total_annotations: int
