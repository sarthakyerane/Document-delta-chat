"""
delta-chat · src/db/models.py
SQLAlchemy ORM models for run history, eval results, and metadata.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Float, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    """Tracks each ingest+delta pipeline run."""
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    pid_a: Mapped[str] = mapped_column(String(512))
    pid_b: Mapped[str] = mapped_column(String(512))
    path_a: Mapped[str] = mapped_column(Text)
    path_b: Mapped[str] = mapped_column(Text)
    format_a: Mapped[str] = mapped_column(String(32))
    format_b: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    total_changes: Mapped[int] = mapped_column(Integer, default=0)
    added: Mapped[int] = mapped_column(Integer, default=0)
    removed: Mapped[int] = mapped_column(Integer, default=0)
    modified: Mapped[int] = mapped_column(Integer, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    delta_json_path: Mapped[str] = mapped_column(Text, default="")
    delta_md_path: Mapped[str] = mapped_column(Text, default="")
    trace_path: Mapped[str] = mapped_column(Text, default="")
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[str] = mapped_column(Text, default="")
    stage_timings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class EvalRun(Base):
    """Stores eval scorecard results for regression comparison."""
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True,
                                     default=lambda: str(uuid.uuid4()))
    run_timestamp: Mapped[str] = mapped_column(String(64))  # ISO timestamp
    mode: Mapped[str] = mapped_column(String(32))  # "delta" | "chat" | "full"
    dataset_version: Mapped[str] = mapped_column(String(32), default="v1")

    # Delta metrics
    delta_precision: Mapped[float] = mapped_column(Float, default=0.0)
    delta_recall: Mapped[float] = mapped_column(Float, default=0.0)
    delta_f1: Mapped[float] = mapped_column(Float, default=0.0)
    delta_pairs_tested: Mapped[int] = mapped_column(Integer, default=0)

    # Chat metrics
    chat_correctness: Mapped[float] = mapped_column(Float, default=0.0)
    chat_groundedness: Mapped[float] = mapped_column(Float, default=0.0)
    chat_qa_tested: Mapped[int] = mapped_column(Integer, default=0)

    # Cost / latency
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Raw scorecard
    scorecard_json: Mapped[dict] = mapped_column(JSON, default=dict)
    result_path: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(tz=timezone.utc),
    )
