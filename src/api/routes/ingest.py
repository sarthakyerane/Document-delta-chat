"""delta-chat · src/api/routes/ingest.py"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.api.schemas import IngestRequest, IngestResponse
from src.ingest.base import resolve_format_type
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer
from src.pipeline.graph import run_pipeline

router = APIRouter()
log = get_logger(__name__)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    """
    Full pipeline: ingest two document revisions → compute delta → index for chat.
    Returns run_id for subsequent /chat and /delta calls.
    """
    # Validate paths exist
    for label, path in [("path_a", request.path_a), ("path_b", request.path_b)]:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"{label} not found: {path}")

    request_id = str(uuid.uuid4())

    try:
        # Detect formats
        fmt_a = resolve_format_type(request.path_a)
        fmt_b = resolve_format_type(request.path_b)

        # Run pipeline (blocking in FastAPI for now — wrap in thread pool for prod)
        final_state = run_pipeline(
            pid_a=request.pid_a, path_a=request.path_a,
            pid_b=request.pid_b, path_b=request.path_b,
            revision_label_a=request.revision_label_a,
            revision_label_b=request.revision_label_b,
            request_id=request_id,
        )

        report = final_state.get("delta_report")
        summary = report.summary if report else {}
        errors = final_state.get("errors", [])
        status = "error" if errors else "complete"

        # Persist run to DB
        try:
            from src.db.session import get_session_factory
            from src.db.models import Run
            factory = get_session_factory()
            with factory() as session:
                run = Run(
                    id=final_state["run_id"],
                    request_id=request_id,
                    pid_a=request.pid_a, pid_b=request.pid_b,
                    path_a=request.path_a, path_b=request.path_b,
                    format_a=fmt_a, format_b=fmt_b,
                    status=status,
                    total_changes=summary.get("total_changes", 0),
                    added=summary.get("added", 0),
                    removed=summary.get("removed", 0),
                    modified=summary.get("modified", 0),
                    avg_confidence=summary.get("avg_confidence", 0.0),
                    delta_json_path=final_state.get("delta_json_path", ""),
                    delta_md_path=final_state.get("delta_md_path", ""),
                    indexed=final_state.get("indexed", False),
                    error_message=str(errors) if errors else "",
                    stage_timings=final_state.get("stage_timings", {}),
                    completed_at=datetime.now(tz=timezone.utc),
                )
                session.add(run)
                session.commit()
        except Exception as db_err:
            log.warning("ingest.db_persist_failed", error=str(db_err))

        trace_path = os.path.join(
            "traces", f"{request_id}.json"
        )

        return IngestResponse(
            run_id=final_state["run_id"],
            request_id=request_id,
            status=status,
            pid_a=request.pid_a,
            pid_b=request.pid_b,
            format_a=fmt_a,
            format_b=fmt_b,
            elements_a=final_state.get("doc_a", None) and
                        final_state["doc_a"].element_count or 0,
            elements_b=final_state.get("doc_b", None) and
                        final_state["doc_b"].element_count or 0,
            total_changes=summary.get("total_changes", 0),
            added=summary.get("added", 0),
            removed=summary.get("removed", 0),
            modified=summary.get("modified", 0),
            avg_confidence=summary.get("avg_confidence", 0.0),
            delta_json_path=final_state.get("delta_json_path", ""),
            delta_md_path=final_state.get("delta_md_path", ""),
            indexed=final_state.get("indexed", False),
            stage_timings=final_state.get("stage_timings", {}),
            errors=errors,
            trace_path=trace_path,
        )

    except Exception as exc:
        log.error("ingest.route.error", error=str(exc), request_id=request_id)
        raise HTTPException(status_code=500, detail=str(exc))
