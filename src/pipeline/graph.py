"""
delta-chat · src/pipeline/graph.py
══════════════════════════════════════════════════════════════════════════════
LangGraph pipeline graph — ingest → delta → report → index.

Why LangGraph for a linear pipeline:
  Node boundaries make stage spans explicit — each node is one OTel span.
  A conditional edge routes errors to an "error_exit" node rather than
  letting exceptions propagate uncontrolled, ensuring every failure is
  visible in the trace.  Future work (retry loops, human-in-the-loop
  review of low-confidence deltas) is straightforward to add as edges.

Node sequence:
  ingest_node → delta_node → report_node → index_node → [done]
                ↓ (on error at any stage)
             error_node

State flows through all nodes — no global mutable state.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.pipeline.state import PipelineState
from src.observability.logging import get_logger, set_request_id
from src.observability.tracing import RequestTracer

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────

def ingest_node(state: PipelineState) -> PipelineState:
    """Resolve PIDs → canonical Documents for both revisions."""
    tracer: RequestTracer = state["tracer"]
    set_request_id(state["request_id"])

    with tracer.span("pipeline.ingest") as span:
        t0 = time.time()
        try:
            from src.ingest.base import AdapterRegistry, resolve_format_type
            import src.ingest  # trigger adapter registration

            for side, pid, path in [
                ("a", state["pid_a"], state["path_a"]),
                ("b", state["pid_b"], state["path_b"]),
            ]:
                fmt = resolve_format_type(path)
                adapter = AdapterRegistry.get(fmt)
                doc = adapter.ingest(
                    pid=pid,
                    source_path=path,
                    tracer=tracer,
                    revision_label=state.get(f"revision_label_{side}"),
                )
                state[f"doc_{side}"] = doc
                log.info(f"pipeline.ingest.{side}.done",
                         pid=pid, format=fmt, pages=doc.page_count)

        except Exception as exc:
            state.setdefault("errors", []).append({
                "stage": "ingest",
                "error": type(exc).__name__,
                "message": str(exc),
            })
            tracer.end_span(span, error=exc)
            raise

        state.setdefault("stage_timings", {})["ingest"] = (time.time() - t0) * 1000

    return state


def delta_node(state: PipelineState) -> PipelineState:
    """Compute structured delta between the two canonical Documents."""
    tracer: RequestTracer = state["tracer"]

    with tracer.span("pipeline.delta") as span:
        t0 = time.time()
        try:
            from src.delta.engine import DeltaEngine

            engine = DeltaEngine(tracer=tracer)
            report = engine.compute(state["doc_a"], state["doc_b"], state["run_id"])
            state["delta_report"] = report

            log.info("pipeline.delta.done",
                     total=report.summary["total_changes"],
                     added=report.summary["added"],
                     removed=report.summary["removed"],
                     modified=report.summary["modified"])

        except Exception as exc:
            state.setdefault("errors", []).append({
                "stage": "delta",
                "error": type(exc).__name__,
                "message": str(exc),
            })
            tracer.end_span(span, error=exc)
            raise

        state.setdefault("stage_timings", {})["delta"] = (time.time() - t0) * 1000

    return state


def report_node(state: PipelineState) -> PipelineState:
    """Render delta report to JSON + Markdown and write to output directory."""
    tracer: RequestTracer = state["tracer"]

    with tracer.span("pipeline.report") as span:
        t0 = time.time()
        try:
            from src.delta.report import DeltaReportRenderer

            renderer = DeltaReportRenderer()
            bundle = renderer.render(state["delta_report"], state["output_dir"])

            state["delta_json_path"] = bundle.json_path
            state["delta_md_path"] = bundle.markdown_path
            state["delta_md_text"] = bundle.markdown_text

            log.info("pipeline.report.done",
                     json=bundle.json_path, md=bundle.markdown_path)

        except Exception as exc:
            state.setdefault("errors", []).append({
                "stage": "report",
                "error": type(exc).__name__,
                "message": str(exc),
            })
            tracer.end_span(span, error=exc)
            raise

        state.setdefault("stage_timings", {})["report"] = (time.time() - t0) * 1000

    return state


def index_node(state: PipelineState) -> PipelineState:
    """Index PID A, PID B, and delta report into ChromaDB."""
    tracer: RequestTracer = state["tracer"]

    with tracer.span("pipeline.index") as span:
        t0 = time.time()
        try:
            from src.chat.index import ChromaIndex

            idx = ChromaIndex(run_id=state["run_id"])
            n_a = idx.index_document(state["doc_a"], "pid_a", tracer=tracer)
            n_b = idx.index_document(state["doc_b"], "pid_b", tracer=tracer)
            n_d = idx.index_delta_report(state["delta_report"], tracer=tracer)

            state["indexed"] = True
            state["index_stats"] = {"pid_a": n_a, "pid_b": n_b, "delta": n_d}

            log.info("pipeline.index.done",
                     pid_a_chunks=n_a, pid_b_chunks=n_b, delta_chunks=n_d)

        except Exception as exc:
            state.setdefault("errors", []).append({
                "stage": "index",
                "error": type(exc).__name__,
                "message": str(exc),
            })
            tracer.end_span(span, error=exc)
            log.warning("pipeline.index.failed_but_continuing", error=str(exc))
            state["indexed"] = False
            # Index failure is non-fatal — report is still written

        state.setdefault("stage_timings", {})["index"] = (time.time() - t0) * 1000

    return state


def error_node(state: PipelineState) -> PipelineState:
    """Terminal error node — logs all errors in state and exports trace."""
    tracer: RequestTracer = state["tracer"]
    errors = state.get("errors", [])
    log.error("pipeline.failed", errors=errors, run_id=state.get("run_id"))
    trace_path = tracer.export()
    log.info("pipeline.trace.exported", path=trace_path)
    return state


def _has_errors(state: PipelineState) -> str:
    """Conditional edge: route to error_node if errors exist, else continue."""
    return "error" if state.get("errors") else "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline_graph() -> Any:
    """Assemble and compile the LangGraph pipeline."""
    g = StateGraph(PipelineState)

    g.add_node("ingest", ingest_node)
    g.add_node("delta", delta_node)
    g.add_node("report", report_node)
    g.add_node("index", index_node)
    g.add_node("error_exit", error_node)

    g.set_entry_point("ingest")

    g.add_conditional_edges("ingest", _has_errors, {"ok": "delta", "error": "error_exit"})
    g.add_conditional_edges("delta", _has_errors, {"ok": "report", "error": "error_exit"})
    g.add_conditional_edges("report", _has_errors, {"ok": "index", "error": "error_exit"})
    g.add_edge("index", END)
    g.add_edge("error_exit", END)

    return g.compile()


_compiled_graph = None


def get_pipeline_graph() -> Any:
    """Singleton compiled graph — safe to call multiple times."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_pipeline_graph()
    return _compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pid_a: str, path_a: str,
    pid_b: str, path_b: str,
    output_dir: Optional[str] = None,
    revision_label_a: Optional[str] = None,
    revision_label_b: Optional[str] = None,
    request_id: Optional[str] = None,
) -> PipelineState:
    """
    Execute the full ingest → delta → report → index pipeline.
    Returns the final state (contains delta_report, paths, timings, errors).
    Writes a trace file to traces/<request_id>.json.
    """
    from typing import Optional

    run_id = str(uuid.uuid4())
    request_id = request_id or run_id

    tracer = RequestTracer(request_id=request_id)
    set_request_id(request_id)

    if output_dir is None:
        output_dir = os.path.join("data", "runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    initial_state: PipelineState = {
        "pid_a": pid_a,
        "pid_b": pid_b,
        "path_a": path_a,
        "path_b": path_b,
        "run_id": run_id,
        "request_id": request_id,
        "revision_label_a": revision_label_a,
        "revision_label_b": revision_label_b,
        "tracer": tracer,
        "errors": [],
        "stage_timings": {},
        "output_dir": output_dir,
        "indexed": False,
    }

    log.info("pipeline.start",
             run_id=run_id, pid_a=pid_a, pid_b=pid_b,
             path_a=path_a, path_b=path_b)

    graph = get_pipeline_graph()
    final_state = graph.invoke(initial_state)

    # Always export trace (even on error)
    trace_path = tracer.export()
    log.info("pipeline.complete",
             run_id=run_id,
             timings=final_state.get("stage_timings", {}),
             errors=len(final_state.get("errors", [])),
             trace_path=trace_path)

    return final_state


from typing import Optional  # noqa: E402 (needed for run_pipeline signature above)
