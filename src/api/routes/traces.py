"""delta-chat · src/api/routes/traces.py"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/traces/{request_id}")
async def get_trace(request_id: str):
    """Return the full trace JSON for a request."""
    path = os.path.join("traces", f"{request_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Trace {request_id} not found")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(content=data)


@router.get("/traces")
async def list_traces():
    """List all available trace IDs."""
    traces_dir = "traces"
    if not os.path.exists(traces_dir):
        return {"traces": []}
    files = sorted(
        [f.replace(".json", "") for f in os.listdir(traces_dir) if f.endswith(".json")],
        reverse=True,
    )
    return {"traces": files, "count": len(files)}


@router.get("/traces/{request_id}/llm-summary")
async def get_trace_llm_summary(request_id: str):
    """Return LLM cost/token summary from a trace."""
    path = os.path.join("traces", f"{request_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Trace {request_id} not found")
    with open(path) as f:
        data = json.load(f)
    return {
        "request_id": request_id,
        "llm_summary": data.get("llm_summary", {}),
        "llm_calls": data.get("llm_calls", []),
        "total_duration_ms": data.get("total_duration_ms"),
        "errors": data.get("errors", []),
    }
