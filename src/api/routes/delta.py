"""delta-chat · src/api/routes/delta.py"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/delta/{run_id}")
async def get_delta_json(run_id: str):
    """Retrieve the delta report JSON for a completed run."""
    path = os.path.join("data", "runs", run_id, "delta_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Delta report not found for run {run_id}")
    return FileResponse(path, media_type="application/json")


@router.get("/delta/{run_id}/markdown")
async def get_delta_markdown(run_id: str):
    """Retrieve the delta report Markdown for a completed run."""
    path = os.path.join("data", "runs", run_id, "delta_report.md")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Delta Markdown not found for run {run_id}")
    return FileResponse(path, media_type="text/markdown")


@router.get("/delta/{run_id}/summary")
async def get_delta_summary(run_id: str):
    """Return just the summary section of the delta report."""
    path = os.path.join("data", "runs", run_id, "delta_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(path) as f:
        data = json.load(f)
    return {"run_id": run_id, "summary": data.get("summary", {})}
