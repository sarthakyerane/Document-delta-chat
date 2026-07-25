"""delta-chat · src/api/routes/chat.py"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse, CitationResponse
from src.chat.answer import AnswerEngine
from src.chat.cache import SemanticCache
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

router = APIRouter()
log = get_logger(__name__)

# Shared cache instance (process-level, tracks hit rate across calls)
_cache = SemanticCache()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Grounded chat — answer a question over PID A, PID B, and the delta report.
    Every answer carries citations back to source documents + similarity scores.
    """
    request_id = str(uuid.uuid4())
    tracer = RequestTracer(request_id=request_id)

    try:
        engine = AnswerEngine(run_id=request.run_id, tracer=tracer)
        grounded = engine.answer(request.query)

        tracer.export()

        return ChatResponse(
            run_id=request.run_id,
            query=request.query,
            answer=grounded.answer,
            citations=[
                CitationResponse(
                    label=c.label,
                    source_type=c.source_type,
                    source_pid=c.source_pid,
                    page_index=c.page_index,
                    similarity=c.similarity,
                    snippet=c.snippet,
                )
                for c in grounded.citations
            ],
            from_cache=grounded.from_cache,
            cache_similarity=grounded.cache_similarity,
            insufficient_grounding=grounded.insufficient_grounding,
            cache_stats=engine.cache.stats(),
        )
    except Exception as exc:
        log.error("chat.route.error", error=str(exc), run_id=request.run_id)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/chat/cache-stats")
async def cache_stats():
    """Return current cache hit rate for observability."""
    return _cache.stats()
