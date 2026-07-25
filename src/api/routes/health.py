"""delta-chat · src/api/routes/health.py"""
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.config import get_settings
from src.db.session import health_check as db_health

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Service health check — checks DB, Redis, ChromaDB connectivity."""
    # DB
    db_ok = db_health()

    # Redis
    redis_ok = False
    if settings.redis_enabled:
        try:
            import redis
            r = redis.from_url(settings.redis_url)
            r.ping()
            redis_ok = True
        except Exception:
            pass

    # ChromaDB (embedded — just check if persist dir exists)
    import os
    chroma_ok = os.path.exists(settings.chroma_persist_dir)

    # LLM providers available
    available: list[str] = []
    from src.chat.llm import GroqProvider, GeminiProvider, OllamaProvider
    for p in [GroqProvider(), GeminiProvider(), OllamaProvider()]:
        if p.available():
            available.append(p.name)

    overall = "healthy" if db_ok else "degraded"
    return HealthResponse(
        status=overall, db=db_ok, redis=redis_ok,
        chroma=chroma_ok, llm_providers=available,
    )
