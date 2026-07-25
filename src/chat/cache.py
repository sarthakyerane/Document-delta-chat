"""
delta-chat · src/chat/cache.py
══════════════════════════════════════════════════════════════════════════════
Redis semantic cache for chat answers.

Pattern reused from Meeting Intelligence Agent (cosine ≥ 0.92 there;
here 0.90 per config, since document queries are slightly more varied).

Storage layout in Redis:
    Key: dc:cache:{run_id}:{cache_entry_id}
    Value: JSON {embedding: [...], answer: {...}, query: "...", created_at: "..."}

Lookup: iterate over all keys for this run_id, compute cosine similarity
against query embedding, return best match above threshold.

Note: O(n) per lookup — acceptable for take-home scale (n typically < 100
queries per session). Production would use Redis Stack vector index (HNSW).
This is documented in the README as a known limitation.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import get_settings
from src.observability.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_CACHE_KEY_PREFIX = "dc:cache"


@dataclass
class CacheHit:
    answer: dict
    similarity: float
    query: str


class SemanticCache:
    """Redis-backed semantic cache with cosine similarity lookup."""

    def __init__(self):
        self._client: Optional[object] = None
        self._enabled = settings.redis_enabled
        self.hits = 0
        self.misses = 0

    def _get_client(self):
        if not self._enabled:
            return None
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(settings.redis_url, decode_responses=True)
                self._client.ping()
            except Exception as e:
                log.warning("cache.redis_connect_failed", error=str(e))
                self._enabled = False
                return None
        return self._client

    def get(
        self,
        query_embedding: list[float],
        run_id: str,
        threshold: Optional[float] = None,
    ) -> Optional[CacheHit]:
        """Return a cached answer if cosine similarity ≥ threshold."""
        r = self._get_client()
        if r is None:
            return None

        threshold = threshold or settings.redis_semantic_cache_threshold
        pattern = f"{_CACHE_KEY_PREFIX}:{run_id}:*"
        try:
            keys = r.keys(pattern)
            if not keys:
                self.misses += 1
                return None

            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return None

            best_sim = 0.0
            best_entry: Optional[dict] = None

            for key in keys:
                raw = r.get(key)
                if not raw:
                    continue
                entry = json.loads(raw)
                c_vec = np.array(entry["embedding"], dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_sim >= threshold and best_entry:
                log.info("cache.hit", similarity=round(best_sim, 4), run_id=run_id)
                self.hits += 1
                return CacheHit(
                    answer=best_entry["answer"],
                    similarity=best_sim,
                    query=best_entry.get("query", ""),
                )
        except Exception as e:
            log.warning("cache.get_failed", error=str(e))

        self.misses += 1
        return None

    def set(
        self,
        query: str,
        query_embedding: list[float],
        run_id: str,
        answer: dict,
    ) -> None:
        """Store a query+answer pair in the cache."""
        r = self._get_client()
        if r is None:
            return

        entry_id = uuid.uuid4().hex[:8]
        key = f"{_CACHE_KEY_PREFIX}:{run_id}:{entry_id}"
        payload = json.dumps({
            "query": query,
            "embedding": query_embedding,
            "answer": answer,
            "created_at": time.time(),
        })
        try:
            r.set(key, payload, ex=settings.redis_cache_ttl)
            log.debug("cache.set", key=key, ttl=settings.redis_cache_ttl)
        except Exception as e:
            log.warning("cache.set_failed", error=str(e))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
            "enabled": self._enabled,
        }
