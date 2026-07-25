"""
delta-chat · src/chat/answer.py
══════════════════════════════════════════════════════════════════════════════
Grounded answer generation with citations + "insufficient grounding" path.

Design contract:
  1. Every answer carries explicit citations back to PID+page+location
     or a delta entry — never naked LLM generation.
  2. If retrieval returns nothing above the similarity threshold, the system
     emits an explicit "insufficient grounding" response rather than letting
     the LLM hallucinate.  This path is tested in eval.
  3. Redis semantic cache wraps the retrieval+LLM step.  Cache hit → instant
     response; cache miss → full pipeline.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from src.canonical.model import DeltaReport
from src.chat.cache import SemanticCache
from src.chat.index import ChromaIndex, RetrievedChunk
from src.chat.llm import LLMClient, LLMResponse
from src.config import get_settings
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()

# ── Singleton cache instance (shared across chat sessions in a process) ────────
_cache = SemanticCache()


@dataclass
class Citation:
    label: str          # e.g. "[PID A (rev_a.pdf), page 3]"
    source_type: str    # "pid_a" | "pid_b" | "delta"
    source_pid: str
    page_index: int
    similarity: float
    snippet: str        # first 120 chars of the retrieved chunk


@dataclass
class GroundedAnswer:
    query: str
    answer: str
    citations: list[Citation]
    from_cache: bool = False
    cache_similarity: Optional[float] = None
    insufficient_grounding: bool = False
    run_id: str = ""


_GROUNDED_SYSTEM_PROMPT = """You are a precise document analysis assistant for engineering and technical documents.

You are given context retrieved from two document revisions (PID A and PID B) and a delta report describing what changed between them.

Rules:
1. Answer ONLY from the provided context. Do not use general knowledge or fabricate details.
2. Every factual claim must cite its source using [Citation N] notation.
3. If the context does not contain enough information to answer the question, say so explicitly.
4. Be specific: mention page numbers, element types, and values when available.
5. For "what changed" questions, describe changes concisely with old and new values.
6. If asked about something not in the context, say: "The retrieved context does not contain information about [topic]."

NEVER make up measurements, dimensions, notes, or any document content."""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Citation {i}] {chunk.citation_label} (similarity={chunk.similarity:.3f})"
        )
        lines.append(chunk.text[:800])  # cap per chunk to respect token budget
        lines.append("")
    return "\n".join(lines)


class AnswerEngine:
    """
    Orchestrates: cache lookup → retrieve → LLM → cache store → return.
    """

    def __init__(self, run_id: str, tracer: Optional[RequestTracer] = None):
        self.run_id = run_id
        self.tracer = tracer
        self.index = ChromaIndex(run_id)
        self.llm = LLMClient(tracer=tracer)
        self.cache = _cache

    def answer(self, query: str) -> GroundedAnswer:
        """Full grounded answer pipeline with tracing."""
        span = self.tracer.start_span(
            "chat.answer", {"query_len": len(query), "run_id": self.run_id}
        ) if self.tracer else None

        # ── Step 1: Embed query ────────────────────────────────────────────────
        try:
            query_embedding = self.llm.embed_query(query)
        except Exception as e:
            log.error("answer.embed_query_failed", error=str(e))
            result = GroundedAnswer(
                query=query, run_id=self.run_id,
                answer="Unable to process query: embedding service unavailable.",
                citations=[], insufficient_grounding=True,
            )
            if span and self.tracer:
                self.tracer.end_span(span, error=e)
            return result

        # ── Step 2: Cache lookup ───────────────────────────────────────────────
        hit = self.cache.get(query_embedding, self.run_id)
        if hit:
            log.info("answer.cache_hit",
                     similarity=hit.similarity, run_id=self.run_id)
            ans_data = hit.answer
            result = GroundedAnswer(
                query=query,
                answer=ans_data["answer"],
                citations=[Citation(**c) for c in ans_data.get("citations", [])],
                from_cache=True,
                cache_similarity=hit.similarity,
                insufficient_grounding=ans_data.get("insufficient_grounding", False),
                run_id=self.run_id,
            )
            if span and self.tracer:
                self.tracer.end_span(span, attributes={"source": "cache"})
            return result

        # ── Step 3: Retrieve ───────────────────────────────────────────────────
        chunks = self.index.retrieve(
            query=query,
            query_embedding=query_embedding,
            tracer=self.tracer,
        )

        if not chunks:
            log.info("answer.insufficient_grounding", query=query[:80])
            result = GroundedAnswer(
                query=query, run_id=self.run_id,
                answer=(
                    "I was unable to find relevant information in the indexed documents "
                    f"for your query: '{query}'. "
                    "Please try rephrasing or asking about specific pages, elements, or changes."
                ),
                citations=[],
                insufficient_grounding=True,
            )
            if span and self.tracer:
                self.tracer.end_span(span, attributes={"source": "insufficient_grounding"})
            return result

        # ── Step 4: LLM grounded generation ───────────────────────────────────
        context_block = _build_context_block(chunks[:settings.retrieval_top_k * 2])
        messages = [
            {"role": "system", "content": _GROUNDED_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context from indexed documents:\n\n{context_block}\n\n"
                f"Question: {query}\n\n"
                "Answer the question using only the context above. "
                "Use [Citation N] to cite each source. "
                "If context is insufficient, say so explicitly."
            )},
        ]

        llm_response: LLMResponse = self.llm.complete_sync(messages, temperature=0.1)
        answer_text = llm_response.content.strip()

        # ── Step 5: Build citations ────────────────────────────────────────────
        citations = [
            Citation(
                label=chunk.citation_label,
                source_type=chunk.source_type,
                source_pid=chunk.source_pid,
                page_index=chunk.page_index,
                similarity=chunk.similarity,
                snippet=chunk.text[:120],
            )
            for chunk in chunks[:settings.retrieval_top_k]
        ]

        result = GroundedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            insufficient_grounding=False,
            run_id=self.run_id,
        )

        # ── Step 6: Cache the result ───────────────────────────────────────────
        self.cache.set(
            query=query,
            query_embedding=query_embedding,
            run_id=self.run_id,
            answer={
                "answer": answer_text,
                "citations": [
                    {
                        "label": c.label, "source_type": c.source_type,
                        "source_pid": c.source_pid, "page_index": c.page_index,
                        "similarity": c.similarity, "snippet": c.snippet,
                    }
                    for c in citations
                ],
                "insufficient_grounding": False,
            },
        )

        if span and self.tracer:
            self.tracer.end_span(span, attributes={
                "source": "llm",
                "chunks_used": len(chunks),
                "citations": len(citations),
                "cache_stats": str(self.cache.stats()),
            })

        log.info("answer.complete",
                 query_len=len(query),
                 answer_len=len(answer_text),
                 citations=len(citations),
                 cache_stats=self.cache.stats())

        return result

    def format_answer_for_display(self, answer: GroundedAnswer) -> str:
        """Format a GroundedAnswer as readable text for CLI output."""
        lines = []
        if answer.from_cache:
            lines.append(f"⚡ [Cache hit — similarity {answer.cache_similarity:.3f}]")
        if answer.insufficient_grounding:
            lines.append("⚠️  Insufficient grounding — retrieval returned no relevant context.")
        lines.append(f"\n**Answer:**\n{answer.answer}")
        if answer.citations:
            lines.append("\n**Sources:**")
            for i, c in enumerate(answer.citations, 1):
                lines.append(f"  [{i}] {c.label} (sim={c.similarity:.3f})")
                lines.append(f"      …{c.snippet}…")
        return "\n".join(lines)
