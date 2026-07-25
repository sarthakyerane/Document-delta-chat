"""
delta-chat · src/chat/index.py
══════════════════════════════════════════════════════════════════════════════
ChromaDB indexing and retrieval.

Three SEPARATE collections per run (never blended — citation needs source):
    dc_{run_id}_pid_a   — chunks from document A
    dc_{run_id}_pid_b   — chunks from document B
    dc_{run_id}_delta   — delta report chunks

Each chunk metadata:
    source_pid: str         — which PID this came from
    source_type: str        — "pid_a" | "pid_b" | "delta"
    page_index: int         — page number
    element_id: str         — element hash (for citation link-back)
    bbox: str               — "x0,y0,x1,y1" stringified
    delta_id: str           — for delta chunks
    run_id: str

Why three collections: keeping them separate lets the retriever query all
three in parallel and return cited results tagged by source, satisfying the
grounding requirement ("citations back to a PID + location or delta entry").
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import chromadb

from src.canonical.model import Document, DeltaReport
from src.chat.llm import LLMClient
from src.config import get_settings
from src.delta.report import DeltaReportRenderer
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    text: str
    source_type: str   # "pid_a" | "pid_b" | "delta"
    source_pid: str
    page_index: int
    element_id: Optional[str]
    delta_id: Optional[str]
    bbox: Optional[str]
    run_id: str
    similarity: float

    @property
    def citation_label(self) -> str:
        """Human-readable citation string for LLM prompt and answer."""
        if self.source_type == "delta":
            return f"[Delta Report, page {self.page_index + 1}, delta_id={self.delta_id}]"
        pid_label = "PID A" if self.source_type == "pid_a" else "PID B"
        return f"[{pid_label} ({self.source_pid}), page {self.page_index + 1}]"


class ChromaIndex:
    """
    Manages ChromaDB collections for a single ingest run.
    ChromaDB runs in embedded/persistent mode — no server container needed.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._llm = LLMClient()

    def _collection_name(self, source_type: str) -> str:
        # ChromaDB names must be [3, 512] chars and valid identifiers
        safe_id = self.run_id.replace("-", "_")[:40]
        return f"dc_{safe_id}_{source_type}"

    def _get_or_create(self, source_type: str) -> chromadb.Collection:
        name = self._collection_name(source_type)
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_document(
        self,
        doc: Document,
        source_type: str,  # "pid_a" or "pid_b"
        tracer: Optional[RequestTracer] = None,
    ) -> int:
        """Index all elements of a document into the appropriate collection."""
        span = tracer.start_span(f"index.{source_type}",
                                 {"pid": doc.pid, "elements": doc.element_count}) if tracer else None
        collection = self._get_or_create(source_type)
        indexed = 0

        # Chunk by page (group all elements on a page into one text chunk)
        for page in doc.pages:
            elems = page.all_elements
            if not elems:
                continue

            # One chunk per page (aggregated text for better retrieval)
            chunk_parts = []
            for elem in elems:
                if elem.text or elem.raw_value:
                    chunk_parts.append(
                        f"[{elem.type}] {elem.display_text}"
                    )

            if not chunk_parts:
                continue

            chunk_text = "\n".join(chunk_parts)
            chunk_id = f"{source_type}_{doc.pid}_{page.page_index}"

            # Embedding via Google text-embedding-004
            try:
                embedding = self._llm.embed_query(chunk_text)
            except Exception as e:
                log.warning("index.embed_failed",
                            pid=doc.pid, page=page.page_index, error=str(e))
                continue

            meta = {
                "source_pid": doc.pid,
                "source_type": source_type,
                "page_index": page.page_index,
                "page_label": page.page_label or str(page.page_index + 1),
                "run_id": self.run_id,
                "element_ids": json.dumps([e.element_id for e in elems[:20]]),
                "chunk_type": "page",
            }

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[meta],
            )
            indexed += 1

        log.info("index.document.complete",
                 pid=doc.pid, source_type=source_type, chunks=indexed)

        if span and tracer:
            tracer.end_span(span, attributes={"chunks_indexed": indexed})

        return indexed

    def index_delta_report(
        self,
        report: DeltaReport,
        tracer: Optional[RequestTracer] = None,
    ) -> int:
        """Index delta report chunks into the delta collection."""
        span = tracer.start_span("index.delta",
                                 {"items": len(report.items)}) if tracer else None
        collection = self._get_or_create("delta")
        renderer = DeltaReportRenderer()
        chunks = renderer.render_summary_for_indexing(report)
        indexed = 0

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"delta_{self.run_id}_{i}"
            # Determine page_index from chunk (first delta item's page in chunk)
            page_index = 0
            delta_id = ""

            try:
                embedding = self._llm.embed_query(chunk_text)
            except Exception as e:
                log.warning("index.delta.embed_failed", chunk_idx=i, error=str(e))
                continue

            meta = {
                "source_pid": f"{report.pid_a}+{report.pid_b}",
                "source_type": "delta",
                "page_index": i,  # chunk index as page
                "run_id": self.run_id,
                "delta_id": delta_id,
                "chunk_type": "delta_page" if i > 0 else "delta_summary",
            }

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk_text],
                metadatas=[meta],
            )
            indexed += 1

        log.info("index.delta.complete", chunks=indexed, run_id=self.run_id)
        if span and tracer:
            tracer.end_span(span, attributes={"chunks_indexed": indexed})
        return indexed

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        query_embedding: Optional[list[float]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        tracer: Optional[RequestTracer] = None,
    ) -> list[RetrievedChunk]:
        """
        Query all three collections (pid_a, pid_b, delta) in parallel and
        return merged, sorted results above the similarity threshold.

        Each result carries its source metadata for citation generation.
        """
        span = tracer.start_span("retrieve",
                                 {"query_len": len(query)}) if tracer else None
        top_k = top_k or settings.retrieval_top_k
        sim_threshold = similarity_threshold or settings.retrieval_similarity_threshold

        if query_embedding is None:
            try:
                query_embedding = self._llm.embed_query(query)
            except Exception as e:
                log.error("retrieve.embed_query_failed", error=str(e))
                if span and tracer:
                    tracer.end_span(span, error=e)
                return []

        all_chunks: list[RetrievedChunk] = []

        for source_type in ["pid_a", "pid_b", "delta"]:
            try:
                collection = self._get_or_create(source_type)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                docs = results["documents"][0] if results["documents"] else []
                metas = results["metadatas"][0] if results["metadatas"] else []
                distances = results["distances"][0] if results["distances"] else []

                for doc_text, meta, dist in zip(docs, metas, distances):
                    # ChromaDB cosine distance = 1 - cosine_similarity
                    similarity = 1.0 - dist
                    if similarity < sim_threshold:
                        continue
                    all_chunks.append(RetrievedChunk(
                        text=doc_text,
                        source_type=meta.get("source_type", source_type),
                        source_pid=meta.get("source_pid", ""),
                        page_index=meta.get("page_index", 0),
                        element_id=None,
                        delta_id=meta.get("delta_id"),
                        bbox=None,
                        run_id=meta.get("run_id", self.run_id),
                        similarity=round(similarity, 4),
                    ))
            except Exception as e:
                log.warning("retrieve.collection_failed",
                            source_type=source_type, error=str(e))

        # Sort by similarity descending
        all_chunks.sort(key=lambda c: c.similarity, reverse=True)

        log.info("retrieve.complete",
                 query_len=len(query),
                 results=len(all_chunks),
                 threshold=sim_threshold)

        if span and tracer:
            tracer.end_span(span, attributes={
                "retrieved": len(all_chunks),
                "from_pid_a": sum(1 for c in all_chunks if c.source_type == "pid_a"),
                "from_pid_b": sum(1 for c in all_chunks if c.source_type == "pid_b"),
                "from_delta": sum(1 for c in all_chunks if c.source_type == "delta"),
            })

        return all_chunks

    def delete_run(self) -> None:
        """Clean up all collections for this run (e.g., after eval)."""
        for source_type in ["pid_a", "pid_b", "delta"]:
            try:
                self._client.delete_collection(self._collection_name(source_type))
            except Exception:
                pass
