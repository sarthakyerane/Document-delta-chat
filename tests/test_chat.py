"""
delta-chat · tests/test_chat.py
Tests for the chat layer — grounded answers and insufficient-grounding path.
These tests mock the LLM and ChromaDB to be deterministic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.chat.answer import AnswerEngine, GroundedAnswer
from src.chat.index import RetrievedChunk


@pytest.fixture
def mock_chunks():
    return [
        RetrievedChunk(
            text="[dimension] 42.5 mm on page 1. Changed to 45.0 mm in Rev B.",
            source_type="delta",
            source_pid="pair01_a+pair01_b",
            page_index=0,
            element_id=None,
            delta_id="delta_abc",
            bbox=None,
            run_id="test-run",
            similarity=0.92,
        )
    ]


class TestAnswerEngine:
    @patch("src.chat.answer.ChromaIndex")
    @patch("src.chat.answer.LLMClient")
    def test_grounded_answer_has_citations(
        self, mock_llm_cls, mock_chroma_cls, mock_chunks
    ):
        """A grounded answer must have at least one citation."""
        # Mock retrieval
        mock_chroma = MagicMock()
        mock_chroma.retrieve.return_value = mock_chunks
        mock_chroma_cls.return_value = mock_chroma

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.embed_query.return_value = [0.1] * 768
        mock_llm.complete_sync.return_value = MagicMock(
            content="The flange diameter changed from 42.5 mm to 45.0 mm. [Citation 1]"
        )
        mock_llm_cls.return_value = mock_llm

        # Mock cache (miss)
        with patch("src.chat.answer._cache") as mock_cache:
            mock_cache.get.return_value = None
            engine = AnswerEngine(run_id="test-run")
            engine.llm = mock_llm
            engine.index = mock_chroma
            answer = engine.answer("What changed?")

        assert len(answer.citations) > 0
        assert not answer.insufficient_grounding
        assert answer.answer  # non-empty

    @patch("src.chat.answer.ChromaIndex")
    @patch("src.chat.answer.LLMClient")
    def test_insufficient_grounding_on_no_chunks(
        self, mock_llm_cls, mock_chroma_cls
    ):
        """If retrieval returns nothing, emit insufficient_grounding=True."""
        mock_chroma = MagicMock()
        mock_chroma.retrieve.return_value = []
        mock_chroma_cls.return_value = mock_chroma

        mock_llm = MagicMock()
        mock_llm.embed_query.return_value = [0.1] * 768
        mock_llm_cls.return_value = mock_llm

        with patch("src.chat.answer._cache") as mock_cache:
            mock_cache.get.return_value = None
            engine = AnswerEngine(run_id="test-run")
            engine.llm = mock_llm
            engine.index = mock_chroma
            answer = engine.answer("What is the capital of France?")

        assert answer.insufficient_grounding is True
        assert len(answer.citations) == 0

    @patch("src.chat.answer.ChromaIndex")
    @patch("src.chat.answer.LLMClient")
    def test_cache_hit_returns_instantly(
        self, mock_llm_cls, mock_chroma_cls
    ):
        """A cache hit should not invoke the LLM."""
        mock_llm = MagicMock()
        mock_llm.embed_query.return_value = [0.1] * 768
        mock_llm_cls.return_value = mock_llm

        cached_answer = {
            "answer": "Cached: dimension changed.",
            "citations": [{"label": "[Delta]", "source_type": "delta",
                           "source_pid": "test", "page_index": 0,
                           "similarity": 0.95, "snippet": "42.5 mm → 45.0 mm"}],
            "insufficient_grounding": False,
        }

        with patch("src.chat.answer._cache") as mock_cache:
            from src.chat.cache import CacheHit
            mock_cache.get.return_value = CacheHit(
                answer=cached_answer, similarity=0.93, query="What changed?"
            )
            engine = AnswerEngine(run_id="test-run")
            engine.llm = mock_llm
            answer = engine.answer("What changed?")

        assert answer.from_cache is True
        assert answer.cache_similarity == pytest.approx(0.93)
        mock_llm.complete_sync.assert_not_called()


class TestCitationLabel:
    def test_pid_citation_label(self):
        chunk = RetrievedChunk(
            text="sample", source_type="pid_a", source_pid="rev_a.pdf",
            page_index=2, element_id=None, delta_id=None, bbox=None,
            run_id="r1", similarity=0.90,
        )
        label = chunk.citation_label
        assert "PID A" in label
        assert "rev_a.pdf" in label
        assert "3" in label  # page_index 2 → page 3

    def test_delta_citation_label(self):
        chunk = RetrievedChunk(
            text="delta chunk", source_type="delta", source_pid="a+b",
            page_index=1, element_id=None, delta_id="abc123", bbox=None,
            run_id="r1", similarity=0.88,
        )
        label = chunk.citation_label
        assert "Delta Report" in label
        assert "abc123" in label
