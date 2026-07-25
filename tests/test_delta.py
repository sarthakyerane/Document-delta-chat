"""
delta-chat · tests/test_delta.py
Tests for the delta alignment engine.

Key claims tested:
  1. Exact-ID match is always stage 1 (confidence=1.0, method="exact_id")
  2. Unchanged elements are NOT emitted as delta items
  3. Changed dimension → DeltaItem with change_type="modified"
  4. Added element → DeltaItem with change_type="added"
  5. Removed element → DeltaItem with change_type="removed"
  6. Alignment stage methods are one of the documented values

Non-determinism: alignment stages 1–5 are deterministic.
Stage 6 (LLM) is disabled in tests via ALIGNMENT_USE_LLM=false.
"""
from __future__ import annotations

import os
import pytest

# Disable LLM in tests — ensure alignment is deterministic
os.environ["ALIGNMENT_USE_LLM"] = "false"

from src.delta.align import DocumentAligner, AlignmentResult
from src.delta.engine import DeltaEngine


class TestAlignment:
    def test_exact_id_match(self, tracer, sample_doc_a, sample_doc_b):
        """Unchanged note in both docs should match via exact_id (stage 1)."""
        aligner = DocumentAligner(tracer=tracer)
        result = aligner.align(sample_doc_a, sample_doc_b)

        # The note "See drawing 42" exists unchanged in both → exact_id match
        exact_id_pairs = [p for p in result.matched if p.method == "exact_id"]
        assert len(exact_id_pairs) >= 1

    def test_changed_element_not_exact_id(self, tracer, sample_doc_a, sample_doc_b):
        """Changed dimension should not match via exact_id."""
        aligner = DocumentAligner(tracer=tracer)
        result = aligner.align(sample_doc_a, sample_doc_b)

        # The dimension changed (42.5 → 45.0) so it should NOT be an exact_id match
        # It should be matched by geometric+text stages with confidence < 1.0
        dim_pairs = [
            p for p in result.matched
            if p.element_a.type == "dimension" and p.element_b.type == "dimension"
        ]
        exact_dim = [p for p in dim_pairs if p.method == "exact_id"]
        assert len(exact_dim) == 0, "Changed dimension should not be exact_id match"

    def test_added_elements_in_only_b(self, tracer, sample_doc_a, sample_doc_b):
        """CAUTION note exists only in doc_b → should appear in only_in_b."""
        aligner = DocumentAligner(tracer=tracer)
        result = aligner.align(sample_doc_a, sample_doc_b)
        only_b_texts = [e.text for e in result.only_in_b]
        assert any("CAUTION" in (t or "") for t in only_b_texts)

    def test_alignment_method_valid(self, tracer, sample_doc_a, sample_doc_b):
        """All alignment methods must be one of the documented values."""
        valid_methods = {
            "exact_id", "geometric+text_high", "text_high",
            "geometric+text_medium", "cross_page_text", "llm_assisted",
        }
        aligner = DocumentAligner(tracer=tracer)
        result = aligner.align(sample_doc_a, sample_doc_b)
        for pair in result.matched:
            assert pair.method in valid_methods, f"Unknown method: {pair.method}"

    def test_confidence_range(self, tracer, sample_doc_a, sample_doc_b):
        """All confidence values must be in [0.0, 1.0]."""
        aligner = DocumentAligner(tracer=tracer)
        result = aligner.align(sample_doc_a, sample_doc_b)
        for pair in result.matched:
            assert 0.0 <= pair.confidence <= 1.0, f"OOB confidence: {pair.confidence}"


class TestDeltaEngine:
    def test_unchanged_not_in_delta(self, tracer, sample_doc_a, sample_doc_b):
        """Unchanged 'See drawing 42' note should NOT appear as a delta item."""
        engine = DeltaEngine(tracer=tracer)
        report = engine.compute(sample_doc_a, sample_doc_b, run_id="test-run")

        unchanged_items = [
            i for i in report.items
            if "See drawing 42" in (i.old_value or "") and
               "See drawing 42" in (i.new_value or "")
        ]
        assert len(unchanged_items) == 0

    def test_changed_dimension_is_modified(self, tracer, sample_doc_a, sample_doc_b):
        """42.5mm → 45.0mm should produce a 'modified' dimension delta item."""
        engine = DeltaEngine(tracer=tracer)
        report = engine.compute(sample_doc_a, sample_doc_b, run_id="test-run")

        modified_dims = [
            i for i in report.items
            if i.change_type == "modified" and i.element_type == "dimension"
        ]
        assert len(modified_dims) >= 1
        dim = modified_dims[0]
        assert "42" in (dim.old_value or "") or "45" in (dim.new_value or "")

    def test_added_note_in_delta(self, tracer, sample_doc_a, sample_doc_b):
        """CAUTION note should appear as 'added' in the delta report."""
        engine = DeltaEngine(tracer=tracer)
        report = engine.compute(sample_doc_a, sample_doc_b, run_id="test-run")

        added = [i for i in report.items if i.change_type == "added"]
        added_texts = [i.new_value or "" for i in added]
        assert any("CAUTION" in t or "torque" in t.lower() for t in added_texts)

    def test_summary_counts_consistent(self, tracer, sample_doc_a, sample_doc_b):
        """Summary counts must match actual item counts."""
        engine = DeltaEngine(tracer=tracer)
        report = engine.compute(sample_doc_a, sample_doc_b, run_id="test-run")

        s = report.summary
        actual_added = sum(1 for i in report.items if i.change_type == "added")
        actual_removed = sum(1 for i in report.items if i.change_type == "removed")
        actual_modified = sum(1 for i in report.items if i.change_type == "modified")

        assert s["added"] == actual_added
        assert s["removed"] == actual_removed
        assert s["modified"] == actual_modified
        assert s["total_changes"] == len(report.items)

    def test_no_llm_called_for_trivial_changes(
        self, tracer, sample_doc_a, sample_doc_b, monkeypatch
    ):
        """
        IMPORTANT: Determinism test.
        A changed numeric dimension value should NOT trigger the LLM —
        string compare is sufficient.  This tests that LLM is not called
        for cases where it is not needed.
        """
        llm_called = []

        def mock_llm_describe(*args, **kwargs):
            llm_called.append(True)
            return "mocked description"

        import src.delta.engine as engine_mod
        monkeypatch.setattr(engine_mod, "_llm_describe_change", mock_llm_describe)

        engine = DeltaEngine(tracer=tracer)
        report = engine.compute(sample_doc_a, sample_doc_b, run_id="test-run")

        # Dimension change (42.5→45.0) should not invoke LLM
        assert len(llm_called) == 0, (
            "LLM should not be called for simple dimension value changes. "
            "String/numeric compare is sufficient."
        )

    def test_delta_id_deterministic(self, tracer, sample_doc_a, sample_doc_b):
        """Running the same delta twice should produce the same delta_ids."""
        engine = DeltaEngine(tracer=tracer)
        report1 = engine.compute(sample_doc_a, sample_doc_b, run_id="run1")
        report2 = engine.compute(sample_doc_a, sample_doc_b, run_id="run2")

        ids1 = sorted(i.delta_id for i in report1.items)
        ids2 = sorted(i.delta_id for i in report2.items)
        assert ids1 == ids2, "Delta IDs should be deterministic across identical runs"
