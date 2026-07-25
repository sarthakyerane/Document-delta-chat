"""
delta-chat · tests/conftest.py
Shared pytest fixtures.
"""
from __future__ import annotations

import os
import pytest

from src.canonical.model import BoundingBox, Element, Page, Block, Document
from src.observability.tracing import RequestTracer


@pytest.fixture
def tracer() -> RequestTracer:
    return RequestTracer(request_id="test-request-id")


@pytest.fixture
def sample_bbox() -> BoundingBox:
    return BoundingBox(x0=100, y0=200, x1=300, y1=220, page_width=595, page_height=842)


@pytest.fixture
def sample_element(sample_bbox) -> Element:
    eid = Element.make_id(0, "dimension", sample_bbox, "42.5 mm")
    return Element(
        element_id=eid,
        type="dimension",
        bbox=sample_bbox,
        text="42.5 mm",
        raw_value="42.5 mm",
        numeric_value=42.5,
        unit="mm",
        confidence=1.0,
    )


@pytest.fixture
def sample_doc_a() -> Document:
    """Minimal document A for alignment/delta tests."""
    bbox_a = BoundingBox(x0=100, y0=200, x1=300, y1=220, page_width=595, page_height=842)
    e1 = Element(
        element_id=Element.make_id(0, "dimension", bbox_a, "42.5 mm"),
        type="dimension", bbox=bbox_a,
        text="42.5 mm", raw_value="42.5 mm", numeric_value=42.5, unit="mm", confidence=1.0,
    )
    bbox_a2 = BoundingBox(x0=100, y0=250, x1=300, y1=270, page_width=595, page_height=842)
    e2 = Element(
        element_id=Element.make_id(0, "note", bbox_a2, "See drawing 42"),
        type="note", bbox=bbox_a2,
        text="See drawing 42", confidence=1.0,
    )
    block = Block(block_id="blk_a", type="text_block", bbox=bbox_a, elements=[e1, e2])
    page = Page(page_index=0, page_label="1", width=595, height=842, blocks=[block])
    return Document(
        pid="doc_a", format="pdf_native",
        revision_label="Rev A", page_count=1, pages=[page],
        source_path="test_a.pdf",
    )


@pytest.fixture
def sample_doc_b() -> Document:
    """Document B — same structure as A but with changed dimension and added note."""
    # Changed dimension: 42.5 → 45.0
    bbox_b1 = BoundingBox(x0=100, y0=200, x1=300, y1=220, page_width=595, page_height=842)
    e1 = Element(
        element_id=Element.make_id(0, "dimension", bbox_b1, "45.0 mm"),
        type="dimension", bbox=bbox_b1,
        text="45.0 mm", raw_value="45.0 mm", numeric_value=45.0, unit="mm", confidence=1.0,
    )
    # Same note (unchanged)
    bbox_b2 = BoundingBox(x0=100, y0=250, x1=300, y1=270, page_width=595, page_height=842)
    e2 = Element(
        element_id=Element.make_id(0, "note", bbox_b2, "See drawing 42"),
        type="note", bbox=bbox_b2,
        text="See drawing 42", confidence=1.0,
    )
    # New added element
    bbox_b3 = BoundingBox(x0=100, y0=300, x1=350, y1=320, page_width=595, page_height=842)
    e3 = Element(
        element_id=Element.make_id(0, "note", bbox_b3, "CAUTION: Min torque 45 Nm"),
        type="note", bbox=bbox_b3,
        text="CAUTION: Min torque 45 Nm", confidence=1.0,
    )
    block = Block(block_id="blk_b", type="text_block", bbox=bbox_b1, elements=[e1, e2, e3])
    page = Page(page_index=0, page_label="1", width=595, height=842, blocks=[block])
    return Document(
        pid="doc_b", format="pdf_native",
        revision_label="Rev B", page_count=1, pages=[page],
        source_path="test_b.pdf",
    )
