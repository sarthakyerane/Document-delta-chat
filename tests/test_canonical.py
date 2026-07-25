"""
delta-chat · tests/test_canonical.py
Tests for the canonical model — the central seam.
"""
from __future__ import annotations

import pytest
from src.canonical.model import BoundingBox, Element, Page, Block, Document


def test_bbox_iou_full_overlap():
    a = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    b = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    assert a.iou(b) == pytest.approx(1.0)


def test_bbox_iou_no_overlap():
    a = BoundingBox(x0=0, y0=0, x1=50, y1=50)
    b = BoundingBox(x0=60, y0=60, x1=100, y1=100)
    assert a.iou(b) == pytest.approx(0.0)


def test_bbox_iou_partial():
    a = BoundingBox(x0=0, y0=0, x1=100, y1=100)
    b = BoundingBox(x0=50, y0=50, x1=150, y1=150)
    assert 0.0 < a.iou(b) < 1.0


def test_bbox_clamp_inverted():
    """BoundingBox should auto-correct inverted coordinates."""
    b = BoundingBox(x0=100, y0=200, x1=50, y1=100)  # inverted
    assert b.x0 < b.x1
    assert b.y0 < b.y1


def test_bbox_normalized():
    b = BoundingBox(x0=100, y0=200, x1=200, y1=300, page_width=400, page_height=800)
    norm = b.normalized()
    assert norm.x0 == pytest.approx(0.25)
    assert norm.y0 == pytest.approx(0.25)
    assert norm.x1 == pytest.approx(0.5)


def test_element_id_deterministic():
    """Same inputs → same element_id across calls."""
    bbox = BoundingBox(x0=100, y0=200, x1=300, y1=220)
    id1 = Element.make_id(0, "dimension", bbox, "42.5 mm")
    id2 = Element.make_id(0, "dimension", bbox, "42.5 mm")
    assert id1 == id2


def test_element_id_different_content():
    """Different text → different element_id."""
    bbox = BoundingBox(x0=100, y0=200, x1=300, y1=220)
    id1 = Element.make_id(0, "dimension", bbox, "42.5 mm")
    id2 = Element.make_id(0, "dimension", bbox, "45.0 mm")
    assert id1 != id2


def test_element_id_whitespace_normalized():
    """Leading/trailing whitespace should not change element_id."""
    bbox = BoundingBox(x0=100, y0=200, x1=300, y1=220)
    id1 = Element.make_id(0, "text", bbox, "hello world")
    id2 = Element.make_id(0, "text", bbox, "  hello world  ")
    assert id1 == id2


def test_document_all_elements(sample_doc_a):
    elems = sample_doc_a.all_elements
    assert len(elems) == 2


def test_document_element_count(sample_doc_a):
    assert sample_doc_a.element_count == 2


def test_page_all_elements(sample_doc_a):
    page = sample_doc_a.pages[0]
    assert len(page.all_elements) == 2


def test_element_display_text():
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=10)
    e = Element(
        element_id="abc",
        type="dimension",
        bbox=bbox,
        text=None,
        raw_value="42.5 mm",
    )
    assert e.display_text == "42.5 mm"


def test_document_get_page(sample_doc_a):
    assert sample_doc_a.get_page(0) is not None
    assert sample_doc_a.get_page(99) is None
