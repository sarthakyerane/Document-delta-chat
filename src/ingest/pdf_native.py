"""
delta-chat · src/ingest/pdf_native.py
══════════════════════════════════════════════════════════════════════════════
Native PDF adapter (born-digital PDFs with an extractable text/vector layer).

Libraries:
  • pdfplumber  — word-level bbox extraction + table cell detection
                  (better table structure recovery than PyMuPDF's text layer)
  • PyMuPDF (fitz) — vector/geometry path extraction; page dimensions
                  (pdfplumber doesn't expose raw vector paths)

Pipeline per page:
  1. Extract words + bboxes (pdfplumber)           → text / note elements
  2. Detect & extract tables (pdfplumber)           → table_cell elements
  3. Extract vector drawings (PyMuPDF)              → geometry elements
  4. Classify text items as 'dimension' vs 'note' vs 'text' (regex + heuristic)
  5. Assemble into Block → Page → Document

Determinism: all steps are fully deterministic (no LLM involved).
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import re
import uuid
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF

from src.canonical.model import (
    Block,
    BoundingBox,
    Document,
    Element,
    Page,
)
from src.config import get_settings
from src.ingest.base import AdapterRegistry, FormatAdapter, IngestionError
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()

# ── Regex patterns for dimension detection ────────────────────────────────────
# Matches patterns like: 42.5 mm, 1/2", 3.14 in, 500mm, Ø25, R10, 45°
_DIM_RE = re.compile(
    r"""
    (?:
        [ØRø]?\s*          # optional diameter/radius prefix
        \d+(?:[.,]\d+)?    # integer or decimal number
        \s*
        (?:mm|cm|m|in|ft|″|'|°|deg|inch(?:es)?|feet|meter|metre)
        \b
    )
    | (?:\d+\s*/\s*\d+\s*(?:in|inch|″|'))  # fractions with unit
    | (?:±\s*\d+(?:[.,]\d+)?)              # tolerances
    """,
    re.VERBOSE | re.IGNORECASE,
)

_NOTE_KEYWORDS = frozenset([
    "note", "notes", "warning", "caution", "see", "ref", "reference",
    "general", "typical", "typ", "nts", "not to scale", "unless", "all",
])


def _classify_text(text: str) -> str:
    """
    Classify a text string as 'dimension' | 'note' | 'text'.
    Purely deterministic — no LLM.
    """
    clean = text.strip()
    if _DIM_RE.search(clean):
        return "dimension"
    lowered = clean.lower()
    first_word = lowered.split()[0] if lowered.split() else ""
    if first_word in _NOTE_KEYWORDS or lowered.startswith("note"):
        return "note"
    return "text"


def _plumber_bbox_to_canonical(
    x0: float, y0: float, x1: float, y1: float,
    page_width: float, page_height: float,
) -> BoundingBox:
    """pdfplumber uses top-left origin; convert to standard bbox."""
    return BoundingBox(
        x0=x0, y0=y0, x1=x1, y1=y1,
        page_width=page_width, page_height=page_height,
    )


@AdapterRegistry.register
class NativePDFAdapter(FormatAdapter):
    """
    Adapter for born-digital PDFs.
    Registered automatically via @AdapterRegistry.register.
    """

    @property
    def supported_format(self) -> str:
        return "pdf_native"

    def ingest(
        self,
        pid: str,
        source_path: str,
        tracer: RequestTracer,
        revision_label: Optional[str] = None,
    ) -> Document:
        log.info("ingest.pdf_native.start", pid=pid, source_path=source_path)

        with tracer.span("ingest.pdf_native", {"pid": pid, "path": source_path}):
            try:
                pages = self._extract_pages(pid, source_path, tracer)
            except Exception as exc:
                raise IngestionError(pid, f"Failed to ingest native PDF: {exc}", cause=exc) from exc

        doc = Document(
            pid=pid,
            format="pdf_native",
            revision_label=revision_label,
            title=self._extract_title(source_path),
            page_count=len(pages),
            pages=pages,
            source_path=source_path,
            metadata={"adapter": "NativePDFAdapter"},
        )
        log.info(
            "ingest.pdf_native.complete",
            pid=pid,
            pages=len(pages),
            elements=doc.element_count,
        )
        return doc

    def _extract_title(self, path: str) -> Optional[str]:
        try:
            doc = fitz.open(path)
            meta = doc.metadata or {}
            doc.close()
            return meta.get("title") or None
        except Exception:
            return None

    def _extract_pages(
        self, pid: str, path: str, tracer: RequestTracer
    ) -> list[Page]:
        pages: list[Page] = []

        # Open both libraries once
        fitz_doc = fitz.open(path)

        with pdfplumber.open(path) as pdf:
            for page_idx, pl_page in enumerate(pdf.pages):
                with tracer.span(
                    f"ingest.pdf_native.page",
                    {"pid": pid, "page_index": page_idx},
                ):
                    page = self._process_page(
                        pid, page_idx, pl_page, fitz_doc[page_idx]
                    )
                    pages.append(page)

        fitz_doc.close()
        return pages

    def _process_page(
        self,
        pid: str,
        page_idx: int,
        pl_page: pdfplumber.page.Page,
        fitz_page: fitz.Page,
    ) -> Page:
        pw = float(pl_page.width)
        ph = float(pl_page.height)
        blocks: list[Block] = []

        # ── 1. Tables (pdfplumber table detection) ────────────────────────────
        table_bboxes: list[tuple[float, float, float, float]] = []
        try:
            tables = pl_page.extract_tables(
                table_settings={"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            table_regions = pl_page.find_tables()
            for tbl_obj, rows in zip(table_regions, tables):
                tbl_block = self._extract_table(pid, page_idx, tbl_obj, rows, pw, ph)
                if tbl_block:
                    blocks.append(tbl_block)
                    table_bboxes.append(tbl_obj.bbox)
        except Exception as e:
            log.warning("ingest.pdf_native.table_extract_failed",
                        pid=pid, page_index=page_idx, error=str(e))

        # ── 2. Words (pdfplumber word extraction) ─────────────────────────────
        word_elements: list[Element] = []
        try:
            words = pl_page.extract_words(
                x_tolerance=3,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True,
                extra_attrs=["fontname", "size"],
            )
            for w in words:
                # Skip words inside table regions
                wx0, wy0, wx1, wy1 = w["x0"], w["top"], w["x1"], w["bottom"]
                if self._in_table(wx0, wy0, wx1, wy1, table_bboxes):
                    continue
                text = w.get("text", "").strip()
                if not text:
                    continue
                elem_type = _classify_text(text)
                bbox = _plumber_bbox_to_canonical(wx0, wy0, wx1, wy1, pw, ph)
                eid = Element.make_id(page_idx, elem_type, bbox, text)
                word_elements.append(Element(
                    element_id=eid,
                    type=elem_type,
                    bbox=bbox,
                    text=text,
                    raw_value=text if elem_type == "dimension" else None,
                    numeric_value=self._parse_numeric(text) if elem_type == "dimension" else None,
                    unit=self._parse_unit(text) if elem_type == "dimension" else None,
                    confidence=1.0,
                    font_name=w.get("fontname"),
                    font_size=w.get("size"),
                ))
        except Exception as e:
            log.warning("ingest.pdf_native.word_extract_failed",
                        pid=pid, page_index=page_idx, error=str(e))

        # Group words into text blocks (cluster by y-proximity)
        if word_elements:
            text_blocks = self._cluster_into_blocks(page_idx, word_elements, pw, ph)
            blocks.extend(text_blocks)

        # ── 3. Vector geometry (PyMuPDF) ──────────────────────────────────────
        geom_elements: list[Element] = []
        try:
            drawings = fitz_page.get_drawings()
            for drw in drawings:
                rect = drw.get("rect")
                if not rect:
                    continue
                area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
                if area < 4:  # filter noise (< 2x2 pt paths)
                    continue
                bbox = BoundingBox(
                    x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1,
                    page_width=pw, page_height=ph,
                )
                eid = Element.make_id(page_idx, "geometry", bbox)
                geom_elements.append(Element(
                    element_id=eid,
                    type="geometry",
                    bbox=bbox,
                    confidence=1.0,
                    metadata={"fill": drw.get("fill"), "color": drw.get("color")},
                ))
        except Exception as e:
            log.warning("ingest.pdf_native.geom_extract_failed",
                        pid=pid, page_index=page_idx, error=str(e))

        if geom_elements:
            geom_bbox = self._union_bbox(geom_elements, pw, ph)
            blocks.append(Block(
                block_id=hashlib.sha256(f"{page_idx}_geom".encode()).hexdigest()[:12],
                type="figure",
                bbox=geom_bbox,
                elements=geom_elements,
            ))

        return Page(
            page_index=page_idx,
            page_label=str(page_idx + 1),
            width=pw,
            height=ph,
            blocks=blocks,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_table(
        self, pid: str, page_idx: int,
        tbl_obj: object, rows: list[list[Optional[str]]],
        pw: float, ph: float,
    ) -> Optional[Block]:
        try:
            bbox_t = tbl_obj.bbox  # (x0, top, x1, bottom)
            tbl_bbox = _plumber_bbox_to_canonical(
                bbox_t[0], bbox_t[1], bbox_t[2], bbox_t[3], pw, ph
            )
            elements: list[Element] = []
            for row_i, row in enumerate(rows):
                for col_i, cell_text in enumerate(row):
                    if not cell_text:
                        continue
                    cell_text = cell_text.strip()
                    # Approximate cell bbox (evenly divided for now)
                    cell_w = tbl_bbox.width / max(len(row), 1)
                    cell_h = tbl_bbox.height / max(len(rows), 1)
                    cell_bbox = BoundingBox(
                        x0=tbl_bbox.x0 + col_i * cell_w,
                        y0=tbl_bbox.y0 + row_i * cell_h,
                        x1=tbl_bbox.x0 + (col_i + 1) * cell_w,
                        y1=tbl_bbox.y0 + (row_i + 1) * cell_h,
                        page_width=pw, page_height=ph,
                    )
                    eid = Element.make_id(page_idx, "table_cell", cell_bbox, cell_text)
                    elements.append(Element(
                        element_id=eid,
                        type="table_cell",
                        bbox=cell_bbox,
                        text=cell_text,
                        raw_value=cell_text,
                        confidence=1.0,
                        metadata={"row": row_i, "col": col_i},
                    ))
            if not elements:
                return None
            return Block(
                block_id=hashlib.sha256(
                    f"{page_idx}_tbl_{tbl_bbox.x0:.0f}".encode()
                ).hexdigest()[:12],
                type="table",
                bbox=tbl_bbox,
                elements=elements,
            )
        except Exception as e:
            log.warning("ingest.pdf_native.table_block_failed",
                        pid=pid, page_index=page_idx, error=str(e))
            return None

    def _in_table(
        self, x0: float, y0: float, x1: float, y1: float,
        table_bboxes: list[tuple],
    ) -> bool:
        for tb in table_bboxes:
            tx0, ty0, tx1, ty1 = tb
            if x0 >= tx0 - 2 and y0 >= ty0 - 2 and x1 <= tx1 + 2 and y1 <= ty1 + 2:
                return True
        return False

    def _cluster_into_blocks(
        self, page_idx: int, elements: list[Element],
        pw: float, ph: float,
        y_gap_threshold: float = 12.0,  # pt
    ) -> list[Block]:
        """Cluster word elements into line groups by vertical proximity."""
        if not elements:
            return []
        sorted_elems = sorted(elements, key=lambda e: (e.bbox.y0, e.bbox.x0))
        clusters: list[list[Element]] = []
        current: list[Element] = [sorted_elems[0]]
        for elem in sorted_elems[1:]:
            if abs(elem.bbox.y0 - current[-1].bbox.y0) <= y_gap_threshold:
                current.append(elem)
            else:
                clusters.append(current)
                current = [elem]
        clusters.append(current)

        blocks: list[Block] = []
        for cluster in clusters:
            cluster_bbox = self._union_bbox(cluster, pw, ph)
            bid = hashlib.sha256(
                f"{page_idx}_{cluster_bbox.x0:.0f}_{cluster_bbox.y0:.0f}".encode()
            ).hexdigest()[:12]
            
            combined_text = " ".join(e.text for e in cluster if e.text)
            elem_type = _classify_text(combined_text)
            combined_element = Element(
                element_id=Element.make_id(page_idx, elem_type, cluster_bbox, combined_text),
                type=elem_type,
                bbox=cluster_bbox,
                text=combined_text,
                raw_value=combined_text if elem_type == "dimension" else None,
                numeric_value=self._parse_numeric(combined_text) if elem_type == "dimension" else None,
                unit=self._parse_unit(combined_text) if elem_type == "dimension" else None,
                confidence=1.0,
            )
            
            blocks.append(Block(
                block_id=bid,
                type="text_block",
                bbox=cluster_bbox,
                elements=[combined_element],
            ))
        return blocks

    def _union_bbox(
        self, elements: list[Element], pw: float, ph: float
    ) -> BoundingBox:
        x0 = min(e.bbox.x0 for e in elements)
        y0 = min(e.bbox.y0 for e in elements)
        x1 = max(e.bbox.x1 for e in elements)
        y1 = max(e.bbox.y1 for e in elements)
        return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page_width=pw, page_height=ph)

    def _parse_numeric(self, text: str) -> Optional[float]:
        m = re.search(r"(\d+(?:[.,]\d+)?)", text.replace(",", "."))
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None

    def _parse_unit(self, text: str) -> Optional[str]:
        units = ["mm", "cm", "m", "in", "ft", "°", "deg", "inch", "feet", "meter", "metre", "″", "'"]
        lowered = text.lower()
        for u in units:
            if u in lowered:
                return u
        return None
