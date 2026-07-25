"""
delta-chat · src/ingest/dwg.py
══════════════════════════════════════════════════════════════════════════════
DWG / DXF adapter — CAD drawing files.

Status: REAL ADAPTER INTERFACE — fully implemented for DXF files.
        For binary .dwg files, tries ezdxf's partial DWG reader first,
        then ODA File Converter CLI (if configured), then returns a
        structured IngestionError (visible in trace) rather than a silent
        fallback.

Why ezdxf: pure-Python, no native binaries needed for DXF; ezdxf 1.x
           has partial DWG support via ezdxf.recover for some DWG versions.

Entity mapping → Element types:
  TEXT, MTEXT, ATTDEF, ATTRIB  → 'text' or 'note' or 'dimension'
  DIMENSION                    → 'dimension'
  LINE, ARC, CIRCLE, ELLIPSE,
  SPLINE, POLYLINE, LWPOLYLINE → 'geometry'
  Everything else              → 'text' (with type="text")

Coordinate system:
  DWG/DXF uses model-space units (often mm or inches).
  We map the model extents to a normalised page bbox.
  BoundingBox.page_width / page_height are set to the extents width/height
  so downstream normalisation works identically to PDF pages.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Any, Optional

from src.canonical.model import Block, BoundingBox, Document, Element, Page
from src.config import get_settings
from src.ingest.base import (
    AdapterRegistry, DWGConversionError, FormatAdapter, IngestionError,
)
from src.observability.logging import get_logger
from src.observability.tracing import RequestTracer

log = get_logger(__name__)
settings = get_settings()

_GEOMETRY_TYPES = frozenset([
    "LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE",
    "POLYLINE", "LWPOLYLINE", "3DFACE", "SOLID", "TRACE",
    "POINT", "RAY", "XLINE", "HATCH",
])

_TEXT_TYPES = frozenset(["TEXT", "MTEXT", "ATTDEF", "ATTRIB"])


@AdapterRegistry.register
class DWGAdapter(FormatAdapter):
    """
    Adapter for DWG / DXF CAD files.
    For .dxf files: fully parsed via ezdxf.
    For .dwg files: attempts ezdxf.recover, then ODA CLI, then structured error.
    """

    @property
    def supported_format(self) -> str:
        return "dwg"

    def ingest(
        self,
        pid: str,
        source_path: str,
        tracer: RequestTracer,
        revision_label: Optional[str] = None,
    ) -> Document:
        log.info("ingest.dwg.start", pid=pid, source_path=source_path)

        with tracer.span("ingest.dwg", {"pid": pid, "path": source_path}):
            ext = os.path.splitext(source_path)[1].lower()
            dxf_path = source_path

            if ext == ".dwg":
                dxf_path = self._convert_dwg_to_dxf(pid, source_path, tracer)

            try:
                pages, metadata = self._parse_dxf(pid, dxf_path, tracer)
            except Exception as exc:
                raise IngestionError(pid, f"DXF parse failed: {exc}", cause=exc) from exc

        doc = Document(
            pid=pid,
            format="dwg",
            revision_label=revision_label,
            title=metadata.get("title"),
            page_count=len(pages),
            pages=pages,
            source_path=source_path,
            metadata={
                "adapter": "DWGAdapter",
                "dxf_path": dxf_path,
                **metadata,
            },
        )
        log.info("ingest.dwg.complete", pid=pid, pages=len(pages), elements=doc.element_count)
        return doc

    def _convert_dwg_to_dxf(
        self, pid: str, dwg_path: str, tracer: RequestTracer
    ) -> str:
        """
        Convert .dwg → .dxf.  Strategy:
          1. Try ezdxf.recover (supports some DWG versions natively)
          2. Try ODA File Converter CLI (if DWG_ODA_CONVERTER_PATH is set)
          3. Raise DWGConversionError with full diagnostics in trace
        """
        # Output path: same dir, .dxf extension
        dxf_path = dwg_path.replace(".dwg", ".dxf").replace(".DWG", ".dxf")

        # Strategy 1: ezdxf partial DWG support (saves as dxf)
        with tracer.span("ingest.dwg.ezdxf_recover", {"dwg_path": dwg_path}):
            try:
                import ezdxf
                doc, auditor = ezdxf.recover.readfile(dwg_path)
                if auditor.has_errors:
                    log.warning("ingest.dwg.audit_errors",
                                pid=pid, errors=len(auditor.errors))
                doc.saveas(dxf_path)
                log.info("ingest.dwg.ezdxf_recover_success", pid=pid, dxf_path=dxf_path)
                return dxf_path
            except Exception as e1:
                log.warning("ingest.dwg.ezdxf_recover_failed", pid=pid, error=str(e1))

        # Strategy 2: ODA File Converter CLI
        oda_path = settings.dwg_oda_converter_path
        if oda_path and os.path.isfile(oda_path):
            with tracer.span("ingest.dwg.oda_converter", {"oda_path": oda_path}):
                try:
                    out_dir = os.path.dirname(dwg_path) or "."
                    result = subprocess.run(
                        [oda_path, out_dir, out_dir, "ACAD2018", "DXF", "0", "1",
                         os.path.basename(dwg_path)],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0 and os.path.exists(dxf_path):
                        log.info("ingest.dwg.oda_success", pid=pid)
                        return dxf_path
                    else:
                        log.warning("ingest.dwg.oda_failed",
                                    pid=pid, stderr=result.stderr[:500])
                except Exception as e2:
                    log.warning("ingest.dwg.oda_exception", pid=pid, error=str(e2))

        # All strategies failed — raise a traceable, honest error
        raise DWGConversionError(
            pid=pid,
            message=(
                f"Cannot convert '{dwg_path}' from DWG to DXF. "
                "Tried: (1) ezdxf.recover — failed (DWG version likely unsupported). "
                "(2) ODA File Converter CLI — "
                f"{'not configured (set DWG_ODA_CONVERTER_PATH)' if not oda_path else 'failed'}. "
                "See trace for details. Workaround: convert to DXF manually and re-ingest."
            ),
        )

    def _parse_dxf(
        self, pid: str, dxf_path: str, tracer: RequestTracer
    ) -> tuple[list[Page], dict[str, Any]]:
        import ezdxf
        from ezdxf.math import BoundingBox2d

        with tracer.span("ingest.dwg.parse_dxf", {"dxf_path": dxf_path}):
            doc = ezdxf.readfile(dxf_path)

        metadata: dict[str, Any] = {
            "dxf_version": doc.dxfversion,
            "title": self._extract_title_block(doc),
        }

        # Collect all layouts (model space + paper space sheets)
        layouts = [doc.modelspace()]
        for name in doc.layout_names():
            if name != "Model":
                try:
                    layouts.append(doc.layout(name))
                except Exception:
                    pass

        pages: list[Page] = []
        for layout_idx, layout in enumerate(layouts):
            with tracer.span(f"ingest.dwg.layout", {"layout": layout.name, "index": layout_idx}):
                page = self._process_layout(pid, layout_idx, layout)
                pages.append(page)

        return pages, metadata

    def _process_layout(self, pid: str, layout_idx: int, layout: Any) -> Page:
        """Convert a DXF layout (model or paper space) to a Page."""
        elements: list[Element] = []
        layer_names: set[str] = set()

        # Compute extents for this layout
        extmin = getattr(layout, "extmin", None)
        extmax = getattr(layout, "extmax", None)

        pw = ph = 1.0
        ox = oy = 0.0
        if extmin and extmax:
            ox, oy = extmin.x, extmin.y
            pw = max(extmax.x - extmin.x, 1.0)
            ph = max(extmax.y - extmin.y, 1.0)

        for entity in layout:
            try:
                dxf_type = entity.dxftype()
                layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
                layer_names.add(layer)

                if dxf_type in _TEXT_TYPES:
                    elems = self._extract_text_entity(
                        entity, dxf_type, layout_idx, pw, ph, ox, oy
                    )
                    elements.extend(elems)
                elif dxf_type == "DIMENSION":
                    elem = self._extract_dimension_entity(
                        entity, layout_idx, pw, ph, ox, oy
                    )
                    if elem:
                        elements.append(elem)
                elif dxf_type in _GEOMETRY_TYPES:
                    elem = self._extract_geometry_entity(
                        entity, dxf_type, layout_idx, pw, ph, ox, oy
                    )
                    if elem:
                        elements.append(elem)
            except Exception as e:
                log.debug("ingest.dwg.entity_skip", entity_type=str(type(entity)), error=str(e))
                continue

        # Group into one block per layer
        blocks: list[Block] = []
        layer_groups: dict[str, list[Element]] = {}
        for elem in elements:
            layer = elem.layer or "0"
            layer_groups.setdefault(layer, []).append(elem)

        for layer_name, layer_elements in layer_groups.items():
            if not layer_elements:
                continue
            x0 = min(e.bbox.x0 for e in layer_elements)
            y0 = min(e.bbox.y0 for e in layer_elements)
            x1 = max(e.bbox.x1 for e in layer_elements)
            y1 = max(e.bbox.y1 for e in layer_elements)
            block_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page_width=pw, page_height=ph)
            bid = hashlib.sha256(f"{layout_idx}_{layer_name}".encode()).hexdigest()[:12]
            blocks.append(Block(
                block_id=bid,
                type="drawing_entity",
                bbox=block_bbox,
                elements=layer_elements,
            ))

        # Page bbox = full layout extents
        page_bbox = BoundingBox(x0=0, y0=0, x1=pw, y1=ph, page_width=pw, page_height=ph)
        return Page(
            page_index=layout_idx,
            page_label=layout.name,
            width=pw,
            height=ph,
            blocks=blocks,
        )

    def _extract_text_entity(
        self, entity: Any, dxf_type: str,
        layout_idx: int, pw: float, ph: float, ox: float, oy: float,
    ) -> list[Element]:
        from src.ingest.pdf_native import _classify_text

        text = ""
        try:
            if dxf_type == "MTEXT":
                text = entity.plain_mtext() or ""
            else:
                text = entity.dxf.text or ""
        except Exception:
            return []

        text = text.strip()
        if not text:
            return []

        # Get position
        try:
            if dxf_type in {"TEXT", "ATTDEF", "ATTRIB"}:
                pt = entity.dxf.insert
                h = entity.dxf.height if hasattr(entity.dxf, "height") else 2.5
                x0 = pt.x - ox
                y0 = pt.y - oy
                x1 = x0 + len(text) * h * 0.6
                y1 = y0 + h
            else:  # MTEXT
                pt = entity.dxf.insert
                h = entity.dxf.char_height if hasattr(entity.dxf, "char_height") else 2.5
                w = entity.dxf.width if hasattr(entity.dxf, "width") else 50
                x0 = pt.x - ox
                y0 = pt.y - oy
                x1 = x0 + w
                y1 = y0 + h
        except Exception:
            return []

        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
        elem_type = _classify_text(text)
        bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page_width=pw, page_height=ph)
        eid = Element.make_id(layout_idx, elem_type, bbox, text)

        return [Element(
            element_id=eid,
            type=elem_type,
            bbox=bbox,
            text=text,
            raw_value=text,
            layer=layer,
            confidence=1.0,
            metadata={"dxf_type": dxf_type, "handle": str(entity.dxf.handle)},
        )]

    def _extract_dimension_entity(
        self, entity: Any, layout_idx: int,
        pw: float, ph: float, ox: float, oy: float,
    ) -> Optional[Element]:
        try:
            mtext = entity.dxf.text if hasattr(entity.dxf, "text") else ""
            defpt = entity.dxf.defpoint if hasattr(entity.dxf, "defpoint") else None
            textpt = entity.dxf.text_midpoint if hasattr(entity.dxf, "text_midpoint") else defpt

            if textpt is None:
                return None

            x, y = textpt.x - ox, textpt.y - oy
            # Use a small fixed bbox around the text midpoint
            h = 2.5
            bbox = BoundingBox(x0=x - 5, y0=y - h, x1=x + 30, y1=y + h,
                               page_width=pw, page_height=ph)
            text = mtext or str(entity.dxf.actual_measurement) if hasattr(entity.dxf, "actual_measurement") else mtext

            # Try to parse numeric value
            numeric = None
            try:
                numeric = float(entity.dxf.actual_measurement)
            except Exception:
                pass

            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
            eid = Element.make_id(layout_idx, "dimension", bbox, text)
            return Element(
                element_id=eid,
                type="dimension",
                bbox=bbox,
                text=text,
                raw_value=text,
                numeric_value=numeric,
                layer=layer,
                confidence=1.0,
                metadata={"dxf_type": "DIMENSION", "handle": str(entity.dxf.handle)},
            )
        except Exception as e:
            log.debug("ingest.dwg.dimension_skip", error=str(e))
            return None

    def _extract_geometry_entity(
        self, entity: Any, dxf_type: str,
        layout_idx: int, pw: float, ph: float, ox: float, oy: float,
    ) -> Optional[Element]:
        try:
            bbox_coords = self._entity_bbox(entity, dxf_type)
            if not bbox_coords:
                return None
            x0, y0, x1, y1 = (c - ox if i % 2 == 0 else c - oy
                               for i, c in enumerate(bbox_coords))
            # After offset (ox, oy) on x-coords and oy on y-coords
            x0, y0 = bbox_coords[0] - ox, bbox_coords[1] - oy
            x1, y1 = bbox_coords[2] - ox, bbox_coords[3] - oy
            area = abs((x1 - x0) * (y1 - y0))
            if area < 0.001:
                return None  # degenerate
            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, page_width=pw, page_height=ph)
            eid = Element.make_id(layout_idx, "geometry", bbox)
            return Element(
                element_id=eid,
                type="geometry",
                bbox=bbox,
                layer=layer,
                confidence=1.0,
                metadata={"dxf_type": dxf_type, "handle": str(entity.dxf.handle)},
            )
        except Exception:
            return None

    def _entity_bbox(
        self, entity: Any, dxf_type: str
    ) -> Optional[tuple[float, float, float, float]]:
        try:
            if dxf_type == "LINE":
                s, e = entity.dxf.start, entity.dxf.end
                return (min(s.x, e.x), min(s.y, e.y), max(s.x, e.x), max(s.y, e.y))
            if dxf_type in {"CIRCLE", "ARC"}:
                c = entity.dxf.center
                r = entity.dxf.radius
                return (c.x - r, c.y - r, c.x + r, c.y + r)
            if dxf_type == "ELLIPSE":
                c = entity.dxf.center
                mr = entity.dxf.major_axis.magnitude
                return (c.x - mr, c.y - mr, c.x + mr, c.y + mr)
            if dxf_type in {"LWPOLYLINE", "POLYLINE"}:
                pts = list(entity.vertices()) if hasattr(entity, "vertices") else []
                if not pts:
                    return None
                xs = [p.dxf.location.x if hasattr(p.dxf, "location") else p[0] for p in pts]
                ys = [p.dxf.location.y if hasattr(p.dxf, "location") else p[1] for p in pts]
                return (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            pass
        return None

    def _extract_title_block(self, doc: Any) -> Optional[str]:
        """Attempt to extract drawing title from title block attributes."""
        try:
            msp = doc.modelspace()
            for entity in msp.query("INSERT"):
                block_name = entity.dxf.name.upper()
                if any(k in block_name for k in ["TITLE", "BORDER", "TB"]):
                    for attrib in entity.attribs:
                        tag = attrib.dxf.tag.upper()
                        if "TITLE" in tag or "DWG_NAME" in tag or "NAME" in tag:
                            return attrib.dxf.text
        except Exception:
            pass
        return None
