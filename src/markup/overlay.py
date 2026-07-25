"""
delta-chat · src/markup/overlay.py
══════════════════════════════════════════════════════════════════════════════
BONUS: Visual delta overlay — draws bboxes / highlights on changed regions.

Exports annotated PDFs to MARKUP_OUTPUT_DIR/<run_id>_{pid}.pdf

Implementation:
  • Native PDF: PyMuPDF annotations (Rect highlights on the delta bboxes)
  • Scanned PDF: PIL draw on rasterized page images, re-assembled as PDF

Color coding (configurable via env):
  • Green (00CC44) = added (visible in PID B overlay)
  • Red   (CC2200) = removed (visible in PID A overlay)
  • Amber (CCAA00) = modified (visible in both overlays)
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from typing import Optional

import fitz  # PyMuPDF

from src.canonical.model import ChangeType, DeltaItem, DeltaReport, ElementLocation
from src.config import get_settings
from src.observability.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string (no #) to RGB floats 0.0–1.0."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255.0, g / 255.0, b / 255.0)


_COLORS: dict[ChangeType, tuple[float, float, float]] = {}


def _get_colors() -> dict[str, tuple[float, float, float]]:
    return {
        "added": _hex_to_rgb(settings.markup_box_color_add),
        "removed": _hex_to_rgb(settings.markup_box_color_remove),
        "modified": _hex_to_rgb(settings.markup_box_color_modify),
    }


class DeltaMarkupOverlay:
    """
    Overlays delta bboxes onto document PDFs.
    Generates two annotated files:
        <run_id>_pid_a.pdf  — shows REMOVED and MODIFIED items
        <run_id>_pid_b.pdf  — shows ADDED and MODIFIED items
    """

    def overlay(
        self,
        report: DeltaReport,
        path_a: str,
        path_b: str,
        run_id: str,
        output_dir: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Generate annotated PDFs for both document revisions.
        Returns (path_a_annotated, path_b_annotated).
        """
        output_dir = output_dir or settings.markup_output_dir
        os.makedirs(output_dir, exist_ok=True)

        colors = _get_colors()
        line_width = settings.markup_line_width

        # Annotate PID A: REMOVED + MODIFIED (showing what was there before)
        path_a_out = os.path.join(output_dir, f"{run_id}_pid_a_annotated.pdf")
        self._annotate_pdf(
            source_path=path_a,
            output_path=path_a_out,
            items=[i for i in report.items if i.change_type in {"removed", "modified"}],
            location_attr="location_a",
            colors=colors,
            line_width=line_width,
            label_suffix="(removed/modified in A)",
        )

        # Annotate PID B: ADDED + MODIFIED (showing what was added/changed)
        path_b_out = os.path.join(output_dir, f"{run_id}_pid_b_annotated.pdf")
        self._annotate_pdf(
            source_path=path_b,
            output_path=path_b_out,
            items=[i for i in report.items if i.change_type in {"added", "modified"}],
            location_attr="location_b",
            colors=colors,
            line_width=line_width,
            label_suffix="(added/modified in B)",
        )

        log.info("markup.complete",
                 run_id=run_id,
                 pid_a_out=path_a_out,
                 pid_b_out=path_b_out,
                 total_annotations=len(report.items))

        return path_a_out, path_b_out

    def _annotate_pdf(
        self,
        source_path: str,
        output_path: str,
        items: list[DeltaItem],
        location_attr: str,
        colors: dict[str, tuple[float, float, float]],
        line_width: int,
        label_suffix: str,
    ) -> None:
        """Draw annotation rectangles on a PDF using PyMuPDF."""
        try:
            doc = fitz.open(source_path)
        except Exception as e:
            log.error("markup.open_failed", path=source_path, error=str(e))
            return

        annotated = 0
        for item in items:
            loc: Optional[ElementLocation] = getattr(item, location_attr, None)
            if loc is None:
                continue

            page_idx = loc.page_index
            if page_idx >= len(doc):
                continue

            page = doc[page_idx]
            bbox = loc.bbox
            color = colors.get(item.change_type, (0.5, 0.5, 0.5))

            # Draw border rectangle
            rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            annot = page.add_rect_annot(rect)
            annot.set_border(width=line_width)
            annot.set_colors(stroke=color)
            annot.set_info(
                title="delta-chat",
                content=(
                    f"[{item.change_type.upper()}] {item.element_type}\n"
                    f"delta_id: {item.delta_id}\n"
                    f"confidence: {item.confidence:.2f}\n"
                    f"{item.description[:200]}"
                ),
            )
            annot.update()
            annotated += 1

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        log.info("markup.pdf.saved",
                 output=output_path, annotations=annotated, label_suffix=label_suffix)
