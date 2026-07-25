"""
delta-chat · scripts/generate_sample_pdfs.py
══════════════════════════════════════════════════════════════════════════════
Generates synthetic PDF document pairs for the eval dataset.

Pair 01: Engineering flange assembly specification (2 pages)
  doc_a.pdf — Rev A: original dimensions, title, table
  doc_b.pdf — Rev B: changed dimensions, new notes, removed table row

Pair 02: Process flow diagram labels (2 pages)
  doc_a.pdf — Rev 1: original component labels
  doc_b.pdf — Rev 2: relabelled components, added annotations

Requires: pip install reportlab

Provenance (documented for eval): all content is synthetic, generated
deterministically from this script.  No real engineering documents used.
Re-run this script to regenerate identical pairs.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys

# Allow running as script from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "eval", "datasets")
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def _write_pdf(path: str, pages: list[list[dict]]) -> None:
    """
    Write a multi-page PDF using reportlab.
    pages: list of pages, each page is a list of {type, x, y, text, size?}
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfgen import canvas

    W, H = A4  # 595 x 842 pts

    c = canvas.Canvas(path, pagesize=A4)
    styles = getSampleStyleSheet()

    for page_items in pages:
        for item in page_items:
            t = item.get("type", "text")
            x, y = item.get("x", 60), H - item.get("y", 100)  # flip Y (PDF is bottom-up)
            text = item.get("text", "")
            size = item.get("size", 10)
            bold = item.get("bold", False)

            c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.setFillColor(colors.black)

            if t == "line":
                x2, y2 = item.get("x2", x + 100), H - item.get("y2", item.get("y", 100))
                c.line(x, y, x2, y2)
            elif t == "rect":
                w, h = item.get("w", 100), item.get("h", 20)
                c.rect(x, y - h, w, h)
            else:
                c.drawString(x, y, text)

        c.showPage()

    c.save()


def generate_pair_01():
    """Pair 01: Flange Assembly Specification Rev A → Rev B"""
    base_dir = os.path.join(EVAL_DIR, "pair_01")
    os.makedirs(base_dir, exist_ok=True)

    # ── Rev A ──────────────────────────────────────────────────────────────────
    page1_a = [
        {"type": "text", "x": 60, "y": 60, "text": "Flange Assembly Specification", "size": 16, "bold": True},
        {"type": "text", "x": 60, "y": 85, "text": "Document No.: FAS-001 | Revision: A | Date: 2024-01-10", "size": 9},
        {"type": "line", "x": 60, "y": 95, "x2": 535, "y2": 95},
        # Dimensions
        {"type": "text", "x": 60, "y": 130, "text": "1. DIMENSIONS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 155, "text": "Flange Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 155, "text": "42.5 mm", "size": 10},
        {"type": "text", "x": 80, "y": 175, "text": "Bolt Circle Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 175, "text": "80.0 mm", "size": 10},
        {"type": "text", "x": 80, "y": 195, "text": "Bolt Hole Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 195, "text": "14.0 mm", "size": 10},
        {"type": "text", "x": 80, "y": 215, "text": "Number of Bolts:", "size": 10},
        {"type": "text", "x": 220, "y": 215, "text": "8", "size": 10},
        # Spec table
        {"type": "text", "x": 60, "y": 250, "text": "2. MATERIALS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 275, "text": "Component", "size": 10, "bold": True},
        {"type": "text", "x": 250, "y": 275, "text": "Material Grade", "size": 10, "bold": True},
        {"type": "text", "x": 400, "y": 275, "text": "Standard", "size": 10, "bold": True},
        {"type": "line", "x": 80, "y": 285, "x2": 500, "y2": 285},
        {"type": "text", "x": 80, "y": 300, "text": "Flange Body", "size": 10},
        {"type": "text", "x": 250, "y": 300, "text": "A105", "size": 10},
        {"type": "text", "x": 400, "y": 300, "text": "ASME B16.5", "size": 10},
        {"type": "text", "x": 80, "y": 320, "text": "Bolts", "size": 10},
        {"type": "text", "x": 250, "y": 320, "text": "A193 B7", "size": 10},
        {"type": "text", "x": 400, "y": 320, "text": "ASTM A193", "size": 10},
        {"type": "text", "x": 80, "y": 340, "text": "Gasket", "size": 10},
        {"type": "text", "x": 250, "y": 340, "text": "316 SS Spiral Wound", "size": 10},
        {"type": "text", "x": 400, "y": 340, "text": "ASME B16.20", "size": 10},
    ]
    page2_a = [
        {"type": "text", "x": 60, "y": 60, "text": "FAS-001 Rev A — Page 2", "size": 9},
        {"type": "text", "x": 60, "y": 90, "text": "3. TOLERANCES AND REQUIREMENTS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 115, "text": "Wall Thickness:", "size": 10},
        {"type": "text", "x": 220, "y": 115, "text": "6.0 mm", "size": 10},
        {"type": "text", "x": 80, "y": 135, "text": "Surface Finish:", "size": 10},
        {"type": "text", "x": 220, "y": 135, "text": "Ra 3.2 µm (125 µin)", "size": 10},
        {"type": "text", "x": 80, "y": 155, "text": "Pressure Rating:", "size": 10},
        {"type": "text", "x": 220, "y": 155, "text": "ASME Class 300", "size": 10},
        {"type": "text", "x": 60, "y": 200, "text": "4. INSPECTION REQUIREMENTS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 225, "text": "All welds to be inspected per ASME IX.", "size": 10},
        {"type": "text", "x": 80, "y": 245, "text": "Hydrostatic test at 1.5x design pressure.", "size": 10},
    ]

    # ── Rev B ──────────────────────────────────────────────────────────────────
    page1_b = [
        {"type": "text", "x": 60, "y": 60, "text": "Flange Assembly Specification Rev B", "size": 16, "bold": True},
        {"type": "text", "x": 60, "y": 85, "text": "Document No.: FAS-001 | Revision: B | Date: 2024-03-15", "size": 9},
        {"type": "line", "x": 60, "y": 95, "x2": 535, "y2": 95},
        # Changed dimensions
        {"type": "text", "x": 60, "y": 130, "text": "1. DIMENSIONS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 155, "text": "Flange Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 155, "text": "45.0 mm", "size": 10},  # CHANGED
        {"type": "text", "x": 80, "y": 175, "text": "Bolt Circle Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 175, "text": "85.0 mm", "size": 10},  # CHANGED
        {"type": "text", "x": 80, "y": 195, "text": "Bolt Hole Diameter:", "size": 10},
        {"type": "text", "x": 220, "y": 195, "text": "14.0 mm", "size": 10},
        {"type": "text", "x": 80, "y": 215, "text": "Number of Bolts:", "size": 10},
        {"type": "text", "x": 220, "y": 215, "text": "8", "size": 10},
        # Spec table — A105 row REMOVED
        {"type": "text", "x": 60, "y": 250, "text": "2. MATERIALS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 275, "text": "Component", "size": 10, "bold": True},
        {"type": "text", "x": 250, "y": 275, "text": "Material Grade", "size": 10, "bold": True},
        {"type": "text", "x": 400, "y": 275, "text": "Standard", "size": 10, "bold": True},
        {"type": "line", "x": 80, "y": 285, "x2": 500, "y2": 285},
        # A105 row removed
        {"type": "text", "x": 80, "y": 300, "text": "Bolts", "size": 10},
        {"type": "text", "x": 250, "y": 300, "text": "A193 B7", "size": 10},
        {"type": "text", "x": 400, "y": 300, "text": "ASTM A193", "size": 10},
        {"type": "text", "x": 80, "y": 320, "text": "Gasket", "size": 10},
        {"type": "text", "x": 250, "y": 320, "text": "316 SS Spiral Wound", "size": 10},
        {"type": "text", "x": 400, "y": 320, "text": "ASME B16.20", "size": 10},
    ]
    page2_b = [
        {"type": "text", "x": 60, "y": 60, "text": "FAS-001 Rev B — Page 2", "size": 9},
        {"type": "text", "x": 60, "y": 90, "text": "3. TOLERANCES AND REQUIREMENTS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 115, "text": "Wall Thickness:", "size": 10},
        {"type": "text", "x": 220, "y": 115, "text": "8.0 mm", "size": 10},  # CHANGED
        {"type": "text", "x": 80, "y": 135, "text": "Surface Finish:", "size": 10},
        {"type": "text", "x": 220, "y": 135, "text": "Ra 3.2 µm (125 µin)", "size": 10},
        {"type": "text", "x": 80, "y": 155, "text": "Pressure Rating:", "size": 10},
        {"type": "text", "x": 220, "y": 155, "text": "ASME Class 300", "size": 10},
        {"type": "text", "x": 60, "y": 200, "text": "4. INSPECTION REQUIREMENTS", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 225, "text": "All welds to be inspected per ASME IX.", "size": 10},
        {"type": "text", "x": 80, "y": 245, "text": "Hydrostatic test at 1.5x design pressure.", "size": 10},
        # NEW NOTES
        {"type": "text", "x": 60, "y": 420, "text": "CAUTION: Minimum torque 45 N·m", "size": 10, "bold": True},
        {"type": "text", "x": 60, "y": 445, "text": "See DWG-042 for assembly details", "size": 10},
    ]

    _write_pdf(os.path.join(base_dir, "doc_a.pdf"), [page1_a, page2_a])
    _write_pdf(os.path.join(base_dir, "doc_b.pdf"), [page1_b, page2_b])
    # Also copy to samples for demo
    os.makedirs(os.path.join(SAMPLES_DIR, "pair_01"), exist_ok=True)
    import shutil
    shutil.copy(os.path.join(base_dir, "doc_a.pdf"),
                os.path.join(SAMPLES_DIR, "pair_01", "doc_a.pdf"))
    shutil.copy(os.path.join(base_dir, "doc_b.pdf"),
                os.path.join(SAMPLES_DIR, "pair_01", "doc_b.pdf"))
    print(f"[OK] Pair 01 generated: {base_dir}")


def generate_pair_02():
    """Pair 02: Process flow diagram Rev 1 → Rev 2"""
    base_dir = os.path.join(EVAL_DIR, "pair_02")
    os.makedirs(base_dir, exist_ok=True)

    page1_a = [
        {"type": "text", "x": 60, "y": 50, "text": "PROCESS FLOW DIAGRAM — REV 1", "size": 14, "bold": True},
        {"type": "text", "x": 60, "y": 75, "text": "Project: Alpha Plant | Drawing No.: PFD-001", "size": 9},
        {"type": "line", "x": 60, "y": 85, "x2": 535, "y2": 85},
        # Component labels
        {"type": "text", "x": 100, "y": 180, "text": "FEED TANK", "size": 10, "bold": True},
        {"type": "text", "x": 100, "y": 195, "text": "TK-101", "size": 10},
        {"type": "rect", "x": 90, "y": 165, "w": 100, "h": 40},
        {"type": "text", "x": 280, "y": 180, "text": "CENTRIFUGAL PUMP", "size": 10, "bold": True},
        {"type": "text", "x": 280, "y": 195, "text": "P-101", "size": 10},
        {"type": "rect", "x": 270, "y": 165, "w": 120, "h": 40},
        {"type": "text", "x": 450, "y": 180, "text": "PRODUCT TANK", "size": 10, "bold": True},
        {"type": "text", "x": 450, "y": 195, "text": "TK-201", "size": 10},
        {"type": "rect", "x": 440, "y": 165, "w": 90, "h": 40},
        # Flow lines
        {"type": "line", "x": 190, "y": 185, "x2": 270, "y2": 185},
        {"type": "line", "x": 390, "y": 185, "x2": 440, "y2": 185},
    ]
    page2_a = [
        {"type": "text", "x": 60, "y": 60, "text": "PFD-001 Rev 1 — Notes Page", "size": 9},
        {"type": "text", "x": 60, "y": 90, "text": "GENERAL NOTES", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 115, "text": "1. All piping ANSI B31.3 unless noted.", "size": 10},
        {"type": "text", "x": 80, "y": 135, "text": "2. Design pressure: 10 barg.", "size": 10},
        {"type": "text", "x": 80, "y": 155, "text": "3. Design temperature: 150°C.", "size": 10},
        {"type": "text", "x": 60, "y": 200, "text": "Cooling Water Return (CWR)", "size": 10},
        {"type": "text", "x": 60, "y": 218, "text": "CWR supply at 20°C, return max 35°C.", "size": 10},
    ]

    page1_b = [
        {"type": "text", "x": 60, "y": 50, "text": "PROCESS FLOW DIAGRAM — REV 2", "size": 14, "bold": True},
        {"type": "text", "x": 60, "y": 75, "text": "Project: Alpha Plant | Drawing No.: PFD-001 | Rev: 2", "size": 9},
        {"type": "line", "x": 60, "y": 85, "x2": 535, "y2": 85},
        {"type": "text", "x": 100, "y": 180, "text": "FEED TANK", "size": 10, "bold": True},
        {"type": "text", "x": 100, "y": 195, "text": "TK-101", "size": 10},
        {"type": "rect", "x": 90, "y": 165, "w": 100, "h": 40},
        {"type": "text", "x": 280, "y": 180, "text": "CENTRIFUGAL PUMP", "size": 10, "bold": True},
        {"type": "text", "x": 280, "y": 195, "text": "P-101A", "size": 10},   # CHANGED
        {"type": "rect", "x": 270, "y": 165, "w": 120, "h": 40},
        {"type": "text", "x": 450, "y": 180, "text": "PRODUCT TANK", "size": 10, "bold": True},
        {"type": "text", "x": 450, "y": 195, "text": "TK-201B (Revised)", "size": 10},  # CHANGED
        {"type": "rect", "x": 440, "y": 165, "w": 90, "h": 40},
        {"type": "line", "x": 190, "y": 185, "x2": 270, "y2": 185},
        {"type": "line", "x": 390, "y": 185, "x2": 440, "y2": 185},
        # ADDED annotations
        {"type": "text", "x": 300, "y": 155, "text": "120 L/min", "size": 9},
        {"type": "text", "x": 300, "y": 170, "text": "Max 10 bar", "size": 9},
    ]
    page2_b = [
        {"type": "text", "x": 60, "y": 60, "text": "PFD-001 Rev 2 — Notes Page", "size": 9},
        {"type": "text", "x": 60, "y": 90, "text": "GENERAL NOTES", "size": 12, "bold": True},
        {"type": "text", "x": 80, "y": 115, "text": "1. All piping ANSI B31.3 unless noted.", "size": 10},
        {"type": "text", "x": 80, "y": 135, "text": "2. Design pressure: 10 barg.", "size": 10},
        {"type": "text", "x": 80, "y": 155, "text": "3. Design temperature: 150°C.", "size": 10},
        # CWR REMOVED
        # NEW note added
        {"type": "text", "x": 60, "y": 360, "text": "NOTE: All lines ANSI 150 class unless noted", "size": 10, "bold": True},
    ]

    _write_pdf(os.path.join(base_dir, "doc_a.pdf"), [page1_a, page2_a])
    _write_pdf(os.path.join(base_dir, "doc_b.pdf"), [page1_b, page2_b])
    print(f"[OK] Pair 02 generated: {base_dir}")


if __name__ == "__main__":
    try:
        from reportlab.pdfgen import canvas  # type: ignore  # noqa
        print("Generating synthetic PDF document pairs...")
        generate_pair_01()
        generate_pair_02()
        print("\n[OK] All document pairs generated.")
        print("  eval/datasets/pair_01/doc_{a,b}.pdf")
        print("  eval/datasets/pair_02/doc_{a,b}.pdf")
        print("  data/samples/pair_01/doc_{a,b}.pdf")
    except ImportError:
        print("ERROR: reportlab not installed. Run: pip install reportlab")
        print("Or: pip install -e '.[dev]'")
        sys.exit(1)
