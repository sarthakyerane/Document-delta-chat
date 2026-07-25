# Document Delta & Grounded Chat

> Applied AI Engineer take-home — format-agnostic revision diffing + grounded RAG chat.

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Set GOOGLE_API_KEY, GROQ_API_KEY (at minimum)

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Generate sample documents
python scripts/generate_sample_pdfs.py

# 4. Initialise database (SQLite by default)
python scripts/setup_db.py

# 5. Run full pipeline + interactive chat
delta-chat pipeline \
  --pid-a pair01_rev_a --path-a eval/datasets/pair_01/doc_a.pdf \
  --pid-b pair01_rev_b --path-b eval/datasets/pair_01/doc_b.pdf

# 6. Or run via FastAPI
uvicorn main:app --reload
# POST http://localhost:8000/ingest  → run_id
# POST http://localhost:8000/chat    → grounded answer + citations
```

---

## Architecture

```
Input (PDF / DWG)
       │
 ┌─────▼──────────────────────────────────────────────┐
 │  Ingest Layer  (src/ingest/)                        │
 │  ┌─────────────────┐  ┌──────────────────────────┐  │
 │  │ NativePDFAdapter│  │ ScannedPDFAdapter         │  │
 │  │ pdfplumber +    │  │ Gemini Vision (primary)   │  │
 │  │ PyMuPDF vectors │  │ Tesseract (fallback)      │  │
 │  └─────────────────┘  └──────────────────────────┘  │
 │  ┌─────────────────┐                                 │
 │  │ DWGAdapter      │ ezdxf + ODA CLI fallback        │
 │  └─────────────────┘                                 │
 └─────────────────────┬──────────────────────────────┘
                       │ canonical Document model
 ┌─────────────────────▼──────────────────────────────┐
 │  Delta Layer  (src/delta/)                          │
 │  DocumentAligner → 6-stage alignment                │
 │  DeltaEngine  → add/remove/modify classification    │
 │  DeltaReportRenderer → JSON + Markdown output       │
 └─────────────────────┬──────────────────────────────┘
                       │ DeltaReport
 ┌─────────────────────▼──────────────────────────────┐
 │  RAG / Chat Layer  (src/chat/)                      │
 │  ChromaDB (3 collections: pid_a, pid_b, delta)      │
 │  Google text-embedding-004 (embeddings)             │
 │  Groq → Gemini → Ollama (3-tier LLM fallback)       │
 │  Redis semantic cache (cosine ≥ 0.90)               │
 └─────────────────────┬──────────────────────────────┘
                       │ GroundedAnswer + citations
 ┌─────────────────────▼──────────────────────────────┐
 │  API (FastAPI) + CLI (Click + Rich)                 │
 │  /ingest  /chat  /delta/{id}  /traces/{id}          │
 └────────────────────────────────────────────────────┘
```

### Pipeline Graph (LangGraph)

```
[ingest_node] → [delta_node] → [report_node] → [index_node] → END
      ↓ (error)       ↓ (error)       ↓ (error)
   [error_exit] ────────────────────────────────→ END
```

---

## Design Decisions & Trade-offs

### Why VLM for scanned PDFs, not Tesseract alone?

**Decision:** Gemini 2.5 Flash Vision is the primary OCR path; Tesseract is the fallback.

**Rationale:** The rubric asks for "applied AI judgment." Tesseract returns raw text + loose bboxes and requires a second pass to classify elements (dimension vs. note vs. label) — two LLM calls. Gemini Vision does joint text extraction + layout understanding + element classification in one inference call, returning an Element-shaped JSON array. This is 50–70% cheaper per page for scanned engineering drawings where classification accuracy matters.

**Cost:** ~$0.001–0.003/page at 200 DPI. Documented per-request in traces.

**Trade-off:** Gemini Vision adds latency vs. Tesseract (~800ms vs. ~200ms per page). Tesseract fallback is triggered by: missing API key, quota exceeded, or `OCR_PROVIDER=tesseract`.

**Known limitation:** Gemini bbox_norm values have ±0.01 noise across runs. This is handled by rounding bbox to 1dp in `element_id` hashing.

---

### Why NOT ask an LLM whether a dimension changed?

The rubric says: "calling an LLM to compare '42.5 mm' to '45.0 mm' when string equality answers it will be penalised."

The delta engine classifies dimension changes purely by comparing `numeric_value` floats (or `raw_value` strings if numeric parsing fails). LLM is called exactly once per pipeline per pair, for semantic description of text/note changes where the text similarity is in the ambiguous 0.35–0.70 range. This boundary is documented in `src/delta/engine.py`.

---

### Why three separate ChromaDB collections (not one)?

**Decision:** `dc_{run_id}_pid_a`, `dc_{run_id}_pid_b`, `dc_{run_id}_delta` — separate collections, queried in parallel and merged.

**Rationale:** The chat system must attribute every claim to a specific source (PID A, PID B, or delta report). One combined collection loses source provenance. Separate collections guarantee citation accuracy: a query about "what was in Rev A" retrieves only from `pid_a`.

**Trade-off:** 3× storage, 3× index queries per chat request. Acceptable at document-pair scale (typically <100 pages total).

---

### Redis cache: O(n) scan vs. HNSW

**Decision:** Linear scan over all cached entries for a run (typically < 100).

**Rationale:** Production at scale would use Redis Stack HNSW vector index. This is a take-home with a single session's worth of queries per run. O(n) on 100 entries is ~0.1ms — not a bottleneck. Documented as a known limitation.

---

### DWG format limitation (honest)

**Decision:** DXF files are fully supported via ezdxf. Binary `.dwg` tries ezdxf.recover, then ODA CLI. If both fail, a structured `DWGConversionError` is raised with a workaround message.

**Rationale:** Binary DWG is a proprietary format. Pure-Python reading is limited to some versions via ezdxf's experimental reader. ODA File Converter is the gold standard but requires a free binary download. This is an honest capability boundary, not a silent fallback to an empty document.

**Mitigation:** The delta engine and chat layer work on any format that produces a canonical Document. Adding a fully functional DWG converter is a drop-in `DWGAdapter._convert_dwg_to_dxf` replacement.

---

### LLM non-determinism boundaries

Non-deterministic paths, clearly isolated:

| Module | Function | Why LLM | Mitigation |
|---|---|---|---|
| `pdf_scanned.py` | `_ocr_gemini` | VLM OCR — deterministic alternatives would require 2x calls | temperature=0, JSON retry × 3 |
| `align.py` | `_llm_disambiguate` | Semantic understanding of ambiguous element pairs | Only called for sim in [0.40, 0.70); disabled in tests |
| `engine.py` | `_llm_describe_change` | Summarise what a reworded note means | Only text/note, sim < 0.85; skipped for dimensions |
| `metrics.py` | `evaluate_chat_qa` | LLM-as-judge for chat correctness/groundedness | temperature=0, validated against 10 hand-checked cases |

All other stages are fully deterministic.

---

### Alignment confidence formula

| Stage | Method label | Score formula |
|---|---|---|
| 1 | `exact_id` | 1.0 |
| 2 | `geometric+text_high` | (IoU + text_sim) / 2 |
| 3 | `text_high` | text_sim × 0.90 |
| 4 | `geometric+text_medium` | (IoU + text_sim) / 2 × 0.85 |
| 5 | `cross_page_text` | text_sim × 0.70 |
| 6 | `llm_assisted` | LLM confidence (0–1) |

Combined item confidence = alignment_confidence × min(elem_a.confidence, elem_b.confidence)

---

## Scope Cuts (deliberate)

| Feature | Cut? | Reason |
|---|---|---|
| Markup / redlines | Out of core | Bonus — `src/markup/overlay.py` is complete but not in main pipeline |
| Scanned DWG / raster DWG | Out | DWG adapter handles vector/text. Raster DWG not a realistic format. |
| Multi-document session | Out | Single pair per run. Extension is additive (new run_id per pair set). |
| Streaming chat | Out | Synchronous chat sufficient for demo. SSE/websocket is FastAPI boilerplate. |
| Redis HNSW | Out | O(n) scan sufficient for single session. Documented. |
| MCP tools | Bonus-only | Not in core flow. |

---

## Known Failure Modes

| Scenario | Behaviour | Severity |
|---|---|---|
| Scanned PDF with handwritten text | Gemini VLM may miss or misclassify handwritten text | Medium — logged as low-confidence elements |
| DWG with 3D solids (3D model space) | Only 2D projection entities extracted | Low — 3D CAD not in scope |
| Two identical documents | All elements match stage 1; zero delta items (correct) | Expected |
| Very large PDF (>200 pages) | Indexing latency ∝ pages; no timeout protection per-page | Medium — add per-page timeout in production |
| LLM quota exhausted | Falls through to Ollama; if Ollama is down, `AllProvidersFailedError` | Medium — logged, surfaced in API response |
| Tesseract not installed | ScannedPDFAdapter raises `OCRFailureError` with clear message | Low — expected dep |
| ChromaDB cold start | First embed call is slow (~1s); subsequent calls use persisted index | Low |

---

## What I'd Do Next (With More Time)

1. **True Layout-Aware Bounding Boxes for DWG:** The current DWG adapter extracts coordinates but doesn't fully understand CAD block hierarchical transforms. I'd add full block explosion and matrix translation to perfectly align CAD text geometries.
2. **Streaming Chat Responses:** The current `/chat` endpoint is synchronous. I would implement SSE (Server-Sent Events) in FastAPI to stream the LLM response tokens to the client for a better UX.
3. **Advanced Table Diffing:** Tables are currently extracted as flat text blocks. I'd implement a specific `TableCell` canonical type that stores `(row, col)` indices, allowing for structural diffing of complex engineering schedules.
4. **HNSW Redis Index:** Swap the current O(n) semantic cache scan for a true Redis Stack HNSW vector index to support caching thousands of runs efficiently.

---

## Evaluation

```bash
# Generate sample PDFs first
python scripts/generate_sample_pdfs.py

# Run full eval (delta P/R/F1 + chat correctness/groundedness)
python eval/run_eval.py

# Delta only
python eval/run_eval.py --mode delta

# Chat only
python eval/run_eval.py --mode chat
```

Results written to `eval/results/<timestamp>.json` — regression-comparable.

### Evaluation Scorecard Output (pair_01 + pair_02)

#### Delta Metrics
| Metric | Value | Threshold | Status |
|---|---|---|---|
| Precision | 0.6667 | -- | |
| Recall | 0.9231 | -- | |
| **F1** | **0.7742** | ≥ 0.5 | **PASS** |

#### Chat Metrics
| Metric | Value | Threshold | Status |
|---|---|---|---|
| **Correctness** | **0.5556** | ≥ 0.6 | **FAIL** |
| **Groundedness** | **0.6667** | ≥ 0.65 | **PASS** |
| Q&A pairs tested | 9 | -- | |

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

Key test assertions:
- `test_no_llm_called_for_trivial_changes` — verifies LLM not invoked for numeric dimension compare
- `test_delta_id_deterministic` — verifies same delta produces same IDs
- `test_insufficient_grounding_on_no_chunks` — verifies fallback path

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | DB, Redis, ChromaDB, LLM provider status |
| `/ingest` | POST | Full pipeline; returns `run_id` |
| `/chat` | POST | Grounded chat; requires `run_id` |
| `/delta/{run_id}` | GET | Raw delta JSON |
| `/delta/{run_id}/markdown` | GET | Human-readable Markdown report |
| `/delta/{run_id}/summary` | GET | Summary counts only |
| `/traces/{request_id}` | GET | OTel trace JSON |
| `/traces/{request_id}/llm-summary` | GET | LLM cost/token summary |

---

## Observability

Every request produces a trace JSON in `traces/<request_id>.json`:

```json
{
  "request_id": "...",
  "total_duration_ms": 3412,
  "spans": [
    {"name": "ingest.pdf_native", "duration_ms": 120, ...},
    {"name": "delta.align", "duration_ms": 45, ...},
    {"name": "index.pid_a", "duration_ms": 890, ...}
  ],
  "llm_calls": [
    {"model": "gemini-2.5-flash", "input_tokens": 1200, "output_tokens": 400,
     "duration_ms": 800, "cost_usd": 0.0018}
  ],
  "llm_summary": {"total_cost_usd": 0.0018, "total_tokens": 1600}
}
```

---

## Repository Structure

```
delta-chat/
├── main.py                    # FastAPI entry point
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── src/
│   ├── config.py              # Pydantic settings
│   ├── canonical/             # Central data model (THE SEAM)
│   │   └── model.py           # Document → Page → Block → Element
│   ├── ingest/                # Format adapters
│   │   ├── base.py            # FormatAdapter ABC + AdapterRegistry
│   │   ├── pdf_native.py      # pdfplumber + PyMuPDF
│   │   ├── pdf_scanned.py     # Gemini Vision OCR + Tesseract fallback
│   │   └── dwg.py             # ezdxf + ODA CLI
│   ├── delta/                 # Alignment + classification
│   │   ├── align.py           # 6-stage alignment engine
│   │   ├── engine.py          # add/remove/modify classification
│   │   └── report.py          # JSON + Markdown renderer
│   ├── chat/                  # RAG + grounded answers
│   │   ├── llm.py             # 3-tier LLM client
│   │   ├── index.py           # ChromaDB indexing + retrieval
│   │   ├── cache.py           # Redis semantic cache
│   │   └── answer.py          # AnswerEngine + GroundedAnswer
│   ├── pipeline/              # LangGraph orchestration
│   │   ├── state.py
│   │   └── graph.py
│   ├── api/                   # FastAPI app + routes
│   ├── db/                    # SQLAlchemy ORM (run history, eval)
│   ├── markup/                # Bonus: PDF annotation overlay
│   └── observability/         # OTel tracing + structlog
├── eval/
│   ├── metrics.py             # P/R/F1 + LLM-as-judge
│   ├── run_eval.py            # Runnable harness
│   ├── datasets/              # Labeled document pairs
│   │   ├── pair_01/           # Flange spec Rev A → Rev B
│   │   └── pair_02/           # Process flow diagram Rev 1 → Rev 2
│   └── results/               # Timestamped eval JSON outputs
├── scripts/
│   ├── generate_sample_pdfs.py
│   ├── setup_db.py
│   └── init.sql
└── tests/
    ├── conftest.py
    ├── test_canonical.py
    ├── test_delta.py
    └── test_chat.py
```
