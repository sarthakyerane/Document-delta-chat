.PHONY: help install install-prod setup setup-db run run-prod \
        ingest delta chat pipeline demo eval eval-delta eval-chat \
        markup generate-samples traces show-trace \
        test test-fast lint format typecheck \
        docker-up docker-down docker-logs clean clean-all

# ── Colours ────────────────────────────────────────────────────────────────────
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m
BOLD  := \033[1m

PYTHON   := python
PIP      := pip
UVICORN  := uvicorn

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "$(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
install: ## Install all dependencies (incl. dev)
	$(PIP) install -e ".[dev]"

install-prod: ## Install production dependencies only
	$(PIP) install -e .

setup: install setup-db ## Full first-time setup: install + DB + directories
	mkdir -p traces data/chroma data/markup data/samples eval/results
	@if [ ! -f .env ]; then cp .env.example .env; echo "$(GREEN)Created .env from .env.example — fill in API keys$(RESET)"; fi

setup-db: ## Initialise database schema
	$(PYTHON) scripts/setup_db.py

# ─────────────────────────────────────────────────────────────────────────────
# Running
# ─────────────────────────────────────────────────────────────────────────────
run: ## Start FastAPI dev server (hot-reload)
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --reload --log-level info

run-prod: ## Start FastAPI production server
	$(UVICORN) main:app --host 0.0.0.0 --port 8000 --workers 2

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline  (single documented command — spec §04 "Reproducible run")
# ─────────────────────────────────────────────────────────────────────────────
pipeline: ## Full pipeline: ingest → delta → report → interactive chat
           ## Usage: make pipeline PID_A=./doc_a.pdf PID_B=./doc_b.pdf
	@if [ -z "$(PID_A)" ] || [ -z "$(PID_B)" ]; then \
	  echo "$(BOLD)Usage:$(RESET) make pipeline PID_A=<path_a> PID_B=<path_b>"; exit 1; fi
	$(PYTHON) -m src.cli pipeline --pid-a "$(PID_A)" --pid-b "$(PID_B)"

demo: ## Run demo with included sample pair_01 documents
	$(PYTHON) -m src.cli pipeline \
	  --pid-a "data/samples/pair_01/doc_a.pdf" \
	  --pid-b "data/samples/pair_01/doc_b.pdf"

ingest: ## Ingest documents only. Usage: make ingest PID_A=... PID_B=...
	@if [ -z "$(PID_A)" ] || [ -z "$(PID_B)" ]; then \
	  echo "Usage: make ingest PID_A=<path_a> PID_B=<path_b>"; exit 1; fi
	$(PYTHON) -m src.cli ingest --pid-a "$(PID_A)" --pid-b "$(PID_B)"

delta: ## Compute delta for existing run. Usage: make delta RUN_ID=<id>
	@if [ -z "$(RUN_ID)" ]; then echo "Usage: make delta RUN_ID=<run_id>"; exit 1; fi
	$(PYTHON) -m src.cli delta --run-id "$(RUN_ID)"

chat: ## Start interactive chat. Usage: make chat RUN_ID=<id>
	@if [ -z "$(RUN_ID)" ]; then echo "Usage: make chat RUN_ID=<run_id>"; exit 1; fi
	$(PYTHON) -m src.cli chat --run-id "$(RUN_ID)"

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (20% of rubric — first-class deliverable)
# ─────────────────────────────────────────────────────────────────────────────
eval: ## Run full eval harness → scorecard + eval/results/<timestamp>.json
	$(PYTHON) eval/run_eval.py

eval-delta: ## Delta precision/recall/F1 only
	$(PYTHON) eval/run_eval.py --mode delta

eval-chat: ## Chat correctness + groundedness only
	$(PYTHON) eval/run_eval.py --mode chat

# ─────────────────────────────────────────────────────────────────────────────
# Markup (bonus)
# ─────────────────────────────────────────────────────────────────────────────
markup: ## Generate delta markup overlay. Usage: make markup RUN_ID=<id>
	@if [ -z "$(RUN_ID)" ]; then echo "Usage: make markup RUN_ID=<run_id>"; exit 1; fi
	$(PYTHON) -m src.cli markup --run-id "$(RUN_ID)"

# ─────────────────────────────────────────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────────────────────────────────────────
generate-samples: ## Synthesise PDF document pairs for eval dataset
	$(PYTHON) scripts/generate_sample_pdfs.py

# ─────────────────────────────────────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────────────────────────────────────
traces: ## List recent trace files
	ls -lth traces/ | head -20

show-trace: ## Pretty-print a trace. Usage: make show-trace REQUEST_ID=<id>
	@if [ -z "$(REQUEST_ID)" ]; then echo "Usage: make show-trace REQUEST_ID=<id>"; exit 1; fi
	$(PYTHON) -c "import json; print(json.dumps(json.load(open('traces/$(REQUEST_ID).json')), indent=2))"

# ─────────────────────────────────────────────────────────────────────────────
# Quality
# ─────────────────────────────────────────────────────────────────────────────
test: ## Run full test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests excluding slow / integration tests
	pytest tests/ -v -m "not slow and not integration"

lint: ## Lint with ruff
	ruff check src/ eval/ tests/ scripts/

format: ## Auto-format with ruff
	ruff format src/ eval/ tests/ scripts/

typecheck: ## Type-check with mypy
	mypy src/ --ignore-missing-imports

# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────
docker-up: ## Start infrastructure (MySQL + Redis)
	docker compose up -d mysql redis
	@echo "$(GREEN)Waiting for services to be healthy...$(RESET)"
	docker compose ps

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Follow all service logs
	docker compose logs -f

docker-build: ## Build the app Docker image
	docker compose build app

docker-full: ## Start all services including app
	docker compose up -d

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean: ## Remove generated files (traces, markup, pycache)
	rm -f traces/*.json data/markup/*.pdf
	find . -name "*.pyc" -delete 2>/dev/null; true
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; true

clean-all: clean ## Remove everything including chroma data and htmlcov
	rm -rf data/chroma/* htmlcov/ .mypy_cache/ .ruff_cache/
