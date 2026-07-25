#!/usr/bin/env python
"""
delta-chat · eval/run_eval.py
══════════════════════════════════════════════════════════════════════════════
Runnable evaluation harness.

Usage:
    python eval/run_eval.py              # full eval
    python eval/run_eval.py --mode delta # delta metrics only
    python eval/run_eval.py --mode chat  # chat metrics only

Output:
    • Rich scorecard table printed to stdout
    • eval/results/<timestamp>.json  (regression-comparable)
    • Exit code 0 on success, 1 if any metric below threshold

Regression-friendly: each run writes a JSON result keyed by timestamp.
Compare two runs:
    python eval/compare_runs.py eval/results/<ts1>.json eval/results/<ts2>.json

Failure table documented inline and in README.
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Make src importable when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── THRESHOLDS (below these, the eval flags a regression) ────────────────────
MIN_DELTA_F1 = 0.50
MIN_CHAT_CORRECTNESS = 0.60
MIN_CHAT_GROUNDEDNESS = 0.65


def load_dataset_pairs() -> list[dict]:
    """Load all dataset pairs from eval/datasets/*/"""
    pairs = []
    for pair_dir in sorted(os.listdir(DATASETS_DIR)):
        pair_path = os.path.join(DATASETS_DIR, pair_dir)
        if not os.path.isdir(pair_path):
            continue
        gt_path = os.path.join(pair_path, "ground_truth_delta.json")
        qa_path = os.path.join(pair_path, "qa_pairs.json")
        meta_path = os.path.join(pair_path, "metadata.json")
        if not os.path.exists(gt_path):
            continue
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        with open(gt_path) as f:
            gt = json.load(f)
        qa_pairs = []
        if os.path.exists(qa_path):
            with open(qa_path) as f:
                qa_pairs = json.load(f)
        pairs.append({
            "pair_id": pair_dir,
            "path_a": os.path.join(DATASETS_DIR, pair_dir, "doc_a.pdf"),
            "path_b": os.path.join(DATASETS_DIR, pair_dir, "doc_b.pdf"),
            "pid_a": meta.get("pid_a", f"{pair_dir}_a"),
            "pid_b": meta.get("pid_b", f"{pair_dir}_b"),
            "ground_truth_delta": gt,
            "qa_pairs": qa_pairs,
            "meta": meta,
        })
    return pairs


def run_delta_eval(pairs: list[dict]) -> dict:
    """Run delta evaluation across all pairs."""
    from eval.metrics import GTDeltaItem, compute_delta_metrics
    from src.pipeline.graph import run_pipeline

    all_tp = all_fp = all_fn = 0
    pair_results = []

    for pair in pairs:
        pid_a, pid_b = pair["pid_a"], pair["pid_b"]
        path_a, path_b = pair["path_a"], pair["path_b"]

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            console.print(f"[yellow]Skip {pair['pair_id']}: document files not found[/yellow]")
            pair_results.append({
                "pair_id": pair["pair_id"],
                "status": "skipped",
                "reason": "document files not found",
            })
            continue

        console.print(f"  Evaluating pair [cyan]{pair['pair_id']}[/cyan]...")

        try:
            state = run_pipeline(pid_a=pid_a, path_a=path_a,
                                  pid_b=pid_b, path_b=path_b)
            report = state.get("delta_report")
            if not report:
                raise ValueError("No delta report produced")

            predicted = [item.model_dump(
                exclude={"element_a", "element_b"}
            ) for item in report.items]

            gt_items = [GTDeltaItem.from_dict(d) for d in pair["ground_truth_delta"].get("items", [])]
            metrics = compute_delta_metrics(predicted, gt_items)

            all_tp += metrics.true_positives
            all_fp += metrics.false_positives
            all_fn += metrics.false_negatives

            pair_results.append({
                "pair_id": pair["pair_id"],
                "status": "ok",
                "run_id": state["run_id"],
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "fn": metrics.false_negatives,
                "predicted": metrics.total_predicted,
                "ground_truth": metrics.total_ground_truth,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            console.print(f"  [red]Error on {pair['pair_id']}: {e}[/red]")
            pair_results.append({
                "pair_id": pair["pair_id"],
                "status": "error",
                "error": str(e),
            })

    # Aggregate
    total_tp = all_tp
    total_fp = all_fp
    total_fn = all_fn
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = (2 * agg_precision * agg_recall / (agg_precision + agg_recall)
               if (agg_precision + agg_recall) > 0 else 0.0)

    return {
        "aggregate": {
            "precision": round(agg_precision, 4),
            "recall": round(agg_recall, 4),
            "f1": round(agg_f1, 4),
        },
        "pairs": pair_results,
    }


def run_chat_eval(pairs: list[dict]) -> dict:
    """Run chat evaluation across all pairs."""
    from eval.metrics import evaluate_chat_qa
    from src.pipeline.graph import run_pipeline
    from src.chat.answer import AnswerEngine
    from src.observability.tracing import RequestTracer

    all_scores = []
    pair_results = []

    for pair in pairs:
        if not pair.get("qa_pairs"):
            continue

        path_a, path_b = pair["path_a"], pair["path_b"]
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        console.print(f"  Chat eval pair [cyan]{pair['pair_id']}[/cyan]...")

        try:
            # Run pipeline to get run_id
            state = run_pipeline(
                pid_a=pair["pid_a"], path_a=path_a,
                pid_b=pair["pid_b"], path_b=path_b,
            )
            run_id = state["run_id"]

            # Answer each QA pair
            tracer = RequestTracer()
            engine = AnswerEngine(run_id=run_id, tracer=tracer)

            system_answers = []
            for qa in pair["qa_pairs"]:
                ans = engine.answer(qa["question"])
                system_answers.append({
                    "answer": ans.answer,
                    "citations": [
                        {"label": c.label, "snippet": c.snippet}
                        for c in ans.citations
                    ],
                    "insufficient_grounding": ans.insufficient_grounding,
                })

            metrics = evaluate_chat_qa(pair["qa_pairs"], system_answers)
            all_scores.extend(metrics.scores)

            pair_results.append({
                "pair_id": pair["pair_id"],
                "status": "ok",
                "avg_correctness": metrics.avg_correctness,
                "avg_groundedness": metrics.avg_groundedness,
                "qa_count": metrics.total_qa,
            })
        except Exception as e:
            console.print(f"  [red]Error on {pair['pair_id']}: {e}[/red]")
            pair_results.append({"pair_id": pair["pair_id"], "status": "error", "error": str(e)})

    if not all_scores:
        return {"aggregate": {"correctness": 0.0, "groundedness": 0.0}, "pairs": pair_results}

    agg_correctness = sum(s["correctness"] for s in all_scores) / len(all_scores)
    agg_groundedness = sum(s["groundedness"] for s in all_scores) / len(all_scores)

    return {
        "aggregate": {
            "correctness": round(agg_correctness, 4),
            "groundedness": round(agg_groundedness, 4),
            "total_qa": len(all_scores),
        },
        "pairs": pair_results,
        "per_question": all_scores,
    }


def print_scorecard(delta_results: dict, chat_results: dict, mode: str, elapsed_ms: float):
    """Print Rich scorecard table."""
    console.print("\n")
    console.rule("[bold cyan]delta-chat Evaluation Scorecard[/bold cyan]")

    if mode in {"full", "delta"}:
        agg = delta_results.get("aggregate", {})
        delta_table = Table(title="Delta Metrics", show_header=True)
        delta_table.add_column("Metric", style="cyan")
        delta_table.add_column("Value", style="white")
        delta_table.add_column("Threshold", style="dim")
        delta_table.add_column("Status", style="white")

        p = agg.get("precision", 0)
        r = agg.get("recall", 0)
        f = agg.get("f1", 0)
        delta_table.add_row("Precision", f"{p:.4f}", "—", "")
        delta_table.add_row("Recall", f"{r:.4f}", "—", "")
        delta_table.add_row(
            "F1", f"{f:.4f}", f">= {MIN_DELTA_F1}",
            "[green]PASS[/green]" if f >= MIN_DELTA_F1 else "[red]FAIL[/red]",
        )
        console.print(delta_table)

    if mode in {"full", "chat"}:
        agg = chat_results.get("aggregate", {})
        chat_table = Table(title="Chat Metrics", show_header=True)
        chat_table.add_column("Metric", style="cyan")
        chat_table.add_column("Value", style="white")
        chat_table.add_column("Threshold", style="dim")
        chat_table.add_column("Status", style="white")

        c = agg.get("correctness", 0)
        g = agg.get("groundedness", 0)
        chat_table.add_row(
            "Correctness", f"{c:.4f}", f">= {MIN_CHAT_CORRECTNESS}",
            "[green]PASS[/green]" if c >= MIN_CHAT_CORRECTNESS else "[red]FAIL[/red]",
        )
        chat_table.add_row(
            "Groundedness", f"{g:.4f}", f">= {MIN_CHAT_GROUNDEDNESS}",
            "[green]PASS[/green]" if g >= MIN_CHAT_GROUNDEDNESS else "[red]FAIL[/red]",
        )
        chat_table.add_row("Q&A pairs tested", str(agg.get("total_qa", 0)), "—", "")
        console.print(chat_table)

    console.print(f"\n[dim]Evaluation completed in {elapsed_ms:.0f} ms[/dim]")


def main():
    parser = argparse.ArgumentParser(description="delta-chat evaluation harness")
    parser.add_argument("--mode", default="full",
                        choices=["full", "delta", "chat"],
                        help="Evaluation mode")
    args = parser.parse_args()

    from src.observability.logging import configure_logging
    configure_logging()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = load_dataset_pairs()

    if not pairs:
        console.print("[red]No eval datasets found in eval/datasets/[/red]")
        console.print("Run: python scripts/generate_sample_pdfs.py  to create sample data")
        sys.exit(1)

    console.print(f"\n[bold cyan]Running eval ({args.mode} mode) on {len(pairs)} pair(s)...[/bold cyan]\n")

    t0 = time.time()
    delta_results: dict = {}
    chat_results: dict = {}

    if args.mode in {"full", "delta"}:
        console.print("[bold]Delta evaluation:[/bold]")
        delta_results = run_delta_eval(pairs)

    if args.mode in {"full", "chat"}:
        console.print("[bold]Chat evaluation:[/bold]")
        chat_results = run_chat_eval(pairs)

    elapsed_ms = (time.time() - t0) * 1000
    print_scorecard(delta_results, chat_results, args.mode, elapsed_ms)

    # Write result file
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = os.path.join(RESULTS_DIR, f"{timestamp}.json")
    result = {
        "timestamp": timestamp,
        "mode": args.mode,
        "elapsed_ms": round(elapsed_ms, 1),
        "dataset_version": "v1",
        "delta": delta_results,
        "chat": chat_results,
        "thresholds": {
            "min_delta_f1": MIN_DELTA_F1,
            "min_chat_correctness": MIN_CHAT_CORRECTNESS,
            "min_chat_groundedness": MIN_CHAT_GROUNDEDNESS,
        },
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"\n[green]Results written to: {result_path}[/green]")

    # Persist to DB
    try:
        from src.db.session import get_session_factory
        from src.db.models import EvalRun
        factory = get_session_factory()
        with factory() as session:
            er = EvalRun(
                run_timestamp=timestamp,
                mode=args.mode,
                delta_precision=delta_results.get("aggregate", {}).get("precision", 0.0),
                delta_recall=delta_results.get("aggregate", {}).get("recall", 0.0),
                delta_f1=delta_results.get("aggregate", {}).get("f1", 0.0),
                delta_pairs_tested=len([p for p in pairs
                                         if os.path.exists(p["path_a"])]),
                chat_correctness=chat_results.get("aggregate", {}).get("correctness", 0.0),
                chat_groundedness=chat_results.get("aggregate", {}).get("groundedness", 0.0),
                chat_qa_tested=chat_results.get("aggregate", {}).get("total_qa", 0),
                scorecard_json=result,
                result_path=result_path,
            )
            session.add(er)
            session.commit()
    except Exception as e:
        console.print(f"[yellow]Warning: could not persist eval to DB: {e}[/yellow]")

    # Exit code
    failed = False
    if args.mode in {"full", "delta"}:
        if delta_results.get("aggregate", {}).get("f1", 0) < MIN_DELTA_F1:
            failed = True
    if args.mode in {"full", "chat"}:
        if chat_results.get("aggregate", {}).get("correctness", 0) < MIN_CHAT_CORRECTNESS:
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
