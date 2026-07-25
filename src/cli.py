"""
delta-chat · src/cli.py
══════════════════════════════════════════════════════════════════════════════
Click CLI — the primary user interface for this system.
All commands wire the same pipeline/chat/markup code as the FastAPI routes.

Usage examples:
  delta-chat pipeline --pid-a ./doc_a.pdf --pid-b ./doc_b.pdf
  delta-chat chat --run-id <id>
  delta-chat markup --run-id <id>
  delta-chat eval
══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.observability.logging import configure_logging

console = Console()


@click.group()
@click.option("--log-format", default=None, help="Override log format: json|console")
def main(log_format: str | None):
    """delta-chat — Document Delta & Grounded Chat CLI"""
    if log_format:
        os.environ["LOG_FORMAT"] = log_format
    configure_logging()


@main.command()
@click.option("--pid-a", required=True, help="PID for revision A")
@click.option("--pid-b", required=True, help="PID for revision B")
@click.option("--path-a", required=True, help="File path for revision A")
@click.option("--path-b", required=True, help="File path for revision B")
@click.option("--label-a", default=None, help="Revision label for A")
@click.option("--label-b", default=None, help="Revision label for B")
def ingest(pid_a, pid_b, path_a, path_b, label_a, label_b):
    """Ingest two documents and compute delta."""
    from src.pipeline.graph import run_pipeline

    console.print(Panel(
        f"[bold cyan]Ingesting documents[/bold cyan]\n"
        f"PID A: {pid_a} → {path_a}\n"
        f"PID B: {pid_b} → {path_b}",
        title="delta-chat ingest",
    ))

    state = run_pipeline(
        pid_a=pid_a, path_a=path_a,
        pid_b=pid_b, path_b=path_b,
        revision_label_a=label_a,
        revision_label_b=label_b,
    )
    _print_run_summary(state)
    console.print(f"\n[green]Run ID: {state['run_id']}[/green]")
    console.print("Use this run ID with: delta-chat chat --run-id <id>")


@main.command()
@click.option("--pid-a", required=True, help="PID for revision A")
@click.option("--pid-b", required=True, help="PID for revision B")
@click.option(
    "--path-a",
    default=None,
    help="File path for revision A (can also be --pid-a if path)",
)
@click.option("--path-b", default=None, help="File path for revision B")
@click.pass_context
def pipeline(ctx, pid_a, pid_b, path_a, path_b):
    """
    Full pipeline: ingest → delta → report → index → interactive chat.
    This is the single documented command for the acceptance criteria.
    """
    # If no explicit path, treat pid as path (common shorthand)
    path_a = path_a or pid_a
    path_b = path_b or pid_b

    from src.pipeline.graph import run_pipeline

    console.print(Panel(
        f"[bold cyan]Full pipeline: ingest → delta → index → chat[/bold cyan]\n"
        f"A: {path_a}\nB: {path_b}",
        title="delta-chat pipeline",
    ))

    state = run_pipeline(
        pid_a=pid_a, path_a=path_a,
        pid_b=pid_b, path_b=path_b,
    )
    _print_run_summary(state)

    errors = state.get("errors", [])
    if errors:
        console.print(f"[red]Pipeline completed with {len(errors)} error(s)[/red]")
        for e in errors:
            console.print(f"  [red]• [{e['stage']}] {e['message']}[/red]")
    else:
        console.print(f"\n[green]✓ Pipeline complete. Run ID: {state['run_id']}[/green]")
        # Launch interactive chat
        ctx.invoke(chat, run_id=state["run_id"])


@main.command()
@click.option("--run-id", required=True, help="Run ID from a previous ingest")
def chat(run_id: str):
    """Interactive chat session over indexed documents."""
    from src.chat.answer import AnswerEngine
    from src.observability.tracing import RequestTracer

    console.print(Panel(
        f"[bold cyan]Grounded chat — Run ID: {run_id}[/bold cyan]\n"
        "Type your question and press Enter. Type 'quit' or 'exit' to stop.\n"
        "Type 'stats' to see cache hit rate.",
        title="delta-chat chat",
    ))

    tracer = RequestTracer()
    engine = AnswerEngine(run_id=run_id, tracer=tracer)

    while True:
        try:
            query = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Chat session ended.[/dim]")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        if query.lower() == "stats":
            console.print(f"Cache stats: {engine.cache.stats()}")
            continue

        try:
            answer = engine.answer(query)
            console.print(f"\n[bold green]Assistant:[/bold green]")
            console.print(engine.format_answer_for_display(answer))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@main.command()
@click.option("--run-id", required=True, help="Run ID to generate markup for")
@click.option("--path-a", required=True, help="Original path to document A")
@click.option("--path-b", required=True, help="Original path to document B")
def markup(run_id: str, path_a: str, path_b: str):
    """Generate delta markup overlay PDFs (bonus feature)."""
    import json

    delta_path = os.path.join("data", "runs", run_id, "delta_report.json")
    if not os.path.exists(delta_path):
        console.print(f"[red]Delta report not found for run {run_id}[/red]")
        sys.exit(1)

    from src.canonical.model import DeltaReport
    from src.markup.overlay import DeltaMarkupOverlay

    with open(delta_path) as f:
        report = DeltaReport.model_validate(json.load(f))

    overlay = DeltaMarkupOverlay()
    a_out, b_out = overlay.overlay(report, path_a, path_b, run_id)
    console.print(f"[green]✓ Markup complete:[/green]")
    console.print(f"  PID A annotated: {a_out}")
    console.print(f"  PID B annotated: {b_out}")


@main.command()
@click.option("--mode", default="full", type=click.Choice(["full", "delta", "chat"]),
              help="Eval mode")
def eval(mode: str):
    """Run evaluation harness and print scorecard."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "eval/run_eval.py", "--mode", mode],
        cwd=os.getcwd(),
    )
    sys.exit(result.returncode)


def _print_run_summary(state: dict):
    """Print a Rich table summarising a pipeline run."""
    report = state.get("delta_report")
    if not report:
        console.print("[yellow]No delta report generated.[/yellow]")
        return

    s = report.summary
    table = Table(title="Delta Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Changes", str(s.get("total_changes", 0)))
    table.add_row("Added", f"[green]{s.get('added', 0)}[/green]")
    table.add_row("Removed", f"[red]{s.get('removed', 0)}[/red]")
    table.add_row("Modified", f"[yellow]{s.get('modified', 0)}[/yellow]")
    table.add_row("Avg Confidence", f"{s.get('avg_confidence', 0.0):.2f}")
    table.add_row("Low Confidence", str(s.get("low_confidence_count", 0)))

    timings = state.get("stage_timings", {})
    total_ms = sum(timings.values())
    table.add_row("Total Time", f"{total_ms:.0f} ms")
    for stage, ms in timings.items():
        table.add_row(f"  └ {stage}", f"{ms:.0f} ms")

    console.print(table)

    if state.get("delta_md_path"):
        console.print(f"\n📄 Report: [link={state['delta_md_path']}]{state['delta_md_path']}[/link]")
    if state.get("delta_json_path"):
        console.print(f"📊 JSON:   {state['delta_json_path']}")


if __name__ == "__main__":
    main()
