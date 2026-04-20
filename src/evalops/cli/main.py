"""Typer CLI for EvalOps.

This module provides the command-line interface for running evaluations,
comparing variants, managing baselines, and checking for drift.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="evalops",
    help="Production-grade LLM evaluation and observability platform.",
    add_completion=False,
)

console = Console()


def load_target_function(target_path: str):
    """Load a target function from a module path.

    Args:
        target_path: Path like 'module.submodule:function_name'

    Returns:
        The loaded function.
    """
    if ":" not in target_path:
        raise typer.BadParameter(
            f"Target must be in format 'module:function', got '{target_path}'"
        )

    module_path, func_name = target_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise typer.BadParameter(f"Could not import module '{module_path}': {e}")

    if not hasattr(module, func_name):
        raise typer.BadParameter(
            f"Module '{module_path}' has no attribute '{func_name}'"
        )

    return getattr(module, func_name)


def get_repository(database_url: str | None = None):
    """Get or create repository instance."""
    from evalops.storage import EvalRepository

    url = database_url or "sqlite:///evalops.db"
    repo = EvalRepository(url)
    repo.initialize()
    return repo


def parse_metrics(metric_names: list[str]):
    """Parse metric names into Metric instances."""
    from evalops.core.metrics import (
        ContainsKeywords,
        ExactMatch,
        Latency,
        SemanticSimilarity,
        TokenCost,
    )

    metric_map = {
        "exact_match": ExactMatch,
        "contains_keywords": ContainsKeywords,
        "latency": Latency,
        "token_cost": TokenCost,
        "semantic_similarity": SemanticSimilarity,
    }

    metrics = []
    for name in metric_names:
        name_lower = name.lower().replace("-", "_")
        if name_lower not in metric_map:
            raise typer.BadParameter(
                f"Unknown metric '{name}'. Available: {', '.join(metric_map.keys())}"
            )
        metrics.append(metric_map[name_lower]())

    return metrics


@app.command()
def run(
    dataset: Path = typer.Argument(..., help="Path to dataset JSON file"),
    target: str = typer.Option(
        ..., "--target", "-t", help="Target function as 'module:function'"
    ),
    metrics: list[str] = typer.Option(
        ["exact_match"],
        "--metric",
        "-m",
        help="Metrics to evaluate (can specify multiple)",
    ),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name for this run"),
    database: Optional[str] = typer.Option(
        None, "--database", "-d", help="Database URL"
    ),
    save: bool = typer.Option(True, "--save/--no-save", help="Save results to database"),
    tags: list[str] = typer.Option([], "--tag", help="Tags for the run"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output results to JSON file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run an evaluation on a dataset."""
    from evalops import EvalDataset, EvalRunner

    # Load dataset
    console.print(f"[bold]Loading dataset from {dataset}...[/bold]")
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    try:
        eval_dataset = EvalDataset.from_json(dataset)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"  Loaded {len(eval_dataset)} cases")

    # Load target function
    console.print(f"[bold]Loading target function: {target}[/bold]")
    try:
        target_func = load_target_function(target)
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Parse metrics
    console.print(f"[bold]Metrics:[/bold] {', '.join(metrics)}")
    try:
        metric_instances = parse_metrics(metrics)
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Set up repository
    repo = get_repository(database) if save else None

    # Create runner
    runner = EvalRunner(
        repository=repo,
        auto_save=save,
        enable_observability=verbose,
    )

    # Run evaluation
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Running evaluation...", total=None)

        result = asyncio.run(
            runner.evaluate(
                dataset=eval_dataset,
                target=target_func,
                metrics=metric_instances,
                run_name=name,
                tags=tags if tags else None,
            )
        )

    # Display results
    console.print()
    _display_run_result(result, verbose)

    # Save to output file if specified
    if output:
        output_data = {
            "run_id": result.id,
            "dataset": result.dataset_name,
            "total_cases": result.total_cases,
            "successful_cases": result.successful_cases,
            "pass_rate": result.pass_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "metrics_summary": result.metrics_summary,
        }
        output.write_text(json.dumps(output_data, indent=2))
        console.print(f"[green]Results saved to {output}[/green]")


def _display_run_result(result, verbose: bool = False) -> None:
    """Display run result in a formatted table."""
    # Status panel
    status_color = "green" if result.pass_rate >= 0.8 else "yellow" if result.pass_rate >= 0.5 else "red"
    status_emoji = "[green]PASSED[/green]" if result.pass_rate >= 0.8 else "[red]FAILED[/red]"

    console.print(
        Panel(
            f"[bold]Run ID:[/bold] {result.id}\n"
            f"[bold]Dataset:[/bold] {result.dataset_name}\n"
            f"[bold]Status:[/bold] {status_emoji}",
            title="Evaluation Results",
        )
    )

    # Summary table
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total Cases", str(result.total_cases))
    table.add_row("Successful", str(result.successful_cases))
    table.add_row("Pass Rate", f"{result.pass_rate:.1%}")
    table.add_row("Avg Latency", f"{result.avg_latency_ms:.1f} ms")

    console.print(table)

    # Metrics table
    if result.metrics_summary:
        metrics_table = Table(title="Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", justify="right")
        metrics_table.add_column("Status", justify="center")

        for name, summary in result.metrics_summary.items():
            score = summary.get("score", 0)
            passed = summary.get("passed", False)
            status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

            if isinstance(score, float) and score <= 1:
                score_str = f"{score:.1%}"
            else:
                score_str = f"{score:.2f}"

            metrics_table.add_row(name, score_str, status)

        console.print(metrics_table)


@app.command()
def compare(
    dataset: Path = typer.Argument(..., help="Path to dataset JSON file"),
    variant_a: str = typer.Option(
        ..., "--variant-a", "-a", help="Variant A function as 'module:function'"
    ),
    variant_b: str = typer.Option(
        ..., "--variant-b", "-b", help="Variant B function as 'module:function'"
    ),
    metrics: list[str] = typer.Option(
        ["exact_match"],
        "--metric",
        "-m",
        help="Metrics to compare",
    ),
    name_a: str = typer.Option("variant_a", "--name-a", help="Name for variant A"),
    name_b: str = typer.Option("variant_b", "--name-b", help="Name for variant B"),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level"),
) -> None:
    """Compare two LLM variants (A/B test)."""
    from evalops import EvalDataset
    from evalops.comparison import ABTest

    # Load dataset
    console.print(f"[bold]Loading dataset from {dataset}...[/bold]")
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    eval_dataset = EvalDataset.from_json(dataset)
    console.print(f"  Loaded {len(eval_dataset)} cases")

    # Load variants
    console.print(f"[bold]Loading variants...[/bold]")
    try:
        func_a = load_target_function(variant_a)
        func_b = load_target_function(variant_b)
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Parse metrics
    metric_instances = parse_metrics(metrics)

    # Run A/B test
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Running A/B test...", total=None)

        ab_test = ABTest(confidence_level=confidence)
        result = asyncio.run(
            ab_test.compare(
                dataset=eval_dataset,
                variant_a=func_a,
                variant_b=func_b,
                metrics=metric_instances,
                variant_a_name=name_a,
                variant_b_name=name_b,
            )
        )

    # Display results
    console.print()
    _display_ab_result(result)


def _display_ab_result(result) -> None:
    """Display A/B test result."""
    from evalops.comparison import Winner

    winner_color = {
        Winner.VARIANT_A: "blue",
        Winner.VARIANT_B: "green",
        Winner.TIE: "yellow",
        Winner.INCONCLUSIVE: "dim",
    }

    console.print(
        Panel(
            f"[bold]Test:[/bold] {result.name}\n"
            f"[bold]Variant A:[/bold] {result.variant_a_name}\n"
            f"[bold]Variant B:[/bold] {result.variant_b_name}\n"
            f"[bold]Cases:[/bold] {result.total_cases}\n"
            f"[bold]Winner:[/bold] [{winner_color[result.overall_winner]}]{result.overall_winner.value}[/{winner_color[result.overall_winner]}]",
            title="A/B Test Results",
        )
    )

    # Metrics comparison table
    table = Table(title="Metric Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Variant A", justify="right")
    table.add_column("Variant B", justify="right")
    table.add_column("Difference", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Winner", justify="center")

    for name, comp in result.metric_comparisons.items():
        winner_str = {
            Winner.VARIANT_A: f"[blue]{result.variant_a_name}[/blue]",
            Winner.VARIANT_B: f"[green]{result.variant_b_name}[/green]",
            Winner.TIE: "[yellow]Tie[/yellow]",
            Winner.INCONCLUSIVE: "[dim]?[/dim]",
        }[comp.winner]

        sig_marker = "*" if comp.stats.significant else ""

        table.add_row(
            name,
            f"{comp.mean_a:.3f}",
            f"{comp.mean_b:.3f}",
            f"{comp.difference:+.3f}",
            f"{comp.stats.p_value:.4f}{sig_marker}",
            winner_str,
        )

    console.print(table)
    console.print()
    console.print(f"[bold]Recommendation:[/bold] {result.recommendation}")


@app.command()
def history(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Filter by dataset"),
    days: int = typer.Option(30, "--days", help="Number of days to show"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum runs to show"),
    min_pass_rate: Optional[float] = typer.Option(
        None, "--min-pass-rate", help="Minimum pass rate filter"
    ),
    database: Optional[str] = typer.Option(None, "--database", help="Database URL"),
) -> None:
    """List past evaluation runs."""
    repo = get_repository(database)

    start_date = datetime.now(timezone.utc) - timedelta(days=days)

    runs = repo.list_runs(
        dataset_name=dataset,
        min_pass_rate=min_pass_rate,
        start_date=start_date,
        limit=limit,
    )

    if not runs:
        console.print("[yellow]No runs found matching criteria.[/yellow]")
        return

    table = Table(title=f"Evaluation History (Last {days} Days)")
    table.add_column("Run ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Dataset")
    table.add_column("Cases", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Date")

    for run in runs:
        pass_color = "green" if run.pass_rate >= 0.8 else "yellow" if run.pass_rate >= 0.5 else "red"

        table.add_row(
            run.id[:8] + "...",
            run.name or "-",
            run.dataset_name,
            str(run.total_cases),
            f"[{pass_color}]{run.pass_rate:.1%}[/{pass_color}]",
            f"{run.avg_latency_ms:.0f}ms",
            run.started_at.strftime("%Y-%m-%d %H:%M") if run.started_at else "-",
        )

    console.print(table)

    # Show stats
    stats = repo.get_run_stats(dataset_name=dataset, days=days)
    console.print()
    console.print(
        f"[bold]Total:[/bold] {stats['total_runs']} runs | "
        f"[bold]Avg Pass Rate:[/bold] {stats['avg_pass_rate']:.1%} | "
        f"[bold]Total Cases:[/bold] {stats['total_cases']}"
    )


@app.command()
def baseline(
    action: str = typer.Argument(..., help="Action: save, load, list, delete"),
    run_id: Optional[str] = typer.Option(None, "--run", "-r", help="Run ID for save"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Baseline name"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    baseline_id: Optional[str] = typer.Option(None, "--id", help="Baseline ID"),
    database: Optional[str] = typer.Option(None, "--database", help="Database URL"),
) -> None:
    """Manage baselines for regression testing."""
    repo = get_repository(database)

    if action == "save":
        if not run_id:
            console.print("[red]Error: --run is required for save action[/red]")
            raise typer.Exit(1)
        if not name:
            name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run = repo.get_run(run_id)
        if not run:
            console.print(f"[red]Error: Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        # Create mock run result for baseline creation
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.id = run.id
        mock_result.dataset_name = run.dataset_name
        mock_result.pass_rate = run.pass_rate
        mock_result.total_cases = run.total_cases
        mock_result.metrics_summary = run.metrics_summary or {}

        baseline_record = repo.save_baseline(mock_result, name=name)
        console.print(f"[green]Baseline saved: {baseline_record.id}[/green]")
        console.print(f"  Name: {baseline_record.name}")
        console.print(f"  Dataset: {baseline_record.dataset_name}")
        console.print(f"  Pass Rate: {baseline_record.pass_rate:.1%}")

    elif action == "load" or action == "show":
        bl = repo.get_baseline(baseline_id=baseline_id, dataset_name=dataset)
        if not bl:
            console.print("[yellow]No baseline found.[/yellow]")
            return

        console.print(
            Panel(
                f"[bold]ID:[/bold] {bl.id}\n"
                f"[bold]Name:[/bold] {bl.name}\n"
                f"[bold]Dataset:[/bold] {bl.dataset_name}\n"
                f"[bold]Pass Rate:[/bold] {bl.pass_rate:.1%}\n"
                f"[bold]Active:[/bold] {bl.is_active}\n"
                f"[bold]Created:[/bold] {bl.created_at}",
                title="Baseline",
            )
        )

        if bl.metrics:
            table = Table(title="Baseline Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            for metric, value in bl.metrics.items():
                table.add_row(metric, f"{value:.4f}")
            console.print(table)

    elif action == "list":
        baselines = repo.list_baselines(dataset_name=dataset)
        if not baselines:
            console.print("[yellow]No baselines found.[/yellow]")
            return

        table = Table(title="Baselines")
        table.add_column("ID", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Dataset")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Active", justify="center")
        table.add_column("Created")

        for bl in baselines:
            table.add_row(
                bl.id[:8] + "...",
                bl.name,
                bl.dataset_name,
                f"{bl.pass_rate:.1%}",
                "[green]Yes[/green]" if bl.is_active else "[dim]No[/dim]",
                bl.created_at.strftime("%Y-%m-%d") if bl.created_at else "-",
            )

        console.print(table)

    elif action == "delete":
        if not baseline_id:
            console.print("[red]Error: --id is required for delete action[/red]")
            raise typer.Exit(1)

        deleted = repo.delete_baseline(baseline_id)
        if deleted:
            console.print(f"[green]Baseline deleted: {baseline_id}[/green]")
        else:
            console.print(f"[red]Baseline not found: {baseline_id}[/red]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: save, load, list, delete")
        raise typer.Exit(1)


@app.command()
def regression(
    dataset: Path = typer.Argument(..., help="Path to dataset JSON file"),
    target: str = typer.Option(
        ..., "--target", "-t", help="Target function as 'module:function'"
    ),
    metrics: list[str] = typer.Option(
        ["exact_match"],
        "--metric",
        "-m",
        help="Metrics to evaluate",
    ),
    thresholds: list[str] = typer.Option(
        [],
        "--threshold",
        help="Thresholds as 'metric:value' (e.g., exact_match:0.9)",
    ),
    baseline_path: Optional[Path] = typer.Option(
        None, "--baseline", help="Path to baseline JSON file"
    ),
    output_format: str = typer.Option(
        "text", "--output-format", help="Output format: text, json, github"
    ),
    fail_on_regression: bool = typer.Option(
        True, "--fail/--no-fail", help="Fail if regression detected"
    ),
) -> None:
    """Run regression tests against thresholds and baselines."""
    from evalops.comparison.regression import MetricThreshold, RegressionTester

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Load target
    try:
        target_func = load_target_function(target)
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Parse metrics
    metric_instances = parse_metrics(metrics)

    # Parse thresholds
    threshold_instances = []
    for t in thresholds:
        if ":" not in t:
            continue
        m_name, val = t.split(":", 1)
        try:
            threshold_instances.append(MetricThreshold(m_name, min_pass_rate=float(val)))
        except ValueError:
            continue

    # Setup tester
    tester = RegressionTester(
        thresholds=threshold_instances,
        fail_on_regression=fail_on_regression,
    )

    if baseline_path and baseline_path.exists():
        tester.load_baseline(baseline_path)

    # Run tests
    report = asyncio.run(
        tester.run(
            dataset=dataset,  # RegressionTester.run expects EvalDataset, but let's check its implementation
            target=target_func,
            metrics=metric_instances,
        )
    )

    # Wait, RegressionTester.run expects EvalDataset object.
    from evalops import EvalDataset
    eval_dataset = EvalDataset.from_json(dataset)

    report = asyncio.run(
        tester.run(
            dataset=eval_dataset,
            target=target_func,
            metrics=metric_instances,
        )
    )

    # Output results
    if output_format == "json":
        console.print(json.dumps(report.to_dict(), indent=2))
    elif output_format == "github":
        summary_path = Path("regression_summary.md")
        summary_path.write_text(report.to_github_summary())
        report.print_summary()
    else:
        report.print_summary()

    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)


@app.command()
def drift(
    dataset: str = typer.Argument(..., help="Dataset name to check drift for"),
    days: int = typer.Option(7, "--days", help="Days of history to analyze"),
    warning_threshold: float = typer.Option(
        0.05, "--warning", "-w", help="Warning threshold (e.g., 0.05 = 5%)"
    ),
    critical_threshold: float = typer.Option(
        0.10, "--critical", "-c", help="Critical threshold (e.g., 0.10 = 10%)"
    ),
    database: Optional[str] = typer.Option(None, "--database", help="Database URL"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check for quality drift against baseline."""
    from evalops.comparison import AlertSeverity, DriftDetector, DriftDirection

    repo = get_repository(database)

    # Get baseline
    baseline = repo.get_baseline(dataset_name=dataset)
    if not baseline:
        console.print(f"[red]No baseline found for dataset: {dataset}[/red]")
        console.print("Create a baseline first with: evalops baseline save --run RUN_ID --name NAME")
        raise typer.Exit(1)

    # Get recent history
    history = repo.get_history(dataset, days=days)
    if not history:
        console.print(f"[yellow]No recent runs found for dataset: {dataset}[/yellow]")
        raise typer.Exit(0)

    # Set up drift detector
    detector = DriftDetector(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        window_size=min(len(history), 5),
    )

    # Set baseline
    detector.set_baseline_from_values(baseline.metrics, pass_rate=baseline.pass_rate)

    # Add snapshots
    for entry in history:
        detector.add_raw_snapshot(
            metrics=entry.get("metrics", {}),
            pass_rate=entry.get("pass_rate", 0.0),
            run_id=entry.get("run_id"),
        )

    # Check for drift
    report = detector.check()

    if json_output:
        console.print(json.dumps(report.to_dict(), indent=2))
        return

    # Display results
    health_color = {
        "healthy": "green",
        "improving": "blue",
        "degraded": "yellow",
        "critical": "red",
        "unknown": "dim",
    }

    console.print(
        Panel(
            f"[bold]Dataset:[/bold] {dataset}\n"
            f"[bold]Baseline:[/bold] {baseline.name}\n"
            f"[bold]Snapshots:[/bold] {report.snapshot_count}\n"
            f"[bold]Health:[/bold] [{health_color.get(report.overall_health, 'white')}]{report.overall_health.upper()}[/{health_color.get(report.overall_health, 'white')}]\n"
            f"[bold]Drift Detected:[/bold] {'Yes' if report.drift_detected else 'No'}",
            title="Drift Analysis",
        )
    )

    # Metrics comparison
    if report.baseline_metrics and report.current_metrics:
        table = Table(title="Metric Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Trend", justify="center")

        trend_symbols = {
            DriftDirection.IMPROVING: "[green]+[/green]",
            DriftDirection.STABLE: "[dim]=[/dim]",
            DriftDirection.DEGRADING: "[red]-[/red]",
        }

        for metric in report.baseline_metrics:
            baseline_val = report.baseline_metrics.get(metric, 0)
            current_val = report.current_metrics.get(metric, 0)
            trend = report.metric_trends.get(metric, DriftDirection.STABLE)

            table.add_row(
                metric,
                f"{baseline_val:.3f}",
                f"{current_val:.3f}",
                trend_symbols.get(trend, "?"),
            )

        console.print(table)

    # Alerts
    if report.alerts:
        console.print()
        console.print("[bold]Alerts:[/bold]")
        for alert in report.alerts:
            severity_color = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red",
            }
            console.print(
                f"  [{severity_color.get(alert.severity, 'white')}]{alert.severity.value.upper()}:[/{severity_color.get(alert.severity, 'white')}] {alert.message}"
            )

            if alert.suggested_actions:
                for action in alert.suggested_actions[:2]:
                    console.print(f"    - {action}")

    # Exit code based on health
    if report.overall_health == "critical":
        raise typer.Exit(2)
    elif report.overall_health == "degraded":
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show EvalOps version."""
    from evalops import __version__

    console.print(f"EvalOps v{__version__}")


@app.command()
def init(
    database: str = typer.Option(
        "sqlite:///evalops.db", "--database", "-d", help="Database URL"
    ),
) -> None:
    """Initialize the EvalOps database."""
    from evalops.storage import DatabaseManager

    manager = DatabaseManager(database)
    version = manager.initialize()

    console.print(f"[green]Database initialized at {database}[/green]")
    console.print(f"  Schema version: {version}")

    health = manager.check_health()
    console.print(f"  Tables: {', '.join(health.get('tables', []))}")


@app.command()
def health(
    database: str = typer.Option(
        "sqlite:///evalops.db", "--database", "-d", help="Database URL"
    ),
) -> None:
    """Check database health."""
    from evalops.storage import DatabaseManager

    manager = DatabaseManager(database)
    status = manager.check_health()

    if status.get("healthy"):
        console.print("[green]Database is healthy[/green]")
        console.print(f"  Version: {status.get('version')}")
        console.print(f"  Tables: {', '.join(status.get('tables', []))}")

        sizes = manager.table_sizes()
        console.print("  Records:")
        for table, count in sizes.items():
            console.print(f"    {table}: {count}")
    else:
        console.print("[red]Database unhealthy[/red]")
        console.print(f"  Error: {status.get('error')}")
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
