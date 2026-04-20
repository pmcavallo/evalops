"""Regression testing CLI command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from evalops.cli.main import load_target_function, parse_metrics

app = typer.Typer()
console = Console()


@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON file"),
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
    from evalops import EvalDataset
    from evalops.comparison.regression import MetricThreshold, RegressionTester

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    try:
        eval_dataset = EvalDataset.from_json(dataset)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)

    # Load target
    try:
        target_func = load_target_function(target)
    except Exception as e:
        console.print(f"[red]Error loading target: {e}[/red]")
        raise typer.Exit(1)

    # Parse metrics
    try:
        metric_instances = parse_metrics(metrics)
    except Exception as e:
        console.print(f"[red]Error parsing metrics: {e}[/red]")
        raise typer.Exit(1)

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
            dataset=eval_dataset,
            target=target_func,
            metrics=metric_instances,
        )
    )

    # Output results
    if output_format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif output_format == "github":
        summary_path = Path("regression_summary.md")
        summary_path.write_text(report.to_github_summary())
        report.print_summary()
    else:
        report.print_summary()

    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)


if __name__ == "__main__":
    typer.run(main)
