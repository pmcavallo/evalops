"""Regression testing CLI command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="regression", help="Run regression tests.")
console = Console()


def default_target(x: str) -> str:
    """Default target function that returns the input as-is."""
    return x


@app.callback(invoke_without_command=True)
def main(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON file"),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target function as 'module:function'"
    ),
    baseline: Path | None = typer.Option(None, "--baseline", help="Path to baseline JSON file"),
    metrics: list[str] = typer.Option(
        ["exact_match"], "--metric", "-m", help="Metrics to evaluate"
    ),
    thresholds: list[str] = typer.Option(
        [], "--threshold", help="Custom thresholds as 'metric:value'"
    ),
    output_format: str = typer.Option(
        "text", "--output-format", help="Output format: text, github"
    ),
    output_file: Path | None = typer.Option(None, "--output-file", help="Save report to file"),
) -> None:
    """Run regression tests and exit with non-zero if failed."""
    from evalops import EvalDataset
    from evalops.cli.main import load_target_function, parse_metrics
    from evalops.comparison.regression import MetricThreshold, run_regression_test

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    eval_dataset = EvalDataset.from_json(dataset)

    # Load target
    if target:
        target_func = load_target_function(target)
    else:
        target_func = default_target

    # Parse metrics
    metric_instances = parse_metrics(metrics)

    # Parse thresholds
    parsed_thresholds = []
    for t in thresholds:
        if ":" not in t:
            continue
        name, val = t.split(":", 1)
        parsed_thresholds.append(MetricThreshold(name, min_pass_rate=float(val)))

    # Run tests
    report = asyncio.run(
        run_regression_test(
            dataset=eval_dataset,
            target=target_func,
            metrics=metric_instances,
            thresholds=parsed_thresholds,
            baseline_path=baseline,
        )
    )

    # Output results
    if output_format == "github":
        summary = report.to_github_summary()
        Path("regression_summary.md").write_text(summary, encoding="utf-8")
        console.print(summary)
    else:
        report.print_summary()

    if output_file:
        output_file.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)


if __name__ == "__main__":
    app()
