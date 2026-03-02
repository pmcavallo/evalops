"""Regression testing CLI command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(name="regression", help="Run regression tests.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON file"),
    target: str = typer.Option(None, "--target", "-t", help="Target function as 'module:function'"),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help="Path to baseline JSON file"),
    metrics: list[str] = typer.Option(
        ["exact_match"], "--metric", "-m", help="Metrics to evaluate"
    ),
    thresholds: list[str] = typer.Option(
        [], "--threshold", help="Custom thresholds as 'metric:value'"
    ),
    output_format: str = typer.Option("text", "--output-format", help="Output format: text, github"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", help="Save report to file"),
) -> None:
    """Run regression tests and exit with non-zero if failed."""
    from evalops.cli.main import load_target_function, parse_metrics
    from evalops.comparison.regression import MetricThreshold, run_regression_test
    from evalops import EvalDataset

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        raise typer.Exit(1)

    eval_dataset = EvalDataset.from_json(dataset)

    # Load target
    if target:
        target_func = load_target_function(target)
    else:
        target_func = lambda x: x

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
        Path("regression_summary.md").write_text(summary)
        console.print(summary)
    else:
        report.print_summary()

    if output_file:
        import json
        output_file.write_text(json.dumps(report.to_dict(), indent=2))

    if report.exit_code != 0:
        raise typer.Exit(report.exit_code)

if __name__ == "__main__":
    app()
