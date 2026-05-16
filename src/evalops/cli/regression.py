"""CLI for running regression tests in CI/CD pipelines."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from evalops import EvalDataset
from evalops.comparison.regression import MetricThreshold, run_regression_test

app = typer.Typer(help="Run regression tests.")
console = Console()

@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", help="Path to dataset JSON file"),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target function as 'module:function'"
    ),
    metrics: list[str] = typer.Option(
        ["exact_match"],
        "--metric",
        "-m",
        help="Metrics to evaluate",
    ),
    baseline: Path | None = typer.Option(
        None, "--baseline", "-b", help="Path to baseline JSON file"
    ),
    threshold: list[str] = typer.Option(
        [], "--threshold", help="Thresholds as 'metric:value'"
    ),
    output_format: str = typer.Option(
        "github", "--output-format", help="Output format: text, github"
    ),
    name: str | None = typer.Option(None, "--name", help="Name for this test run"),
) -> None:
    """Run regression tests against thresholds or baseline."""

    # Load dataset
    if not dataset.exists():
        console.print(f"[red]Error: Dataset file not found: {dataset}[/red]")
        sys.exit(1)

    eval_dataset = EvalDataset.from_json(dataset)

    # Load target if provided
    target_func = None
    if target:
        import importlib
        if ":" not in target:
            console.print("[red]Error: Target must be in format 'module:function'[/red]")
            sys.exit(1)
        module_path, func_name = target.rsplit(":", 1)
        try:
            module = importlib.import_module(module_path)
            target_func = getattr(module, func_name)
        except Exception as e:
            console.print(f"[red]Error loading target: {e}[/red]")
            sys.exit(1)
    else:
        # Default mock target if none provided (some CI environments might use this)
        def target_func(x):
            return x

    # Parse thresholds
    threshold_configs = []
    for t in threshold:
        if ":" not in t:
            continue
        m_name, m_val = t.split(":", 1)
        try:
            threshold_configs.append(MetricThreshold(m_name, min_pass_rate=float(m_val)))
        except ValueError:
            continue

    # Run regression test
    console.print("[bold]Running regression tests...[/bold]")

    # Note: Using evalops.api.app's load_metrics logic or similar
    from evalops.api.app import load_metrics
    metric_instances = load_metrics(metrics)

    report = asyncio.run(
        run_regression_test(
            dataset=eval_dataset,
            target=target_func,
            metrics=metric_instances,
            thresholds=threshold_configs,
            baseline_path=baseline,
            run_name=name,
        )
    )

    # Display results
    if output_format == "github":
        summary = report.to_github_summary()
        Path("regression_summary.md").write_text(summary)
        console.print("[green]GitHub summary written to regression_summary.md[/green]")

    report.print_summary()

    if report.exit_code != 0:
        sys.exit(report.exit_code)

if __name__ == "__main__":
    app()
