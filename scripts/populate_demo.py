#!/usr/bin/env python
"""Populate the EvalOps database with demo data.

This script creates realistic evaluation data for demonstrating the dashboard:
- Runs evaluations on sample datasets with mock targets
- Creates baselines from good runs
- Simulates runs over time to create history
- Runs A/B comparisons between good and bad models
- Generates drift detection scenarios

Usage:
    python scripts/populate_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evalops.core.dataset import EvalDataset
from evalops.core.metrics import ExactMatch
from evalops.core.runner import EvalRunner
from evalops.storage.repository import EvalRepository

# Import demo targets
sys.path.insert(0, str(Path(__file__).parent.parent / "demo"))
from targets import (
    bad_classifier,
    bad_qa_model,
    bad_summarizer,
    degrading_classifier,
    degrading_qa_model,
    good_classifier,
    good_qa_model,
    good_summarizer,
)


def load_datasets() -> dict[str, EvalDataset]:
    """Load all demo datasets."""
    datasets_dir = Path(__file__).parent.parent / "demo" / "datasets"

    datasets = {}
    for dataset_file in datasets_dir.glob("*.json"):
        dataset = EvalDataset.from_json(dataset_file)
        datasets[dataset.name] = dataset
        print(f"  Loaded {dataset.name}: {len(dataset)} cases")

    return datasets


async def run_qa_evaluations(
    runner: EvalRunner,
    dataset: EvalDataset,
) -> tuple[str, str]:
    """Run Q&A model evaluations and return run IDs."""
    print("\n  Running good Q&A model...")
    good_result = await runner.evaluate(
        dataset=dataset,
        target=good_qa_model,
        metrics=[ExactMatch()],
        run_name="qa_good_model_baseline",
        save=True,
        tags=["qa", "baseline", "good_model"],
    )
    print(f"    Pass rate: {good_result.pass_rate:.1%}")

    print("  Running bad Q&A model...")
    bad_result = await runner.evaluate(
        dataset=dataset,
        target=bad_qa_model,
        metrics=[ExactMatch()],
        run_name="qa_bad_model",
        save=True,
        tags=["qa", "comparison", "bad_model"],
    )
    print(f"    Pass rate: {bad_result.pass_rate:.1%}")

    return good_result.id, bad_result.id


async def run_summarization_evaluations(
    runner: EvalRunner,
    dataset: EvalDataset,
) -> tuple[str, str]:
    """Run summarization evaluations and return run IDs."""
    print("\n  Running good summarizer...")
    good_result = await runner.evaluate(
        dataset=dataset,
        target=good_summarizer,
        metrics=[ExactMatch()],
        run_name="summarization_good_model_baseline",
        save=True,
        tags=["summarization", "baseline", "good_model"],
    )
    print(f"    Pass rate: {good_result.pass_rate:.1%}")

    print("  Running bad summarizer...")
    bad_result = await runner.evaluate(
        dataset=dataset,
        target=bad_summarizer,
        metrics=[ExactMatch()],
        run_name="summarization_bad_model",
        save=True,
        tags=["summarization", "comparison", "bad_model"],
    )
    print(f"    Pass rate: {bad_result.pass_rate:.1%}")

    return good_result.id, bad_result.id


async def run_classification_evaluations(
    runner: EvalRunner,
    dataset: EvalDataset,
) -> tuple[str, str]:
    """Run classification evaluations and return run IDs."""
    print("\n  Running good classifier...")
    good_result = await runner.evaluate(
        dataset=dataset,
        target=good_classifier,
        metrics=[ExactMatch()],
        run_name="classification_good_model_baseline",
        save=True,
        tags=["classification", "baseline", "good_model"],
    )
    print(f"    Pass rate: {good_result.pass_rate:.1%}")

    print("  Running bad classifier...")
    bad_result = await runner.evaluate(
        dataset=dataset,
        target=bad_classifier,
        metrics=[ExactMatch()],
        run_name="classification_bad_model",
        save=True,
        tags=["classification", "comparison", "bad_model"],
    )
    print(f"    Pass rate: {bad_result.pass_rate:.1%}")

    return good_result.id, bad_result.id


def create_baselines(repo: EvalRepository, run_ids: dict[str, str]) -> None:
    """Create baselines from good model runs."""
    print("\n--- Creating Baselines ---")

    for dataset_name, run_id in run_ids.items():
        run = repo.get_run(run_id)
        if run:
            # Extract metrics from the run
            metrics = {}
            if run.metrics_summary:
                for name, data in run.metrics_summary.items():
                    if isinstance(data, dict) and "score" in data:
                        metrics[name] = data["score"]
                    else:
                        metrics[name] = float(data) if data else 0.0
            metrics["pass_rate"] = run.pass_rate

            baseline = repo.save_baseline_from_values(
                dataset_name=run.dataset_name,
                name=f"{dataset_name}_baseline",
                metrics=metrics,
                pass_rate=run.pass_rate,
                total_cases=run.total_cases,
            )
            print(f"  Created baseline: {baseline.name} (pass_rate={baseline.pass_rate:.1%})")


async def simulate_historical_runs(
    runner: EvalRunner,
    datasets: dict[str, EvalDataset],
) -> None:
    """Simulate runs over time to create history for trend charts."""
    print("\n--- Simulating Historical Runs ---")

    # QA runs with slight variations
    qa_dataset = datasets.get("qa_eval")
    if qa_dataset:
        print("  Simulating Q&A history (8 runs)...")
        for day in range(8):
            # Use degrading model to show some variation
            degradation = 0.02 * (day % 3)  # Small oscillation

            def qa_target(input_text: str, deg=degradation) -> str:
                return degrading_qa_model(input_text, degradation=deg)

            result = await runner.evaluate(
                dataset=qa_dataset,
                target=qa_target,
                metrics=[ExactMatch()],
                run_name=f"qa_daily_run_day{day+1}",
                save=True,
                tags=["qa", "daily", f"day_{day+1}"],
            )
            print(f"    Day {day+1}: {result.pass_rate:.1%}")

    # Classification runs showing degradation trend (for drift detection)
    cls_dataset = datasets.get("classification_eval")
    if cls_dataset:
        print("  Simulating classification with degradation (6 runs)...")
        for run_num in range(6):
            # Gradual degradation
            degradation = 0.03 * run_num  # 0%, 3%, 6%, 9%, 12%, 15%

            def cls_target(input_text: str, deg=degradation) -> str:
                return degrading_classifier(input_text, degradation=deg)

            result = await runner.evaluate(
                dataset=cls_dataset,
                target=cls_target,
                metrics=[ExactMatch()],
                run_name=f"classification_run_{run_num+1}",
                save=True,
                tags=["classification", "monitoring", f"run_{run_num+1}"],
            )
            print(f"    Run {run_num+1}: {result.pass_rate:.1%} (degradation: {degradation:.0%})")


async def run_ab_comparisons(
    runner: EvalRunner,
    datasets: dict[str, EvalDataset],
) -> None:
    """Run A/B comparison tests between good and bad models."""
    print("\n--- Running A/B Comparisons ---")

    # Q&A A/B test
    qa_dataset = datasets.get("qa_eval")
    if qa_dataset:
        print("  Q&A Model A/B Test:")
        result_a = await runner.evaluate(
            dataset=qa_dataset,
            target=good_qa_model,
            metrics=[ExactMatch()],
            run_name="qa_ab_variant_a",
            save=True,
            tags=["qa", "ab_test", "variant_a"],
        )

        result_b = await runner.evaluate(
            dataset=qa_dataset,
            target=bad_qa_model,
            metrics=[ExactMatch()],
            run_name="qa_ab_variant_b",
            save=True,
            tags=["qa", "ab_test", "variant_b"],
        )

        print(f"    Variant A (good): {result_a.pass_rate:.1%}")
        print(f"    Variant B (bad):  {result_b.pass_rate:.1%}")
        winner = "A" if result_a.pass_rate > result_b.pass_rate else "B"
        print(f"    Winner: Variant {winner}")

    # Classification A/B test
    cls_dataset = datasets.get("classification_eval")
    if cls_dataset:
        print("  Classification Model A/B Test:")
        result_a = await runner.evaluate(
            dataset=cls_dataset,
            target=good_classifier,
            metrics=[ExactMatch()],
            run_name="classification_ab_variant_a",
            save=True,
            tags=["classification", "ab_test", "variant_a"],
        )

        result_b = await runner.evaluate(
            dataset=cls_dataset,
            target=bad_classifier,
            metrics=[ExactMatch()],
            run_name="classification_ab_variant_b",
            save=True,
            tags=["classification", "ab_test", "variant_b"],
        )

        print(f"    Variant A (good): {result_a.pass_rate:.1%}")
        print(f"    Variant B (bad):  {result_b.pass_rate:.1%}")
        winner = "A" if result_a.pass_rate > result_b.pass_rate else "B"
        print(f"    Winner: Variant {winner}")


def print_summary(repo: EvalRepository) -> None:
    """Print a summary of all data created."""
    print("\n" + "=" * 60)
    print("DEMO DATA SUMMARY")
    print("=" * 60)

    stats = repo.get_run_stats()
    print(f"\nTotal Runs: {stats['total_runs']}")
    print(f"Total Cases Evaluated: {stats['total_cases']}")
    print(f"Average Pass Rate: {stats['avg_pass_rate']:.1%}")
    print(f"Average Latency: {stats['avg_latency_ms']:.1f}ms")

    baselines = repo.list_baselines()
    print(f"\nBaselines Created: {len(baselines)}")
    for baseline in baselines:
        status = "ACTIVE" if baseline.is_active else "inactive"
        print(f"  - {baseline.name}: {baseline.pass_rate:.1%} [{status}]")

    # List recent runs
    runs = repo.list_runs(limit=5)
    print(f"\nRecent Runs (last 5 of {stats['total_runs']}):")
    for run in runs:
        print(f"  - {run.name}: {run.pass_rate:.1%} ({run.total_cases} cases)")

    print("\n" + "=" * 60)
    print("Dashboard ready! Run: evalops-dashboard")
    print("Or: python -m streamlit run src/evalops/dashboard/app.py")
    print("=" * 60)


async def main() -> None:
    """Main function to populate demo data."""
    print("=" * 60)
    print("EvalOps Demo Data Population Script")
    print("=" * 60)

    # Initialize repository
    db_path = Path(__file__).parent.parent / "evalops_demo.db"
    db_url = f"sqlite:///{db_path}"

    print(f"\nDatabase: {db_path}")

    # Remove existing demo database
    if db_path.exists():
        print("  Removing existing demo database...")
        db_path.unlink()

    print("  Initializing repository...")
    repo = EvalRepository(db_url)
    repo.initialize()

    # Create runner with repository
    runner = EvalRunner(
        repository=repo,
        auto_save=False,  # We'll explicitly save with tags
        enable_observability=False,  # Disable for faster demo generation
    )

    # Load datasets
    print("\n--- Loading Datasets ---")
    datasets = load_datasets()

    # Store good run IDs for baselines
    baseline_run_ids: dict[str, str] = {}

    # Run initial evaluations
    print("\n--- Running Initial Evaluations ---")

    if "qa_eval" in datasets:
        good_id, _ = await run_qa_evaluations(runner, datasets["qa_eval"])
        baseline_run_ids["qa_eval"] = good_id

    if "summarization_eval" in datasets:
        good_id, _ = await run_summarization_evaluations(runner, datasets["summarization_eval"])
        baseline_run_ids["summarization_eval"] = good_id

    if "classification_eval" in datasets:
        good_id, _ = await run_classification_evaluations(runner, datasets["classification_eval"])
        baseline_run_ids["classification_eval"] = good_id

    # Create baselines from good runs
    create_baselines(repo, baseline_run_ids)

    # Simulate historical runs
    await simulate_historical_runs(runner, datasets)

    # Run A/B comparisons
    await run_ab_comparisons(runner, datasets)

    # Print summary
    print_summary(repo)


if __name__ == "__main__":
    asyncio.run(main())
