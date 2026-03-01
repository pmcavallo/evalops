"""Regression testing for CI/CD integration.

This module provides tools for running regression tests in CI/CD pipelines,
comparing current performance against baselines and producing actionable
pass/fail results.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from evalops.core.dataset import EvalDataset
from evalops.core.metrics import Metric
from evalops.core.runner import EvalRunner, EvalRunResult, TargetFunc


class RegressionStatus(Enum):
    """Status of a regression test."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class MetricThreshold:
    """Threshold configuration for a single metric.

    Attributes:
        metric_name: Name of the metric.
        min_value: Minimum acceptable value (for higher-is-better metrics).
        max_value: Maximum acceptable value (for lower-is-better metrics).
        min_pass_rate: Minimum pass rate for this metric.
        max_degradation: Maximum allowed degradation from baseline.
    """

    metric_name: str
    min_value: float | None = None
    max_value: float | None = None
    min_pass_rate: float | None = None
    max_degradation: float | None = None


@dataclass
class RegressionResult:
    """Result of a single metric regression check.

    Attributes:
        metric_name: Name of the metric.
        status: Pass/fail status.
        current_value: Current metric value.
        baseline_value: Baseline value (if comparing).
        threshold: The threshold that was applied.
        message: Human-readable result message.
    """

    metric_name: str
    status: RegressionStatus
    current_value: float
    baseline_value: float | None = None
    threshold: float | None = None
    message: str = ""


@dataclass
class RegressionReport:
    """Complete regression test report.

    Attributes:
        status: Overall pass/fail status.
        results: Per-metric regression results.
        summary: Summary of the test run.
        run_result: The evaluation run result.
        baseline_id: ID of baseline run (if comparing).
        exit_code: Exit code for CI/CD (0 = pass, 1 = fail).
        created_at: When the report was generated.
    """

    status: RegressionStatus = RegressionStatus.PASSED
    results: list[RegressionResult] = field(default_factory=list)
    summary: str = ""
    run_result: EvalRunResult | None = None
    baseline_id: str | None = None
    exit_code: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Export report as dictionary."""
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "summary": self.summary,
            "results": [
                {
                    "metric": r.metric_name,
                    "status": r.status.value,
                    "current_value": round(r.current_value, 4),
                    "baseline_value": (
                        round(r.baseline_value, 4) if r.baseline_value else None
                    ),
                    "threshold": round(r.threshold, 4) if r.threshold else None,
                    "message": r.message,
                }
                for r in self.results
            ],
            "baseline_id": self.baseline_id,
            "created_at": self.created_at.isoformat(),
        }

    def to_github_summary(self) -> str:
        """Generate GitHub Actions summary markdown."""
        lines = []

        # Header with status emoji
        status_emoji = "✅" if self.status == RegressionStatus.PASSED else "❌"
        lines.append(f"## {status_emoji} Regression Test Results")
        lines.append("")
        lines.append(f"**Status:** {self.status.value.upper()}")
        lines.append(f"**Summary:** {self.summary}")
        lines.append("")

        # Results table
        lines.append("### Metric Results")
        lines.append("")
        lines.append("| Metric | Status | Current | Baseline | Threshold |")
        lines.append("|--------|--------|---------|----------|-----------|")

        for r in self.results:
            status_icon = "✅" if r.status == RegressionStatus.PASSED else "❌"
            baseline = f"{r.baseline_value:.4f}" if r.baseline_value else "-"
            threshold = f"{r.threshold:.4f}" if r.threshold else "-"
            lines.append(
                f"| {r.metric_name} | {status_icon} | {r.current_value:.4f} | {baseline} | {threshold} |"
            )

        lines.append("")

        # Failed metrics details
        failed = [r for r in self.results if r.status == RegressionStatus.FAILED]
        if failed:
            lines.append("### Failed Metrics")
            lines.append("")
            for r in failed:
                lines.append(f"- **{r.metric_name}**: {r.message}")
            lines.append("")

        return "\n".join(lines)

    def print_summary(self, file: Any = None) -> None:
        """Print summary to stdout/file."""
        if file is None:
            file = sys.stdout

        print("=" * 60, file=file)
        print(f"REGRESSION TEST: {self.status.value.upper()}", file=file)
        print("=" * 60, file=file)
        print(f"Summary: {self.summary}", file=file)
        print("", file=file)

        for r in self.results:
            status_mark = "✓" if r.status == RegressionStatus.PASSED else "✗"
            print(f"  [{status_mark}] {r.metric_name}: {r.message}", file=file)

        print("", file=file)
        print(f"Exit Code: {self.exit_code}", file=file)


class RegressionTester:
    """Regression tester for CI/CD integration.

    Runs evaluations and checks results against thresholds and baselines,
    producing actionable pass/fail results for CI/CD pipelines.

    Example:
        ```python
        tester = RegressionTester(
            thresholds=[
                MetricThreshold("exact_match", min_pass_rate=0.90),
                MetricThreshold("latency", max_value=2000),
            ],
            fail_on_regression=True,
        )

        report = await tester.run(
            dataset=test_dataset,
            target=my_llm,
            metrics=[ExactMatch(), Latency()],
        )

        if report.status == RegressionStatus.FAILED:
            sys.exit(report.exit_code)
        ```

    CI/CD Integration:
        ```yaml
        # .github/workflows/eval.yml
        - name: Run Regression Tests
          run: |
            python -m evalops regression \\
              --dataset tests/fixtures/regression.json \\
              --threshold exact_match:0.9 \\
              --threshold latency:2000
        ```
    """

    def __init__(
        self,
        thresholds: list[MetricThreshold] | None = None,
        default_min_pass_rate: float = 0.80,
        max_degradation: float = 0.05,
        fail_on_regression: bool = True,
        runner: EvalRunner | None = None,
    ) -> None:
        """Initialize the regression tester.

        Args:
            thresholds: Per-metric threshold configurations.
            default_min_pass_rate: Default minimum pass rate if not specified.
            max_degradation: Maximum allowed degradation from baseline.
            fail_on_regression: Whether to fail the test on regression.
            runner: EvalRunner to use.
        """
        self.thresholds = {t.metric_name: t for t in (thresholds or [])}
        self.default_min_pass_rate = default_min_pass_rate
        self.max_degradation = max_degradation
        self.fail_on_regression = fail_on_regression
        self.runner = runner or EvalRunner(enable_observability=False)

        self._baseline: dict[str, float] | None = None
        self._baseline_id: str | None = None

    def set_baseline(self, run_result: EvalRunResult) -> None:
        """Set baseline from a previous run result.

        Args:
            run_result: Previous run to use as baseline.
        """
        self._baseline = {}
        self._baseline_id = run_result.id

        for name, summary in run_result.metrics_summary.items():
            if "pass_rate" in summary.get("details", {}):
                self._baseline[name] = summary["details"]["pass_rate"]
            else:
                self._baseline[name] = summary.get("score", 0.0)

        self._baseline["overall_pass_rate"] = run_result.pass_rate

    def set_baseline_from_values(
        self,
        metrics: dict[str, float],
        baseline_id: str = "manual_baseline",
    ) -> None:
        """Set baseline from raw values.

        Args:
            metrics: Dict of metric name to baseline value.
            baseline_id: Identifier for this baseline.
        """
        self._baseline = metrics.copy()
        self._baseline_id = baseline_id

    def load_baseline(self, path: Path) -> None:
        """Load baseline from a JSON file.

        Args:
            path: Path to baseline JSON file.
        """
        data = json.loads(path.read_text())
        self._baseline = data.get("metrics", {})
        self._baseline_id = data.get("baseline_id", path.stem)

    def save_baseline(self, run_result: EvalRunResult, path: Path) -> None:
        """Save a run result as baseline for future comparisons.

        Args:
            run_result: Run result to save as baseline.
            path: Path to save the baseline JSON.
        """
        metrics = {}
        for name, summary in run_result.metrics_summary.items():
            if "pass_rate" in summary.get("details", {}):
                metrics[name] = summary["details"]["pass_rate"]
            else:
                metrics[name] = summary.get("score", 0.0)

        metrics["overall_pass_rate"] = run_result.pass_rate

        data = {
            "baseline_id": run_result.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": run_result.dataset_name,
            "total_cases": run_result.total_cases,
            "metrics": metrics,
        }

        path.write_text(json.dumps(data, indent=2))

    async def run(
        self,
        dataset: EvalDataset,
        target: TargetFunc,
        metrics: list[Metric],
        run_name: str | None = None,
    ) -> RegressionReport:
        """Run regression test.

        Args:
            dataset: Dataset to evaluate.
            target: Target function to test.
            metrics: Metrics to evaluate.
            run_name: Optional name for the run.

        Returns:
            RegressionReport with pass/fail status and details.
        """
        # Run evaluation
        run_result = await self.runner.evaluate(
            dataset=dataset,
            target=target,
            metrics=metrics,
            run_name=run_name or "regression_test",
        )

        # Check thresholds
        results = self._check_thresholds(run_result)

        # Determine overall status
        failed = [r for r in results if r.status == RegressionStatus.FAILED]

        if failed:
            status = RegressionStatus.FAILED
            exit_code = 1 if self.fail_on_regression else 0
            summary = f"Failed {len(failed)} of {len(results)} metric checks"
        else:
            status = RegressionStatus.PASSED
            exit_code = 0
            summary = f"All {len(results)} metric checks passed"

        return RegressionReport(
            status=status,
            results=results,
            summary=summary,
            run_result=run_result,
            baseline_id=self._baseline_id,
            exit_code=exit_code,
        )

    def _check_thresholds(self, run_result: EvalRunResult) -> list[RegressionResult]:
        """Check all metrics against thresholds."""
        results = []

        # Check overall pass rate
        results.append(
            self._check_metric(
                "overall_pass_rate",
                run_result.pass_rate,
                MetricThreshold("overall_pass_rate", min_pass_rate=self.default_min_pass_rate),
            )
        )

        # Check individual metrics
        for name, summary in run_result.metrics_summary.items():
            threshold = self.thresholds.get(name)

            # Get current value
            if "pass_rate" in summary.get("details", {}):
                current_value = summary["details"]["pass_rate"]
            else:
                current_value = summary.get("score", 0.0)

            # Use default threshold if none specified
            if threshold is None:
                threshold = MetricThreshold(name, min_pass_rate=self.default_min_pass_rate)

            results.append(self._check_metric(name, current_value, threshold))

        return results

    def _check_metric(
        self,
        metric_name: str,
        current_value: float,
        threshold: MetricThreshold,
    ) -> RegressionResult:
        """Check a single metric against its threshold."""
        baseline_value = self._baseline.get(metric_name) if self._baseline else None

        # Check minimum value threshold
        if threshold.min_value is not None:
            if current_value < threshold.min_value:
                return RegressionResult(
                    metric_name=metric_name,
                    status=RegressionStatus.FAILED,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    threshold=threshold.min_value,
                    message=f"Value {current_value:.4f} below minimum {threshold.min_value:.4f}",
                )

        # Check maximum value threshold (for lower-is-better metrics)
        if threshold.max_value is not None:
            if current_value > threshold.max_value:
                return RegressionResult(
                    metric_name=metric_name,
                    status=RegressionStatus.FAILED,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    threshold=threshold.max_value,
                    message=f"Value {current_value:.4f} exceeds maximum {threshold.max_value:.4f}",
                )

        # Check pass rate threshold
        if threshold.min_pass_rate is not None:
            if current_value < threshold.min_pass_rate:
                return RegressionResult(
                    metric_name=metric_name,
                    status=RegressionStatus.FAILED,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    threshold=threshold.min_pass_rate,
                    message=f"Pass rate {current_value:.1%} below threshold {threshold.min_pass_rate:.1%}",
                )

        # Check degradation from baseline
        if baseline_value is not None:
            max_deg = threshold.max_degradation or self.max_degradation
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value
                if degradation > max_deg:
                    return RegressionResult(
                        metric_name=metric_name,
                        status=RegressionStatus.FAILED,
                        current_value=current_value,
                        baseline_value=baseline_value,
                        threshold=max_deg,
                        message=f"Degraded {degradation:.1%} from baseline (max {max_deg:.1%})",
                    )

        # All checks passed
        return RegressionResult(
            metric_name=metric_name,
            status=RegressionStatus.PASSED,
            current_value=current_value,
            baseline_value=baseline_value,
            message=f"Value {current_value:.4f} meets all thresholds",
        )


async def run_regression_test(
    dataset: EvalDataset,
    target: TargetFunc,
    metrics: list[Metric],
    thresholds: list[MetricThreshold] | None = None,
    baseline_path: Path | None = None,
    **kwargs: Any,
) -> RegressionReport:
    """Convenience function for running regression tests.

    Args:
        dataset: Dataset to evaluate.
        target: Target function to test.
        metrics: Metrics to evaluate.
        thresholds: Metric thresholds.
        baseline_path: Path to baseline file (optional).
        **kwargs: Additional arguments for RegressionTester.

    Returns:
        RegressionReport with results.
    """
    tester = RegressionTester(thresholds=thresholds, **kwargs)

    if baseline_path and baseline_path.exists():
        tester.load_baseline(baseline_path)

    return await tester.run(dataset, target, metrics)
