"""A/B testing and statistical significance for evaluations.

This module provides tools for comparing two evaluation results and
determining if differences in performance are statistically significant.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from evalops.core.dataset import EvalDataset
from evalops.core.metrics import Metric
from evalops.core.runner import EvalRunner, EvalRunResult, TargetFunc

if TYPE_CHECKING:
    pass


class Winner(Enum):
    """Winner of an A/B test comparison."""

    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    TIE = "tie"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StatisticalResult:
    """Result of a statistical test comparison.

    Attributes:
        t_statistic: The calculated t-statistic.
        p_value: The calculated p-value.
        significant: Whether the difference is statistically significant.
        effect_size: The calculated Cohen's d effect size.
        effect_magnitude: Interpretation of effect size (none, small, medium, large).
    """

    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_magnitude: str

    @classmethod
    def from_samples(
        cls,
        samples_a: list[float],
        samples_b: list[float],
        confidence_level: float = 0.95,
    ) -> StatisticalResult:
        """Compute statistical significance between two sets of samples.

        Uses a paired t-test for evaluation cases that are run on the
        same inputs for both variants.
        """
        if len(samples_a) != len(samples_b):
            raise ValueError("Sample sizes must match for paired comparison")

        n = len(samples_a)
        if n < 2:
            return cls(0.0, 1.0, False, 0.0, "none")

        differences = [a - b for a, b in zip(samples_a, samples_b)]
        mean_diff = statistics.mean(differences)

        try:
            std_diff = statistics.stdev(differences)
        except statistics.StatisticsError:
            std_diff = 0.0

        if std_diff == 0:
            if mean_diff == 0:
                return cls(0.0, 1.0, False, 0.0, "none")
            else:
                # Consistently different
                return cls(
                    float("inf") if mean_diff > 0 else float("-inf"), 0.0, True, 1.0, "large"
                )

        # t-statistic
        t_stat = mean_diff / (std_diff / math.sqrt(n))

        # Approximate p-value (using normal distribution for n > 30)
        # For small n, this is a rough estimate
        p_val = cls._approx_p_value(t_stat)

        significant = p_val <= (1 - confidence_level)

        # Effect size (Cohen's d)
        std_a = statistics.stdev(samples_a) if len(samples_a) > 1 else 1.0
        std_b = statistics.stdev(samples_b) if len(samples_b) > 1 else 1.0
        pooled_std = math.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            magnitude = "none"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"

        return cls(t_stat, p_val, significant, effect_size, magnitude)

    @staticmethod
    def _approx_p_value(t: float) -> float:
        """Approximate two-tailed p-value for t-statistic."""
        # Simple approximation of the normal CDF for the p-value
        x = abs(t)
        # Using a polynomial approximation of the error function
        p = 1.0 / (1.0 + 0.2316419 * x)
        d = 0.3989423 * math.exp(-x * x / 2.0)
        erf = 1.0 - d * p * (
            0.3193815 + p * (-0.3565638 + p * (1.781478 + p * (-1.821256 + p * 1.330274)))
        )
        # Two-tailed p-value
        return 1.0 - erf


@dataclass
class MetricComparison:
    """Comparison result for a single metric.

    Attributes:
        metric_name: Name of the metric.
        mean_a: Average score for variant A.
        mean_b: Average score for variant B.
        std_a: Standard deviation for variant A.
        std_b: Standard deviation for variant B.
        difference: Absolute difference (A - B).
        relative_difference: Percentage difference.
        stats: Statistical significance results.
        winner: The winning variant for this metric.
    """

    metric_name: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    difference: float
    relative_difference: float
    stats: StatisticalResult
    winner: Winner


@dataclass
class ABTestResult:
    """Complete result of an A/B test comparison.

    Attributes:
        name: Name of the comparison.
        variant_a_name: Name of variant A.
        variant_b_name: Name of variant B.
        dataset_name: Dataset used for comparison.
        total_cases: Number of cases compared.
        metric_comparisons: Comparison results per metric.
        overall_winner: The determined overall winner.
        recommendation: Human-readable recommendation.
    """

    name: str
    variant_a_name: str = "variant_a"
    variant_b_name: str = "variant_b"
    dataset_name: str = "unknown"
    total_cases: int = 0
    metric_comparisons: dict[str, MetricComparison] = field(default_factory=dict)
    overall_winner: Winner = Winner.INCONCLUSIVE
    recommendation: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        """Export result summary as dictionary."""
        return {
            "name": self.name,
            "overall_winner": self.overall_winner.value,
            "total_cases": self.total_cases,
            "metrics": {
                name: {
                    "mean_a": round(c.mean_a, 4),
                    "mean_b": round(c.mean_b, 4),
                    "difference": round(c.difference, 4),
                    "significant": c.stats.significant,
                    "magnitude": c.stats.effect_magnitude,
                    "winner": c.winner.value,
                }
                for name, c in self.metric_comparisons.items()
            },
            "recommendation": self.recommendation,
        }


class ABTest:
    """Conducts A/B tests between LLM variants.

    Runs comparative evaluations and analyzes results for statistical
    significance.

    Example:
        ```python
        ab_test = ABTest(confidence_level=0.95)
        result = await ab_test.compare(
            dataset=dataset,
            variant_a=model_v1,
            variant_b=model_v2,
            metrics=[ExactMatch(), SemanticSimilarity()]
        )
        print(f"Winner: {result.overall_winner}")
        ```
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_effect_size: float = 0.2,
        runner: EvalRunner | None = None,
    ) -> None:
        """Initialize the A/B tester.

        Args:
            confidence_level: Required confidence level (default 0.95).
            min_effect_size: Minimum Cohen's d to consider a meaningful winner.
            runner: EvalRunner to use for evaluations.
        """
        self.confidence_level = confidence_level
        self.min_effect_size = min_effect_size
        self.runner = runner or EvalRunner(enable_observability=False)

    async def compare(
        self,
        dataset: EvalDataset,
        variant_a: TargetFunc,
        variant_b: TargetFunc,
        metrics: list[Metric],
        name: str | None = None,
        variant_a_name: str = "variant_a",
        variant_b_name: str = "variant_b",
    ) -> ABTestResult:
        """Compare two variants on a dataset.

        Runs both variants and computes statistical significance.
        """
        # Run evaluations
        run_a = await self.runner.evaluate(
            dataset, variant_a, metrics, run_name=f"{variant_a_name}_ab"
        )
        run_b = await self.runner.evaluate(
            dataset, variant_b, metrics, run_name=f"{variant_b_name}_ab"
        )

        return self.analyze_runs(
            run_a,
            run_b,
            name=name or f"AB_{variant_a_name}_vs_{variant_b_name}",
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
        )

    def analyze_runs(
        self,
        run_a: EvalRunResult,
        run_b: EvalRunResult,
        name: str,
        variant_a_name: str = "variant_a",
        variant_b_name: str = "variant_b",
    ) -> ABTestResult:
        """Analyze results from two existing runs."""
        comparisons = {}

        # Collect all metric names
        metric_names = set(run_a.metrics_summary.keys()) | set(run_b.metrics_summary.keys())

        for m_name in metric_names:
            # Extract scores for this metric from both runs
            scores_a = [r.metric_results.get(m_name, {}).get("score", 0.0) for r in run_a.results]
            scores_b = [r.metric_results.get(m_name, {}).get("score", 0.0) for r in run_b.results]

            if not scores_a or not scores_b:
                continue

            # Compute statistics
            stats = StatisticalResult.from_samples(
                scores_a, scores_b, confidence_level=self.confidence_level
            )

            # Determine metric winner
            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)
            std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
            std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0

            winner = Winner.TIE
            if stats.significant and abs(stats.effect_size) >= self.min_effect_size:
                winner = Winner.VARIANT_A if mean_a > mean_b else Winner.VARIANT_B
            elif not stats.significant:
                winner = Winner.INCONCLUSIVE

            comparisons[m_name] = MetricComparison(
                metric_name=m_name,
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                difference=mean_a - mean_b,
                relative_difference=((mean_a - mean_b) / mean_b * 100 if mean_b != 0 else 0.0),
                stats=stats,
                winner=winner,
            )

        # Determine overall winner
        a_wins = sum(1 for c in comparisons.values() if c.winner == Winner.VARIANT_A)
        b_wins = sum(1 for c in comparisons.values() if c.winner == Winner.VARIANT_B)

        if a_wins > b_wins:
            overall = Winner.VARIANT_A
        elif b_wins > a_wins:
            overall = Winner.VARIANT_B
        elif a_wins == 0 and b_wins == 0:
            overall = Winner.TIE
        else:
            overall = Winner.TIE

        # Generate recommendation
        if overall == Winner.VARIANT_A:
            rec = (
                f"Recommend {variant_a_name}. It outperformed {variant_b_name} on {a_wins} metrics."
            )
        elif overall == Winner.VARIANT_B:
            rec = (
                f"Recommend {variant_b_name}. It outperformed {variant_a_name} on {b_wins} metrics."
            )
        else:
            rec = (
                "No clear winner detected. "
                "The variants performed similarly or results were inconclusive."
            )

        return ABTestResult(
            name=name,
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
            dataset_name=run_a.dataset_name,
            total_cases=len(run_a.results),
            metric_comparisons=comparisons,
            overall_winner=overall,
            recommendation=rec,
        )


async def compare_variants(
    dataset: EvalDataset,
    variant_a: TargetFunc,
    variant_b: TargetFunc,
    metrics: list[Metric],
    **kwargs: Any,
) -> ABTestResult:
    """Convenience function for A/B testing."""
    ab_test = ABTest(**kwargs)
    return await ab_test.compare(dataset, variant_a, variant_b, metrics)
