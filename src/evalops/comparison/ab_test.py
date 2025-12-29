"""A/B testing framework for comparing LLM variants.

This module provides tools for statistically rigorous comparison of
two LLM variants (prompts, models, configurations) on the same dataset.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from evalops.core.dataset import EvalDataset
from evalops.core.metrics import Metric, MetricResult
from evalops.core.runner import EvalRunResult, EvalRunner, TargetFunc


class Winner(Enum):
    """Enum representing the winner of an A/B test."""

    VARIANT_A = "variant_a"
    VARIANT_B = "variant_b"
    TIE = "tie"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StatisticalResult:
    """Result of a statistical comparison.

    Attributes:
        t_statistic: The t-statistic from the paired t-test.
        p_value: The p-value (probability of observing this difference by chance).
        significant: Whether the difference is statistically significant.
        effect_size: Cohen's d effect size.
        effect_magnitude: Interpretation of effect size (small/medium/large).
        confidence_level: The confidence level used for significance testing.
    """

    t_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_magnitude: str
    confidence_level: float = 0.95

    @classmethod
    def from_samples(
        cls,
        samples_a: list[float],
        samples_b: list[float],
        confidence_level: float = 0.95,
    ) -> StatisticalResult:
        """Calculate statistical result from paired samples.

        Args:
            samples_a: Scores from variant A.
            samples_b: Scores from variant B (same order as A).
            confidence_level: Confidence level for significance (default 0.95).

        Returns:
            StatisticalResult with t-test and effect size calculations.
        """
        if len(samples_a) != len(samples_b):
            raise ValueError("Sample sizes must match for paired t-test")

        n = len(samples_a)
        if n < 2:
            return cls(
                t_statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                effect_magnitude="none",
                confidence_level=confidence_level,
            )

        # Calculate differences
        differences = [a - b for a, b in zip(samples_a, samples_b)]
        mean_diff = statistics.mean(differences)

        # Handle case where all differences are zero
        try:
            std_diff = statistics.stdev(differences)
        except statistics.StatisticsError:
            std_diff = 0.0

        if std_diff == 0:
            # No variance in differences
            if mean_diff == 0:
                return cls(
                    t_statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    effect_size=0.0,
                    effect_magnitude="none",
                    confidence_level=confidence_level,
                )
            else:
                # Perfect separation
                return cls(
                    t_statistic=float("inf") if mean_diff > 0 else float("-inf"),
                    p_value=0.0,
                    significant=True,
                    effect_size=float("inf") if mean_diff > 0 else float("-inf"),
                    effect_magnitude="large",
                    confidence_level=confidence_level,
                )

        # Paired t-test
        t_statistic = mean_diff / (std_diff / math.sqrt(n))

        # Calculate p-value using t-distribution approximation
        # Using a simple approximation for the CDF
        df = n - 1
        p_value = cls._t_test_p_value(abs(t_statistic), df) * 2  # Two-tailed

        # Effect size (Cohen's d for paired samples)
        pooled_std = math.sqrt(
            (statistics.stdev(samples_a) ** 2 + statistics.stdev(samples_b) ** 2) / 2
        )
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Interpret effect size
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            effect_magnitude = "negligible"
        elif abs_effect < 0.5:
            effect_magnitude = "small"
        elif abs_effect < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

        # Significance threshold
        alpha = 1 - confidence_level
        significant = p_value < alpha

        return cls(
            t_statistic=t_statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            effect_magnitude=effect_magnitude,
            confidence_level=confidence_level,
        )

    @staticmethod
    def _t_test_p_value(t: float, df: int) -> float:
        """Approximate p-value from t-statistic using normal approximation.

        For large df, t-distribution approaches normal distribution.
        For smaller df, this is an approximation.
        """
        # Use normal approximation for simplicity
        # More accurate would use scipy.stats.t.sf, but we avoid the dependency
        x = t / math.sqrt(1 + t * t / df)
        # Approximate standard normal CDF
        return 0.5 * math.erfc(x / math.sqrt(2))


@dataclass
class MetricComparison:
    """Comparison result for a single metric.

    Attributes:
        metric_name: Name of the metric being compared.
        mean_a: Mean score for variant A.
        mean_b: Mean score for variant B.
        std_a: Standard deviation for variant A.
        std_b: Standard deviation for variant B.
        difference: Absolute difference (B - A).
        relative_difference: Percentage difference from A.
        stats: Statistical test results.
        winner: Which variant is better for this metric.
        higher_is_better: Whether higher scores are better.
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
    higher_is_better: bool = True


@dataclass
class ABTestResult:
    """Complete result of an A/B test comparison.

    Attributes:
        id: Unique identifier for this test.
        name: Name of the A/B test.
        variant_a_name: Name/description of variant A.
        variant_b_name: Name/description of variant B.
        dataset_name: Name of the dataset used.
        total_cases: Number of cases evaluated.
        run_a: Full evaluation result for variant A.
        run_b: Full evaluation result for variant B.
        metric_comparisons: Per-metric comparison results.
        overall_winner: Overall winner across all metrics.
        recommendation: Human-readable recommendation.
        created_at: When the test was run.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    variant_a_name: str = "variant_a"
    variant_b_name: str = "variant_b"
    dataset_name: str = ""
    total_cases: int = 0
    run_a: EvalRunResult | None = None
    run_b: EvalRunResult | None = None
    metric_comparisons: dict[str, MetricComparison] = field(default_factory=dict)
    overall_winner: Winner = Winner.INCONCLUSIVE
    recommendation: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_summary_dict(self) -> dict[str, Any]:
        """Export a summary suitable for display or logging."""
        return {
            "test_name": self.name,
            "variant_a": self.variant_a_name,
            "variant_b": self.variant_b_name,
            "dataset": self.dataset_name,
            "total_cases": self.total_cases,
            "overall_winner": self.overall_winner.value,
            "recommendation": self.recommendation,
            "metrics": {
                name: {
                    "mean_a": round(comp.mean_a, 4),
                    "mean_b": round(comp.mean_b, 4),
                    "difference": round(comp.difference, 4),
                    "p_value": round(comp.stats.p_value, 4),
                    "significant": comp.stats.significant,
                    "effect_size": round(comp.stats.effect_size, 4),
                    "winner": comp.winner.value,
                }
                for name, comp in self.metric_comparisons.items()
            },
        }


class ABTest:
    """Framework for A/B testing LLM variants.

    Compares two variants (prompts, models, or configurations) on the
    same dataset with statistical rigor.

    Example:
        ```python
        ab_test = ABTest()

        result = await ab_test.compare(
            dataset=eval_dataset,
            variant_a=prompt_v1_system,
            variant_b=prompt_v2_system,
            metrics=[ExactMatch(), Latency()],
            variant_a_name="conservative_prompt",
            variant_b_name="concise_prompt",
        )

        print(f"Winner: {result.overall_winner}")
        print(f"Recommendation: {result.recommendation}")
        ```
    """

    def __init__(
        self,
        runner: EvalRunner | None = None,
        confidence_level: float = 0.95,
        min_effect_size: float = 0.2,
    ) -> None:
        """Initialize the A/B test framework.

        Args:
            runner: EvalRunner to use (creates one if not provided).
            confidence_level: Confidence level for statistical tests.
            min_effect_size: Minimum effect size to consider meaningful.
        """
        self.runner = runner or EvalRunner(enable_observability=False)
        self.confidence_level = confidence_level
        self.min_effect_size = min_effect_size

    async def compare(
        self,
        dataset: EvalDataset,
        variant_a: TargetFunc,
        variant_b: TargetFunc,
        metrics: list[Metric],
        variant_a_name: str = "variant_a",
        variant_b_name: str = "variant_b",
        test_name: str | None = None,
        higher_is_better: dict[str, bool] | None = None,
    ) -> ABTestResult:
        """Compare two variants on the same dataset.

        Args:
            dataset: Dataset to evaluate both variants on.
            variant_a: First variant (target function).
            variant_b: Second variant (target function).
            metrics: Metrics to compare.
            variant_a_name: Human-readable name for variant A.
            variant_b_name: Human-readable name for variant B.
            test_name: Name for this A/B test.
            higher_is_better: Dict mapping metric names to whether higher is better.
                Defaults to True for accuracy metrics, False for latency/cost.

        Returns:
            ABTestResult with detailed comparison and recommendation.
        """
        test_name = test_name or f"ab_test_{dataset.name}"

        # Default higher_is_better settings
        default_higher_is_better = {
            "exact_match": True,
            "contains_keywords": True,
            "latency": False,  # Lower is better
            "token_cost": False,  # Lower is better
            "llm_judge": True,
        }
        if higher_is_better:
            default_higher_is_better.update(higher_is_better)

        # Run both variants
        run_a = await self.runner.evaluate(
            dataset=dataset,
            target=variant_a,
            metrics=metrics,
            run_name=f"{test_name}_{variant_a_name}",
        )

        run_b = await self.runner.evaluate(
            dataset=dataset,
            target=variant_b,
            metrics=metrics,
            run_name=f"{test_name}_{variant_b_name}",
        )

        # Compare metrics
        metric_comparisons = self._compare_metrics(
            run_a, run_b, metrics, default_higher_is_better
        )

        # Determine overall winner
        overall_winner, recommendation = self._determine_winner(
            metric_comparisons, variant_a_name, variant_b_name
        )

        return ABTestResult(
            name=test_name,
            variant_a_name=variant_a_name,
            variant_b_name=variant_b_name,
            dataset_name=dataset.name,
            total_cases=len(dataset),
            run_a=run_a,
            run_b=run_b,
            metric_comparisons=metric_comparisons,
            overall_winner=overall_winner,
            recommendation=recommendation,
        )

    def _compare_metrics(
        self,
        run_a: EvalRunResult,
        run_b: EvalRunResult,
        metrics: list[Metric],
        higher_is_better: dict[str, bool],
    ) -> dict[str, MetricComparison]:
        """Compare metrics between two runs."""
        comparisons = {}

        for metric in metrics:
            metric_name = metric.name
            hib = higher_is_better.get(metric_name, True)

            # Extract per-case scores
            scores_a = []
            scores_b = []

            for result_a, result_b in zip(run_a.results, run_b.results):
                if result_a.success and result_b.success:
                    score_a = result_a.metric_results.get(metric_name, {}).get("score", 0)
                    score_b = result_b.metric_results.get(metric_name, {}).get("score", 0)
                    scores_a.append(score_a)
                    scores_b.append(score_b)

            if not scores_a:
                continue

            mean_a = statistics.mean(scores_a)
            mean_b = statistics.mean(scores_b)
            std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.0
            std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.0

            difference = mean_b - mean_a
            relative_diff = (difference / mean_a * 100) if mean_a != 0 else 0.0

            # Statistical test
            stats = StatisticalResult.from_samples(
                scores_a, scores_b, self.confidence_level
            )

            # Determine winner for this metric
            if not stats.significant:
                winner = Winner.TIE
            elif abs(stats.effect_size) < self.min_effect_size:
                winner = Winner.TIE
            else:
                if hib:
                    # Higher is better
                    winner = Winner.VARIANT_B if mean_b > mean_a else Winner.VARIANT_A
                else:
                    # Lower is better
                    winner = Winner.VARIANT_A if mean_a < mean_b else Winner.VARIANT_B

            comparisons[metric_name] = MetricComparison(
                metric_name=metric_name,
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                difference=difference,
                relative_difference=relative_diff,
                stats=stats,
                winner=winner,
                higher_is_better=hib,
            )

        return comparisons

    def _determine_winner(
        self,
        comparisons: dict[str, MetricComparison],
        variant_a_name: str,
        variant_b_name: str,
    ) -> tuple[Winner, str]:
        """Determine overall winner and generate recommendation."""
        if not comparisons:
            return Winner.INCONCLUSIVE, "No metrics to compare"

        a_wins = sum(1 for c in comparisons.values() if c.winner == Winner.VARIANT_A)
        b_wins = sum(1 for c in comparisons.values() if c.winner == Winner.VARIANT_B)
        ties = sum(1 for c in comparisons.values() if c.winner == Winner.TIE)

        # Check for significant wins
        significant_a = sum(
            1
            for c in comparisons.values()
            if c.winner == Winner.VARIANT_A and c.stats.significant
        )
        significant_b = sum(
            1
            for c in comparisons.values()
            if c.winner == Winner.VARIANT_B and c.stats.significant
        )

        # Build recommendation
        parts = []

        if significant_a > significant_b:
            overall = Winner.VARIANT_A
            parts.append(f"Recommend {variant_a_name}")
            parts.append(f"Wins {a_wins} of {len(comparisons)} metrics")
            if significant_a > 0:
                parts.append(f"{significant_a} statistically significant")
        elif significant_b > significant_a:
            overall = Winner.VARIANT_B
            parts.append(f"Recommend {variant_b_name}")
            parts.append(f"Wins {b_wins} of {len(comparisons)} metrics")
            if significant_b > 0:
                parts.append(f"{significant_b} statistically significant")
        elif a_wins > b_wins:
            overall = Winner.VARIANT_A
            parts.append(f"Slight edge to {variant_a_name}")
            parts.append(f"Wins {a_wins} vs {b_wins} metrics")
            parts.append("No statistically significant differences")
        elif b_wins > a_wins:
            overall = Winner.VARIANT_B
            parts.append(f"Slight edge to {variant_b_name}")
            parts.append(f"Wins {b_wins} vs {a_wins} metrics")
            parts.append("No statistically significant differences")
        elif ties == len(comparisons):
            overall = Winner.TIE
            parts.append("No meaningful difference between variants")
            parts.append("Choose based on other factors (cost, complexity)")
        else:
            overall = Winner.INCONCLUSIVE
            parts.append("Mixed results - no clear winner")
            parts.append(f"A wins: {a_wins}, B wins: {b_wins}, Ties: {ties}")

        return overall, ". ".join(parts)


async def compare_variants(
    dataset: EvalDataset,
    variant_a: TargetFunc,
    variant_b: TargetFunc,
    metrics: list[Metric],
    **kwargs: Any,
) -> ABTestResult:
    """Convenience function for quick A/B testing.

    Args:
        dataset: Dataset to evaluate.
        variant_a: First variant.
        variant_b: Second variant.
        metrics: Metrics to compare.
        **kwargs: Additional arguments passed to ABTest.compare().

    Returns:
        ABTestResult with comparison details.
    """
    ab_test = ABTest()
    return await ab_test.compare(dataset, variant_a, variant_b, metrics, **kwargs)
