"""Metrics collector for aggregating runtime statistics.

This module provides a metrics collector for tracking evaluation performance,
success rates, latency distributions, and token usage over time.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class LatencyHistogram:
    """Histogram for tracking latency distribution.

    Stores individual values and computes percentiles on demand.
    """

    values: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, value: float) -> None:
        """Record a latency value."""
        with self._lock:
            self.values.append(value)

    def count(self) -> int:
        """Get total number of recorded values."""
        return len(self.values)

    def sum(self) -> float:
        """Get sum of all values."""
        return sum(self.values) if self.values else 0.0

    def mean(self) -> float:
        """Get mean of all values."""
        return statistics.mean(self.values) if self.values else 0.0

    def percentile(self, p: float) -> float:
        """Get percentile value (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * (p / 100))
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def p50(self) -> float:
        """Get 50th percentile (median)."""
        return self.percentile(50)

    def p90(self) -> float:
        """Get 90th percentile."""
        return self.percentile(90)

    def p95(self) -> float:
        """Get 95th percentile."""
        return self.percentile(95)

    def p99(self) -> float:
        """Get 99th percentile."""
        return self.percentile(99)

    def min(self) -> float:
        """Get minimum value."""
        return min(self.values) if self.values else 0.0

    def max(self) -> float:
        """Get maximum value."""
        return max(self.values) if self.values else 0.0

    def reset(self) -> None:
        """Reset the histogram."""
        with self._lock:
            self.values.clear()

    def to_dict(self) -> dict[str, float]:
        """Export histogram stats as dictionary."""
        if not self.values:
            return {
                "count": 0,
                "sum": 0.0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        return {
            "count": self.count(),
            "sum": round(self.sum(), 2),
            "mean": round(self.mean(), 2),
            "min": round(self.min(), 2),
            "max": round(self.max(), 2),
            "p50": round(self.p50(), 2),
            "p90": round(self.p90(), 2),
            "p95": round(self.p95(), 2),
            "p99": round(self.p99(), 2),
        }


@dataclass
class Counter:
    """Thread-safe counter."""

    _value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, amount: int = 1) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount

    def value(self) -> int:
        """Get current value."""
        return self._value

    def reset(self) -> None:
        """Reset to zero."""
        with self._lock:
            self._value = 0


@dataclass
class TokenUsageStats:
    """Statistics for token usage."""

    input_tokens: Counter = field(default_factory=Counter)
    output_tokens: Counter = field(default_factory=Counter)

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage."""
        self.input_tokens.increment(input_tokens)
        self.output_tokens.increment(output_tokens)

    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        return self.input_tokens.value() + self.output_tokens.value()

    def reset(self) -> None:
        """Reset all counters."""
        self.input_tokens.reset()
        self.output_tokens.reset()

    def to_dict(self) -> dict[str, int]:
        """Export as dictionary."""
        return {
            "input_tokens": self.input_tokens.value(),
            "output_tokens": self.output_tokens.value(),
            "total_tokens": self.total_tokens(),
        }


class MetricsCollector:
    """Collects and aggregates runtime metrics for evaluations.

    Thread-safe collector for tracking evaluation performance across
    multiple runs. Supports exporting metrics for dashboards.

    Example:
        ```python
        collector = MetricsCollector()

        # Record metrics during evaluation
        collector.record_run_started()
        collector.record_case_result(success=True, latency_ms=150)
        collector.record_case_result(success=True, latency_ms=200)
        collector.record_run_completed(passed=True)

        # Export for dashboard
        metrics = collector.export()
        print(metrics["success_rate"])  # 1.0
        print(metrics["latency"]["p95"])  # ~200
        ```
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._lock = threading.Lock()

        # Counters
        self.total_runs = Counter()
        self.successful_runs = Counter()
        self.failed_runs = Counter()
        self.total_cases = Counter()
        self.successful_cases = Counter()
        self.failed_cases = Counter()
        self.passed_cases = Counter()  # Cases where all metrics passed

        # Histograms
        self.case_latency = LatencyHistogram()
        self.run_duration = LatencyHistogram()

        # Token usage
        self.token_usage = TokenUsageStats()

        # Metric-specific pass rates
        self.metric_passes: dict[str, Counter] = {}
        self.metric_totals: dict[str, Counter] = {}

        # Timestamps
        self.started_at: datetime = datetime.now(timezone.utc)
        self.last_run_at: datetime | None = None

    def record_run_started(self) -> None:
        """Record that an evaluation run has started."""
        self.total_runs.increment()

    def record_run_completed(
        self,
        passed: bool,
        duration_ms: float,
    ) -> None:
        """Record that an evaluation run has completed.

        Args:
            passed: Whether all metrics passed.
            duration_ms: Total run duration in milliseconds.
        """
        if passed:
            self.successful_runs.increment()
        else:
            self.failed_runs.increment()

        self.run_duration.record(duration_ms)
        self.last_run_at = datetime.now(timezone.utc)

    def record_case_result(
        self,
        success: bool,
        latency_ms: float,
        metrics_passed: bool = True,
        metric_results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Record the result of evaluating a single case.

        Args:
            success: Whether the case executed successfully (no errors).
            latency_ms: Case latency in milliseconds.
            metrics_passed: Whether all metrics passed for this case.
            metric_results: Individual metric results.
        """
        self.total_cases.increment()

        if success:
            self.successful_cases.increment()
            self.case_latency.record(latency_ms)

            if metrics_passed:
                self.passed_cases.increment()
        else:
            self.failed_cases.increment()

        # Track per-metric pass rates
        if metric_results:
            for metric_name, result in metric_results.items():
                if metric_name not in self.metric_passes:
                    self.metric_passes[metric_name] = Counter()
                    self.metric_totals[metric_name] = Counter()

                self.metric_totals[metric_name].increment()
                if result.get("passed", False):
                    self.metric_passes[metric_name].increment()

    def record_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from an LLM call.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        self.token_usage.record(input_tokens, output_tokens)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_runs.reset()
            self.successful_runs.reset()
            self.failed_runs.reset()
            self.total_cases.reset()
            self.successful_cases.reset()
            self.failed_cases.reset()
            self.passed_cases.reset()
            self.case_latency.reset()
            self.run_duration.reset()
            self.token_usage.reset()
            self.metric_passes.clear()
            self.metric_totals.clear()
            self.started_at = datetime.now(timezone.utc)
            self.last_run_at = None

    def success_rate(self) -> float:
        """Calculate overall case success rate (execution success)."""
        total = self.total_cases.value()
        if total == 0:
            return 0.0
        return self.successful_cases.value() / total

    def pass_rate(self) -> float:
        """Calculate metric pass rate (cases where all metrics passed)."""
        total = self.successful_cases.value()
        if total == 0:
            return 0.0
        return self.passed_cases.value() / total

    def metric_pass_rate(self, metric_name: str) -> float:
        """Calculate pass rate for a specific metric."""
        if metric_name not in self.metric_totals:
            return 0.0
        total = self.metric_totals[metric_name].value()
        if total == 0:
            return 0.0
        return self.metric_passes[metric_name].value() / total

    def export(self) -> dict[str, Any]:
        """Export all metrics as a dictionary.

        Returns:
            Dictionary containing all collected metrics, suitable for
            dashboard consumption or JSON serialization.
        """
        now = datetime.now(timezone.utc)

        # Calculate per-metric pass rates
        metric_pass_rates = {}
        for name in self.metric_totals:
            metric_pass_rates[name] = {
                "total": self.metric_totals[name].value(),
                "passed": self.metric_passes[name].value(),
                "pass_rate": round(self.metric_pass_rate(name), 4),
            }

        return {
            "collection_period": {
                "started_at": self.started_at.isoformat(),
                "last_run_at": (
                    self.last_run_at.isoformat() if self.last_run_at else None
                ),
                "exported_at": now.isoformat(),
            },
            "runs": {
                "total": self.total_runs.value(),
                "successful": self.successful_runs.value(),
                "failed": self.failed_runs.value(),
            },
            "cases": {
                "total": self.total_cases.value(),
                "successful": self.successful_cases.value(),
                "failed": self.failed_cases.value(),
                "passed": self.passed_cases.value(),
                "success_rate": round(self.success_rate(), 4),
                "pass_rate": round(self.pass_rate(), 4),
            },
            "latency": self.case_latency.to_dict(),
            "run_duration": self.run_duration.to_dict(),
            "tokens": self.token_usage.to_dict(),
            "metrics": metric_pass_rates,
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format.
        """
        lines = []

        # Counters
        lines.append("# HELP evalops_runs_total Total number of evaluation runs")
        lines.append("# TYPE evalops_runs_total counter")
        lines.append(f"evalops_runs_total {self.total_runs.value()}")

        lines.append("# HELP evalops_cases_total Total number of evaluated cases")
        lines.append("# TYPE evalops_cases_total counter")
        lines.append(f'evalops_cases_total{{status="successful"}} {self.successful_cases.value()}')
        lines.append(f'evalops_cases_total{{status="failed"}} {self.failed_cases.value()}')
        lines.append(f'evalops_cases_total{{status="passed"}} {self.passed_cases.value()}')

        # Gauges
        lines.append("# HELP evalops_success_rate Current success rate")
        lines.append("# TYPE evalops_success_rate gauge")
        lines.append(f"evalops_success_rate {self.success_rate():.4f}")

        lines.append("# HELP evalops_pass_rate Current metric pass rate")
        lines.append("# TYPE evalops_pass_rate gauge")
        lines.append(f"evalops_pass_rate {self.pass_rate():.4f}")

        # Latency histogram
        lines.append("# HELP evalops_latency_ms Case latency in milliseconds")
        lines.append("# TYPE evalops_latency_ms summary")
        lines.append(f'evalops_latency_ms{{quantile="0.5"}} {self.case_latency.p50():.2f}')
        lines.append(f'evalops_latency_ms{{quantile="0.9"}} {self.case_latency.p90():.2f}')
        lines.append(f'evalops_latency_ms{{quantile="0.95"}} {self.case_latency.p95():.2f}')
        lines.append(f'evalops_latency_ms{{quantile="0.99"}} {self.case_latency.p99():.2f}')
        lines.append(f"evalops_latency_ms_sum {self.case_latency.sum():.2f}")
        lines.append(f"evalops_latency_ms_count {self.case_latency.count()}")

        # Tokens
        lines.append("# HELP evalops_tokens_total Total tokens used")
        lines.append("# TYPE evalops_tokens_total counter")
        input_tokens = self.token_usage.input_tokens.value()
        output_tokens = self.token_usage.output_tokens.value()
        lines.append(f'evalops_tokens_total{{type="input"}} {input_tokens}')
        lines.append(f'evalops_tokens_total{{type="output"}} {output_tokens}')

        # Per-metric pass rates
        for name in self.metric_totals:
            pass_rate = self.metric_pass_rate(name)
            lines.append(f'evalops_metric_pass_rate{{metric="{name}"}} {pass_rate:.4f}')

        return "\n".join(lines)


# Global collector instance
_default_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get or create the default metrics collector."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def reset_collector() -> None:
    """Reset the global metrics collector."""
    global _default_collector
    if _default_collector is not None:
        _default_collector.reset()
