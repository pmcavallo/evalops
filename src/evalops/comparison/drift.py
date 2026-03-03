"""Drift detection for monitoring LLM quality over time.

This module provides tools for detecting quality degradation in LLM
systems by comparing current performance against established baselines.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class AlertSeverity(Enum):
    """Severity level of a drift alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftDirection(Enum):
    """Direction of drift from baseline."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class MetricSnapshot:
    """A snapshot of metric values at a point in time.

    Attributes:
        timestamp: When the snapshot was taken.
        run_id: ID of the evaluation run.
        metrics: Dict of metric name to score.
        pass_rate: Overall pass rate for this run.
    """

    timestamp: datetime
    run_id: str
    metrics: dict[str, float]
    pass_rate: float

    @classmethod
    def from_run_result(cls, run_result: Any) -> MetricSnapshot:
        """Create a snapshot from an EvalRunResult."""
        metrics = {}
        for name, summary in run_result.metrics_summary.items():
            # Use pass_rate from details if available, otherwise score
            if "pass_rate" in summary.get("details", {}):
                metrics[name] = summary["details"]["pass_rate"]
            else:
                metrics[name] = summary.get("score", 0.0)

        return cls(
            timestamp=run_result.completed_at or datetime.now(timezone.utc),
            run_id=run_result.id,
            metrics=metrics,
            pass_rate=run_result.pass_rate,
        )


@dataclass
class DriftAlert:
    """Alert generated when drift is detected.

    Attributes:
        id: Unique identifier for this alert.
        severity: Alert severity level.
        metric_name: Name of the metric that drifted.
        baseline_value: The baseline value.
        current_value: The current value.
        degradation: Percentage degradation from baseline.
        direction: Direction of drift.
        message: Human-readable alert message.
        suggested_actions: List of suggested remediation actions.
        created_at: When the alert was created.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    severity: AlertSeverity = AlertSeverity.WARNING
    metric_name: str = ""
    baseline_value: float = 0.0
    current_value: float = 0.0
    degradation: float = 0.0
    direction: DriftDirection = DriftDirection.STABLE
    message: str = ""
    suggested_actions: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Export alert as dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "baseline_value": round(self.baseline_value, 4),
            "current_value": round(self.current_value, 4),
            "degradation": round(self.degradation, 4),
            "direction": self.direction.value,
            "message": self.message,
            "suggested_actions": self.suggested_actions,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DriftReport:
    """Complete drift analysis report.

    Attributes:
        id: Unique identifier for this report.
        drift_detected: Whether any drift was detected.
        alerts: List of drift alerts.
        baseline_metrics: Baseline metric values.
        current_metrics: Current metric values (rolling average).
        metric_trends: Trend direction for each metric.
        overall_health: Overall system health assessment.
        snapshot_count: Number of snapshots in the analysis window.
        created_at: When the report was generated.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    drift_detected: bool = False
    alerts: list[DriftAlert] = field(default_factory=list)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    current_metrics: dict[str, float] = field(default_factory=dict)
    metric_trends: dict[str, DriftDirection] = field(default_factory=dict)
    overall_health: str = "healthy"
    snapshot_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Export report as dictionary."""
        return {
            "id": self.id,
            "drift_detected": self.drift_detected,
            "alerts": [a.to_dict() for a in self.alerts],
            "baseline_metrics": {k: round(v, 4) for k, v in self.baseline_metrics.items()},
            "current_metrics": {k: round(v, 4) for k, v in self.current_metrics.items()},
            "metric_trends": {k: v.value for k, v in self.metric_trends.items()},
            "overall_health": self.overall_health,
            "snapshot_count": self.snapshot_count,
            "created_at": self.created_at.isoformat(),
        }


class DriftDetector:
    """Detects quality drift by comparing current metrics against baseline.

    Tracks metrics over time using a rolling window and alerts when
    quality degrades beyond configurable thresholds.

    Example:
        ```python
        detector = DriftDetector(
            warning_threshold=0.05,   # 5% degradation = warning
            critical_threshold=0.10,  # 10% degradation = critical
            window_size=5,            # 5-run rolling average
        )

        # Establish baseline from initial runs
        detector.set_baseline(initial_run_result)

        # Check subsequent runs for drift
        for run_result in new_results:
            detector.add_snapshot(run_result)
            report = detector.check()

            if report.drift_detected:
                for alert in report.alerts:
                    print(f"ALERT: {alert.message}")
        ```
    """

    # Default suggested actions for different scenarios
    DEFAULT_ACTIONS = {
        "retrieval": [
            "Check retrieval pipeline for recent changes",
            "Verify embedding model version",
            "Review recent document additions",
            "Check vector database health",
        ],
        "generation": [
            "Review recent prompt changes",
            "Check model configuration",
            "Verify API endpoint and model version",
            "Review temperature and sampling settings",
        ],
        "latency": [
            "Check for infrastructure issues",
            "Review recent deployments",
            "Monitor API rate limits",
            "Check for increased load",
        ],
        "general": [
            "Review recent changes to the system",
            "Check data pipeline integrity",
            "Verify external dependencies",
            "Run diagnostic tests",
        ],
    }

    def __init__(
        self,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        window_size: int = 3,
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        """Initialize the drift detector.

        Args:
            warning_threshold: Percentage drop to trigger warning (e.g., 0.05 = 5%).
            critical_threshold: Percentage drop to trigger critical alert.
            window_size: Number of recent runs for rolling average.
            higher_is_better: Dict mapping metric names to whether higher is better.
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size

        # Default higher_is_better settings
        self._higher_is_better = {
            "exact_match": True,
            "contains_keywords": True,
            "latency": False,
            "token_cost": False,
            "llm_judge": True,
            "pass_rate": True,
        }
        if higher_is_better:
            self._higher_is_better.update(higher_is_better)

        # Baseline and history
        self._baseline: MetricSnapshot | None = None
        self._history: deque[MetricSnapshot] = deque(maxlen=window_size * 2)

    def set_baseline(self, run_result: Any) -> None:
        """Set the baseline from an evaluation run result.

        Args:
            run_result: EvalRunResult to use as baseline.
        """
        self._baseline = MetricSnapshot.from_run_result(run_result)

    def set_baseline_from_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Set baseline directly from a snapshot."""
        self._baseline = snapshot

    def set_baseline_from_values(self, metrics: dict[str, float], pass_rate: float = 1.0) -> None:
        """Set baseline from raw metric values.

        Args:
            metrics: Dict of metric name to baseline value.
            pass_rate: Overall pass rate baseline.
        """
        self._baseline = MetricSnapshot(
            timestamp=datetime.now(timezone.utc),
            run_id="baseline",
            metrics=metrics,
            pass_rate=pass_rate,
        )

    def add_snapshot(self, run_result: Any) -> MetricSnapshot:
        """Add a new evaluation result to the history.

        Args:
            run_result: EvalRunResult to add.

        Returns:
            The created MetricSnapshot.
        """
        snapshot = MetricSnapshot.from_run_result(run_result)
        self._history.append(snapshot)
        return snapshot

    def add_raw_snapshot(
        self,
        metrics: dict[str, float],
        pass_rate: float,
        run_id: str | None = None,
    ) -> MetricSnapshot:
        """Add a snapshot from raw values.

        Args:
            metrics: Dict of metric name to value.
            pass_rate: Overall pass rate.
            run_id: Optional run identifier.

        Returns:
            The created MetricSnapshot.
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.now(timezone.utc),
            run_id=run_id or str(uuid4()),
            metrics=metrics,
            pass_rate=pass_rate,
        )
        self._history.append(snapshot)
        return snapshot

    def _get_rolling_average(self) -> dict[str, float]:
        """Calculate rolling average of recent snapshots."""
        if not self._history:
            return {}

        recent = list(self._history)[-self.window_size :]

        # Aggregate metrics
        metric_sums: dict[str, list[float]] = {}
        pass_rates: list[float] = []

        for snapshot in recent:
            pass_rates.append(snapshot.pass_rate)
            for name, value in snapshot.metrics.items():
                if name not in metric_sums:
                    metric_sums[name] = []
                metric_sums[name].append(value)

        # Calculate averages
        result = {}
        for name, values in metric_sums.items():
            result[name] = sum(values) / len(values)

        if pass_rates:
            result["pass_rate"] = sum(pass_rates) / len(pass_rates)

        return result

    def check(self) -> DriftReport:
        """Check for drift against baseline.

        Returns:
            DriftReport with drift analysis and any alerts.
        """
        if self._baseline is None:
            return DriftReport(
                drift_detected=False,
                overall_health="unknown",
                alerts=[
                    DriftAlert(
                        severity=AlertSeverity.INFO,
                        message="No baseline established. Set baseline before checking for drift.",
                    )
                ],
            )

        if not self._history:
            return DriftReport(
                drift_detected=False,
                overall_health="unknown",
                baseline_metrics=self._baseline.metrics,
                alerts=[
                    DriftAlert(
                        severity=AlertSeverity.INFO,
                        message="No evaluation history. Add snapshots before checking for drift.",
                    )
                ],
            )

        current_metrics = self._get_rolling_average()
        baseline_metrics = {**self._baseline.metrics, "pass_rate": self._baseline.pass_rate}

        alerts: list[DriftAlert] = []
        trends: dict[str, DriftDirection] = {}

        # Check each metric
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name not in current_metrics:
                continue

            current_value = current_metrics[metric_name]
            higher_is_better = self._higher_is_better.get(metric_name, True)

            # Calculate degradation
            if baseline_value == 0:
                degradation = 0.0 if current_value == 0 else 1.0
            else:
                if higher_is_better:
                    degradation = (baseline_value - current_value) / baseline_value
                else:
                    degradation = (current_value - baseline_value) / baseline_value

            # Determine trend
            if degradation > self.warning_threshold:
                trends[metric_name] = DriftDirection.DEGRADING
            elif degradation < -self.warning_threshold:
                trends[metric_name] = DriftDirection.IMPROVING
            else:
                trends[metric_name] = DriftDirection.STABLE

            # Generate alerts for degradation
            if degradation >= self.critical_threshold:
                alerts.append(
                    self._create_alert(
                        AlertSeverity.CRITICAL,
                        metric_name,
                        baseline_value,
                        current_value,
                        degradation,
                    )
                )
            elif degradation >= self.warning_threshold:
                alerts.append(
                    self._create_alert(
                        AlertSeverity.WARNING,
                        metric_name,
                        baseline_value,
                        current_value,
                        degradation,
                    )
                )

        # Determine overall health
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

        if critical_count > 0:
            overall_health = "critical"
        elif warning_count > 0:
            overall_health = "degraded"
        elif any(t == DriftDirection.IMPROVING for t in trends.values()):
            overall_health = "improving"
        else:
            overall_health = "healthy"

        return DriftReport(
            drift_detected=len(alerts) > 0,
            alerts=alerts,
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            metric_trends=trends,
            overall_health=overall_health,
            snapshot_count=len(self._history),
        )

    def _create_alert(
        self,
        severity: AlertSeverity,
        metric_name: str,
        baseline_value: float,
        current_value: float,
        degradation: float,
    ) -> DriftAlert:
        """Create a drift alert with appropriate message and actions."""
        direction = DriftDirection.DEGRADING

        # Generate message
        pct = abs(degradation) * 100
        if severity == AlertSeverity.CRITICAL:
            message = f"CRITICAL: {metric_name} degraded by {pct:.1f}% from baseline"
        else:
            message = f"WARNING: {metric_name} dropped {pct:.1f}% from baseline"

        # Suggest actions based on metric type
        if "latency" in metric_name.lower():
            actions = self.DEFAULT_ACTIONS["latency"]
        elif any(x in metric_name.lower() for x in ["match", "accuracy", "judge"]):
            actions = self.DEFAULT_ACTIONS["generation"]
        elif "retrieval" in metric_name.lower() or "keyword" in metric_name.lower():
            actions = self.DEFAULT_ACTIONS["retrieval"]
        else:
            actions = self.DEFAULT_ACTIONS["general"]

        return DriftAlert(
            severity=severity,
            metric_name=metric_name,
            baseline_value=baseline_value,
            current_value=current_value,
            degradation=degradation,
            direction=direction,
            message=message,
            suggested_actions=actions,
        )

    def reset(self) -> None:
        """Reset the detector, clearing baseline and history."""
        self._baseline = None
        self._history.clear()

    def clear_history(self) -> None:
        """Clear history but keep baseline."""
        self._history.clear()
