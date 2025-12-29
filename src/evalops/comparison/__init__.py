"""Comparison engine for A/B testing and drift detection.

This module provides:
- A/B testing framework for comparing LLM variants
- Drift detection for monitoring quality over time
- Regression testing for CI/CD pipelines
"""

from evalops.comparison.ab_test import (
    ABTest,
    ABTestResult,
    MetricComparison,
    StatisticalResult,
    Winner,
    compare_variants,
)
from evalops.comparison.drift import (
    AlertSeverity,
    DriftAlert,
    DriftDetector,
    DriftDirection,
    DriftReport,
    MetricSnapshot,
)
from evalops.comparison.regression import (
    MetricThreshold,
    RegressionReport,
    RegressionResult,
    RegressionStatus,
    RegressionTester,
    run_regression_test,
)

__all__ = [
    # A/B Testing
    "ABTest",
    "ABTestResult",
    "MetricComparison",
    "StatisticalResult",
    "Winner",
    "compare_variants",
    # Drift Detection
    "AlertSeverity",
    "DriftAlert",
    "DriftDetector",
    "DriftDirection",
    "DriftReport",
    "MetricSnapshot",
    # Regression Testing
    "MetricThreshold",
    "RegressionReport",
    "RegressionResult",
    "RegressionStatus",
    "RegressionTester",
    "run_regression_test",
]
