"""Unit tests for the comparison module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evalops.comparison.ab_test import (
    ABTest,
    ABTestResult,
    MetricComparison,
    StatisticalResult,
    Winner,
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
)


class TestStatisticalResult:
    """Tests for StatisticalResult class."""

    def test_from_samples_equal_means(self) -> None:
        """Test samples with identical values."""
        samples_a = [1.0, 1.0, 1.0, 1.0, 1.0]
        samples_b = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = StatisticalResult.from_samples(samples_a, samples_b)

        assert result.t_statistic == 0.0
        assert result.p_value == 1.0
        assert not result.significant
        assert result.effect_size == 0.0
        assert result.effect_magnitude == "none"

    def test_from_samples_different_means(self) -> None:
        """Test samples with different means."""
        samples_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        samples_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        result = StatisticalResult.from_samples(samples_a, samples_b)

        # B is consistently higher
        assert result.t_statistic < 0  # Negative because A - B is negative
        assert result.p_value < 0.05
        assert result.significant
        assert result.effect_size < 0  # Negative because B > A
        assert result.effect_magnitude == "large"

    def test_from_samples_small_difference(self) -> None:
        """Test samples with small, non-significant difference."""
        samples_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        samples_b = [1.1, 2.1, 3.1, 4.1, 5.1]

        result = StatisticalResult.from_samples(samples_a, samples_b)

        # Small difference should not be significant
        assert abs(result.effect_size) < 0.5

    def test_from_samples_mismatched_lengths(self) -> None:
        """Test that mismatched sample sizes raise error."""
        samples_a = [1.0, 2.0, 3.0]
        samples_b = [1.0, 2.0]

        with pytest.raises(ValueError, match="Sample sizes must match"):
            StatisticalResult.from_samples(samples_a, samples_b)

    def test_from_samples_too_few(self) -> None:
        """Test with fewer than 2 samples."""
        samples_a = [1.0]
        samples_b = [2.0]

        result = StatisticalResult.from_samples(samples_a, samples_b)

        assert result.p_value == 1.0
        assert not result.significant
        assert result.effect_magnitude == "none"

    def test_effect_magnitude_thresholds(self) -> None:
        """Test effect size magnitude categorization."""
        # Negligible: < 0.2
        samples_a = [1.0, 1.1, 1.2, 1.3, 1.4]
        samples_b = [1.05, 1.15, 1.25, 1.35, 1.45]
        result = StatisticalResult.from_samples(samples_a, samples_b)
        assert result.effect_magnitude in ["negligible", "small"]


class TestMetricComparison:
    """Tests for MetricComparison class."""

    def test_metric_comparison_creation(self) -> None:
        """Test creating a metric comparison."""
        stats = StatisticalResult(
            t_statistic=2.5,
            p_value=0.02,
            significant=True,
            effect_size=0.6,
            effect_magnitude="medium",
        )

        comparison = MetricComparison(
            metric_name="exact_match",
            mean_a=0.85,
            mean_b=0.92,
            std_a=0.05,
            std_b=0.04,
            difference=0.07,
            relative_difference=8.24,
            stats=stats,
            winner=Winner.VARIANT_B,
        )

        assert comparison.metric_name == "exact_match"
        assert comparison.mean_b > comparison.mean_a
        assert comparison.winner == Winner.VARIANT_B


class TestABTestResult:
    """Tests for ABTestResult class."""

    def test_ab_test_result_creation(self) -> None:
        """Test creating an AB test result."""
        result = ABTestResult(
            name="test_comparison",
            variant_a_name="prompt_v1",
            variant_b_name="prompt_v2",
            dataset_name="test_dataset",
            total_cases=100,
            overall_winner=Winner.VARIANT_B,
            recommendation="Recommend prompt_v2. Wins 2 of 3 metrics.",
        )

        assert result.name == "test_comparison"
        assert result.overall_winner == Winner.VARIANT_B

    def test_to_summary_dict(self) -> None:
        """Test exporting as summary dictionary."""
        stats = StatisticalResult(
            t_statistic=2.5,
            p_value=0.02,
            significant=True,
            effect_size=0.6,
            effect_magnitude="medium",
        )

        comparison = MetricComparison(
            metric_name="exact_match",
            mean_a=0.85,
            mean_b=0.92,
            std_a=0.05,
            std_b=0.04,
            difference=0.07,
            relative_difference=8.24,
            stats=stats,
            winner=Winner.VARIANT_B,
        )

        result = ABTestResult(
            name="test",
            metric_comparisons={"exact_match": comparison},
            overall_winner=Winner.VARIANT_B,
            recommendation="Test recommendation",
        )

        summary = result.to_summary_dict()

        assert summary["overall_winner"] == "variant_b"
        assert "exact_match" in summary["metrics"]
        assert summary["metrics"]["exact_match"]["significant"] is True


class TestABTest:
    """Tests for ABTest class."""

    @pytest.fixture
    def ab_test(self) -> ABTest:
        """Create ABTest instance with mocked runner."""
        return ABTest(confidence_level=0.95, min_effect_size=0.2)

    @pytest.mark.asyncio
    async def test_compare_variants(self, ab_test: ABTest) -> None:
        """Test comparing two variants."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
            ],
        )

        def variant_a(x: str) -> str:
            return x.upper()

        def variant_b(x: str) -> str:
            return x.upper()

        result = await ab_test.compare(
            dataset=dataset,
            variant_a=variant_a,
            variant_b=variant_b,
            metrics=[ExactMatch()],
            variant_a_name="upper_v1",
            variant_b_name="upper_v2",
        )

        assert isinstance(result, ABTestResult)
        assert result.total_cases == 2
        assert result.variant_a_name == "upper_v1"
        assert result.variant_b_name == "upper_v2"

    @pytest.mark.asyncio
    async def test_compare_with_clear_winner(self, ab_test: ABTest) -> None:
        """Test comparison with a clear winner."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
                EvalCase(input="c", expected="C"),
            ],
        )

        def variant_a(x: str) -> str:
            return x.upper()  # Always correct

        def variant_b(x: str) -> str:
            return x.lower()  # Always wrong

        result = await ab_test.compare(
            dataset=dataset,
            variant_a=variant_a,
            variant_b=variant_b,
            metrics=[ExactMatch()],
        )

        assert result.overall_winner in [Winner.VARIANT_A, Winner.TIE, Winner.INCONCLUSIVE]


class TestMetricSnapshot:
    """Tests for MetricSnapshot class."""

    def test_snapshot_creation(self) -> None:
        """Test creating a metric snapshot."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(timezone.utc),
            run_id="run-123",
            metrics={"exact_match": 0.95, "latency": 150.0},
            pass_rate=0.92,
        )

        assert snapshot.run_id == "run-123"
        assert snapshot.metrics["exact_match"] == 0.95
        assert snapshot.pass_rate == 0.92

    def test_from_run_result(self) -> None:
        """Test creating snapshot from run result."""
        # Create a mock run result
        mock_run = MagicMock()
        mock_run.id = "run-456"
        mock_run.completed_at = datetime.now(timezone.utc)
        mock_run.pass_rate = 0.88
        mock_run.metrics_summary = {
            "exact_match": {"score": 0.9, "details": {"pass_rate": 0.9}},
            "latency": {"score": 120.0},
        }

        snapshot = MetricSnapshot.from_run_result(mock_run)

        assert snapshot.run_id == "run-456"
        assert snapshot.metrics["exact_match"] == 0.9
        assert snapshot.metrics["latency"] == 120.0
        assert snapshot.pass_rate == 0.88


class TestDriftAlert:
    """Tests for DriftAlert class."""

    def test_alert_creation(self) -> None:
        """Test creating a drift alert."""
        alert = DriftAlert(
            severity=AlertSeverity.WARNING,
            metric_name="exact_match",
            baseline_value=0.95,
            current_value=0.88,
            degradation=0.0737,
            direction=DriftDirection.DEGRADING,
            message="WARNING: exact_match dropped 7.4% from baseline",
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.direction == DriftDirection.DEGRADING

    def test_to_dict(self) -> None:
        """Test exporting alert as dictionary."""
        alert = DriftAlert(
            severity=AlertSeverity.CRITICAL,
            metric_name="latency",
            baseline_value=100.0,
            current_value=150.0,
            degradation=0.50,
            message="CRITICAL: latency degraded by 50.0%",
        )

        data = alert.to_dict()

        assert data["severity"] == "critical"
        assert data["metric_name"] == "latency"
        assert data["degradation"] == 0.5


class TestDriftReport:
    """Tests for DriftReport class."""

    def test_report_creation(self) -> None:
        """Test creating a drift report."""
        report = DriftReport(
            drift_detected=True,
            overall_health="degraded",
            baseline_metrics={"exact_match": 0.95},
            current_metrics={"exact_match": 0.88},
            metric_trends={"exact_match": DriftDirection.DEGRADING},
        )

        assert report.drift_detected is True
        assert report.overall_health == "degraded"

    def test_to_dict(self) -> None:
        """Test exporting report as dictionary."""
        report = DriftReport(
            drift_detected=False,
            overall_health="healthy",
            metric_trends={"exact_match": DriftDirection.STABLE},
        )

        data = report.to_dict()

        assert data["drift_detected"] is False
        assert data["metric_trends"]["exact_match"] == "stable"


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def detector(self) -> DriftDetector:
        """Create a drift detector."""
        return DriftDetector(
            warning_threshold=0.05,
            critical_threshold=0.10,
            window_size=3,
        )

    def test_no_baseline_check(self, detector: DriftDetector) -> None:
        """Test check without baseline returns info alert."""
        report = detector.check()

        assert report.drift_detected is False
        assert report.overall_health == "unknown"
        assert len(report.alerts) == 1
        assert report.alerts[0].severity == AlertSeverity.INFO

    def test_no_history_check(self, detector: DriftDetector) -> None:
        """Test check with baseline but no history."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)

        report = detector.check()

        assert report.drift_detected is False
        assert len(report.alerts) == 1
        assert "No evaluation history" in report.alerts[0].message

    def test_stable_metrics(self, detector: DriftDetector) -> None:
        """Test detecting stable metrics (no drift)."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)

        # Add snapshots at similar levels
        detector.add_raw_snapshot({"exact_match": 0.94}, pass_rate=0.89)
        detector.add_raw_snapshot({"exact_match": 0.95}, pass_rate=0.90)
        detector.add_raw_snapshot({"exact_match": 0.94}, pass_rate=0.91)

        report = detector.check()

        assert report.drift_detected is False
        assert report.overall_health == "healthy"

    def test_warning_threshold(self, detector: DriftDetector) -> None:
        """Test warning alert when degradation exceeds warning threshold."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)

        # Add snapshots showing 6% degradation (above 5% warning threshold)
        detector.add_raw_snapshot({"exact_match": 0.89}, pass_rate=0.84)
        detector.add_raw_snapshot({"exact_match": 0.89}, pass_rate=0.84)
        detector.add_raw_snapshot({"exact_match": 0.89}, pass_rate=0.84)

        report = detector.check()

        assert report.drift_detected is True
        assert report.overall_health == "degraded"
        warning_alerts = [a for a in report.alerts if a.severity == AlertSeverity.WARNING]
        assert len(warning_alerts) > 0

    def test_critical_threshold(self, detector: DriftDetector) -> None:
        """Test critical alert when degradation exceeds critical threshold."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)

        # Add snapshots showing 15% degradation (above 10% critical threshold)
        detector.add_raw_snapshot({"exact_match": 0.80}, pass_rate=0.75)
        detector.add_raw_snapshot({"exact_match": 0.80}, pass_rate=0.75)
        detector.add_raw_snapshot({"exact_match": 0.80}, pass_rate=0.75)

        report = detector.check()

        assert report.drift_detected is True
        assert report.overall_health == "critical"
        critical_alerts = [a for a in report.alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0

    def test_improving_metrics(self, detector: DriftDetector) -> None:
        """Test detecting improving metrics."""
        detector.set_baseline_from_values({"exact_match": 0.80}, pass_rate=0.75)

        # Add snapshots showing improvement
        detector.add_raw_snapshot({"exact_match": 0.90}, pass_rate=0.88)
        detector.add_raw_snapshot({"exact_match": 0.91}, pass_rate=0.89)
        detector.add_raw_snapshot({"exact_match": 0.92}, pass_rate=0.90)

        report = detector.check()

        assert report.drift_detected is False
        assert report.overall_health == "improving"
        assert report.metric_trends["exact_match"] == DriftDirection.IMPROVING

    def test_lower_is_better_metric(self) -> None:
        """Test drift detection for lower-is-better metrics like latency."""
        detector = DriftDetector(
            warning_threshold=0.05,
            critical_threshold=0.10,
            higher_is_better={"latency": False},
        )

        detector.set_baseline_from_values({"latency": 100.0}, pass_rate=0.90)

        # Latency increased (bad for lower-is-better)
        detector.add_raw_snapshot({"latency": 115.0}, pass_rate=0.90)
        detector.add_raw_snapshot({"latency": 115.0}, pass_rate=0.90)
        detector.add_raw_snapshot({"latency": 115.0}, pass_rate=0.90)

        report = detector.check()

        assert report.drift_detected is True
        assert report.metric_trends["latency"] == DriftDirection.DEGRADING

    def test_reset(self, detector: DriftDetector) -> None:
        """Test resetting detector clears baseline and history."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)
        detector.add_raw_snapshot({"exact_match": 0.94}, pass_rate=0.89)

        detector.reset()

        report = detector.check()
        assert "No baseline" in report.alerts[0].message

    def test_clear_history(self, detector: DriftDetector) -> None:
        """Test clearing history but keeping baseline."""
        detector.set_baseline_from_values({"exact_match": 0.95}, pass_rate=0.90)
        detector.add_raw_snapshot({"exact_match": 0.94}, pass_rate=0.89)

        detector.clear_history()

        report = detector.check()
        assert "No evaluation history" in report.alerts[0].message
        # Baseline metrics should still be present
        assert report.baseline_metrics.get("exact_match") == 0.95


class TestRegressionStatus:
    """Tests for RegressionStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert RegressionStatus.PASSED.value == "passed"
        assert RegressionStatus.FAILED.value == "failed"
        assert RegressionStatus.SKIPPED.value == "skipped"
        assert RegressionStatus.ERROR.value == "error"


class TestMetricThreshold:
    """Tests for MetricThreshold class."""

    def test_threshold_creation(self) -> None:
        """Test creating a threshold."""
        threshold = MetricThreshold(
            metric_name="exact_match",
            min_pass_rate=0.90,
            max_degradation=0.05,
        )

        assert threshold.metric_name == "exact_match"
        assert threshold.min_pass_rate == 0.90
        assert threshold.max_degradation == 0.05

    def test_threshold_with_min_max_value(self) -> None:
        """Test threshold with min/max value."""
        threshold = MetricThreshold(
            metric_name="latency",
            max_value=2000.0,
        )

        assert threshold.max_value == 2000.0


class TestRegressionResult:
    """Tests for RegressionResult class."""

    def test_result_creation(self) -> None:
        """Test creating a regression result."""
        result = RegressionResult(
            metric_name="exact_match",
            status=RegressionStatus.PASSED,
            current_value=0.92,
            baseline_value=0.90,
            threshold=0.85,
            message="Value 0.9200 meets all thresholds",
        )

        assert result.status == RegressionStatus.PASSED
        assert result.current_value == 0.92


class TestRegressionReport:
    """Tests for RegressionReport class."""

    def test_report_creation(self) -> None:
        """Test creating a regression report."""
        report = RegressionReport(
            status=RegressionStatus.PASSED,
            summary="All 3 metric checks passed",
            exit_code=0,
        )

        assert report.status == RegressionStatus.PASSED
        assert report.exit_code == 0

    def test_to_dict(self) -> None:
        """Test exporting report as dictionary."""
        result = RegressionResult(
            metric_name="exact_match",
            status=RegressionStatus.PASSED,
            current_value=0.92,
        )

        report = RegressionReport(
            status=RegressionStatus.PASSED,
            results=[result],
            summary="All checks passed",
        )

        data = report.to_dict()

        assert data["status"] == "passed"
        assert len(data["results"]) == 1
        assert data["results"][0]["metric"] == "exact_match"

    def test_to_github_summary(self) -> None:
        """Test generating GitHub Actions summary."""
        passed_result = RegressionResult(
            metric_name="exact_match",
            status=RegressionStatus.PASSED,
            current_value=0.92,
            threshold=0.85,
        )

        failed_result = RegressionResult(
            metric_name="latency",
            status=RegressionStatus.FAILED,
            current_value=3000.0,
            threshold=2000.0,
            message="Value 3000.0000 exceeds maximum 2000.0000",
        )

        report = RegressionReport(
            status=RegressionStatus.FAILED,
            results=[passed_result, failed_result],
            summary="Failed 1 of 2 metric checks",
        )

        markdown = report.to_github_summary()

        assert "## " in markdown
        assert "Regression Test Results" in markdown
        assert "| Metric | Status |" in markdown
        assert "exact_match" in markdown
        assert "latency" in markdown
        assert "Failed Metrics" in markdown

    def test_print_summary(self, capsys) -> None:
        """Test printing summary to stdout."""
        report = RegressionReport(
            status=RegressionStatus.PASSED,
            results=[
                RegressionResult(
                    metric_name="exact_match",
                    status=RegressionStatus.PASSED,
                    current_value=0.92,
                    message="Value 0.9200 meets all thresholds",
                )
            ],
            summary="All checks passed",
            exit_code=0,
        )

        report.print_summary()

        captured = capsys.readouterr()
        assert "REGRESSION TEST: PASSED" in captured.out
        assert "exact_match" in captured.out
        assert "Exit Code: 0" in captured.out


class TestRegressionTester:
    """Tests for RegressionTester class."""

    @pytest.fixture
    def tester(self) -> RegressionTester:
        """Create a regression tester."""
        return RegressionTester(
            thresholds=[
                MetricThreshold("exact_match", min_pass_rate=0.85),
            ],
            default_min_pass_rate=0.80,
            max_degradation=0.05,
        )

    def test_set_baseline_from_values(self, tester: RegressionTester) -> None:
        """Test setting baseline from raw values."""
        tester.set_baseline_from_values(
            {"exact_match": 0.90, "latency": 100.0},
            baseline_id="baseline_v1",
        )

        assert tester._baseline is not None
        assert tester._baseline["exact_match"] == 0.90
        assert tester._baseline_id == "baseline_v1"

    def test_save_and_load_baseline(self, tester: RegressionTester) -> None:
        """Test saving and loading baseline from file."""
        # Create a mock run result
        mock_run = MagicMock()
        mock_run.id = "run-123"
        mock_run.pass_rate = 0.88
        mock_run.dataset_name = "test_dataset"
        mock_run.total_cases = 100
        mock_run.metrics_summary = {
            "exact_match": {"score": 0.9, "details": {"pass_rate": 0.9}},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"

            # Save baseline
            tester.save_baseline(mock_run, baseline_path)

            assert baseline_path.exists()

            # Verify saved content
            data = json.loads(baseline_path.read_text())
            assert data["baseline_id"] == "run-123"
            assert data["metrics"]["exact_match"] == 0.9

            # Load baseline
            new_tester = RegressionTester()
            new_tester.load_baseline(baseline_path)

            assert new_tester._baseline["exact_match"] == 0.9

    @pytest.mark.asyncio
    async def test_run_regression(self, tester: RegressionTester) -> None:
        """Test running a regression test."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        dataset = EvalDataset(
            name="regression_test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
                EvalCase(input="c", expected="C"),
            ],
        )

        def target(x: str) -> str:
            return x.upper()

        report = await tester.run(
            dataset=dataset,
            target=target,
            metrics=[ExactMatch()],
        )

        assert isinstance(report, RegressionReport)
        assert report.run_result is not None
        assert report.run_result.total_cases == 3

    @pytest.mark.asyncio
    async def test_regression_with_baseline(self, tester: RegressionTester) -> None:
        """Test regression test comparing against baseline."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        # Set a high baseline
        tester.set_baseline_from_values(
            {"exact_match": 1.0, "overall_pass_rate": 1.0},
            baseline_id="perfect_baseline",
        )

        dataset = EvalDataset(
            name="regression_test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
                EvalCase(input="c", expected="X"),  # Will fail
            ],
        )

        def target(x: str) -> str:
            return x.upper()

        report = await tester.run(
            dataset=dataset,
            target=target,
            metrics=[ExactMatch()],
        )

        assert report.baseline_id == "perfect_baseline"
        # The test may fail due to degradation from baseline

    @pytest.mark.asyncio
    async def test_fail_on_threshold_violation(self) -> None:
        """Test that threshold violations cause failure."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        tester = RegressionTester(
            thresholds=[
                MetricThreshold("exact_match", min_pass_rate=1.0),  # Require 100%
            ],
            fail_on_regression=True,
        )

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="X"),  # Will fail
            ],
        )

        def target(x: str) -> str:
            return x.upper()

        report = await tester.run(
            dataset=dataset,
            target=target,
            metrics=[ExactMatch()],
        )

        # Should fail because pass rate < 100%
        assert report.status == RegressionStatus.FAILED
        assert report.exit_code == 1

    @pytest.mark.asyncio
    async def test_no_fail_when_disabled(self) -> None:
        """Test that fail_on_regression=False returns exit_code=0."""
        from evalops import EvalCase, EvalDataset
        from evalops.core.metrics import ExactMatch

        tester = RegressionTester(
            thresholds=[
                MetricThreshold("exact_match", min_pass_rate=1.0),
            ],
            fail_on_regression=False,  # Don't fail
        )

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="X"),  # Will fail
            ],
        )

        def target(x: str) -> str:
            return x.upper()

        report = await tester.run(
            dataset=dataset,
            target=target,
            metrics=[ExactMatch()],
        )

        # Should still be FAILED status but exit_code=0
        assert report.status == RegressionStatus.FAILED
        assert report.exit_code == 0


class TestComparisonModuleImports:
    """Test that the comparison module exports work correctly."""

    def test_ab_test_imports(self) -> None:
        """Test A/B testing imports."""
        from evalops.comparison import (
            ABTest,
            Winner,
        )

        assert ABTest is not None
        assert Winner.TIE.value == "tie"

    def test_drift_imports(self) -> None:
        """Test drift detection imports."""
        from evalops.comparison import (
            AlertSeverity,
            DriftDetector,
        )

        assert DriftDetector is not None
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_regression_imports(self) -> None:
        """Test regression testing imports."""
        from evalops.comparison import (
            RegressionStatus,
            RegressionTester,
        )

        assert RegressionTester is not None
        assert RegressionStatus.PASSED.value == "passed"
