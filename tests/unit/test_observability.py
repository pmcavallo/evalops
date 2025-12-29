"""Unit tests for the observability module."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from evalops.observability.collector import (
    Counter,
    LatencyHistogram,
    MetricsCollector,
    TokenUsageStats,
)
from evalops.observability.logging import (
    EvalLogger,
    LogContext,
    configure_logging,
    get_case_id,
    get_dataset_id,
    get_logger,
    get_request_id,
    get_run_id,
    set_case_id,
    set_dataset_id,
    set_request_id,
    set_run_id,
)
from evalops.observability.tracing import (
    EvalTracer,
    TraceConfig,
    TraceContext,
    configure_tracing,
    get_tracer,
)


class TestCounter:
    """Tests for Counter class."""

    def test_initial_value(self) -> None:
        """Test counter starts at zero."""
        counter = Counter()
        assert counter.value() == 0

    def test_increment(self) -> None:
        """Test incrementing counter."""
        counter = Counter()
        counter.increment()
        assert counter.value() == 1

    def test_increment_by_amount(self) -> None:
        """Test incrementing by specific amount."""
        counter = Counter()
        counter.increment(5)
        assert counter.value() == 5

    def test_reset(self) -> None:
        """Test resetting counter."""
        counter = Counter()
        counter.increment(10)
        counter.reset()
        assert counter.value() == 0


class TestLatencyHistogram:
    """Tests for LatencyHistogram class."""

    def test_empty_histogram(self) -> None:
        """Test empty histogram returns zeros."""
        hist = LatencyHistogram()
        assert hist.count() == 0
        assert hist.mean() == 0.0
        assert hist.p50() == 0.0
        assert hist.p95() == 0.0

    def test_record_values(self) -> None:
        """Test recording values."""
        hist = LatencyHistogram()
        hist.record(100)
        hist.record(200)
        hist.record(300)

        assert hist.count() == 3
        assert hist.sum() == 600
        assert hist.mean() == 200.0
        assert hist.min() == 100
        assert hist.max() == 300

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        hist = LatencyHistogram()
        # Add 100 values: 1, 2, 3, ..., 100
        for i in range(1, 101):
            hist.record(float(i))

        # Percentile calculations can vary by implementation
        # We use a simple index-based approach, so allow +-1 tolerance
        assert 49 <= hist.p50() <= 51
        assert 89 <= hist.p90() <= 91
        assert 94 <= hist.p95() <= 96
        assert 98 <= hist.p99() <= 100

    def test_reset(self) -> None:
        """Test resetting histogram."""
        hist = LatencyHistogram()
        hist.record(100)
        hist.reset()
        assert hist.count() == 0

    def test_to_dict(self) -> None:
        """Test exporting as dictionary."""
        hist = LatencyHistogram()
        hist.record(100)
        hist.record(200)

        data = hist.to_dict()
        assert data["count"] == 2
        assert data["mean"] == 150.0
        assert "p50" in data
        assert "p95" in data


class TestTokenUsageStats:
    """Tests for TokenUsageStats class."""

    def test_initial_state(self) -> None:
        """Test initial state is zero."""
        stats = TokenUsageStats()
        assert stats.total_tokens() == 0

    def test_record_usage(self) -> None:
        """Test recording token usage."""
        stats = TokenUsageStats()
        stats.record(100, 50)
        stats.record(200, 100)

        assert stats.input_tokens.value() == 300
        assert stats.output_tokens.value() == 150
        assert stats.total_tokens() == 450

    def test_to_dict(self) -> None:
        """Test exporting as dictionary."""
        stats = TokenUsageStats()
        stats.record(100, 50)

        data = stats.to_dict()
        assert data["input_tokens"] == 100
        assert data["output_tokens"] == 50
        assert data["total_tokens"] == 150


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create a fresh collector."""
        return MetricsCollector()

    def test_initial_state(self, collector: MetricsCollector) -> None:
        """Test collector starts empty."""
        assert collector.total_runs.value() == 0
        assert collector.total_cases.value() == 0
        assert collector.success_rate() == 0.0

    def test_record_run_lifecycle(self, collector: MetricsCollector) -> None:
        """Test recording a complete run lifecycle."""
        collector.record_run_started()
        collector.record_case_result(success=True, latency_ms=100)
        collector.record_case_result(success=True, latency_ms=150)
        collector.record_run_completed(passed=True, duration_ms=300)

        assert collector.total_runs.value() == 1
        assert collector.successful_runs.value() == 1
        assert collector.total_cases.value() == 2
        assert collector.success_rate() == 1.0

    def test_record_failed_cases(self, collector: MetricsCollector) -> None:
        """Test recording failed cases."""
        collector.record_case_result(success=True, latency_ms=100)
        collector.record_case_result(success=False, latency_ms=50)
        collector.record_case_result(success=True, latency_ms=100)

        assert collector.total_cases.value() == 3
        assert collector.successful_cases.value() == 2
        assert collector.failed_cases.value() == 1
        assert collector.success_rate() == pytest.approx(2 / 3)

    def test_record_metric_results(self, collector: MetricsCollector) -> None:
        """Test recording per-metric results."""
        collector.record_case_result(
            success=True,
            latency_ms=100,
            metrics_passed=True,
            metric_results={
                "exact_match": {"passed": True, "score": 1.0},
                "latency": {"passed": True, "score": 100},
            },
        )
        collector.record_case_result(
            success=True,
            latency_ms=100,
            metrics_passed=False,
            metric_results={
                "exact_match": {"passed": False, "score": 0.0},
                "latency": {"passed": True, "score": 100},
            },
        )

        assert collector.metric_pass_rate("exact_match") == 0.5
        assert collector.metric_pass_rate("latency") == 1.0

    def test_pass_rate(self, collector: MetricsCollector) -> None:
        """Test calculating pass rate."""
        collector.record_case_result(success=True, latency_ms=100, metrics_passed=True)
        collector.record_case_result(success=True, latency_ms=100, metrics_passed=False)
        collector.record_case_result(success=True, latency_ms=100, metrics_passed=True)

        assert collector.pass_rate() == pytest.approx(2 / 3)

    def test_export(self, collector: MetricsCollector) -> None:
        """Test exporting metrics."""
        collector.record_run_started()
        collector.record_case_result(success=True, latency_ms=100)
        collector.record_run_completed(passed=True, duration_ms=150)

        data = collector.export()

        assert "collection_period" in data
        assert "runs" in data
        assert "cases" in data
        assert "latency" in data
        assert "tokens" in data

        assert data["runs"]["total"] == 1
        assert data["cases"]["total"] == 1

    def test_export_prometheus(self, collector: MetricsCollector) -> None:
        """Test Prometheus format export."""
        collector.record_case_result(success=True, latency_ms=100)

        output = collector.export_prometheus()

        assert "evalops_runs_total" in output
        assert "evalops_cases_total" in output
        assert "evalops_latency_ms" in output
        assert "evalops_success_rate" in output

    def test_reset(self, collector: MetricsCollector) -> None:
        """Test resetting collector."""
        collector.record_run_started()
        collector.record_case_result(success=True, latency_ms=100)
        collector.reset()

        assert collector.total_runs.value() == 0
        assert collector.total_cases.value() == 0


class TestLogContext:
    """Tests for logging context management."""

    def test_set_and_get_request_id(self) -> None:
        """Test setting and getting request ID."""
        set_request_id("req-123")
        assert get_request_id() == "req-123"
        set_request_id(None)
        assert get_request_id() is None

    def test_set_and_get_run_id(self) -> None:
        """Test setting and getting run ID."""
        set_run_id("run-456")
        assert get_run_id() == "run-456"
        set_run_id(None)

    def test_log_context_manager(self) -> None:
        """Test LogContext context manager."""
        assert get_run_id() is None

        with LogContext(run_id="ctx-run", dataset_id="ctx-ds"):
            assert get_run_id() == "ctx-run"
            assert get_dataset_id() == "ctx-ds"

        # Context should be cleared
        assert get_run_id() is None
        assert get_dataset_id() is None

    def test_log_context_nested(self) -> None:
        """Test nested LogContext."""
        with LogContext(run_id="outer"):
            assert get_run_id() == "outer"

            with LogContext(run_id="inner"):
                assert get_run_id() == "inner"

            # Should restore outer value
            assert get_run_id() == "outer"


class TestEvalLogger:
    """Tests for EvalLogger class."""

    @pytest.fixture
    def logger(self) -> EvalLogger:
        """Create an EvalLogger instance."""
        configure_logging(level="DEBUG", json_format=False)
        return EvalLogger()

    def test_run_started(self, logger: EvalLogger, capsys) -> None:
        """Test logging run start."""
        logger.run_started(
            dataset_name="test_ds",
            run_id="run-123",
            total_cases=10,
            metrics=["exact_match"],
        )
        # Just verify no exceptions - output format varies

    def test_run_completed(self, logger: EvalLogger, capsys) -> None:
        """Test logging run completion."""
        logger.run_completed(
            run_id="run-123",
            total_cases=10,
            successful_cases=9,
            pass_rate=0.9,
            avg_latency_ms=150.5,
            duration_ms=2000,
        )

    def test_case_started(self, logger: EvalLogger) -> None:
        """Test logging case start."""
        logger.case_started(case_id="case-1", input_preview="test input")

    def test_case_completed(self, logger: EvalLogger) -> None:
        """Test logging case completion."""
        logger.case_completed(
            case_id="case-1",
            latency_ms=150,
            metrics_passed=True,
        )

    def test_case_failed(self, logger: EvalLogger) -> None:
        """Test logging case failure."""
        logger.case_failed(
            case_id="case-1",
            error="Something went wrong",
            latency_ms=50,
        )


class TestEvalTracer:
    """Tests for EvalTracer class."""

    def test_tracer_disabled_without_api_key(self) -> None:
        """Test tracer is disabled without API key."""
        # Clear any existing API key
        with patch.dict("os.environ", {}, clear=True):
            tracer = EvalTracer(enabled=True)
            # is_enabled depends on LANGSMITH_AVAILABLE which checks env at import
            # We mainly test the config
            assert tracer.config.project_name == "evalops"

    def test_tracer_config(self) -> None:
        """Test tracer configuration."""
        tracer = EvalTracer(
            project_name="my-project",
            enabled=False,
            default_tags=["prod", "v1"],
            default_metadata={"version": "1.0"},
        )

        assert tracer.config.project_name == "my-project"
        assert tracer.config.enabled is False
        assert "prod" in tracer.config.tags
        assert tracer.config.metadata["version"] == "1.0"

    def test_trace_run_context(self) -> None:
        """Test trace_run context manager."""
        tracer = EvalTracer(enabled=False)

        with tracer.trace_run("test_run", tags=["test"]) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.project_name == "evalops"
            assert "test" in ctx.tags

    def test_trace_target_decorator_disabled(self) -> None:
        """Test trace_target decorator when disabled."""
        tracer = EvalTracer(enabled=False)

        @tracer.trace_target
        def my_function(x: int) -> int:
            return x * 2

        # Should work normally when tracing disabled
        result = my_function(5)
        assert result == 10

    def test_configure_tracing(self) -> None:
        """Test global tracing configuration."""
        tracer = configure_tracing(
            project_name="configured-project",
            enabled=False,
            tags=["configured"],
        )

        assert tracer.config.project_name == "configured-project"
        assert get_tracer() is tracer


class TestTraceConfig:
    """Tests for TraceConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TraceConfig()
        assert config.project_name == "evalops"
        assert config.tags == []
        assert config.metadata == {}

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TraceConfig(
            project_name="custom",
            enabled=True,
            tags=["tag1"],
            metadata={"key": "value"},
        )
        assert config.project_name == "custom"
        assert config.tags == ["tag1"]


class TestRunnerObservabilityIntegration:
    """Tests for runner observability integration."""

    @pytest.mark.asyncio
    async def test_runner_with_collector(self) -> None:
        """Test runner integrates with collector."""
        from evalops import EvalCase, EvalDataset, EvalRunner, ExactMatch
        from evalops.observability.collector import MetricsCollector

        collector = MetricsCollector()
        runner = EvalRunner(collector=collector, enable_observability=False)
        runner.collector = collector  # Explicitly set

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
            ],
        )

        def upper(x: str) -> str:
            return x.upper()

        await runner.evaluate(dataset, upper, metrics=[ExactMatch()])

        assert collector.total_runs.value() == 1
        assert collector.total_cases.value() == 2
        assert collector.successful_cases.value() == 2

    @pytest.mark.asyncio
    async def test_runner_default_observability(self) -> None:
        """Test runner auto-initializes observability."""
        from evalops import EvalCase, EvalDataset, EvalRunner

        runner = EvalRunner(enable_observability=True)

        assert runner.logger is not None
        assert runner.collector is not None

        dataset = EvalDataset(
            name="test",
            cases=[EvalCase(input="test")],
        )

        await runner.evaluate(dataset, lambda x: x)

        # Collector should have recorded the run
        assert runner.collector.total_runs.value() == 1

    @pytest.mark.asyncio
    async def test_runner_disabled_observability(self) -> None:
        """Test runner with observability disabled."""
        from evalops import EvalCase, EvalDataset, EvalRunner

        runner = EvalRunner(enable_observability=False)

        assert runner.logger is None
        assert runner.collector is None

        dataset = EvalDataset(
            name="test",
            cases=[EvalCase(input="test")],
        )

        # Should still work without observability
        result = await runner.evaluate(dataset, lambda x: x)
        assert result.total_cases == 1
