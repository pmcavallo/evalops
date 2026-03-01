"""Unit tests for the Runner module."""

import pytest

from evalops import (
    ContainsKeywords,
    EvalCase,
    EvalDataset,
    EvalResult,
    EvalRunner,
    ExactMatch,
    Latency,
)


class TestEvalRunner:
    """Tests for EvalRunner."""

    @pytest.fixture
    def runner(self) -> EvalRunner:
        """Create a runner instance."""
        return EvalRunner()

    @pytest.mark.asyncio
    async def test_run_case_sync_target(
        self, runner: EvalRunner, sample_case: EvalCase
    ) -> None:
        """Test running a case with a sync target function."""

        def echo(input: str) -> str:
            return f"Echo: {input}"

        result = await runner.run_case(sample_case, echo)

        assert result.success is True
        assert result.output == "Echo: What is 2 + 2?"
        assert result.input == "What is 2 + 2?"
        assert result.expected == "4"
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_run_case_async_target(
        self, runner: EvalRunner, sample_case: EvalCase
    ) -> None:
        """Test running a case with an async target function."""

        async def async_echo(input: str) -> str:
            return f"Async: {input}"

        result = await runner.run_case(sample_case, async_echo)

        assert result.success is True
        assert result.output == "Async: What is 2 + 2?"

    @pytest.mark.asyncio
    async def test_run_case_error_handling(
        self, runner: EvalRunner, sample_case: EvalCase
    ) -> None:
        """Test that errors in target are captured gracefully."""

        def failing_target(input: str) -> str:
            raise ValueError("Something went wrong")

        result = await runner.run_case(sample_case, failing_target)

        assert result.success is False
        assert result.error is not None
        assert "ValueError" in result.error
        assert "Something went wrong" in result.error
        assert result.output is None

    @pytest.mark.asyncio
    async def test_run_dataset(
        self, runner: EvalRunner, sample_dataset: EvalDataset
    ) -> None:
        """Test running a full dataset."""

        def echo(input: str) -> str:
            return input.upper()

        run_result = await runner.run_dataset(sample_dataset, echo)

        assert run_result.total_cases == 3
        assert run_result.successful_cases == 3
        assert run_result.success_rate == 1.0
        assert len(run_result.results) == 3
        assert run_result.avg_latency_ms > 0

    @pytest.mark.asyncio
    async def test_run_dataset_with_failures(self, runner: EvalRunner) -> None:
        """Test dataset run with some failures."""
        dataset = EvalDataset(
            name="mixed",
            cases=[
                EvalCase(input="good"),
                EvalCase(input="fail"),
                EvalCase(input="good2"),
            ],
        )

        def selective_fail(input: str) -> str:
            if "fail" in input:
                raise RuntimeError("Intentional failure")
            return input

        run_result = await runner.run_dataset(dataset, selective_fail)

        assert run_result.total_cases == 3
        assert run_result.successful_cases == 2
        assert run_result.success_rate == pytest.approx(2 / 3)


class TestEvalResult:
    """Tests for EvalResult."""

    def test_success_property(self) -> None:
        """Test the success property."""
        success_result = EvalResult(
            case_id="1", input="test", output="result"
        )
        assert success_result.success is True

        failed_result = EvalResult(
            case_id="2", input="test", error="failed"
        )
        assert failed_result.success is False

    def test_metrics_passed_no_metrics(self) -> None:
        """Test metrics_passed when no metrics applied."""
        result = EvalResult(case_id="1", input="test", output="result")
        assert result.metrics_passed is True

    def test_metrics_passed_all_pass(self) -> None:
        """Test metrics_passed when all metrics pass."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="result",
            metric_results={
                "exact_match": {"score": 1.0, "passed": True, "details": {}},
                "latency": {"score": 100, "passed": True, "details": {}},
            },
        )
        assert result.metrics_passed is True

    def test_metrics_passed_some_fail(self) -> None:
        """Test metrics_passed when some metrics fail."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="result",
            metric_results={
                "exact_match": {"score": 0.0, "passed": False, "details": {}},
                "latency": {"score": 100, "passed": True, "details": {}},
            },
        )
        assert result.metrics_passed is False


class TestRunnerWithMetrics:
    """Tests for runner with metrics integration."""

    @pytest.fixture
    def runner(self) -> EvalRunner:
        return EvalRunner()

    @pytest.mark.asyncio
    async def test_run_case_with_exact_match(self, runner: EvalRunner) -> None:
        """Test running a case with ExactMatch metric."""
        case = EvalCase(input="What is 2+2?", expected="4")

        def target(input: str) -> str:
            return "4"

        result = await runner.run_case(case, target, metrics=[ExactMatch()])

        assert result.success is True
        assert "exact_match" in result.metric_results
        assert result.metric_results["exact_match"]["passed"] is True
        assert result.metric_results["exact_match"]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_run_case_with_failing_metric(self, runner: EvalRunner) -> None:
        """Test running a case where metric fails."""
        case = EvalCase(input="What is 2+2?", expected="4")

        def target(input: str) -> str:
            return "5"  # Wrong answer

        result = await runner.run_case(case, target, metrics=[ExactMatch()])

        assert result.success is True  # Execution succeeded
        assert result.metric_results["exact_match"]["passed"] is False
        assert result.metric_results["exact_match"]["score"] == 0.0
        assert result.metrics_passed is False

    @pytest.mark.asyncio
    async def test_run_case_with_multiple_metrics(self, runner: EvalRunner) -> None:
        """Test running a case with multiple metrics."""
        case = EvalCase(
            input="Explain AI",
            expected="Artificial Intelligence",
            expected_keywords=["artificial", "intelligence"],
        )

        def target(input: str) -> str:
            return "AI stands for Artificial Intelligence"

        metrics = [ExactMatch(), ContainsKeywords(), Latency(max_ms=1000)]
        result = await runner.run_case(case, target, metrics=metrics)

        assert "exact_match" in result.metric_results
        assert "contains_keywords" in result.metric_results
        assert "latency" in result.metric_results

        # Keywords should pass
        assert result.metric_results["contains_keywords"]["passed"] is True
        # Exact match should fail (output != expected)
        assert result.metric_results["exact_match"]["passed"] is False
        # Latency should pass (fast operation)
        assert result.metric_results["latency"]["passed"] is True

    @pytest.mark.asyncio
    async def test_run_case_no_metrics_on_error(self, runner: EvalRunner) -> None:
        """Test that metrics are not applied when target errors."""
        case = EvalCase(input="test", expected="result")

        def failing_target(input: str) -> str:
            raise ValueError("Error")

        result = await runner.run_case(case, failing_target, metrics=[ExactMatch()])

        assert result.success is False
        assert result.metric_results == {}  # No metrics on error

    @pytest.mark.asyncio
    async def test_run_dataset_with_metrics(self, runner: EvalRunner) -> None:
        """Test running dataset with metrics and aggregation."""
        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="q1", expected="a1"),
                EvalCase(input="q2", expected="a2"),
                EvalCase(input="q3", expected="a3"),
            ],
        )

        def target(input: str) -> str:
            # Return correct answer for q1 and q2, wrong for q3
            if input == "q1":
                return "a1"
            elif input == "q2":
                return "a2"
            return "wrong"

        result = await runner.run_dataset(dataset, target, metrics=[ExactMatch()])

        assert result.total_cases == 3
        assert result.successful_cases == 3

        # Check metrics summary
        assert "exact_match" in result.metrics_summary
        summary = result.metrics_summary["exact_match"]
        assert summary["details"]["pass_rate"] == pytest.approx(2 / 3)
        assert summary["details"]["passed_count"] == 2

    @pytest.mark.asyncio
    async def test_evaluate_alias(self, runner: EvalRunner) -> None:
        """Test that evaluate() works as alias for run_dataset()."""
        dataset = EvalDataset(
            name="test",
            cases=[EvalCase(input="hi", expected="HI")],
        )

        def upper(input: str) -> str:
            return input.upper()

        result = await runner.evaluate(
            dataset=dataset,
            target=upper,
            metrics=[ExactMatch()],
        )

        assert result.total_cases == 1
        assert result.metrics_summary["exact_match"]["passed"] is True

    @pytest.mark.asyncio
    async def test_pass_rate_property(self, runner: EvalRunner) -> None:
        """Test pass_rate property calculation."""
        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
                EvalCase(input="c", expected="WRONG"),  # Will fail
            ],
        )

        def upper(input: str) -> str:
            return input.upper()

        result = await runner.evaluate(
            dataset=dataset,
            target=upper,
            metrics=[ExactMatch()],
        )

        # 2 out of 3 pass
        assert result.pass_rate == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_all_metrics_passed_property(self, runner: EvalRunner) -> None:
        """Test all_metrics_passed property."""
        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
            ],
        )

        def upper(input: str) -> str:
            return input.upper()

        result = await runner.evaluate(
            dataset=dataset,
            target=upper,
            metrics=[ExactMatch(), Latency(p95_target_ms=1000)],
        )

        # All should pass
        assert result.all_metrics_passed is True
