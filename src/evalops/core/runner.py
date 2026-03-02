"""Evaluation runner for executing test cases against LLM systems.

This module provides the core execution engine that runs evaluation cases
against a target callable and collects results with full observability.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from evalops.core.dataset import EvalCase, EvalDataset

if TYPE_CHECKING:
    from evalops.core.metrics import Metric, MetricResult
    from evalops.observability.collector import MetricsCollector
    from evalops.observability.logging import EvalLogger
    from evalops.observability.tracing import EvalTracer
    from evalops.storage.repository import EvalRepository


# Type alias for target functions
# Can be sync: (str) -> str
# Or async: (str) -> Awaitable[str]
TargetFunc = Callable[[str], str | Awaitable[str]]


class EvalResult(BaseModel):
    """Result of evaluating a single test case.

    Attributes:
        id: Unique identifier for this result.
        case_id: ID of the EvalCase that was evaluated.
        input: The input that was sent to the target.
        output: The output received from the target.
        expected: The expected output (if defined).
        latency_ms: Time taken in milliseconds.
        error: Error message if execution failed.
        timestamp: When the evaluation was run.
        metadata: Arbitrary metadata from the case.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    case_id: str
    input: str
    output: str | None = None
    expected: str | None = None
    expected_keywords: list[str] | None = None
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    metric_results: dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the evaluation executed without error."""
        return self.error is None

    @property
    def metrics_passed(self) -> bool:
        """Check if all metrics passed."""
        if not self.metric_results:
            return True
        return all(
            m.get("passed", True) for m in self.metric_results.values()
        )


class EvalRunResult(BaseModel):
    """Result of running a full evaluation.

    Attributes:
        id: Unique identifier for this run.
        dataset_id: ID of the dataset that was evaluated.
        dataset_name: Name of the dataset.
        results: List of individual case results.
        total_cases: Total number of cases evaluated.
        successful_cases: Number of cases that executed without error.
        total_latency_ms: Total time for all evaluations.
        started_at: When the run started.
        completed_at: When the run completed.
        metrics_summary: Aggregated metrics across all results.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_id: str
    dataset_name: str
    results: list[EvalResult] = Field(default_factory=list)
    total_cases: int = 0
    successful_cases: int = 0
    total_latency_ms: float = 0.0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    metrics_summary: dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate the execution success rate."""
        if self.total_cases == 0:
            return 0.0
        return self.successful_cases / self.total_cases

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency across all cases."""
        if self.successful_cases == 0:
            return 0.0
        return self.total_latency_ms / self.successful_cases

    @property
    def all_metrics_passed(self) -> bool:
        """Check if all metrics passed in the summary."""
        if not self.metrics_summary:
            return True
        return all(
            m.get("passed", True) for m in self.metrics_summary.values()
        )

    @property
    def pass_rate(self) -> float:
        """Calculate the rate of results where all metrics passed."""
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.metrics_passed)
        return passed / len(self.results)


class EvalRunner:
    """Executes evaluation cases against a target LLM system.

    The runner supports both synchronous and asynchronous target functions,
    handles errors gracefully, collects timing information, and provides
    full observability through tracing, logging, and metrics collection.

    Example:
        ```python
        from evalops import EvalRunner, EvalDataset
        from evalops.core.metrics import ExactMatch, Latency

        runner = EvalRunner()

        async def my_llm(input: str) -> str:
            return await call_claude(input)

        result = await runner.evaluate(
            dataset=dataset,
            target=my_llm,
            metrics=[ExactMatch(), Latency(p95_target_ms=2000)],
        )
        print(f"Pass rate: {result.pass_rate:.1%}")
        ```

    With observability:
        ```python
        from evalops.observability import (
            EvalTracer, EvalLogger, MetricsCollector
        )

        runner = EvalRunner(
            tracer=EvalTracer(project_name="my-evals"),
            logger=EvalLogger(),
            collector=MetricsCollector(),
        )

        result = await runner.evaluate(dataset, target, metrics)
        print(runner.collector.export())  # Runtime metrics
        ```
    """

    def __init__(
        self,
        tracer: EvalTracer | None = None,
        logger: EvalLogger | None = None,
        collector: MetricsCollector | None = None,
        repository: EvalRepository | None = None,
        enable_observability: bool = True,
        auto_save: bool = False,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            tracer: LangSmith tracer for distributed tracing.
            logger: Structured logger for evaluation events.
            collector: Metrics collector for runtime statistics.
            repository: Repository for persisting run results.
            enable_observability: If True, auto-initialize observability
                components when not provided.
            auto_save: If True and repository is provided, automatically
                save run results after each evaluation.
        """
        self.tracer = tracer
        self.logger = logger
        self.collector = collector
        self.repository = repository
        self.auto_save = auto_save

        # Auto-initialize observability if enabled and not provided
        if enable_observability:
            if self.logger is None:
                from evalops.observability.logging import EvalLogger
                self.logger = EvalLogger()

            if self.collector is None:
                from evalops.observability.collector import MetricsCollector
                self.collector = MetricsCollector()

    async def run_case(
        self,
        case: EvalCase,
        target: TargetFunc,
        metrics: list[Metric] | None = None,
    ) -> EvalResult:
        """Run a single evaluation case against the target.

        Args:
            case: The evaluation case to run.
            target: The target function to evaluate.
            metrics: Optional list of metrics to compute.

        Returns:
            EvalResult with the output, timing, and metric results.
        """
        start_time = time.perf_counter()
        output: str | None = None
        error: str | None = None

        try:
            result = target(case.input)
            # Handle both sync and async targets
            if asyncio.iscoroutine(result):
                output = await result
            else:
                output = result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        eval_result = EvalResult(
            case_id=case.id,
            input=case.input,
            output=output,
            expected=case.expected,
            expected_keywords=case.expected_keywords,
            latency_ms=elapsed_ms,
            error=error,
            metadata=case.metadata,
        )

        # Apply metrics if provided
        if metrics and eval_result.success:
            for metric in metrics:
                metric_result = metric.compute(eval_result)
                eval_result.metric_results[metric.name] = {
                    "score": metric_result.score,
                    "passed": metric_result.passed,
                    "details": metric_result.details,
                }

        return eval_result

    async def run_dataset(
        self,
        dataset: EvalDataset,
        target: TargetFunc,
        metrics: list[Metric] | None = None,
        run_name: str | None = None,
    ) -> EvalRunResult:
        """Run all cases in a dataset against the target.

        Args:
            dataset: The dataset containing cases to evaluate.
            target: The target function to evaluate.
            metrics: Optional list of metrics to compute for each result.
            run_name: Optional name for this run (for tracing/logging).

        Returns:
            EvalRunResult with all individual results and summary stats.
        """
        from evalops.observability.logging import set_dataset_id, set_run_id

        run_id = str(uuid4())
        run_name = run_name or f"eval_{dataset.name}"
        started_at = datetime.now(timezone.utc)
        results: list[EvalResult] = []
        metric_names = [m.name for m in metrics] if metrics else []

        # Set up log context
        set_run_id(run_id)
        set_dataset_id(dataset.id)

        # Log and record run start
        if self.logger:
            self.logger.run_started(
                dataset_name=dataset.name,
                run_id=run_id,
                total_cases=len(dataset),
                metrics=metric_names,
            )

        if self.collector:
            self.collector.record_run_started()

        # Execute cases (optionally with tracing)
        trace_context = None
        if self.tracer and self.tracer.is_enabled:
            trace_context = self.tracer.trace_run(
                run_name,
                tags=["eval_run"],
                metadata={"dataset_id": dataset.id, "dataset_name": dataset.name},
            )
            trace_context.__enter__()

        try:
            for case in dataset:
                # Log case start
                if self.logger:
                    self.logger.case_started(
                        case_id=case.id,
                        input_preview=case.input[:100] if case.input else None,
                    )

                result = await self.run_case(case, target, metrics)
                results.append(result)

                # Log case completion
                if self.logger:
                    if result.success:
                        self.logger.case_completed(
                            case_id=case.id,
                            latency_ms=result.latency_ms,
                            metrics_passed=result.metrics_passed,
                            metric_results=result.metric_results,
                        )
                    else:
                        self.logger.case_failed(
                            case_id=case.id,
                            error=result.error or "Unknown error",
                            latency_ms=result.latency_ms,
                        )

                # Record metrics
                if self.collector:
                    self.collector.record_case_result(
                        success=result.success,
                        latency_ms=result.latency_ms,
                        metrics_passed=result.metrics_passed,
                        metric_results=result.metric_results,
                    )

        finally:
            if trace_context:
                trace_context.__exit__(None, None, None)

        completed_at = datetime.now(timezone.utc)
        successful = [r for r in results if r.success]
        duration_ms = (completed_at - started_at).total_seconds() * 1000

        # Aggregate metrics if provided
        metrics_summary: dict[str, Any] = {}
        if metrics and successful:
            from evalops.core.metrics import MetricResult

            for metric in metrics:
                # Collect individual metric results for this metric
                individual_results: list[MetricResult] = []
                for r in successful:
                    if metric.name in r.metric_results:
                        mr = r.metric_results[metric.name]
                        individual_results.append(
                            MetricResult(
                                name=metric.name,
                                score=mr["score"],
                                passed=mr["passed"],
                                details=mr["details"],
                            )
                        )

                if individual_results:
                    aggregated = metric.aggregate(individual_results)
                    metrics_summary[metric.name] = {
                        "score": aggregated.score,
                        "passed": aggregated.passed,
                        "details": aggregated.details,
                    }

        run_result = EvalRunResult(
            id=run_id,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            results=results,
            total_cases=len(results),
            successful_cases=len(successful),
            total_latency_ms=sum(r.latency_ms for r in successful),
            started_at=started_at,
            completed_at=completed_at,
            metrics_summary=metrics_summary,
        )

        # Log and record run completion
        if self.logger:
            self.logger.run_completed(
                run_id=run_id,
                total_cases=run_result.total_cases,
                successful_cases=run_result.successful_cases,
                pass_rate=run_result.pass_rate,
                avg_latency_ms=run_result.avg_latency_ms,
                duration_ms=duration_ms,
                metrics_summary=metrics_summary,
            )

        if self.collector:
            self.collector.record_run_completed(
                passed=run_result.all_metrics_passed,
                duration_ms=duration_ms,
            )

        # Trace complete run result
        if self.tracer:
            self.tracer.trace_eval_run_result(run_result, run_name)

        # Clear log context
        set_run_id(None)
        set_dataset_id(None)

        return run_result

    async def evaluate(
        self,
        dataset: EvalDataset,
        target: TargetFunc,
        metrics: list[Metric] | None = None,
        run_name: str | None = None,
        save: bool | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalRunResult:
        """Evaluate a dataset against a target with metrics.

        This is the primary API for running evaluations. It's an alias
        for run_dataset with a more intuitive name.

        Args:
            dataset: The dataset containing cases to evaluate.
            target: The target function to evaluate.
            metrics: List of metrics to compute.
            run_name: Optional name for this run (for tracing/logging).
            save: Whether to save results to repository. If None, uses
                the runner's auto_save setting.
            tags: Optional tags for the saved run.
            metadata: Optional metadata for the saved run.

        Returns:
            EvalRunResult with results and aggregated metrics.

        Example:
            ```python
            result = await runner.evaluate(
                dataset=my_dataset,
                target=my_llm,
                metrics=[ExactMatch(), Latency(p95_target_ms=2000)],
            )
            print(f"Pass rate: {result.pass_rate:.1%}")
            for name, summary in result.metrics_summary.items():
                print(f"{name}: {summary['score']:.2f}")
            ```

        With persistence:
            ```python
            runner = EvalRunner(
                repository=EvalRepository("sqlite:///evalops.db"),
                auto_save=True,
            )
            result = await runner.evaluate(
                dataset=my_dataset,
                target=my_llm,
                metrics=[ExactMatch()],
                tags=["production", "nightly"],
            )
            # Result is automatically saved to database
            ```
        """
        run_result = await self.run_dataset(dataset, target, metrics, run_name)

        # Determine if we should save
        should_save = save if save is not None else self.auto_save

        if should_save and self.repository is not None:
            self.repository.save_run(
                run_result,
                name=run_name,
                tags=tags,
                metadata=metadata,
            )

        return run_result
