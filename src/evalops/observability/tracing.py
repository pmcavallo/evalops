"""LangSmith tracing integration for LLM observability.

This module provides integration with LangSmith for tracing LLM calls,
capturing inputs, outputs, latency, and token usage.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from evalops.core.runner import EvalResult, EvalRunResult

# Check if LangSmith is available and configured
LANGSMITH_AVAILABLE = False
try:
    from langsmith import Client, traceable
    from langsmith.run_trees import RunTree

    LANGSMITH_AVAILABLE = bool(os.getenv("LANGSMITH_API_KEY"))
except ImportError:
    traceable = None
    Client = None
    RunTree = None


T = TypeVar("T")


@dataclass
class TraceConfig:
    """Configuration for LangSmith tracing.

    Attributes:
        project_name: LangSmith project name for organizing traces.
        enabled: Whether tracing is enabled.
        tags: Default tags to apply to all traces.
        metadata: Default metadata to include in traces.
    """

    project_name: str = "evalops"
    enabled: bool = True
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enabled and not LANGSMITH_AVAILABLE:
            self.enabled = False


@dataclass
class TraceContext:
    """Context for a trace run.

    Attributes:
        run_id: Unique identifier for this trace run.
        project_name: LangSmith project name.
        tags: Tags for this trace.
        metadata: Metadata for this trace.
        started_at: When the trace started.
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    project_name: str = "evalops"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EvalTracer:
    """Manages LangSmith tracing for evaluation runs.

    Provides decorators and context managers for tracing LLM calls
    and evaluation runs with automatic capture of metrics.

    Example:
        ```python
        tracer = EvalTracer(project_name="my-evals")

        @tracer.trace_target
        async def my_llm(input: str) -> str:
            return await call_claude(input)

        with tracer.trace_run("accuracy_test", tags=["prod"]):
            result = await runner.evaluate(dataset, my_llm)
        ```
    """

    def __init__(
        self,
        project_name: str = "evalops",
        enabled: bool = True,
        default_tags: list[str] | None = None,
        default_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tracer.

        Args:
            project_name: LangSmith project name.
            enabled: Whether to enable tracing.
            default_tags: Default tags for all traces.
            default_metadata: Default metadata for all traces.
        """
        self.config = TraceConfig(
            project_name=project_name,
            enabled=enabled and LANGSMITH_AVAILABLE,
            tags=default_tags or [],
            metadata=default_metadata or {},
        )
        self._client: Client | None = None
        self._current_context: TraceContext | None = None

    @property
    def client(self) -> Client | None:
        """Lazy-initialize LangSmith client."""
        if self._client is None and self.config.enabled and Client is not None:
            self._client = Client()
        return self._client

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and available."""
        return self.config.enabled and LANGSMITH_AVAILABLE

    @contextmanager
    def trace_run(
        self,
        run_name: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Context manager for tracing an evaluation run.

        Args:
            run_name: Name for this evaluation run.
            tags: Additional tags for this run.
            metadata: Additional metadata for this run.

        Yields:
            TraceContext for the current run.
        """
        all_tags = self.config.tags + (tags or [])
        all_metadata = {**self.config.metadata, **(metadata or {})}

        context = TraceContext(
            project_name=self.config.project_name,
            tags=all_tags,
            metadata={
                "run_name": run_name,
                **all_metadata,
            },
        )

        self._current_context = context

        if self.is_enabled and RunTree is not None:
            # Create a LangSmith run tree for the evaluation
            run_tree = RunTree(
                name=run_name,
                run_type="chain",
                project_name=self.config.project_name,
                tags=all_tags,
                extra={"metadata": all_metadata},
            )
            try:
                yield context
                run_tree.end()
                run_tree.post()
            except Exception as e:
                run_tree.end(error=str(e))
                run_tree.post()
                raise
        else:
            yield context

        self._current_context = None

    def trace_target(
        self,
        func: Callable[..., T] | None = None,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> Callable[..., T]:
        """Decorator to trace a target function (LLM call).

        Args:
            func: The function to trace.
            name: Custom name for the trace (defaults to function name).
            tags: Additional tags for this trace.

        Returns:
            Wrapped function with tracing.

        Example:
            ```python
            @tracer.trace_target
            async def call_claude(input: str) -> str:
                ...

            @tracer.trace_target(name="custom-llm", tags=["important"])
            def my_function(x):
                ...
            ```
        """

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            trace_name = name or fn.__name__
            all_tags = (self.config.tags or []) + (tags or [])

            if self.is_enabled and traceable is not None:
                # Use LangSmith's traceable decorator
                return traceable(
                    name=trace_name,
                    tags=all_tags,
                    project_name=self.config.project_name,
                )(fn)
            else:
                # No-op wrapper when tracing is disabled
                return fn

        if func is not None:
            return decorator(func)
        return decorator

    def trace_eval_result(self, result: EvalResult) -> None:
        """Record an individual evaluation result to LangSmith.

        Args:
            result: The evaluation result to trace.
        """
        if not self.is_enabled or self.client is None:
            return

        # Log as a feedback/evaluation in LangSmith
        try:
            self.client.create_run(
                name=f"eval_case_{result.case_id}",
                run_type="evaluation",
                project_name=self.config.project_name,
                inputs={"input": result.input, "expected": result.expected},
                outputs={"output": result.output},
                error=result.error,
                extra={
                    "latency_ms": result.latency_ms,
                    "metrics": result.metric_results,
                    "metadata": result.metadata,
                },
                tags=self.config.tags + ["eval_case"],
            )
        except Exception:
            # Don't fail the eval if tracing fails
            pass

    def trace_eval_run_result(
        self,
        result: EvalRunResult,
        run_name: str | None = None,
    ) -> None:
        """Record a complete evaluation run to LangSmith.

        Args:
            result: The evaluation run result to trace.
            run_name: Optional name for the run.
        """
        if not self.is_enabled or self.client is None:
            return

        name = run_name or f"eval_run_{result.dataset_name}"

        try:
            self.client.create_run(
                name=name,
                run_type="evaluation",
                project_name=self.config.project_name,
                inputs={
                    "dataset_id": result.dataset_id,
                    "dataset_name": result.dataset_name,
                    "total_cases": result.total_cases,
                },
                outputs={
                    "successful_cases": result.successful_cases,
                    "success_rate": result.success_rate,
                    "pass_rate": result.pass_rate,
                    "avg_latency_ms": result.avg_latency_ms,
                    "metrics_summary": result.metrics_summary,
                },
                extra={
                    "started_at": result.started_at.isoformat(),
                    "completed_at": (
                        result.completed_at.isoformat() if result.completed_at else None
                    ),
                },
                tags=self.config.tags + ["eval_run"],
            )
        except Exception:
            # Don't fail the eval if tracing fails
            pass


# Global tracer instance for convenience
_default_tracer: EvalTracer | None = None


def get_tracer() -> EvalTracer:
    """Get or create the default tracer instance."""
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = EvalTracer()
    return _default_tracer


def configure_tracing(
    project_name: str = "evalops",
    enabled: bool = True,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvalTracer:
    """Configure the global tracer.

    Args:
        project_name: LangSmith project name.
        enabled: Whether to enable tracing.
        tags: Default tags for all traces.
        metadata: Default metadata for all traces.

    Returns:
        Configured EvalTracer instance.
    """
    global _default_tracer
    _default_tracer = EvalTracer(
        project_name=project_name,
        enabled=enabled,
        default_tags=tags,
        default_metadata=metadata,
    )
    return _default_tracer


def trace_target(
    func: Callable[..., T] | None = None,
    *,
    name: str | None = None,
    tags: list[str] | None = None,
) -> Callable[..., T]:
    """Convenience decorator using the global tracer.

    Example:
        ```python
        @trace_target
        async def my_llm(input: str) -> str:
            return await call_claude(input)
        ```
    """
    tracer = get_tracer()
    if func is not None:
        return tracer.trace_target(func, name=name, tags=tags)
    return tracer.trace_target(name=name, tags=tags)
