"""Observability components for LLM evaluation.

This module provides:
- LangSmith tracing integration
- Structured logging with structlog
- Metrics collection for runtime statistics
"""

from evalops.observability.collector import (
    Counter,
    LatencyHistogram,
    MetricsCollector,
    TokenUsageStats,
    get_collector,
    reset_collector,
)
from evalops.observability.logging import (
    EvalLogger,
    LogConfig,
    LogContext,
    configure_logging,
    get_logger,
    get_case_id,
    get_dataset_id,
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
    trace_target,
)

__all__ = [
    # Tracing
    "EvalTracer",
    "TraceConfig",
    "TraceContext",
    "configure_tracing",
    "get_tracer",
    "trace_target",
    # Logging
    "EvalLogger",
    "LogConfig",
    "LogContext",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "set_request_id",
    "get_run_id",
    "set_run_id",
    "get_dataset_id",
    "set_dataset_id",
    "get_case_id",
    "set_case_id",
    # Collector
    "MetricsCollector",
    "Counter",
    "LatencyHistogram",
    "TokenUsageStats",
    "get_collector",
    "reset_collector",
]
