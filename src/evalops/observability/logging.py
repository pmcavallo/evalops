"""Structured logging configuration for EvalOps.

This module provides structured JSON logging with request ID correlation
and contextual fields for evaluation runs.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from structlog.types import EventDict, Processor

# Context variables for request correlation
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_dataset_id: ContextVar[str | None] = ContextVar("dataset_id", default=None)
_case_id: ContextVar[str | None] = ContextVar("case_id", default=None)


def get_request_id() -> str | None:
    """Get the current request ID from context."""
    return _request_id.get()


def set_request_id(request_id: str | None) -> None:
    """Set the request ID in context."""
    _request_id.set(request_id)


def get_run_id() -> str | None:
    """Get the current run ID from context."""
    return _run_id.get()


def set_run_id(run_id: str | None) -> None:
    """Set the run ID in context."""
    _run_id.set(run_id)


def get_dataset_id() -> str | None:
    """Get the current dataset ID from context."""
    return _dataset_id.get()


def set_dataset_id(dataset_id: str | None) -> None:
    """Set the dataset ID in context."""
    _dataset_id.set(dataset_id)


def get_case_id() -> str | None:
    """Get the current case ID from context."""
    return _case_id.get()


def set_case_id(case_id: str | None) -> None:
    """Set the case ID in context."""
    _case_id.set(case_id)


def add_context_fields(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Processor to add context fields to log events."""
    request_id = get_request_id()
    run_id = get_run_id()
    dataset_id = get_dataset_id()
    case_id = get_case_id()

    if request_id:
        event_dict["request_id"] = request_id
    if run_id:
        event_dict["run_id"] = run_id
    if dataset_id:
        event_dict["dataset_id"] = dataset_id
    if case_id:
        event_dict["case_id"] = case_id

    return event_dict


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Processor to add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Processor to add service information."""
    event_dict["service"] = "evalops"
    return event_dict


class LogConfig:
    """Configuration for structured logging.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to output JSON format.
        include_timestamp: Whether to include timestamps.
        include_caller: Whether to include caller info.
    """

    def __init__(
        self,
        level: str = "INFO",
        json_format: bool = True,
        include_timestamp: bool = True,
        include_caller: bool = False,
    ) -> None:
        self.level = level.upper()
        self.json_format = json_format
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller


_configured = False


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_timestamp: bool = True,
    include_caller: bool = False,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to output JSON format (True for production).
        include_timestamp: Whether to include timestamps.
        include_caller: Whether to include caller information.
    """
    global _configured

    # Build processor chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_service_info,
        add_context_fields,
    ]

    if include_timestamp:
        processors.append(add_timestamp)

    if include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ))

    processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging
    if json_format:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                add_timestamp,
                add_service_info,
            ],
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                add_timestamp,
            ],
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Set evalops logger level
    evalops_logger = logging.getLogger("evalops")
    evalops_logger.setLevel(getattr(logging, level.upper()))

    _configured = True


def get_logger(name: str = "evalops") -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (default: evalops).

    Returns:
        Configured structlog logger.
    """
    global _configured
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)


class LogContext:
    """Context manager for setting log context fields.

    Example:
        ```python
        with LogContext(run_id="abc123", dataset_id="ds1"):
            logger.info("Starting evaluation")
            # All logs in this block will have run_id and dataset_id
        ```
    """

    def __init__(
        self,
        request_id: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
        case_id: str | None = None,
        auto_request_id: bool = False,
    ) -> None:
        """Initialize log context.

        Args:
            request_id: Request correlation ID.
            run_id: Evaluation run ID.
            dataset_id: Dataset ID being evaluated.
            case_id: Current case ID.
            auto_request_id: Auto-generate request ID if not provided.
        """
        self.request_id = request_id
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.case_id = case_id
        self.auto_request_id = auto_request_id

        self._prev_request_id: str | None = None
        self._prev_run_id: str | None = None
        self._prev_dataset_id: str | None = None
        self._prev_case_id: str | None = None

    def __enter__(self) -> LogContext:
        # Save previous values
        self._prev_request_id = get_request_id()
        self._prev_run_id = get_run_id()
        self._prev_dataset_id = get_dataset_id()
        self._prev_case_id = get_case_id()

        # Set new values
        if self.request_id:
            set_request_id(self.request_id)
        elif self.auto_request_id:
            set_request_id(str(uuid4()))

        if self.run_id:
            set_run_id(self.run_id)
        if self.dataset_id:
            set_dataset_id(self.dataset_id)
        if self.case_id:
            set_case_id(self.case_id)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore previous values
        set_request_id(self._prev_request_id)
        set_run_id(self._prev_run_id)
        set_dataset_id(self._prev_dataset_id)
        set_case_id(self._prev_case_id)


class EvalLogger:
    """High-level logging interface for evaluation operations.

    Provides structured logging methods specific to evaluation workflows.

    Example:
        ```python
        eval_logger = EvalLogger()

        eval_logger.run_started("my_dataset", run_id="abc123")
        eval_logger.case_started(case_id="case1")
        eval_logger.case_completed(case_id="case1", latency_ms=150, passed=True)
        eval_logger.run_completed(total=50, passed=45, avg_latency=120)
        ```
    """

    def __init__(self, logger_name: str = "evalops") -> None:
        self._logger = get_logger(logger_name)

    def run_started(
        self,
        dataset_name: str,
        run_id: str,
        total_cases: int,
        metrics: list[str] | None = None,
    ) -> None:
        """Log the start of an evaluation run."""
        self._logger.info(
            "evaluation_run_started",
            dataset_name=dataset_name,
            run_id=run_id,
            total_cases=total_cases,
            metrics=metrics or [],
        )

    def run_completed(
        self,
        run_id: str,
        total_cases: int,
        successful_cases: int,
        pass_rate: float,
        avg_latency_ms: float,
        duration_ms: float,
        metrics_summary: dict[str, Any] | None = None,
    ) -> None:
        """Log the completion of an evaluation run."""
        self._logger.info(
            "evaluation_run_completed",
            run_id=run_id,
            total_cases=total_cases,
            successful_cases=successful_cases,
            pass_rate=round(pass_rate, 4),
            avg_latency_ms=round(avg_latency_ms, 2),
            duration_ms=round(duration_ms, 2),
            metrics_summary=metrics_summary,
        )

    def run_failed(
        self,
        run_id: str,
        error: str,
        cases_completed: int = 0,
    ) -> None:
        """Log a failed evaluation run."""
        self._logger.error(
            "evaluation_run_failed",
            run_id=run_id,
            error=error,
            cases_completed=cases_completed,
        )

    def case_started(
        self,
        case_id: str,
        input_preview: str | None = None,
    ) -> None:
        """Log the start of evaluating a case."""
        self._logger.debug(
            "evaluation_case_started",
            case_id=case_id,
            input_preview=input_preview[:100] if input_preview else None,
        )

    def case_completed(
        self,
        case_id: str,
        latency_ms: float,
        metrics_passed: bool,
        metric_results: dict[str, Any] | None = None,
    ) -> None:
        """Log the completion of evaluating a case."""
        level = "info" if metrics_passed else "warning"
        getattr(self._logger, level)(
            "evaluation_case_completed",
            case_id=case_id,
            latency_ms=round(latency_ms, 2),
            metrics_passed=metrics_passed,
            metric_results=metric_results,
        )

    def case_failed(
        self,
        case_id: str,
        error: str,
        latency_ms: float = 0,
    ) -> None:
        """Log a failed case evaluation."""
        self._logger.error(
            "evaluation_case_failed",
            case_id=case_id,
            error=error,
            latency_ms=round(latency_ms, 2),
        )

    def metric_computed(
        self,
        metric_name: str,
        score: float,
        passed: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log a computed metric result."""
        self._logger.debug(
            "metric_computed",
            metric_name=metric_name,
            score=round(score, 4),
            passed=passed,
            details=details,
        )

    def llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Log an LLM API call."""
        if success:
            self._logger.debug(
                "llm_call_completed",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
            )
        else:
            self._logger.error(
                "llm_call_failed",
                model=model,
                error=error,
                latency_ms=round(latency_ms, 2),
            )
