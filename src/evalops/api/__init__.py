"""FastAPI REST API for EvalOps.

This module provides:
- FastAPI application for running evaluations
- Pydantic schemas for request/response models
- Endpoints for runs, baselines, drift detection, and health checks
"""

from evalops.api.app import app
from evalops.api.schemas import (
    AlertSeverity,
    BaselineListResponse,
    BaselineRequest,
    BaselineResponse,
    CaseListResponse,
    CaseResult,
    CompareRequest,
    CompareResponse,
    DatabaseHealth,
    DriftAlert,
    DriftCheckRequest,
    DriftDirection,
    DriftResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HealthStatus,
    HistoryPoint,
    HistoryResponse,
    MetricComparison,
    RunDetail,
    RunListResponse,
    RunRequest,
    RunResponse,
    RunStats,
    RunSummary,
)

__all__ = [
    # App
    "app",
    # Enums
    "AlertSeverity",
    "DriftDirection",
    "HealthStatus",
    # Run schemas
    "RunRequest",
    "RunResponse",
    "RunSummary",
    "RunDetail",
    "RunListResponse",
    "CaseResult",
    "CaseListResponse",
    # Baseline schemas
    "BaselineRequest",
    "BaselineResponse",
    "BaselineListResponse",
    # Drift schemas
    "DriftAlert",
    "DriftCheckRequest",
    "DriftResponse",
    # Compare schemas
    "CompareRequest",
    "CompareResponse",
    "MetricComparison",
    # Health schemas
    "DatabaseHealth",
    "HealthResponse",
    # Error schemas
    "ErrorDetail",
    "ErrorResponse",
    # History schemas
    "RunStats",
    "HistoryPoint",
    "HistoryResponse",
]
