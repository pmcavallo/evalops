"""Pydantic schemas for the EvalOps REST API.

This module defines request and response models for all API endpoints,
ensuring type safety and automatic OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Common Enums
# ============================================================================


class HealthStatus(str, Enum):
    """Overall health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Drift alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftDirection(str, Enum):
    """Direction of metric drift."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


# ============================================================================
# Run Schemas
# ============================================================================


class RunRequest(BaseModel):
    """Request to trigger an evaluation run."""

    dataset_path: str = Field(..., description="Path to the dataset file (JSON/YAML)")
    target_module: str = Field(..., description="Module path to the target function (e.g., 'myapp.llm:query')")
    metrics: list[str] = Field(
        default_factory=list,
        description="List of metric names to compute (e.g., ['exact_match', 'latency'])",
    )
    name: str | None = Field(None, description="Optional name for this run")
    tags: list[str] | None = Field(None, description="Optional tags for filtering")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")
    save: bool = Field(True, description="Whether to save results to database")


class CaseResult(BaseModel):
    """Result of evaluating a single case."""

    id: str
    case_id: str
    input: str
    output: str | None = None
    expected: str | None = None
    expected_keywords: list[str] | None = None
    latency_ms: float = 0.0
    success: bool = True
    metrics_passed: bool = True
    error: str | None = None
    metric_results: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


class RunSummary(BaseModel):
    """Summary of an evaluation run (for list views)."""

    id: str
    name: str | None = None
    dataset_id: str
    dataset_name: str
    total_cases: int = 0
    successful_cases: int = 0
    pass_rate: float = 0.0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    tags: list[str] | None = None
    started_at: datetime
    completed_at: datetime | None = None
    created_at: datetime


class RunDetail(RunSummary):
    """Detailed view of an evaluation run."""

    total_latency_ms: float = 0.0
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


class RunResponse(BaseModel):
    """Response after triggering a run."""

    id: str
    status: str = "completed"
    pass_rate: float
    success_rate: float
    total_cases: int
    successful_cases: int
    avg_latency_ms: float
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    message: str = "Evaluation completed successfully"


class RunListResponse(BaseModel):
    """Paginated list of runs."""

    runs: list[RunSummary]
    total: int
    limit: int
    offset: int


class CaseListResponse(BaseModel):
    """Paginated list of case results."""

    cases: list[CaseResult]
    total: int
    limit: int
    offset: int
    run_id: str


# ============================================================================
# Baseline Schemas
# ============================================================================


class BaselineRequest(BaseModel):
    """Request to create a baseline."""

    run_id: str = Field(..., description="ID of the run to use as baseline")
    name: str = Field(..., description="Name for this baseline")
    deactivate_existing: bool = Field(
        True, description="Deactivate other baselines for this dataset"
    )
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class BaselineResponse(BaseModel):
    """Baseline details."""

    id: str
    name: str
    dataset_name: str
    source_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    pass_rate: float = 0.0
    total_cases: int = 0
    is_active: bool = True
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] | None = None


class BaselineListResponse(BaseModel):
    """List of baselines."""

    baselines: list[BaselineResponse]
    total: int


# ============================================================================
# Drift Schemas
# ============================================================================


class DriftAlert(BaseModel):
    """A drift alert for a specific metric."""

    id: str
    severity: AlertSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    degradation: float
    direction: DriftDirection
    message: str
    suggested_actions: list[str] = Field(default_factory=list)
    created_at: datetime


class DriftCheckRequest(BaseModel):
    """Request to check for drift."""

    dataset_name: str = Field(..., description="Dataset to check drift for")
    days: int = Field(30, description="Number of days to analyze")
    warning_threshold: float = Field(0.05, description="Degradation threshold for warnings (e.g., 0.05 = 5%)")
    critical_threshold: float = Field(0.10, description="Degradation threshold for critical (e.g., 0.10 = 10%)")


class DriftResponse(BaseModel):
    """Drift analysis results."""

    id: str
    drift_detected: bool
    alerts: list[DriftAlert] = Field(default_factory=list)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    current_metrics: dict[str, float] = Field(default_factory=dict)
    metric_trends: dict[str, DriftDirection] = Field(default_factory=dict)
    overall_health: HealthStatus = HealthStatus.HEALTHY
    snapshot_count: int = 0
    created_at: datetime


# ============================================================================
# Compare (A/B Test) Schemas
# ============================================================================


class CompareRequest(BaseModel):
    """Request to run an A/B comparison."""

    dataset_path: str = Field(..., description="Path to the dataset file")
    variant_a: str = Field(..., description="Module path to variant A (e.g., 'myapp.v1:query')")
    variant_b: str = Field(..., description="Module path to variant B (e.g., 'myapp.v2:query')")
    metrics: list[str] = Field(
        default_factory=list,
        description="List of metrics to compare",
    )
    confidence_level: float = Field(0.95, description="Statistical confidence level")


class MetricComparison(BaseModel):
    """Comparison of a single metric between variants."""

    metric_name: str
    variant_a_mean: float
    variant_b_mean: float
    difference: float
    percent_change: float
    p_value: float
    is_significant: bool
    winner: str | None = None


class CompareResponse(BaseModel):
    """A/B comparison results."""

    id: str
    dataset_name: str
    variant_a_name: str
    variant_b_name: str
    variant_a_pass_rate: float
    variant_b_pass_rate: float
    comparisons: list[MetricComparison] = Field(default_factory=list)
    overall_winner: str | None = None
    recommendation: str
    confidence_level: float
    created_at: datetime


# ============================================================================
# Health Check Schemas
# ============================================================================


class DatabaseHealth(BaseModel):
    """Database health information."""

    healthy: bool
    version: int | None = None
    current_version: int | None = None
    needs_migration: bool = False
    tables: list[str] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    """Overall system health."""

    status: HealthStatus
    version: str
    database: DatabaseHealth
    uptime_seconds: float = 0.0
    checks: dict[str, bool] = Field(default_factory=dict)


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None


# ============================================================================
# Stats and History Schemas
# ============================================================================


class RunStats(BaseModel):
    """Aggregated run statistics."""

    total_runs: int = 0
    total_cases: int = 0
    avg_pass_rate: float = 0.0
    avg_latency_ms: float = 0.0
    min_pass_rate: float = 0.0
    max_pass_rate: float = 0.0
    first_run_at: datetime | None = None
    last_run_at: datetime | None = None


class HistoryPoint(BaseModel):
    """Single point in run history."""

    run_id: str
    timestamp: datetime
    pass_rate: float
    avg_latency_ms: float
    metrics: dict[str, float] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    """Historical run data for trend analysis."""

    dataset_name: str
    points: list[HistoryPoint] = Field(default_factory=list)
    stats: RunStats
    days: int
