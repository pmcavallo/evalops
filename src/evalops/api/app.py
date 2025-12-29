"""FastAPI application for EvalOps REST API.

This module provides the main API application with endpoints for:
- Running evaluations
- Managing runs and their results
- Managing baselines
- Checking for drift
- System health monitoring

Example:
    Run the API server:
    ```bash
    uvicorn evalops.api.app:app --reload
    ```

    Or with Python:
    ```python
    import uvicorn
    uvicorn.run("evalops.api.app:app", reload=True)
    ```
"""

from __future__ import annotations

import asyncio
import importlib
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from evalops import __version__
from evalops.api.schemas import (
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
    AlertSeverity,
)
from evalops.storage import DatabaseManager, EvalRepository

# ============================================================================
# Application State
# ============================================================================


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        self.start_time: datetime = datetime.now(timezone.utc)
        self.database_url: str = "sqlite:///evalops.db"
        self._db_manager: DatabaseManager | None = None
        self._repository: EvalRepository | None = None

    @property
    def db_manager(self) -> DatabaseManager:
        if self._db_manager is None:
            self._db_manager = DatabaseManager(self.database_url)
        return self._db_manager

    @property
    def repository(self) -> EvalRepository:
        if self._repository is None:
            self._repository = EvalRepository(self.database_url)
        return self._repository

    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


app_state = AppState()


# ============================================================================
# Lifespan and App Setup
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize database if needed
    try:
        app_state.db_manager.initialize()
    except Exception:
        pass  # Database might not be configured yet
    yield
    # Shutdown: Cleanup if needed
    pass


app = FastAPI(
    title="EvalOps API",
    description="Production-grade LLM evaluation and observability platform",
    version=__version__,
    lifespan=lifespan,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependencies
# ============================================================================


def get_repository() -> EvalRepository:
    """Dependency to get the repository."""
    return app_state.repository


def get_db_manager() -> DatabaseManager:
    """Dependency to get the database manager."""
    return app_state.db_manager


# ============================================================================
# Helper Functions
# ============================================================================


def load_target(target_path: str) -> Any:
    """Load a target function from a module path.

    Args:
        target_path: Module path in format 'module.path:function_name'

    Returns:
        The loaded callable.

    Raises:
        HTTPException: If the target cannot be loaded.
    """
    try:
        if ":" not in target_path:
            raise ValueError("Target must be in format 'module.path:function_name'")

        module_path, func_name = target_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        target = getattr(module, func_name)
        return target
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load target '{target_path}': {e}",
        )


def load_metrics(metric_names: list[str]) -> list[Any]:
    """Load metric instances from names.

    Args:
        metric_names: List of metric names.

    Returns:
        List of Metric instances.
    """
    from evalops.core.metrics import (
        ContainsKeywords,
        ExactMatch,
        Latency,
        SemanticSimilarity,
        TokenCost,
    )

    metrics_map = {
        "exact_match": ExactMatch,
        "contains_keywords": ContainsKeywords,
        "latency": Latency,
        "token_cost": TokenCost,
        "semantic_similarity": SemanticSimilarity,
    }

    metrics = []
    for name in metric_names:
        name_lower = name.lower().replace("-", "_")
        if name_lower in metrics_map:
            metrics.append(metrics_map[name_lower]())

    return metrics


def record_to_run_summary(record: Any) -> RunSummary:
    """Convert a database record to a RunSummary."""
    return RunSummary(
        id=record.id,
        name=record.name,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        total_cases=record.total_cases,
        successful_cases=record.successful_cases,
        pass_rate=record.pass_rate,
        success_rate=record.success_rate,
        avg_latency_ms=record.avg_latency_ms,
        tags=record.tags,
        started_at=record.started_at,
        completed_at=record.completed_at,
        created_at=record.created_at,
    )


def record_to_run_detail(record: Any) -> RunDetail:
    """Convert a database record to a RunDetail."""
    return RunDetail(
        id=record.id,
        name=record.name,
        dataset_id=record.dataset_id,
        dataset_name=record.dataset_name,
        total_cases=record.total_cases,
        successful_cases=record.successful_cases,
        pass_rate=record.pass_rate,
        success_rate=record.success_rate,
        avg_latency_ms=record.avg_latency_ms,
        total_latency_ms=record.total_latency_ms,
        metrics_summary=record.metrics_summary or {},
        tags=record.tags,
        metadata=record.metadata_json,
        started_at=record.started_at,
        completed_at=record.completed_at,
        created_at=record.created_at,
    )


def record_to_baseline(record: Any) -> BaselineResponse:
    """Convert a database record to a BaselineResponse."""
    return BaselineResponse(
        id=record.id,
        name=record.name,
        dataset_name=record.dataset_name,
        source_run_id=record.source_run_id,
        metrics=record.metrics or {},
        pass_rate=record.pass_rate,
        total_cases=record.total_cases,
        is_active=record.is_active,
        created_at=record.created_at,
        expires_at=record.expires_at,
        metadata=record.metadata_json,
    )


# ============================================================================
# Health Endpoints
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check system health",
)
async def health_check(
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> HealthResponse:
    """Check the health of the system including database connectivity."""
    db_health_info = db_manager.check_health()

    db_health = DatabaseHealth(
        healthy=db_health_info.get("healthy", False),
        version=db_health_info.get("version"),
        current_version=db_health_info.get("current_version"),
        needs_migration=db_health_info.get("needs_migration", False),
        tables=db_health_info.get("tables", []),
        error=db_health_info.get("error"),
    )

    overall_status = HealthStatus.HEALTHY if db_health.healthy else HealthStatus.DEGRADED

    return HealthResponse(
        status=overall_status,
        version=__version__,
        database=db_health,
        uptime_seconds=app_state.uptime_seconds(),
        checks={"database": db_health.healthy},
    )


@app.get(
    "/",
    tags=["Health"],
    summary="API root",
)
async def root():
    """API root endpoint with basic info."""
    return {
        "name": "EvalOps API",
        "version": __version__,
        "docs_url": "/docs",
        "health_url": "/health",
    }


# ============================================================================
# Run Endpoints
# ============================================================================


@app.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Runs"],
    summary="Trigger evaluation run",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def create_run(
    request: RunRequest,
    repository: EvalRepository = Depends(get_repository),
) -> RunResponse:
    """Trigger a new evaluation run.

    Loads the dataset from the specified path, imports the target function,
    and runs the evaluation with the specified metrics.
    """
    from pathlib import Path

    from evalops.core.dataset import EvalDataset
    from evalops.core.runner import EvalRunner

    # Load dataset
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.dataset_path}",
        )

    try:
        dataset = EvalDataset.from_json(dataset_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load dataset: {e}",
        )

    # Load target
    target = load_target(request.target_module)

    # Load metrics
    metrics = load_metrics(request.metrics) if request.metrics else []

    # Create runner
    runner = EvalRunner(
        repository=repository if request.save else None,
        auto_save=request.save,
    )

    # Run evaluation
    try:
        result = await runner.evaluate(
            dataset=dataset,
            target=target,
            metrics=metrics,
            run_name=request.name,
            tags=request.tags,
            metadata=request.metadata,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {e}",
        )

    return RunResponse(
        id=result.id,
        status="completed",
        pass_rate=result.pass_rate,
        success_rate=result.success_rate,
        total_cases=result.total_cases,
        successful_cases=result.successful_cases,
        avg_latency_ms=result.avg_latency_ms,
        metrics_summary=result.metrics_summary,
        message="Evaluation completed successfully",
    )


@app.get(
    "/runs",
    response_model=RunListResponse,
    tags=["Runs"],
    summary="List evaluation runs",
)
async def list_runs(
    dataset_name: str | None = Query(None, description="Filter by dataset name"),
    tag: str | None = Query(None, description="Filter by tag"),
    min_pass_rate: float | None = Query(None, ge=0, le=1, description="Minimum pass rate"),
    days: int | None = Query(None, gt=0, description="Only runs from last N days"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    repository: EvalRepository = Depends(get_repository),
) -> RunListResponse:
    """List evaluation runs with optional filters."""
    start_date = None
    if days:
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

    records = repository.list_runs(
        dataset_name=dataset_name,
        tag=tag,
        min_pass_rate=min_pass_rate,
        start_date=start_date,
        limit=limit,
        offset=offset,
    )

    runs = [record_to_run_summary(r) for r in records]

    # Get total count
    all_records = repository.list_runs(
        dataset_name=dataset_name,
        tag=tag,
        min_pass_rate=min_pass_rate,
        start_date=start_date,
        limit=10000,
        offset=0,
    )

    return RunListResponse(
        runs=runs,
        total=len(all_records),
        limit=limit,
        offset=offset,
    )


@app.get(
    "/runs/{run_id}",
    response_model=RunDetail,
    tags=["Runs"],
    summary="Get run details",
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
)
async def get_run(
    run_id: str,
    repository: EvalRepository = Depends(get_repository),
) -> RunDetail:
    """Get detailed information about a specific run."""
    record = repository.get_run(run_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    return record_to_run_detail(record)


@app.get(
    "/runs/{run_id}/cases",
    response_model=CaseListResponse,
    tags=["Runs"],
    summary="Get run case results",
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
)
async def get_run_cases(
    run_id: str,
    success_only: bool = Query(False, description="Only successful cases"),
    failed_only: bool = Query(False, description="Only failed cases"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset"),
    repository: EvalRepository = Depends(get_repository),
) -> CaseListResponse:
    """Get individual case results for a run."""
    run = repository.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    # Get cases from the run
    cases_query = run.cases

    if success_only:
        cases_query = cases_query.filter_by(success=True)
    elif failed_only:
        cases_query = cases_query.filter_by(success=False)

    total = cases_query.count()
    case_records = cases_query.offset(offset).limit(limit).all()

    cases = [
        CaseResult(
            id=c.id,
            case_id=c.case_id,
            input=c.input,
            output=c.output,
            expected=c.expected,
            expected_keywords=c.expected_keywords,
            latency_ms=c.latency_ms,
            success=c.success,
            metrics_passed=c.metrics_passed,
            error=c.error,
            metric_results=c.metric_results or {},
            metadata=c.metadata_json,
        )
        for c in case_records
    ]

    return CaseListResponse(
        cases=cases,
        total=total,
        limit=limit,
        offset=offset,
        run_id=run_id,
    )


@app.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Runs"],
    summary="Delete a run",
    responses={
        404: {"model": ErrorResponse, "description": "Run not found"},
    },
)
async def delete_run(
    run_id: str,
    repository: EvalRepository = Depends(get_repository),
):
    """Delete a run and all its cases."""
    record = repository.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}",
        )

    repository.delete_run(run_id)


# ============================================================================
# Baseline Endpoints
# ============================================================================


@app.post(
    "/baselines",
    response_model=BaselineResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Baselines"],
    summary="Create a baseline",
    responses={
        404: {"model": ErrorResponse, "description": "Source run not found"},
    },
)
async def create_baseline(
    request: BaselineRequest,
    repository: EvalRepository = Depends(get_repository),
) -> BaselineResponse:
    """Create a new baseline from an existing run."""
    from evalops.core.runner import EvalRunResult

    # Get the source run
    run_record = repository.get_run(request.run_id)
    if run_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source run not found: {request.run_id}",
        )

    # Create a mock EvalRunResult from the record for the repository method
    run_result = EvalRunResult(
        id=run_record.id,
        dataset_id=run_record.dataset_id,
        dataset_name=run_record.dataset_name,
        total_cases=run_record.total_cases,
        successful_cases=run_record.successful_cases,
        total_latency_ms=run_record.total_latency_ms,
        started_at=run_record.started_at,
        completed_at=run_record.completed_at,
        metrics_summary=run_record.metrics_summary or {},
    )

    baseline = repository.save_baseline(
        run_result=run_result,
        name=request.name,
        deactivate_existing=request.deactivate_existing,
        metadata=request.metadata,
    )

    return record_to_baseline(baseline)


@app.get(
    "/baselines",
    response_model=BaselineListResponse,
    tags=["Baselines"],
    summary="List baselines",
)
async def list_baselines(
    dataset_name: str | None = Query(None, description="Filter by dataset"),
    active_only: bool = Query(True, description="Only active baselines"),
    repository: EvalRepository = Depends(get_repository),
) -> BaselineListResponse:
    """List baselines with optional filters."""
    baselines = repository.list_baselines(
        dataset_name=dataset_name,
        active_only=active_only,
    )

    return BaselineListResponse(
        baselines=[record_to_baseline(b) for b in baselines],
        total=len(baselines),
    )


@app.get(
    "/baselines/{baseline_id}",
    response_model=BaselineResponse,
    tags=["Baselines"],
    summary="Get baseline details",
    responses={
        404: {"model": ErrorResponse, "description": "Baseline not found"},
    },
)
async def get_baseline(
    baseline_id: str,
    repository: EvalRepository = Depends(get_repository),
) -> BaselineResponse:
    """Get details of a specific baseline."""
    baseline = repository.get_baseline(baseline_id=baseline_id)
    if baseline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline not found: {baseline_id}",
        )

    return record_to_baseline(baseline)


@app.delete(
    "/baselines/{baseline_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Baselines"],
    summary="Delete a baseline",
    responses={
        404: {"model": ErrorResponse, "description": "Baseline not found"},
    },
)
async def delete_baseline(
    baseline_id: str,
    repository: EvalRepository = Depends(get_repository),
):
    """Delete a baseline."""
    baseline = repository.get_baseline(baseline_id=baseline_id)
    if baseline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline not found: {baseline_id}",
        )

    repository.delete_baseline(baseline_id)


# ============================================================================
# Drift Detection Endpoints
# ============================================================================


@app.get(
    "/drift",
    response_model=DriftResponse,
    tags=["Drift"],
    summary="Check for drift",
)
async def check_drift(
    dataset_name: str = Query(..., description="Dataset to check"),
    days: int = Query(30, gt=0, description="Days to analyze"),
    warning_threshold: float = Query(0.05, ge=0, le=1, description="Warning threshold"),
    critical_threshold: float = Query(0.10, ge=0, le=1, description="Critical threshold"),
    repository: EvalRepository = Depends(get_repository),
) -> DriftResponse:
    """Check for quality drift against the baseline."""
    from evalops.comparison.drift import DriftDetector

    # Get baseline for dataset
    baseline = repository.get_baseline(dataset_name=dataset_name, active_only=True)
    if baseline is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active baseline found for dataset: {dataset_name}",
        )

    # Get recent run history
    history = repository.get_history(
        dataset_name=dataset_name,
        days=days,
    )

    if not history:
        return DriftResponse(
            id=str(uuid4()),
            drift_detected=False,
            baseline_metrics=baseline.metrics,
            overall_health=HealthStatus.UNKNOWN,
            created_at=datetime.now(timezone.utc),
        )

    # Create detector and set baseline
    detector = DriftDetector(
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    detector.set_baseline_from_values(
        metrics=baseline.metrics,
        pass_rate=baseline.pass_rate,
    )

    # Add snapshots from history
    for point in history:
        metrics = point.get("metrics", {})
        metrics["pass_rate"] = point.get("pass_rate", 0.0)
        detector.add_raw_snapshot(
            metrics=metrics,
            pass_rate=point.get("pass_rate", 0.0),
            run_id=point.get("run_id"),
        )

    # Check for drift
    report = detector.check()

    # Convert alerts to schema format
    alerts = [
        DriftAlert(
            id=a.id,
            severity=AlertSeverity(a.severity.value),
            metric_name=a.metric_name,
            baseline_value=a.baseline_value,
            current_value=a.current_value,
            degradation=a.degradation,
            direction=DriftDirection(a.direction.value),
            message=a.message,
            suggested_actions=a.suggested_actions,
            created_at=a.created_at,
        )
        for a in report.alerts
    ]

    # Map health status
    health_map = {
        "healthy": HealthStatus.HEALTHY,
        "degraded": HealthStatus.DEGRADED,
        "critical": HealthStatus.CRITICAL,
        "improving": HealthStatus.HEALTHY,
        "unknown": HealthStatus.UNKNOWN,
    }

    return DriftResponse(
        id=report.id,
        drift_detected=report.drift_detected,
        alerts=alerts,
        baseline_metrics=report.baseline_metrics,
        current_metrics=report.current_metrics,
        metric_trends={k: DriftDirection(v.value) for k, v in report.metric_trends.items()},
        overall_health=health_map.get(report.overall_health, HealthStatus.UNKNOWN),
        snapshot_count=report.snapshot_count,
        created_at=report.created_at,
    )


@app.post(
    "/drift/check",
    response_model=DriftResponse,
    tags=["Drift"],
    summary="Check for drift (POST)",
)
async def check_drift_post(
    request: DriftCheckRequest,
    repository: EvalRepository = Depends(get_repository),
) -> DriftResponse:
    """Check for quality drift (POST version for complex requests)."""
    return await check_drift(
        dataset_name=request.dataset_name,
        days=request.days,
        warning_threshold=request.warning_threshold,
        critical_threshold=request.critical_threshold,
        repository=repository,
    )


# ============================================================================
# History and Stats Endpoints
# ============================================================================


@app.get(
    "/history/{dataset_name}",
    response_model=HistoryResponse,
    tags=["History"],
    summary="Get run history",
)
async def get_history(
    dataset_name: str,
    days: int = Query(30, gt=0, description="Days of history"),
    repository: EvalRepository = Depends(get_repository),
) -> HistoryResponse:
    """Get historical run data for trend analysis."""
    history = repository.get_history(dataset_name=dataset_name, days=days)

    points = [
        HistoryPoint(
            run_id=h.get("run_id", ""),
            timestamp=h.get("timestamp", datetime.now(timezone.utc)),
            pass_rate=h.get("pass_rate", 0.0),
            avg_latency_ms=h.get("avg_latency_ms", 0.0),
            metrics=h.get("metrics", {}),
        )
        for h in history
    ]

    # Calculate stats
    stats_data = repository.get_run_stats(dataset_name=dataset_name, days=days)

    stats = RunStats(
        total_runs=stats_data.get("total_runs", 0),
        total_cases=stats_data.get("total_cases", 0),
        avg_pass_rate=stats_data.get("avg_pass_rate", 0.0),
        avg_latency_ms=stats_data.get("avg_latency_ms", 0.0),
        min_pass_rate=stats_data.get("min_pass_rate", 0.0),
        max_pass_rate=stats_data.get("max_pass_rate", 0.0),
        first_run_at=stats_data.get("first_run_at"),
        last_run_at=stats_data.get("last_run_at"),
    )

    return HistoryResponse(
        dataset_name=dataset_name,
        points=points,
        stats=stats,
        days=days,
    )


@app.get(
    "/stats",
    response_model=RunStats,
    tags=["History"],
    summary="Get overall statistics",
)
async def get_stats(
    dataset_name: str | None = Query(None, description="Filter by dataset"),
    days: int | None = Query(None, gt=0, description="Only last N days"),
    repository: EvalRepository = Depends(get_repository),
) -> RunStats:
    """Get aggregated run statistics."""
    stats_data = repository.get_run_stats(dataset_name=dataset_name, days=days)

    return RunStats(
        total_runs=stats_data.get("total_runs", 0),
        total_cases=stats_data.get("total_cases", 0),
        avg_pass_rate=stats_data.get("avg_pass_rate", 0.0),
        avg_latency_ms=stats_data.get("avg_latency_ms", 0.0),
        min_pass_rate=stats_data.get("min_pass_rate", 0.0),
        max_pass_rate=stats_data.get("max_pass_rate", 0.0),
        first_run_at=stats_data.get("first_run_at"),
        last_run_at=stats_data.get("last_run_at"),
    )


# ============================================================================
# Compare (A/B Test) Endpoint
# ============================================================================


@app.post(
    "/compare",
    response_model=CompareResponse,
    tags=["Compare"],
    summary="Run A/B comparison",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Dataset not found"},
    },
)
async def run_comparison(
    request: CompareRequest,
) -> CompareResponse:
    """Run an A/B comparison between two variants."""
    from pathlib import Path

    from evalops.comparison.ab_test import ABTest
    from evalops.core.dataset import EvalDataset
    from evalops.core.runner import EvalRunner

    # Load dataset
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {request.dataset_path}",
        )

    try:
        dataset = EvalDataset.from_json(dataset_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load dataset: {e}",
        )

    # Load targets
    target_a = load_target(request.variant_a)
    target_b = load_target(request.variant_b)

    # Load metrics
    metrics = load_metrics(request.metrics) if request.metrics else []

    # Create runners
    runner_a = EvalRunner(enable_observability=False)
    runner_b = EvalRunner(enable_observability=False)

    # Run evaluations
    try:
        result_a = await runner_a.evaluate(dataset=dataset, target=target_a, metrics=metrics)
        result_b = await runner_b.evaluate(dataset=dataset, target=target_b, metrics=metrics)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {e}",
        )

    # Run A/B test
    ab_test = ABTest(confidence_level=request.confidence_level)
    ab_result = ab_test.compare(result_a, result_b)

    # Convert comparisons
    comparisons = [
        MetricComparison(
            metric_name=c.metric_name,
            variant_a_mean=c.variant_a_mean,
            variant_b_mean=c.variant_b_mean,
            difference=c.difference,
            percent_change=c.percent_change,
            p_value=c.p_value,
            is_significant=c.is_significant,
            winner=c.winner,
        )
        for c in ab_result.comparisons
    ]

    return CompareResponse(
        id=ab_result.id,
        dataset_name=dataset.name,
        variant_a_name=request.variant_a,
        variant_b_name=request.variant_b,
        variant_a_pass_rate=result_a.pass_rate,
        variant_b_pass_rate=result_b.pass_rate,
        comparisons=comparisons,
        overall_winner=ab_result.overall_winner,
        recommendation=ab_result.recommendation,
        confidence_level=request.confidence_level,
        created_at=datetime.now(timezone.utc),
    )


# ============================================================================
# Database Management Endpoints
# ============================================================================


@app.post(
    "/db/init",
    tags=["Database"],
    summary="Initialize database",
)
async def init_database(
    force: bool = Query(False, description="Force re-initialization"),
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Initialize or re-initialize the database schema."""
    version = db_manager.initialize(force=force)
    return {
        "status": "initialized",
        "version": version,
        "message": "Database initialized successfully",
    }


@app.get(
    "/db/migrations",
    tags=["Database"],
    summary="Get migration history",
)
async def get_migrations(
    db_manager: DatabaseManager = Depends(get_db_manager),
):
    """Get the history of applied migrations."""
    return {
        "current_version": db_manager.get_version(),
        "migrations": db_manager.get_migration_history(),
    }


# ============================================================================
# Configuration endpoint for client setup
# ============================================================================


@app.post(
    "/config/database",
    tags=["Configuration"],
    summary="Configure database URL",
)
async def configure_database(database_url: str):
    """Configure the database connection URL.

    This allows changing the database without restarting the server.
    """
    app_state.database_url = database_url
    app_state._db_manager = None
    app_state._repository = None

    # Test the connection
    try:
        health = app_state.db_manager.check_health()
        if not health.get("healthy"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Database connection failed: {health.get('error')}",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid database URL: {e}",
        )

    return {
        "status": "configured",
        "database_url": app_state.db_manager._safe_url(),
    }
