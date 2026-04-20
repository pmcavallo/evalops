"""Repository pattern for evaluation data access.

This module provides a clean interface for persisting and querying
evaluation runs, cases, and baselines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import and_, desc, func, select
from sqlalchemy.orm import Session

from evalops.storage.models import (
    Base,
    BaselineRecord,
    EvalCaseRecord,
    EvalRunRecord,
    SchemaVersion,
    get_engine,
    get_session_factory,
)

if TYPE_CHECKING:
    from evalops.core.runner import EvalResult, EvalRunResult


class EvalRepository:
    """Repository for evaluation data persistence.

    Provides CRUD operations for evaluation runs, cases, and baselines
    with support for filtering and historical queries.

    Example:
        ```python
        repo = EvalRepository("sqlite:///evalops.db")
        repo.initialize()

        # Save a run
        repo.save_run(run_result, name="nightly_eval", tags=["production"])

        # Query runs
        runs = repo.list_runs(
            dataset_name="qa_dataset",
            min_pass_rate=0.8,
            limit=10,
        )

        # Get history for drift analysis
        history = repo.get_history(
            dataset_name="qa_dataset",
            days=30,
        )
        ```
    """

    def __init__(
        self,
        database_url: str = "sqlite:///evalops.db",
        echo: bool = False,
    ) -> None:
        """Initialize the repository.

        Args:
            database_url: Database connection string.
            echo: Whether to log SQL statements.
        """
        self.database_url = database_url
        self.engine = get_engine(database_url, echo=echo)
        self.session_factory = get_session_factory(self.engine)

    def initialize(self) -> None:
        """Initialize the database schema.

        Creates all tables if they don't exist.
        """
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get a new database session.

        Returns:
            SQLAlchemy Session instance.
        """
        return self.session_factory()

    # =========================================================================
    # Run Operations
    # =========================================================================

    def save_run(
        self,
        run_result: "EvalRunResult",
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        save_cases: bool = True,
    ) -> EvalRunRecord:
        """Save an evaluation run to the database.

        Args:
            run_result: EvalRunResult to persist.
            name: Optional name for the run.
            tags: Optional tags for filtering.
            metadata: Optional metadata.
            save_cases: Whether to save individual case results.

        Returns:
            The persisted EvalRunRecord.
        """
        with self.get_session() as session:
            # Create run record
            run_record = EvalRunRecord.from_run_result(
                run_result, name=name, tags=tags, metadata=metadata
            )
            session.add(run_record)

            # Save individual case results
            if save_cases:
                for eval_result in run_result.results:
                    case_record = EvalCaseRecord.from_eval_result(
                        eval_result, run_id=run_result.id
                    )
                    session.add(case_record)

            session.commit()
            session.refresh(run_record)
            return run_record

    def get_run(self, run_id: str) -> EvalRunRecord | None:
        """Retrieve a run by ID.

        Args:
            run_id: The run's unique identifier.

        Returns:
            EvalRunRecord if found, None otherwise.
        """
        with self.get_session() as session:
            stmt = select(EvalRunRecord).where(EvalRunRecord.id == run_id)
            return session.scalar(stmt)

    def get_run_with_cases(
        self, run_id: str
    ) -> tuple[EvalRunRecord | None, list[EvalCaseRecord]]:
        """Retrieve a run with all its case results.

        Args:
            run_id: The run's unique identifier.

        Returns:
            Tuple of (run record, list of case records).
        """
        with self.get_session() as session:
            run_stmt = select(EvalRunRecord).where(EvalRunRecord.id == run_id)
            run = session.scalar(run_stmt)

            if run is None:
                return None, []

            cases_stmt = (
                select(EvalCaseRecord)
                .where(EvalCaseRecord.run_id == run_id)
                .order_by(EvalCaseRecord.created_at)
            )
            cases = list(session.scalars(cases_stmt))

            return run, cases

    def list_runs(
        self,
        dataset_name: str | None = None,
        name_contains: str | None = None,
        tags: list[str] | None = None,
        min_pass_rate: float | None = None,
        max_pass_rate: float | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "started_at",
        descending: bool = True,
    ) -> list[EvalRunRecord]:
        """Query runs with filters.

        Args:
            dataset_name: Filter by dataset name.
            name_contains: Filter by run name containing string.
            tags: Filter by tags (any match).
            min_pass_rate: Minimum pass rate.
            max_pass_rate: Maximum pass rate.
            start_date: Filter runs after this date.
            end_date: Filter runs before this date.
            limit: Maximum number of results.
            offset: Number of results to skip.
            order_by: Column to order by.
            descending: Whether to order descending.

        Returns:
            List of matching EvalRunRecords.
        """
        with self.get_session() as session:
            stmt = select(EvalRunRecord)

            # Apply filters
            conditions = []

            if dataset_name is not None:
                conditions.append(EvalRunRecord.dataset_name == dataset_name)

            if name_contains is not None:
                conditions.append(EvalRunRecord.name.ilike(f"%{name_contains}%"))

            if min_pass_rate is not None:
                conditions.append(EvalRunRecord.pass_rate >= min_pass_rate)

            if max_pass_rate is not None:
                conditions.append(EvalRunRecord.pass_rate <= max_pass_rate)

            if start_date is not None:
                conditions.append(EvalRunRecord.started_at >= start_date)

            if end_date is not None:
                conditions.append(EvalRunRecord.started_at <= end_date)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            # Apply ordering
            order_column = getattr(EvalRunRecord, order_by, EvalRunRecord.started_at)
            if descending:
                stmt = stmt.order_by(desc(order_column))
            else:
                stmt = stmt.order_by(order_column)

            # Apply pagination
            stmt = stmt.limit(limit).offset(offset)

            return list(session.scalars(stmt))

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and its cases.

        Args:
            run_id: The run's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self.get_session() as session:
            stmt = select(EvalRunRecord).where(EvalRunRecord.id == run_id)
            run = session.scalar(stmt)

            if run is None:
                return False

            session.delete(run)
            session.commit()
            return True

    def get_history(
        self,
        dataset_name: str,
        days: int = 30,
        metric_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical run data for drift analysis.

        Args:
            dataset_name: Dataset to get history for.
            days: Number of days of history.
            metric_names: Specific metrics to include (None for all).

        Returns:
            List of dicts with run summaries ordered by date.
        """
        start_date = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        from datetime import timedelta

        start_date = start_date - timedelta(days=days)

        runs = self.list_runs(
            dataset_name=dataset_name,
            start_date=start_date,
            order_by="started_at",
            descending=False,
            limit=1000,
        )

        history = []
        for run in runs:
            entry = {
                "run_id": run.id,
                "name": run.name,
                "started_at": run.started_at,
                "pass_rate": run.pass_rate,
                "total_cases": run.total_cases,
                "avg_latency_ms": run.avg_latency_ms,
                "metrics": {},
            }

            # Extract metric values
            if run.metrics_summary:
                for name, summary in run.metrics_summary.items():
                    if metric_names is None or name in metric_names:
                        if "pass_rate" in summary.get("details", {}):
                            entry["metrics"][name] = summary["details"]["pass_rate"]
                        else:
                            entry["metrics"][name] = summary.get("score", 0.0)

            history.append(entry)

        return history

    def get_run_stats(
        self,
        dataset_name: str | None = None,
        days: int | None = None,
    ) -> dict[str, Any]:
        """Get aggregate statistics for runs.

        Args:
            dataset_name: Filter by dataset name.
            days: Limit to recent days.

        Returns:
            Dict with aggregate statistics.
        """
        with self.get_session() as session:
            stmt = select(
                func.count(EvalRunRecord.id).label("total_runs"),
                func.avg(EvalRunRecord.pass_rate).label("avg_pass_rate"),
                func.min(EvalRunRecord.pass_rate).label("min_pass_rate"),
                func.max(EvalRunRecord.pass_rate).label("max_pass_rate"),
                func.avg(EvalRunRecord.avg_latency_ms).label("avg_latency"),
                func.sum(EvalRunRecord.total_cases).label("total_cases"),
            )

            conditions = []

            if dataset_name is not None:
                conditions.append(EvalRunRecord.dataset_name == dataset_name)

            if days is not None:
                from datetime import timedelta

                start_date = datetime.now(timezone.utc) - timedelta(days=days)
                conditions.append(EvalRunRecord.started_at >= start_date)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            row = session.execute(stmt).first()

            if row is None or row.total_runs == 0:
                return {
                    "total_runs": 0,
                    "avg_pass_rate": 0.0,
                    "min_pass_rate": 0.0,
                    "max_pass_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "total_cases": 0,
                }

            return {
                "total_runs": row.total_runs,
                "avg_pass_rate": float(row.avg_pass_rate or 0),
                "min_pass_rate": float(row.min_pass_rate or 0),
                "max_pass_rate": float(row.max_pass_rate or 0),
                "avg_latency_ms": float(row.avg_latency or 0),
                "total_cases": int(row.total_cases or 0),
            }

    # =========================================================================
    # Baseline Operations
    # =========================================================================

    def save_baseline(
        self,
        run_result: "EvalRunResult",
        name: str,
        deactivate_existing: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineRecord:
        """Save a run result as a baseline.

        Args:
            run_result: EvalRunResult to use as baseline.
            name: Name for the baseline.
            deactivate_existing: Whether to deactivate existing baselines.
            metadata: Optional metadata.

        Returns:
            The persisted BaselineRecord.
        """
        with self.get_session() as session:
            # Deactivate existing baselines for this dataset
            if deactivate_existing:
                update_stmt = (
                    select(BaselineRecord)
                    .where(
                        and_(
                            BaselineRecord.dataset_name == run_result.dataset_name,
                            BaselineRecord.is_active.is_(True),
                        )
                    )
                )
                for baseline in session.scalars(update_stmt):
                    baseline.is_active = False

            # Create new baseline
            baseline = BaselineRecord.from_run_result(
                run_result, name=name, metadata=metadata
            )
            session.add(baseline)
            session.commit()
            session.refresh(baseline)
            return baseline

    def save_baseline_from_values(
        self,
        dataset_name: str,
        name: str,
        metrics: dict[str, float],
        pass_rate: float = 0.0,
        total_cases: int = 0,
        deactivate_existing: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineRecord:
        """Save a baseline from raw metric values.

        Args:
            dataset_name: Dataset this baseline applies to.
            name: Name for the baseline.
            metrics: Dict of metric name to value.
            pass_rate: Overall pass rate.
            total_cases: Number of cases.
            deactivate_existing: Whether to deactivate existing baselines.
            metadata: Optional metadata.

        Returns:
            The persisted BaselineRecord.
        """
        with self.get_session() as session:
            # Deactivate existing baselines
            if deactivate_existing:
                update_stmt = select(BaselineRecord).where(
                    and_(
                        BaselineRecord.dataset_name == dataset_name,
                        BaselineRecord.is_active.is_(True),
                    )
                )
                for baseline in session.scalars(update_stmt):
                    baseline.is_active = False

            # Create new baseline
            baseline = BaselineRecord(
                name=name,
                dataset_name=dataset_name,
                metrics=metrics,
                pass_rate=pass_rate,
                total_cases=total_cases,
                is_active=True,
                metadata_json=metadata,
            )
            session.add(baseline)
            session.commit()
            session.refresh(baseline)
            return baseline

    def get_baseline(
        self,
        baseline_id: str | None = None,
        dataset_name: str | None = None,
        active_only: bool = True,
    ) -> BaselineRecord | None:
        """Retrieve a baseline.

        Args:
            baseline_id: Specific baseline ID to retrieve.
            dataset_name: Get active baseline for this dataset.
            active_only: Only return active baselines.

        Returns:
            BaselineRecord if found, None otherwise.
        """
        with self.get_session() as session:
            if baseline_id is not None:
                stmt = select(BaselineRecord).where(BaselineRecord.id == baseline_id)
                return session.scalar(stmt)

            if dataset_name is not None:
                conditions = [BaselineRecord.dataset_name == dataset_name]
                if active_only:
                    conditions.append(BaselineRecord.is_active.is_(True))

                stmt = (
                    select(BaselineRecord)
                    .where(and_(*conditions))
                    .order_by(desc(BaselineRecord.created_at))
                    .limit(1)
                )
                return session.scalar(stmt)

            return None

    def list_baselines(
        self,
        dataset_name: str | None = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> list[BaselineRecord]:
        """List baselines with optional filtering.

        Args:
            dataset_name: Filter by dataset name.
            active_only: Only return active baselines.
            limit: Maximum number of results.

        Returns:
            List of BaselineRecords.
        """
        with self.get_session() as session:
            stmt = select(BaselineRecord)

            conditions = []
            if dataset_name is not None:
                conditions.append(BaselineRecord.dataset_name == dataset_name)
            if active_only:
                conditions.append(BaselineRecord.is_active.is_(True))

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(BaselineRecord.created_at)).limit(limit)
            return list(session.scalars(stmt))

    def delete_baseline(self, baseline_id: str) -> bool:
        """Delete a baseline.

        Args:
            baseline_id: The baseline's unique identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self.get_session() as session:
            stmt = select(BaselineRecord).where(BaselineRecord.id == baseline_id)
            baseline = session.scalar(stmt)

            if baseline is None:
                return False

            session.delete(baseline)
            session.commit()
            return True

    # =========================================================================
    # Case Operations
    # =========================================================================

    def get_cases(
        self,
        run_id: str,
        passed_only: bool | None = None,
        failed_only: bool | None = None,
        limit: int = 1000,
    ) -> list[EvalCaseRecord]:
        """Get case results for a run.

        Args:
            run_id: The run's unique identifier.
            passed_only: Only return passed cases.
            failed_only: Only return failed cases.
            limit: Maximum number of results.

        Returns:
            List of EvalCaseRecords.
        """
        with self.get_session() as session:
            stmt = select(EvalCaseRecord).where(EvalCaseRecord.run_id == run_id)

            if passed_only:
                stmt = stmt.where(EvalCaseRecord.metrics_passed.is_(True))
            elif failed_only:
                stmt = stmt.where(EvalCaseRecord.metrics_passed.is_(False))

            stmt = stmt.order_by(EvalCaseRecord.created_at).limit(limit)
            return list(session.scalars(stmt))

    def get_failed_cases(
        self,
        run_id: str | None = None,
        dataset_name: str | None = None,
        days: int | None = None,
        limit: int = 100,
    ) -> list[EvalCaseRecord]:
        """Get failed cases across runs.

        Args:
            run_id: Filter by specific run.
            dataset_name: Filter by dataset.
            days: Limit to recent days.
            limit: Maximum number of results.

        Returns:
            List of failed EvalCaseRecords.
        """
        with self.get_session() as session:
            stmt = (
                select(EvalCaseRecord)
                .join(EvalRunRecord)
                .where(EvalCaseRecord.metrics_passed.is_(False))
            )

            conditions = []
            if run_id is not None:
                conditions.append(EvalCaseRecord.run_id == run_id)
            if dataset_name is not None:
                conditions.append(EvalRunRecord.dataset_name == dataset_name)
            if days is not None:
                from datetime import timedelta

                start_date = datetime.now(timezone.utc) - timedelta(days=days)
                conditions.append(EvalCaseRecord.created_at >= start_date)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(desc(EvalCaseRecord.created_at)).limit(limit)
            return list(session.scalars(stmt))
