"""SQLAlchemy ORM models for persisting evaluation data.

This module defines the database schema for storing evaluation runs,
individual case results, and baselines for regression testing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class EvalRunRecord(Base):
    """Database record for an evaluation run.

    Stores metadata about evaluation runs including pass rates,
    metrics summaries, and timing information.

    Attributes:
        id: Unique identifier for the run.
        name: Human-readable name for the run.
        dataset_id: ID of the dataset that was evaluated.
        dataset_name: Name of the dataset.
        total_cases: Total number of cases evaluated.
        successful_cases: Number of cases that executed without error.
        pass_rate: Percentage of cases where all metrics passed.
        success_rate: Percentage of cases that executed without error.
        avg_latency_ms: Average latency across all cases.
        total_latency_ms: Total latency for the run.
        metrics_summary: JSON blob of aggregated metrics.
        tags: JSON array of tags for filtering.
        metadata: JSON blob of arbitrary metadata.
        started_at: When the run started.
        completed_at: When the run completed.
        created_at: When the record was created.
    """

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), index=True)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    successful_cases: Mapped[int] = mapped_column(Integer, default=0)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    total_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    metrics_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    cases: Mapped[list["EvalCaseRecord"]] = relationship(
        "EvalCaseRecord",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    # Indexes for common queries
    __table_args__ = (
        Index("ix_eval_runs_dataset_date", "dataset_name", "started_at"),
        Index("ix_eval_runs_pass_rate_date", "pass_rate", "started_at"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "total_cases": self.total_cases,
            "successful_cases": self.successful_cases,
            "pass_rate": self.pass_rate,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "metrics_summary": self.metrics_summary,
            "tags": self.tags,
            "metadata": self.metadata_json,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_run_result(
        cls,
        run_result: Any,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvalRunRecord":
        """Create a record from an EvalRunResult.

        Args:
            run_result: EvalRunResult to persist.
            name: Optional name for the run.
            tags: Optional tags for filtering.
            metadata: Optional metadata.

        Returns:
            EvalRunRecord ready for persistence.
        """
        return cls(
            id=run_result.id,
            name=name,
            dataset_id=run_result.dataset_id,
            dataset_name=run_result.dataset_name,
            total_cases=run_result.total_cases,
            successful_cases=run_result.successful_cases,
            pass_rate=run_result.pass_rate,
            success_rate=run_result.success_rate,
            avg_latency_ms=run_result.avg_latency_ms,
            total_latency_ms=run_result.total_latency_ms,
            metrics_summary=run_result.metrics_summary,
            tags=tags,
            metadata_json=metadata,
            started_at=run_result.started_at,
            completed_at=run_result.completed_at,
        )


class EvalCaseRecord(Base):
    """Database record for an individual evaluation case result.

    Stores the input, output, expected values, and metric results
    for a single evaluation case.

    Attributes:
        id: Unique identifier for this result.
        run_id: Foreign key to the parent run.
        case_id: ID of the original EvalCase.
        input: The input sent to the target.
        output: The output received from the target.
        expected: The expected output.
        expected_keywords: JSON array of expected keywords.
        latency_ms: Time taken in milliseconds.
        success: Whether execution completed without error.
        metrics_passed: Whether all metrics passed.
        error: Error message if execution failed.
        metric_results: JSON blob of per-metric results.
        metadata_json: JSON blob of case metadata.
        created_at: When the record was created.
    """

    __tablename__ = "eval_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    input: Mapped[str] = mapped_column(Text)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    metrics_passed: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric_results: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    run: Mapped["EvalRunRecord"] = relationship("EvalRunRecord", back_populates="cases")

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "input": self.input,
            "output": self.output,
            "expected": self.expected,
            "expected_keywords": self.expected_keywords,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "metrics_passed": self.metrics_passed,
            "error": self.error,
            "metric_results": self.metric_results,
            "metadata": self.metadata_json,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_eval_result(cls, eval_result: Any, run_id: str) -> "EvalCaseRecord":
        """Create a record from an EvalResult.

        Args:
            eval_result: EvalResult to persist.
            run_id: ID of the parent run.

        Returns:
            EvalCaseRecord ready for persistence.
        """
        return cls(
            id=eval_result.id,
            run_id=run_id,
            case_id=eval_result.case_id,
            input=eval_result.input,
            output=eval_result.output,
            expected=eval_result.expected,
            expected_keywords=eval_result.expected_keywords,
            latency_ms=eval_result.latency_ms,
            success=eval_result.success,
            metrics_passed=eval_result.metrics_passed,
            error=eval_result.error,
            metric_results=eval_result.metric_results,
            metadata_json=eval_result.metadata,
        )


class BaselineRecord(Base):
    """Database record for regression testing baselines.

    Stores baseline metric values for comparing against new runs.

    Attributes:
        id: Unique identifier for the baseline.
        name: Human-readable name for the baseline.
        dataset_name: Name of the dataset this baseline applies to.
        source_run_id: ID of the run this baseline was created from.
        metrics: JSON blob of metric name to baseline value.
        pass_rate: Overall pass rate at baseline.
        total_cases: Number of cases in the baseline run.
        is_active: Whether this is the active baseline for the dataset.
        created_at: When the baseline was created.
        expires_at: Optional expiration date.
        metadata_json: JSON blob of arbitrary metadata.
    """

    __tablename__ = "baselines"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), index=True)
    dataset_name: Mapped[str] = mapped_column(String(255), index=True)
    source_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metrics: Mapped[dict[str, float]] = mapped_column(JSON)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSON, nullable=True
    )

    # Index for finding active baseline for a dataset
    __table_args__ = (
        Index("ix_baselines_dataset_active", "dataset_name", "is_active"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "source_run_id": self.source_run_id,
            "metrics": self.metrics,
            "pass_rate": self.pass_rate,
            "total_cases": self.total_cases,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata_json,
        }

    @classmethod
    def from_run_result(
        cls,
        run_result: Any,
        name: str,
        is_active: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> "BaselineRecord":
        """Create a baseline from an EvalRunResult.

        Args:
            run_result: EvalRunResult to use as baseline.
            name: Name for the baseline.
            is_active: Whether this should be the active baseline.
            metadata: Optional metadata.

        Returns:
            BaselineRecord ready for persistence.
        """
        # Extract metric values from summary
        metrics: dict[str, float] = {}
        for metric_name, summary in run_result.metrics_summary.items():
            if "pass_rate" in summary.get("details", {}):
                metrics[metric_name] = summary["details"]["pass_rate"]
            else:
                metrics[metric_name] = summary.get("score", 0.0)

        metrics["overall_pass_rate"] = run_result.pass_rate

        return cls(
            name=name,
            dataset_name=run_result.dataset_name,
            source_run_id=run_result.id,
            metrics=metrics,
            pass_rate=run_result.pass_rate,
            total_cases=run_result.total_cases,
            is_active=is_active,
            metadata_json=metadata,
        )


class SchemaVersion(Base):
    """Tracks database schema version for migrations.

    Attributes:
        version: Current schema version number.
        applied_at: When this version was applied.
        description: Description of the migration.
    """

    __tablename__ = "schema_versions"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)


def get_engine(database_url: str = "sqlite:///evalops.db", echo: bool = False):
    """Create a SQLAlchemy engine.

    Args:
        database_url: Database connection string.
            SQLite: "sqlite:///path/to/db.db"
            PostgreSQL: "postgresql://user:pass@host:port/db"
        echo: Whether to log SQL statements.

    Returns:
        SQLAlchemy Engine instance.
    """
    # Handle SQLite-specific settings
    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            echo=echo,
            connect_args={"check_same_thread": False},
        )
    else:
        return create_engine(database_url, echo=echo)


def get_session_factory(engine) -> sessionmaker[Session]:
    """Create a session factory bound to an engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        Session factory for creating database sessions.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)
