"""Database schema initialization and migration utilities.

This module provides simple schema management for the EvalOps database,
including table creation and version tracking.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session

from evalops.storage.models import (
    Base,
    SchemaVersion,
    get_engine,
    get_session_factory,
)

# Current schema version
CURRENT_VERSION = 1

# Migration descriptions
MIGRATIONS = {
    1: "Initial schema with eval_runs, eval_cases, baselines, and schema_versions",
}


class DatabaseManager:
    """Manages database schema and migrations.

    Provides utilities for initializing the database schema,
    checking version, and running migrations.

    Example:
        ```python
        manager = DatabaseManager("sqlite:///evalops.db")

        # Initialize or migrate database
        manager.initialize()

        # Check current version
        version = manager.get_version()
        print(f"Database version: {version}")

        # Check if migration needed
        if manager.needs_migration():
            manager.migrate()
        ```
    """

    def __init__(
        self,
        database_url: str = "sqlite:///evalops.db",
        echo: bool = False,
    ) -> None:
        """Initialize the database manager.

        Args:
            database_url: Database connection string.
            echo: Whether to log SQL statements.
        """
        self.database_url = database_url
        self.engine = get_engine(database_url, echo=echo)
        self.session_factory = get_session_factory(self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()

    def initialize(self, force: bool = False) -> int:
        """Initialize the database schema.

        Creates all tables if they don't exist. If the database
        already exists, checks for needed migrations.

        Args:
            force: If True, drop and recreate all tables.

        Returns:
            The current schema version after initialization.
        """
        if force:
            Base.metadata.drop_all(self.engine)

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Ensure schema version is set
        with self.get_session() as session:
            current = self._get_version_record(session)

            if current is None:
                # New database - set to current version
                version_record = SchemaVersion(
                    version=CURRENT_VERSION,
                    description=MIGRATIONS.get(CURRENT_VERSION),
                )
                session.add(version_record)
                session.commit()
                return CURRENT_VERSION
            else:
                return current.version

    def get_version(self) -> int:
        """Get the current schema version.

        Returns:
            Current version number, or 0 if not initialized.
        """
        # Check if schema_versions table exists
        inspector = inspect(self.engine)
        if "schema_versions" not in inspector.get_table_names():
            return 0

        with self.get_session() as session:
            record = self._get_version_record(session)
            return record.version if record else 0

    def _get_version_record(self, session: Session) -> SchemaVersion | None:
        """Get the latest version record."""
        stmt = select(SchemaVersion).order_by(SchemaVersion.version.desc()).limit(1)
        return session.scalar(stmt)

    def needs_migration(self) -> bool:
        """Check if database needs migration.

        Returns:
            True if current version is behind CURRENT_VERSION.
        """
        return self.get_version() < CURRENT_VERSION

    def migrate(self) -> list[int]:
        """Run pending migrations.

        Returns:
            List of versions that were applied.
        """
        current = self.get_version()
        applied = []

        if current == 0:
            # Database not initialized
            self.initialize()
            return [CURRENT_VERSION]

        # Run each migration in order
        for version in range(current + 1, CURRENT_VERSION + 1):
            self._run_migration(version)
            applied.append(version)

        return applied

    def _run_migration(self, version: int) -> None:
        """Run a specific migration.

        Args:
            version: Version number to migrate to.
        """
        with self.get_session() as session:
            # Run migration-specific logic
            migration_func = getattr(self, f"_migrate_to_v{version}", None)
            if migration_func:
                migration_func(session)

            # Record the migration
            version_record = SchemaVersion(
                version=version,
                description=MIGRATIONS.get(version),
            )
            session.add(version_record)
            session.commit()

    def get_migration_history(self) -> list[dict[str, Any]]:
        """Get history of applied migrations.

        Returns:
            List of migration records.
        """
        with self.get_session() as session:
            stmt = select(SchemaVersion).order_by(SchemaVersion.version)
            records = session.scalars(stmt)
            return [
                {
                    "version": r.version,
                    "applied_at": r.applied_at.isoformat() if r.applied_at else None,
                    "description": r.description,
                }
                for r in records
            ]

    def check_health(self) -> dict[str, Any]:
        """Check database health and status.

        Returns:
            Dict with health information.
        """
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()

            with self.get_session() as session:
                # Test a simple query
                session.execute(text("SELECT 1"))

            version = self.get_version()

            return {
                "healthy": True,
                "version": version,
                "current_version": CURRENT_VERSION,
                "needs_migration": version < CURRENT_VERSION,
                "tables": tables,
                "database_url": self._safe_url(),
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "database_url": self._safe_url(),
            }

    def _safe_url(self) -> str:
        """Get database URL with password masked."""
        url = self.database_url
        # Mask password in connection strings like postgresql://user:pass@host/db
        if "@" in url and "://" in url:
            parts = url.split("://", 1)
            if len(parts) == 2 and "@" in parts[1]:
                auth_host = parts[1].split("@", 1)
                if ":" in auth_host[0]:
                    user = auth_host[0].split(":")[0]
                    return f"{parts[0]}://{user}:***@{auth_host[1]}"
        return url

    def drop_all(self) -> None:
        """Drop all tables. Use with caution!"""
        Base.metadata.drop_all(self.engine)

    def table_sizes(self) -> dict[str, int]:
        """Get row counts for all tables.

        Returns:
            Dict of table name to row count.
        """
        sizes = {}
        with self.get_session() as session:
            for table in Base.metadata.tables.values():
                try:
                    result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table.name}")
                    )
                    sizes[table.name] = result.scalar() or 0
                except Exception:
                    sizes[table.name] = -1
        return sizes


def init_db(database_url: str = "sqlite:///evalops.db") -> DatabaseManager:
    """Convenience function to initialize database.

    Args:
        database_url: Database connection string.

    Returns:
        Initialized DatabaseManager.
    """
    manager = DatabaseManager(database_url)
    manager.initialize()
    return manager


def check_db(database_url: str = "sqlite:///evalops.db") -> dict[str, Any]:
    """Convenience function to check database health.

    Args:
        database_url: Database connection string.

    Returns:
        Health check results.
    """
    manager = DatabaseManager(database_url)
    return manager.check_health()
