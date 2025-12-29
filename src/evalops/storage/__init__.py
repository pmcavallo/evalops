"""Storage layer for persisting evaluation results.

This module provides:
- SQLAlchemy ORM models for evaluation data
- Repository pattern for data access
- Database schema management and migrations
"""

from evalops.storage.migrations import (
    CURRENT_VERSION,
    DatabaseManager,
    check_db,
    init_db,
)
from evalops.storage.models import (
    Base,
    BaselineRecord,
    EvalCaseRecord,
    EvalRunRecord,
    SchemaVersion,
    get_engine,
    get_session_factory,
)
from evalops.storage.repository import EvalRepository

__all__ = [
    # Models
    "Base",
    "EvalRunRecord",
    "EvalCaseRecord",
    "BaselineRecord",
    "SchemaVersion",
    # Repository
    "EvalRepository",
    # Migrations
    "DatabaseManager",
    "CURRENT_VERSION",
    "init_db",
    "check_db",
    # Engine
    "get_engine",
    "get_session_factory",
]
