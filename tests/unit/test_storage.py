"""Unit tests for the storage module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from evalops.storage.migrations import CURRENT_VERSION, DatabaseManager
from evalops.storage.models import (
    BaselineRecord,
    EvalCaseRecord,
    EvalRunRecord,
)
from evalops.storage.repository import EvalRepository


class TestEvalRunRecord:
    """Tests for EvalRunRecord model."""

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        now = datetime.now(timezone.utc)
        record = EvalRunRecord(
            id="run-123",
            name="test_run",
            dataset_id="ds-456",
            dataset_name="my_dataset",
            total_cases=10,
            successful_cases=9,
            pass_rate=0.8,
            success_rate=0.9,
            avg_latency_ms=150.5,
            total_latency_ms=1355.0,
            metrics_summary={"exact_match": {"score": 0.8}},
            tags=["production", "nightly"],
            started_at=now,
            completed_at=now,
        )

        data = record.to_dict()

        assert data["id"] == "run-123"
        assert data["name"] == "test_run"
        assert data["dataset_name"] == "my_dataset"
        assert data["pass_rate"] == 0.8
        assert data["tags"] == ["production", "nightly"]

    def test_from_run_result(self) -> None:
        """Test creating record from EvalRunResult."""
        mock_result = MagicMock()
        mock_result.id = "run-789"
        mock_result.dataset_id = "ds-123"
        mock_result.dataset_name = "test_dataset"
        mock_result.total_cases = 5
        mock_result.successful_cases = 4
        mock_result.pass_rate = 0.75
        mock_result.success_rate = 0.8
        mock_result.avg_latency_ms = 100.0
        mock_result.total_latency_ms = 400.0
        mock_result.metrics_summary = {"exact_match": {"score": 0.9}}
        mock_result.started_at = datetime.now(timezone.utc)
        mock_result.completed_at = datetime.now(timezone.utc)

        record = EvalRunRecord.from_run_result(
            mock_result,
            name="test",
            tags=["test"],
        )

        assert record.id == "run-789"
        assert record.dataset_name == "test_dataset"
        assert record.pass_rate == 0.75
        assert record.tags == ["test"]


class TestEvalCaseRecord:
    """Tests for EvalCaseRecord model."""

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = EvalCaseRecord(
            id="case-123",
            run_id="run-456",
            case_id="orig-case-1",
            input="What is 2+2?",
            output="4",
            expected="4",
            latency_ms=50.5,
            success=True,
            metrics_passed=True,
            metric_results={"exact_match": {"score": 1.0, "passed": True}},
        )

        data = record.to_dict()

        assert data["id"] == "case-123"
        assert data["input"] == "What is 2+2?"
        assert data["output"] == "4"
        assert data["metrics_passed"] is True

    def test_from_eval_result(self) -> None:
        """Test creating record from EvalResult."""
        mock_result = MagicMock()
        mock_result.id = "result-123"
        mock_result.case_id = "case-456"
        mock_result.input = "test input"
        mock_result.output = "test output"
        mock_result.expected = "expected"
        mock_result.expected_keywords = ["key1", "key2"]
        mock_result.latency_ms = 75.0
        mock_result.success = True
        mock_result.metrics_passed = True
        mock_result.error = None
        mock_result.metric_results = {}
        mock_result.metadata = {"source": "test"}

        record = EvalCaseRecord.from_eval_result(mock_result, run_id="run-789")

        assert record.id == "result-123"
        assert record.run_id == "run-789"
        assert record.input == "test input"
        assert record.expected_keywords == ["key1", "key2"]


class TestBaselineRecord:
    """Tests for BaselineRecord model."""

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = BaselineRecord(
            id="baseline-123",
            name="v1.0 baseline",
            dataset_name="qa_dataset",
            source_run_id="run-456",
            metrics={"exact_match": 0.95, "latency": 100.0},
            pass_rate=0.92,
            total_cases=100,
            is_active=True,
        )

        data = record.to_dict()

        assert data["id"] == "baseline-123"
        assert data["name"] == "v1.0 baseline"
        assert data["metrics"]["exact_match"] == 0.95
        assert data["is_active"] is True

    def test_from_run_result(self) -> None:
        """Test creating baseline from run result."""
        mock_result = MagicMock()
        mock_result.id = "run-123"
        mock_result.dataset_name = "test_dataset"
        mock_result.pass_rate = 0.88
        mock_result.total_cases = 50
        mock_result.metrics_summary = {
            "exact_match": {"score": 0.9, "details": {"pass_rate": 0.9}},
            "latency": {"score": 120.0, "details": {}},
        }

        record = BaselineRecord.from_run_result(
            mock_result,
            name="test_baseline",
        )

        assert record.source_run_id == "run-123"
        assert record.dataset_name == "test_dataset"
        assert record.metrics["exact_match"] == 0.9
        assert record.metrics["overall_pass_rate"] == 0.88


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary in-memory database with unique name."""
        # Use in-memory database to avoid file locking issues on Windows
        yield "sqlite:///:memory:"

    def test_initialize(self, temp_db: str) -> None:
        """Test database initialization."""
        manager = DatabaseManager(temp_db)
        version = manager.initialize()

        assert version == CURRENT_VERSION

    def test_get_version(self, temp_db: str) -> None:
        """Test getting schema version."""
        manager = DatabaseManager(temp_db)

        # Before init
        assert manager.get_version() == 0

        # After init
        manager.initialize()
        assert manager.get_version() == CURRENT_VERSION

    def test_needs_migration(self, temp_db: str) -> None:
        """Test migration detection."""
        manager = DatabaseManager(temp_db)

        # Before init
        assert manager.needs_migration() is True

        # After init
        manager.initialize()
        assert manager.needs_migration() is False

    def test_check_health(self, temp_db: str) -> None:
        """Test health check."""
        manager = DatabaseManager(temp_db)
        manager.initialize()

        health = manager.check_health()

        assert health["healthy"] is True
        assert health["version"] == CURRENT_VERSION
        assert "eval_runs" in health["tables"]
        assert "baselines" in health["tables"]

    def test_migration_history(self, temp_db: str) -> None:
        """Test getting migration history."""
        manager = DatabaseManager(temp_db)
        manager.initialize()

        history = manager.get_migration_history()

        assert len(history) >= 1
        assert history[0]["version"] == CURRENT_VERSION

    def test_table_sizes(self, temp_db: str) -> None:
        """Test getting table sizes."""
        manager = DatabaseManager(temp_db)
        manager.initialize()

        sizes = manager.table_sizes()

        assert sizes["eval_runs"] == 0
        assert sizes["eval_cases"] == 0
        assert sizes["baselines"] == 0

    def test_drop_all(self, temp_db: str) -> None:
        """Test dropping all tables."""
        manager = DatabaseManager(temp_db)
        manager.initialize()
        manager.drop_all()

        # Tables should not exist
        health = manager.check_health()
        assert "eval_runs" not in health.get("tables", [])

    def test_safe_url(self) -> None:
        """Test URL password masking."""
        with (
            patch("evalops.storage.migrations.get_engine"),
            patch("evalops.storage.migrations.get_session_factory"),
        ):
            # SQLite (no password)
            manager = DatabaseManager("sqlite:///evalops.db")
            assert manager._safe_url() == "sqlite:///evalops.db"

            # PostgreSQL with password
            manager = DatabaseManager("postgresql://synth_user:synth_pass@localhost/synth_db")
            assert manager._safe_url() == "postgresql://synth_user:***@localhost/synth_db"

            # MySQL with password and port
            manager = DatabaseManager("mysql+pymysql://admin:secret123@127.0.0.1:3306/testdb")
            assert manager._safe_url() == "mysql+pymysql://admin:***@127.0.0.1:3306/testdb"

            # User but no password
            manager = DatabaseManager("postgresql://synth_user@localhost/synth_db")
            assert manager._safe_url() == "postgresql://synth_user@localhost/synth_db"

            # Non-URL string
            manager = DatabaseManager("not-a-url")
            assert manager._safe_url() == "not-a-url"


class TestEvalRepository:
    """Tests for EvalRepository class."""

    @pytest.fixture
    def repo(self):
        """Create a repository with in-memory database."""
        repo = EvalRepository("sqlite:///:memory:")
        repo.initialize()
        return repo

    @pytest.fixture
    def sample_run_result(self):
        """Create a sample run result."""
        mock = MagicMock()
        mock.id = "run-test-123"
        mock.dataset_id = "ds-456"
        mock.dataset_name = "test_dataset"
        mock.total_cases = 3
        mock.successful_cases = 2
        mock.pass_rate = 0.67
        mock.success_rate = 0.67
        mock.avg_latency_ms = 100.0
        mock.total_latency_ms = 200.0
        mock.metrics_summary = {"exact_match": {"score": 0.8, "details": {}}}
        mock.started_at = datetime.now(timezone.utc)
        mock.completed_at = datetime.now(timezone.utc)
        mock.results = []
        return mock

    def test_save_and_get_run(self, repo: EvalRepository, sample_run_result) -> None:
        """Test saving and retrieving a run."""
        saved = repo.save_run(
            sample_run_result,
            name="test_run",
            tags=["test"],
        )

        assert saved.id == "run-test-123"
        assert saved.name == "test_run"

        retrieved = repo.get_run("run-test-123")
        assert retrieved is not None
        assert retrieved.dataset_name == "test_dataset"

    def test_save_run_with_cases(self, repo: EvalRepository, sample_run_result) -> None:
        """Test saving run with case results."""
        # Add mock case results
        case1 = MagicMock()
        case1.id = "case-1"
        case1.case_id = "orig-1"
        case1.input = "input1"
        case1.output = "output1"
        case1.expected = "expected1"
        case1.expected_keywords = None
        case1.latency_ms = 50.0
        case1.success = True
        case1.metrics_passed = True
        case1.error = None
        case1.metric_results = {}
        case1.metadata = {}

        case2 = MagicMock()
        case2.id = "case-2"
        case2.case_id = "orig-2"
        case2.input = "input2"
        case2.output = "output2"
        case2.expected = "expected2"
        case2.expected_keywords = None
        case2.latency_ms = 75.0
        case2.success = True
        case2.metrics_passed = False
        case2.error = None
        case2.metric_results = {}
        case2.metadata = {}

        sample_run_result.results = [case1, case2]

        repo.save_run(sample_run_result, save_cases=True)

        run, cases = repo.get_run_with_cases("run-test-123")

        assert run is not None
        assert len(cases) == 2

    def test_list_runs(self, repo: EvalRepository, sample_run_result) -> None:
        """Test listing runs with filters."""
        # Save multiple runs
        sample_run_result.id = "run-1"
        sample_run_result.pass_rate = 0.9
        repo.save_run(sample_run_result, name="run1")

        sample_run_result.id = "run-2"
        sample_run_result.pass_rate = 0.5
        repo.save_run(sample_run_result, name="run2")

        # List all
        all_runs = repo.list_runs()
        assert len(all_runs) == 2

        # Filter by min pass rate
        good_runs = repo.list_runs(min_pass_rate=0.8)
        assert len(good_runs) == 1
        assert good_runs[0].id == "run-1"

        # Filter by dataset
        dataset_runs = repo.list_runs(dataset_name="test_dataset")
        assert len(dataset_runs) == 2

    def test_delete_run(self, repo: EvalRepository, sample_run_result) -> None:
        """Test deleting a run."""
        repo.save_run(sample_run_result)

        deleted = repo.delete_run("run-test-123")
        assert deleted is True

        # Should not exist
        assert repo.get_run("run-test-123") is None

        # Delete non-existent
        deleted = repo.delete_run("non-existent")
        assert deleted is False

    def test_get_history(self, repo: EvalRepository, sample_run_result) -> None:
        """Test getting run history."""
        # Save runs at different times
        sample_run_result.id = "run-old"
        sample_run_result.started_at = datetime.now(timezone.utc) - timedelta(days=5)
        repo.save_run(sample_run_result)

        sample_run_result.id = "run-new"
        sample_run_result.started_at = datetime.now(timezone.utc)
        repo.save_run(sample_run_result)

        history = repo.get_history("test_dataset", days=30)

        assert len(history) == 2
        assert history[0]["run_id"] == "run-old"  # Ordered by date ascending

    def test_get_run_stats(self, repo: EvalRepository, sample_run_result) -> None:
        """Test getting aggregate statistics."""
        sample_run_result.id = "run-1"
        sample_run_result.pass_rate = 0.8
        sample_run_result.avg_latency_ms = 100.0
        sample_run_result.total_cases = 10
        repo.save_run(sample_run_result)

        sample_run_result.id = "run-2"
        sample_run_result.pass_rate = 0.6
        sample_run_result.avg_latency_ms = 200.0
        sample_run_result.total_cases = 10
        repo.save_run(sample_run_result)

        stats = repo.get_run_stats(dataset_name="test_dataset")

        assert stats["total_runs"] == 2
        assert stats["avg_pass_rate"] == 0.7
        assert stats["min_pass_rate"] == 0.6
        assert stats["max_pass_rate"] == 0.8

    def test_save_and_get_baseline(self, repo: EvalRepository, sample_run_result) -> None:
        """Test saving and retrieving baselines."""
        baseline = repo.save_baseline(
            sample_run_result,
            name="v1.0",
        )

        assert baseline.name == "v1.0"
        assert baseline.is_active is True

        # Retrieve by dataset
        retrieved = repo.get_baseline(dataset_name="test_dataset")
        assert retrieved is not None
        assert retrieved.name == "v1.0"

    def test_baseline_deactivation(self, repo: EvalRepository, sample_run_result) -> None:
        """Test that saving new baseline deactivates old one."""
        # Save first baseline
        first = repo.save_baseline(sample_run_result, name="v1.0")
        assert first.is_active is True

        # Save second baseline
        sample_run_result.id = "run-2"
        repo.save_baseline(sample_run_result, name="v2.0")

        # First should be deactivated
        baselines = repo.list_baselines(dataset_name="test_dataset")
        active = [b for b in baselines if b.is_active]
        assert len(active) == 1
        assert active[0].name == "v2.0"

    def test_save_baseline_from_values(self, repo: EvalRepository) -> None:
        """Test saving baseline from raw values."""
        baseline = repo.save_baseline_from_values(
            dataset_name="test_dataset",
            name="manual_baseline",
            metrics={"exact_match": 0.95, "latency": 100.0},
            pass_rate=0.92,
            total_cases=50,
        )

        assert baseline.name == "manual_baseline"
        assert baseline.metrics["exact_match"] == 0.95

    def test_list_baselines(self, repo: EvalRepository, sample_run_result) -> None:
        """Test listing baselines."""
        repo.save_baseline(sample_run_result, name="v1.0")

        sample_run_result.id = "run-2"
        repo.save_baseline(sample_run_result, name="v2.0")

        all_baselines = repo.list_baselines(dataset_name="test_dataset")
        assert len(all_baselines) == 2

        active_only = repo.list_baselines(
            dataset_name="test_dataset",
            active_only=True,
        )
        assert len(active_only) == 1

    def test_delete_baseline(self, repo: EvalRepository, sample_run_result) -> None:
        """Test deleting a baseline."""
        baseline = repo.save_baseline(sample_run_result, name="to_delete")

        deleted = repo.delete_baseline(baseline.id)
        assert deleted is True

        assert repo.get_baseline(baseline_id=baseline.id) is None

    def test_get_cases(self, repo: EvalRepository, sample_run_result) -> None:
        """Test getting case results."""
        case1 = MagicMock()
        case1.id = "case-1"
        case1.case_id = "orig-1"
        case1.input = "input1"
        case1.output = "output1"
        case1.expected = "expected1"
        case1.expected_keywords = None
        case1.latency_ms = 50.0
        case1.success = True
        case1.metrics_passed = True
        case1.error = None
        case1.metric_results = {}
        case1.metadata = {}

        case2 = MagicMock()
        case2.id = "case-2"
        case2.case_id = "orig-2"
        case2.input = "input2"
        case2.output = "output2"
        case2.expected = "expected2"
        case2.expected_keywords = None
        case2.latency_ms = 75.0
        case2.success = True
        case2.metrics_passed = False
        case2.error = None
        case2.metric_results = {}
        case2.metadata = {}

        sample_run_result.results = [case1, case2]
        repo.save_run(sample_run_result, save_cases=True)

        # Get all cases
        all_cases = repo.get_cases("run-test-123")
        assert len(all_cases) == 2

        # Get only passed
        passed = repo.get_cases("run-test-123", passed_only=True)
        assert len(passed) == 1

        # Get only failed
        failed = repo.get_cases("run-test-123", failed_only=True)
        assert len(failed) == 1

    def test_get_failed_cases(self, repo: EvalRepository, sample_run_result) -> None:
        """Test getting failed cases across runs."""
        case = MagicMock()
        case.id = "failed-case"
        case.case_id = "orig-failed"
        case.input = "input"
        case.output = "wrong"
        case.expected = "expected"
        case.expected_keywords = None
        case.latency_ms = 50.0
        case.success = True
        case.metrics_passed = False
        case.error = None
        case.metric_results = {}
        case.metadata = {}

        sample_run_result.results = [case]
        repo.save_run(sample_run_result, save_cases=True)

        failed = repo.get_failed_cases(dataset_name="test_dataset")
        assert len(failed) == 1
        assert failed[0].case_id == "orig-failed"


class TestRunnerStorageIntegration:
    """Tests for Runner integration with storage."""

    @pytest.fixture
    def repo(self):
        """Create a repository with in-memory database."""
        repo = EvalRepository("sqlite:///:memory:")
        repo.initialize()
        return repo

    @pytest.mark.asyncio
    async def test_runner_auto_save(self, repo: EvalRepository) -> None:
        """Test runner auto-saves to repository."""
        from evalops import EvalCase, EvalDataset, EvalRunner, ExactMatch

        runner = EvalRunner(
            repository=repo,
            auto_save=True,
            enable_observability=False,
        )

        dataset = EvalDataset(
            name="test_dataset",
            cases=[
                EvalCase(input="a", expected="A"),
                EvalCase(input="b", expected="B"),
            ],
        )

        result = await runner.evaluate(
            dataset=dataset,
            target=lambda x: x.upper(),
            metrics=[ExactMatch()],
            run_name="auto_save_test",
            tags=["test"],
        )

        # Check run was saved
        runs = repo.list_runs()
        assert len(runs) == 1
        assert runs[0].id == result.id
        assert runs[0].name == "auto_save_test"
        assert runs[0].tags == ["test"]

    @pytest.mark.asyncio
    async def test_runner_explicit_save(self, repo: EvalRepository) -> None:
        """Test runner with explicit save=True."""
        from evalops import EvalCase, EvalDataset, EvalRunner

        runner = EvalRunner(
            repository=repo,
            auto_save=False,  # Auto-save disabled
            enable_observability=False,
        )

        dataset = EvalDataset(
            name="test",
            cases=[EvalCase(input="test")],
        )

        # Without save=True, should not save
        await runner.evaluate(
            dataset=dataset,
            target=lambda x: x,
        )
        assert len(repo.list_runs()) == 0

        # With save=True, should save
        await runner.evaluate(
            dataset=dataset,
            target=lambda x: x,
            save=True,
        )
        assert len(repo.list_runs()) == 1

    @pytest.mark.asyncio
    async def test_runner_no_repository(self) -> None:
        """Test runner without repository doesn't crash."""
        from evalops import EvalCase, EvalDataset, EvalRunner

        runner = EvalRunner(
            repository=None,
            auto_save=True,  # Should be no-op without repository
            enable_observability=False,
        )

        dataset = EvalDataset(
            name="test",
            cases=[EvalCase(input="test")],
        )

        # Should complete without error
        result = await runner.evaluate(
            dataset=dataset,
            target=lambda x: x,
        )
        assert result.total_cases == 1
