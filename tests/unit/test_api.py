"""Tests for the EvalOps FastAPI application."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from evalops.api import app as fastapi_app
from evalops.api.app import get_db_manager, get_repository


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    repo = MagicMock()
    repo.list_runs.return_value = []
    repo.list_baselines.return_value = []
    repo.get_run.return_value = None
    repo.get_baseline.return_value = None
    repo.get_history.return_value = []
    repo.get_run_stats.return_value = {
        "total_runs": 0,
        "total_cases": 0,
        "avg_pass_rate": 0.0,
        "avg_latency_ms": 0.0,
        "min_pass_rate": 0.0,
        "max_pass_rate": 0.0,
        "first_run_at": None,
        "last_run_at": None,
    }
    return repo


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    manager = MagicMock()
    manager.check_health.return_value = {
        "healthy": True,
        "version": 1,
        "current_version": 1,
        "needs_migration": False,
        "tables": ["eval_runs", "eval_cases"],
    }
    manager.initialize.return_value = 1
    manager.get_version.return_value = 1
    manager.get_migration_history.return_value = []
    return manager


@pytest.fixture
def client(mock_repository, mock_db_manager):
    """Create a test client with mocked dependencies."""
    # Override dependencies
    fastapi_app.dependency_overrides[get_repository] = lambda: mock_repository
    fastapi_app.dependency_overrides[get_db_manager] = lambda: mock_db_manager

    yield TestClient(fastapi_app)

    # Clean up overrides
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def sample_run_record():
    """Create a sample run record."""
    record = MagicMock()
    record.id = "test-run-id-12345678"
    record.name = "test_run"
    record.dataset_id = "dataset-id-123"
    record.dataset_name = "test_dataset"
    record.total_cases = 10
    record.successful_cases = 9
    record.pass_rate = 0.9
    record.success_rate = 0.9
    record.avg_latency_ms = 150.0
    record.total_latency_ms = 1350.0
    record.metrics_summary = {"exact_match": {"score": 0.9, "passed": True}}
    record.tags = ["test", "unit"]
    record.metadata_json = {"env": "test"}
    record.started_at = datetime.now(timezone.utc)
    record.completed_at = datetime.now(timezone.utc)
    record.created_at = datetime.now(timezone.utc)
    return record


@pytest.fixture
def sample_baseline_record():
    """Create a sample baseline record."""
    record = MagicMock()
    record.id = "baseline-id-12345678"
    record.name = "test_baseline"
    record.dataset_name = "test_dataset"
    record.source_run_id = "source-run-id"
    record.metrics = {"exact_match": 0.9, "pass_rate": 0.9}
    record.pass_rate = 0.9
    record.total_cases = 10
    record.is_active = True
    record.created_at = datetime.now(timezone.utc)
    record.expires_at = None
    record.metadata_json = None
    return record


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "EvalOps API"
        assert "version" in data
        assert data["docs_url"] == "/docs"


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_healthy(self, client, mock_db_manager):
        """Test health endpoint with healthy database."""
        mock_db_manager.check_health.return_value = {
            "healthy": True,
            "version": 1,
            "current_version": 1,
            "needs_migration": False,
            "tables": ["eval_runs", "eval_cases"],
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["healthy"] is True
        assert "uptime_seconds" in data

    def test_health_degraded(self, client, mock_db_manager):
        """Test health endpoint with degraded database."""
        mock_db_manager.check_health.return_value = {
            "healthy": False,
            "error": "Connection timeout",
        }

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["database"]["healthy"] is False


class TestRunEndpoints:
    """Tests for run-related endpoints."""

    def test_list_runs_empty(self, client, mock_repository):
        """Test listing runs when none exist."""
        response = client.get("/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_list_runs_with_results(self, client, mock_repository, sample_run_record):
        """Test listing runs with existing data."""
        mock_repository.list_runs.return_value = [sample_run_record]

        response = client.get("/runs")
        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["name"] == "test_run"
        assert data["runs"][0]["pass_rate"] == 0.9

    def test_list_runs_with_filters(self, client, mock_repository):
        """Test listing runs with query filters."""
        response = client.get(
            "/runs",
            params={
                "dataset_name": "test_dataset",
                "min_pass_rate": 0.8,
                "days": 7,
                "limit": 10,
            },
        )
        assert response.status_code == 200

        # Verify filters were passed to repository (first call has the user's limit)
        # The endpoint makes two calls: one with user limit, one to count total
        first_call_kwargs = mock_repository.list_runs.call_args_list[0][1]
        assert first_call_kwargs["dataset_name"] == "test_dataset"
        assert first_call_kwargs["min_pass_rate"] == 0.8
        assert first_call_kwargs["limit"] == 10

    def test_get_run_not_found(self, client, mock_repository):
        """Test getting a non-existent run."""
        response = client.get("/runs/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_run_found(self, client, mock_repository, sample_run_record):
        """Test getting an existing run."""
        mock_repository.get_run.return_value = sample_run_record

        response = client.get("/runs/test-run-id")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_run"
        assert data["pass_rate"] == 0.9
        assert "metrics_summary" in data

    def test_get_run_cases(self, client, mock_repository, sample_run_record):
        """Test getting cases for a run."""
        mock_case = MagicMock()
        mock_case.id = "case-id-1"
        mock_case.case_id = "original-case-1"
        mock_case.input = "test input"
        mock_case.output = "test output"
        mock_case.expected = "test expected"
        mock_case.expected_keywords = None
        mock_case.latency_ms = 50.0
        mock_case.success = True
        mock_case.metrics_passed = True
        mock_case.error = None
        mock_case.metric_results = {}
        mock_case.metadata_json = None

        mock_cases_query = MagicMock()
        mock_cases_query.count.return_value = 1
        mock_cases_query.offset.return_value = mock_cases_query
        mock_cases_query.limit.return_value = mock_cases_query
        mock_cases_query.filter_by.return_value = mock_cases_query
        mock_cases_query.all.return_value = [mock_case]

        sample_run_record.cases = mock_cases_query
        mock_repository.get_run.return_value = sample_run_record

        response = client.get("/runs/test-run-id/cases")
        assert response.status_code == 200
        data = response.json()
        assert len(data["cases"]) == 1
        assert data["cases"][0]["input"] == "test input"

    def test_delete_run_not_found(self, client, mock_repository):
        """Test deleting a non-existent run."""
        response = client.delete("/runs/nonexistent-id")
        assert response.status_code == 404

    def test_delete_run_success(self, client, mock_repository, sample_run_record):
        """Test successfully deleting a run."""
        mock_repository.get_run.return_value = sample_run_record

        response = client.delete("/runs/test-run-id")
        assert response.status_code == 204
        mock_repository.delete_run.assert_called_once_with("test-run-id")


class TestBaselineEndpoints:
    """Tests for baseline-related endpoints."""

    def test_list_baselines_empty(self, client, mock_repository):
        """Test listing baselines when none exist."""
        response = client.get("/baselines")
        assert response.status_code == 200
        data = response.json()
        assert data["baselines"] == []
        assert data["total"] == 0

    def test_list_baselines_with_results(self, client, mock_repository, sample_baseline_record):
        """Test listing baselines with existing data."""
        mock_repository.list_baselines.return_value = [sample_baseline_record]

        response = client.get("/baselines")
        assert response.status_code == 200
        data = response.json()
        assert len(data["baselines"]) == 1
        assert data["baselines"][0]["name"] == "test_baseline"

    def test_get_baseline_not_found(self, client, mock_repository):
        """Test getting a non-existent baseline."""
        response = client.get("/baselines/nonexistent-id")
        assert response.status_code == 404

    def test_get_baseline_found(self, client, mock_repository, sample_baseline_record):
        """Test getting an existing baseline."""
        mock_repository.get_baseline.return_value = sample_baseline_record

        response = client.get("/baselines/test-baseline-id")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_baseline"
        assert data["is_active"] is True

    def test_create_baseline_run_not_found(self, client, mock_repository):
        """Test creating baseline with non-existent source run."""
        response = client.post(
            "/baselines",
            json={"run_id": "nonexistent-run", "name": "test_baseline"},
        )
        assert response.status_code == 404
        assert "Source run not found" in response.json()["detail"]

    def test_create_baseline_success(
        self, client, mock_repository, sample_run_record, sample_baseline_record
    ):
        """Test successfully creating a baseline."""
        mock_repository.get_run.return_value = sample_run_record
        mock_repository.save_baseline.return_value = sample_baseline_record

        response = client.post(
            "/baselines",
            json={"run_id": "test-run-id", "name": "new_baseline"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_baseline"

    def test_delete_baseline_not_found(self, client, mock_repository):
        """Test deleting a non-existent baseline."""
        response = client.delete("/baselines/nonexistent-id")
        assert response.status_code == 404

    def test_delete_baseline_success(self, client, mock_repository, sample_baseline_record):
        """Test successfully deleting a baseline."""
        mock_repository.get_baseline.return_value = sample_baseline_record

        response = client.delete("/baselines/test-baseline-id")
        assert response.status_code == 204
        mock_repository.delete_baseline.assert_called_once()


class TestDriftEndpoints:
    """Tests for drift detection endpoints."""

    def test_drift_no_baseline(self, client, mock_repository):
        """Test drift check with no baseline."""
        response = client.get(
            "/drift",
            params={"dataset_name": "test_dataset"},
        )
        assert response.status_code == 404
        assert "No active baseline" in response.json()["detail"]

    def test_drift_no_history(self, client, mock_repository, sample_baseline_record):
        """Test drift check with no run history."""
        mock_repository.get_baseline.return_value = sample_baseline_record
        mock_repository.get_history.return_value = []

        response = client.get(
            "/drift",
            params={"dataset_name": "test_dataset"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["drift_detected"] is False
        assert data["overall_health"] == "unknown"

    def test_drift_healthy(self, client, mock_repository, sample_baseline_record):
        """Test drift check with healthy metrics."""
        mock_repository.get_baseline.return_value = sample_baseline_record
        mock_repository.get_history.return_value = [
            {
                "run_id": "run1",
                "timestamp": datetime.now(timezone.utc),
                "pass_rate": 0.89,
                "avg_latency_ms": 100,
                "metrics": {"exact_match": 0.88},
            },
            {
                "run_id": "run2",
                "timestamp": datetime.now(timezone.utc),
                "pass_rate": 0.91,
                "avg_latency_ms": 95,
                "metrics": {"exact_match": 0.90},
            },
        ]

        response = client.get(
            "/drift",
            params={"dataset_name": "test_dataset"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["overall_health"] in ["healthy", "improving"]

    def test_drift_with_custom_thresholds(self, client, mock_repository, sample_baseline_record):
        """Test drift check with custom thresholds."""
        mock_repository.get_baseline.return_value = sample_baseline_record
        mock_repository.get_history.return_value = [
            {"run_id": "run1", "pass_rate": 0.7, "metrics": {"exact_match": 0.7}},
        ]

        response = client.get(
            "/drift",
            params={
                "dataset_name": "test_dataset",
                "warning_threshold": 0.1,
                "critical_threshold": 0.2,
            },
        )
        assert response.status_code == 200

    def test_drift_post_endpoint(self, client, mock_repository, sample_baseline_record):
        """Test POST version of drift check."""
        mock_repository.get_baseline.return_value = sample_baseline_record
        mock_repository.get_history.return_value = []

        response = client.post(
            "/drift/check",
            json={
                "dataset_name": "test_dataset",
                "days": 7,
                "warning_threshold": 0.05,
                "critical_threshold": 0.10,
            },
        )
        assert response.status_code == 200


class TestHistoryEndpoints:
    """Tests for history endpoints."""

    def test_get_history(self, client, mock_repository):
        """Test getting run history."""
        mock_repository.get_history.return_value = [
            {
                "run_id": "run1",
                "timestamp": datetime.now(timezone.utc),
                "pass_rate": 0.9,
                "avg_latency_ms": 100,
                "metrics": {"exact_match": 0.9},
            },
        ]
        mock_repository.get_run_stats.return_value = {
            "total_runs": 1,
            "total_cases": 10,
            "avg_pass_rate": 0.9,
            "avg_latency_ms": 100,
            "min_pass_rate": 0.9,
            "max_pass_rate": 0.9,
            "first_run_at": datetime.now(timezone.utc),
            "last_run_at": datetime.now(timezone.utc),
        }

        response = client.get("/history/test_dataset")
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_name"] == "test_dataset"
        assert len(data["points"]) == 1
        assert data["stats"]["total_runs"] == 1

    def test_get_stats(self, client, mock_repository):
        """Test getting overall statistics."""
        mock_repository.get_run_stats.return_value = {
            "total_runs": 50,
            "total_cases": 500,
            "avg_pass_rate": 0.85,
            "avg_latency_ms": 120,
            "min_pass_rate": 0.6,
            "max_pass_rate": 1.0,
            "first_run_at": datetime.now(timezone.utc),
            "last_run_at": datetime.now(timezone.utc),
        }

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_runs"] == 50
        assert data["avg_pass_rate"] == 0.85


class TestDatabaseEndpoints:
    """Tests for database management endpoints."""

    def test_init_database(self, client, mock_db_manager):
        """Test database initialization endpoint."""
        mock_db_manager.initialize.return_value = 1

        response = client.post("/db/init")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "initialized"
        assert data["version"] == 1

    def test_get_migrations(self, client, mock_db_manager):
        """Test getting migration history."""
        mock_db_manager.get_version.return_value = 1
        mock_db_manager.get_migration_history.return_value = [
            {"version": 1, "applied_at": "2024-01-01T00:00:00", "description": "Initial"}
        ]

        response = client.get("/db/migrations")
        assert response.status_code == 200
        data = response.json()
        assert data["current_version"] == 1
        assert len(data["migrations"]) == 1


class TestCreateRunEndpoint:
    """Tests for the create run endpoint."""

    def test_create_run_dataset_not_found(self, client, mock_repository):
        """Test creating run with non-existent dataset."""
        response = client.post(
            "/runs",
            json={
                "dataset_path": "/nonexistent/dataset.json",
                "target_module": "json:dumps",
                "metrics": ["exact_match"],
            },
        )
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]

    def test_create_run_invalid_target(self, client, mock_repository):
        """Test creating run with invalid target."""
        # Create a temporary dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"name": "test", "cases": [{"input": "test", "expected": "test"}]},
                f,
            )
            dataset_path = f.name

        try:
            response = client.post(
                "/runs",
                json={
                    "dataset_path": dataset_path,
                    "target_module": "nonexistent_module:func",
                    "metrics": [],
                },
            )
            assert response.status_code == 400
            assert "Failed to load target" in response.json()["detail"]
        finally:
            Path(dataset_path).unlink(missing_ok=True)


class TestSchemaValidation:
    """Tests for request/response schema validation."""

    def test_run_request_validation(self, client, mock_repository):
        """Test run request schema validation."""
        # Missing required field
        response = client.post(
            "/runs",
            json={"metrics": ["exact_match"]},
        )
        assert response.status_code == 422  # Validation error

    def test_baseline_request_validation(self, client, mock_repository):
        """Test baseline request schema validation."""
        # Missing required fields
        response = client.post(
            "/baselines",
            json={},
        )
        assert response.status_code == 422

    def test_drift_request_missing_dataset(self, client, mock_repository):
        """Test drift request requires dataset_name."""
        response = client.get("/drift")
        assert response.status_code == 422


class TestLoadMetrics:
    """Tests for the load_metrics helper function."""

    def test_load_semantic_similarity(self):
        """Test that semantic_similarity metric is available via the API."""
        from evalops.api.app import load_metrics

        metrics = load_metrics(["semantic_similarity"])
        assert len(metrics) == 1
        assert metrics[0].name == "semantic_similarity"

    def test_load_all_metrics(self):
        """Test loading all available metrics."""
        from evalops.api.app import load_metrics

        all_metric_names = [
            "exact_match",
            "contains_keywords",
            "latency",
            "token_cost",
            "semantic_similarity",
        ]
        metrics = load_metrics(all_metric_names)
        assert len(metrics) == 5
        loaded_names = [m.name for m in metrics]
        for name in all_metric_names:
            assert name in loaded_names
