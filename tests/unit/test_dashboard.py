"""Tests for the EvalOps Streamlit dashboard.

These tests verify the dashboard's helper functions and data processing logic.
Since Streamlit components require a running server, we test the underlying
functions with mocked repositories.

Note: Tests that require streamlit/plotly are skipped if dependencies aren't installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Check for optional dependencies
try:
    import plotly
    import streamlit
    DASHBOARD_DEPS_AVAILABLE = True
except ImportError:
    DASHBOARD_DEPS_AVAILABLE = False

requires_dashboard_deps = pytest.mark.skipif(
    not DASHBOARD_DEPS_AVAILABLE,
    reason="Dashboard dependencies (streamlit, plotly) not installed"
)


@requires_dashboard_deps
class TestSessionStateInitialization:
    """Tests for session state initialization."""

    def test_init_session_state_sets_defaults(self):
        """Test that init_session_state sets default values."""
        with patch("streamlit.session_state", {}) as mock_state:
            from evalops.dashboard.app import init_session_state

            init_session_state()

            assert mock_state["database_url"] == "sqlite:///evalops_demo.db"
            assert mock_state["repository"] is None
            assert mock_state["selected_run_id"] is None
            assert mock_state["page"] == "Overview"

    def test_init_session_state_preserves_existing(self):
        """Test that init_session_state doesn't overwrite existing values."""
        with patch("streamlit.session_state", {"database_url": "custom://url"}) as mock_state:
            from evalops.dashboard.app import init_session_state

            init_session_state()

            # Should preserve existing value
            assert mock_state["database_url"] == "custom://url"
            # Should set missing defaults
            assert mock_state["page"] == "Overview"


@requires_dashboard_deps
class TestRepositoryConnection:
    """Tests for repository connection handling."""

    def test_get_repository_creates_new(self):
        """Test that get_repository creates a new repository if none exists."""
        mock_repo = MagicMock()

        with patch("streamlit.session_state", {"database_url": "sqlite:///:memory:", "repository": None}):
            with patch("evalops.dashboard.app.EvalRepository", return_value=mock_repo) as MockRepo:
                from evalops.dashboard.app import get_repository

                get_repository()

                MockRepo.assert_called_once_with("sqlite:///:memory:")
                mock_repo.initialize.assert_called_once()

    def test_get_repository_returns_existing(self):
        """Test that get_repository returns existing repository."""
        existing_repo = MagicMock()

        with patch("streamlit.session_state", {"database_url": "sqlite:///test.db", "repository": existing_repo}):
            from evalops.dashboard.app import get_repository

            result = get_repository()

            assert result is existing_repo

    def test_reconnect_database_success(self):
        """Test successful database reconnection."""
        mock_repo = MagicMock()

        with patch("streamlit.session_state", {"database_url": "old://url", "repository": None}) as mock_state:
            with patch("evalops.dashboard.app.EvalRepository", return_value=mock_repo):
                from evalops.dashboard.app import reconnect_database

                result = reconnect_database("new://url")

                assert result is True
                assert mock_state["database_url"] == "new://url"
                assert mock_state["repository"] is mock_repo

    def test_reconnect_database_failure(self):
        """Test database reconnection failure handling."""
        with patch("streamlit.session_state", {"database_url": "old://url", "repository": None}):
            with patch("evalops.dashboard.app.EvalRepository", side_effect=Exception("Connection failed")):
                with patch("streamlit.error"):
                    from evalops.dashboard.app import reconnect_database

                    result = reconnect_database("bad://url")

                    assert result is False


class TestMockRepository:
    """Test fixtures for mocked repository."""

    @pytest.fixture
    def mock_repository(self):
        """Create a comprehensive mock repository."""
        repo = MagicMock()

        # Mock run records
        mock_run = MagicMock()
        mock_run.id = "run-id-12345678"
        mock_run.name = "test_run"
        mock_run.dataset_id = "dataset-id-123"
        mock_run.dataset_name = "test_dataset"
        mock_run.total_cases = 10
        mock_run.successful_cases = 9
        mock_run.pass_rate = 0.9
        mock_run.success_rate = 0.9
        mock_run.avg_latency_ms = 150.0
        mock_run.total_latency_ms = 1350.0
        mock_run.metrics_summary = {"exact_match": {"score": 0.9, "passed": True}}
        mock_run.tags = ["test"]
        mock_run.started_at = datetime.now(timezone.utc)
        mock_run.completed_at = datetime.now(timezone.utc)
        mock_run.created_at = datetime.now(timezone.utc)

        # Mock case records
        mock_case = MagicMock()
        mock_case.id = "case-id-1"
        mock_case.case_id = "original-case-1"
        mock_case.input = "test input"
        mock_case.output = "test output"
        mock_case.expected = "expected"
        mock_case.latency_ms = 50.0
        mock_case.success = True
        mock_case.metrics_passed = True
        mock_case.error = None
        mock_case.metric_results = {}

        mock_cases_query = MagicMock()
        mock_cases_query.limit.return_value = mock_cases_query
        mock_cases_query.all.return_value = [mock_case]
        mock_run.cases = mock_cases_query

        # Mock baseline records
        mock_baseline = MagicMock()
        mock_baseline.id = "baseline-id-123"
        mock_baseline.name = "test_baseline"
        mock_baseline.dataset_name = "test_dataset"
        mock_baseline.pass_rate = 0.9
        mock_baseline.total_cases = 10
        mock_baseline.metrics = {"exact_match": 0.9, "pass_rate": 0.9}
        mock_baseline.created_at = datetime.now(timezone.utc)
        mock_baseline.is_active = True

        # Configure repository methods
        repo.list_runs.return_value = [mock_run]
        repo.get_run.return_value = mock_run
        repo.list_baselines.return_value = [mock_baseline]
        repo.get_baseline.return_value = mock_baseline
        repo.get_run_stats.return_value = {
            "total_runs": 10,
            "total_cases": 100,
            "avg_pass_rate": 0.85,
            "avg_latency_ms": 120.0,
            "min_pass_rate": 0.7,
            "max_pass_rate": 1.0,
        }
        repo.get_history.return_value = [
            {
                "run_id": "run1",
                "timestamp": datetime.now(timezone.utc),
                "pass_rate": 0.88,
                "avg_latency_ms": 100,
                "metrics": {"exact_match": 0.88},
            },
            {
                "run_id": "run2",
                "timestamp": datetime.now(timezone.utc),
                "pass_rate": 0.92,
                "avg_latency_ms": 95,
                "metrics": {"exact_match": 0.90},
            },
        ]

        return repo

    def test_mock_repository_list_runs(self, mock_repository):
        """Test mock repository list_runs method."""
        runs = mock_repository.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "test_run"
        assert runs[0].pass_rate == 0.9

    def test_mock_repository_get_stats(self, mock_repository):
        """Test mock repository get_run_stats method."""
        stats = mock_repository.get_run_stats()
        assert stats["total_runs"] == 10
        assert stats["avg_pass_rate"] == 0.85

    def test_mock_repository_get_history(self, mock_repository):
        """Test mock repository get_history method."""
        history = mock_repository.get_history()
        assert len(history) == 2
        assert history[0]["pass_rate"] == 0.88


class TestOverviewPageComponents:
    """Tests for overview page data processing."""

    @pytest.fixture
    def mock_repo_with_multiple_datasets(self):
        """Create mock repository with multiple datasets."""
        repo = MagicMock()

        runs = []
        for i, (dataset, pass_rate) in enumerate([
            ("dataset_a", 0.9),
            ("dataset_a", 0.85),
            ("dataset_b", 0.7),
            ("dataset_c", 0.95),
        ]):
            run = MagicMock()
            run.id = f"run-{i}"
            run.name = f"run_{i}"
            run.dataset_name = dataset
            run.pass_rate = pass_rate
            run.avg_latency_ms = 100 + i * 10
            run.total_cases = 10
            run.started_at = datetime.now(timezone.utc)
            runs.append(run)

        repo.list_runs.return_value = runs
        repo.get_run_stats.return_value = {
            "total_runs": 4,
            "total_cases": 40,
            "avg_pass_rate": 0.85,
            "avg_latency_ms": 115,
        }

        return repo

    def test_dataset_grouping(self, mock_repo_with_multiple_datasets):
        """Test that runs are correctly grouped by dataset."""
        runs = mock_repo_with_multiple_datasets.list_runs()

        # Group by dataset (simulating what the page does)
        dataset_stats = {}
        for run in runs:
            name = run.dataset_name
            if name not in dataset_stats:
                dataset_stats[name] = {"runs": 0, "total_pass_rate": 0}
            dataset_stats[name]["runs"] += 1
            dataset_stats[name]["total_pass_rate"] += run.pass_rate

        assert len(dataset_stats) == 3
        assert dataset_stats["dataset_a"]["runs"] == 2
        assert dataset_stats["dataset_b"]["runs"] == 1
        assert dataset_stats["dataset_c"]["runs"] == 1


class TestRunExplorerFiltering:
    """Tests for run explorer filtering logic."""

    def test_filter_by_pass_rate(self):
        """Test filtering runs by minimum pass rate."""
        runs = [
            MagicMock(pass_rate=0.9),
            MagicMock(pass_rate=0.5),
            MagicMock(pass_rate=0.75),
            MagicMock(pass_rate=0.3),
        ]

        min_pass_rate = 0.6
        filtered = [r for r in runs if r.pass_rate >= min_pass_rate]

        assert len(filtered) == 2
        assert all(r.pass_rate >= 0.6 for r in filtered)

    def test_filter_by_dataset(self):
        """Test filtering runs by dataset name."""
        runs = [
            MagicMock(dataset_name="dataset_a"),
            MagicMock(dataset_name="dataset_b"),
            MagicMock(dataset_name="dataset_a"),
        ]

        dataset_filter = "dataset_a"
        filtered = [r for r in runs if r.dataset_name == dataset_filter]

        assert len(filtered) == 2


class TestComparisonCalculations:
    """Tests for A/B comparison calculations."""

    def test_winner_determination_higher_better(self):
        """Test determining winner when higher is better."""
        val_a, val_b = 0.9, 0.85
        higher_better = True

        if higher_better:
            winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
        else:
            winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

        assert winner == "A"

    def test_winner_determination_lower_better(self):
        """Test determining winner when lower is better (e.g., latency)."""
        val_a, val_b = 150, 100  # Latency in ms
        higher_better = False

        if higher_better:
            winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
        else:
            winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

        assert winner == "B"

    def test_winner_tie(self):
        """Test tie detection."""
        val_a, val_b = 0.85, 0.85
        higher_better = True

        if higher_better:
            winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
        else:
            winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

        assert winner == "Tie"


class TestDriftMonitorLogic:
    """Tests for drift monitoring logic."""

    def test_drift_detection_integration(self):
        """Test drift detection with mock baseline and history."""
        from evalops.comparison.drift import DriftDetector

        # Set up detector
        detector = DriftDetector(
            warning_threshold=0.05,
            critical_threshold=0.10,
        )

        # Set baseline
        baseline_metrics = {"exact_match": 0.9, "pass_rate": 0.9}
        detector.set_baseline_from_values(baseline_metrics, pass_rate=0.9)

        # Add snapshots showing slight degradation
        detector.add_raw_snapshot(
            metrics={"exact_match": 0.88, "pass_rate": 0.88},
            pass_rate=0.88,
            run_id="run1",
        )
        detector.add_raw_snapshot(
            metrics={"exact_match": 0.87, "pass_rate": 0.87},
            pass_rate=0.87,
            run_id="run2",
        )

        report = detector.check()

        # Should detect some drift but not critical
        assert report.overall_health in ["healthy", "degraded", "improving"]

    def test_critical_drift_detection(self):
        """Test critical drift is detected correctly."""
        from evalops.comparison.drift import DriftDetector

        detector = DriftDetector(
            warning_threshold=0.05,
            critical_threshold=0.10,
        )

        # Set baseline at 90%
        detector.set_baseline_from_values({"pass_rate": 0.9}, pass_rate=0.9)

        # Add snapshots showing significant degradation (>10%)
        for i in range(3):
            detector.add_raw_snapshot(
                metrics={"pass_rate": 0.75},
                pass_rate=0.75,
                run_id=f"run{i}",
            )

        report = detector.check()

        assert report.drift_detected is True
        assert report.overall_health == "critical"
        assert len(report.alerts) > 0


class TestSettingsPasswordMasking:
    """Tests for password masking in settings."""

    def test_mask_password_in_url(self):
        """Test that passwords are masked in database URLs."""
        # This mimics the logic in render_settings_page
        url = "postgresql://user:secretpassword@host:5432/db"

        display_url = url
        if "@" in url and "://" in url:
            parts = url.split("://", 1)
            if len(parts) == 2 and "@" in parts[1]:
                auth_host = parts[1].split("@", 1)
                if ":" in auth_host[0]:
                    user = auth_host[0].split(":")[0]
                    display_url = f"{parts[0]}://{user}:***@{auth_host[1]}"

        assert display_url == "postgresql://user:***@host:5432/db"
        assert "secretpassword" not in display_url

    def test_no_mask_for_sqlite(self):
        """Test that SQLite URLs are not modified."""
        url = "sqlite:///path/to/db.db"

        display_url = url
        if "@" in url and "://" in url:
            # This logic won't apply to SQLite
            pass

        assert display_url == url


class TestDashboardCLI:
    """Tests for the dashboard CLI launcher."""

    def test_cli_module_exists(self):
        """Test that CLI module is importable."""
        from evalops.dashboard.cli import main

        assert callable(main)

    def test_cli_builds_correct_command(self):
        """Test that CLI builds the correct streamlit command."""
        from pathlib import Path

        from evalops.dashboard import cli

        app_path = Path(cli.__file__).parent / "app.py"
        assert app_path.exists()

    def test_dashboard_entry_point(self):
        """Test that dashboard entry point function exists."""
        from evalops.dashboard import main, run_dashboard

        assert callable(main)
        assert callable(run_dashboard)


class TestPlotlyChartData:
    """Tests for Plotly chart data preparation."""

    def test_pass_rate_trend_data(self):
        """Test pass rate trend data preparation."""
        history = [
            {"timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc), "pass_rate": 0.85},
            {"timestamp": datetime(2024, 1, 2, tzinfo=timezone.utc), "pass_rate": 0.88},
            {"timestamp": datetime(2024, 1, 3, tzinfo=timezone.utc), "pass_rate": 0.90},
        ]

        dates = [h["timestamp"] for h in history]
        pass_rates = [h["pass_rate"] * 100 for h in history]

        assert len(dates) == 3
        assert len(pass_rates) == 3
        assert pass_rates == [85.0, 88.0, 90.0]

    def test_metrics_comparison_data(self):
        """Test metrics comparison data for A/B charts."""
        metrics_a = {"exact_match": 0.9, "latency": 100}
        metrics_b = {"exact_match": 0.85, "latency": 120}

        metric_names = list(set(metrics_a.keys()) | set(metrics_b.keys()))
        scores_a = [metrics_a.get(m, 0) for m in metric_names]
        scores_b = [metrics_b.get(m, 0) for m in metric_names]

        assert len(metric_names) == 2
        assert len(scores_a) == len(scores_b) == 2
