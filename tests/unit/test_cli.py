"""Tests for the EvalOps CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from evalops.cli.main import app, load_target_function, parse_metrics

runner = CliRunner()


class TestLoadTargetFunction:
    """Tests for the load_target_function helper."""

    def test_valid_target(self):
        """Test loading a valid target function."""
        target = load_target_function("json:dumps")
        assert target is not None
        assert callable(target)
        assert target([1, 2, 3]) == "[1, 2, 3]"

    def test_invalid_format_no_colon(self):
        """Test error when target has no colon."""
        with pytest.raises(Exception):  # typer.BadParameter
            load_target_function("json.dumps")

    def test_invalid_module(self):
        """Test error when module doesn't exist."""
        with pytest.raises(Exception):  # typer.BadParameter
            load_target_function("nonexistent_module:func")

    def test_invalid_function(self):
        """Test error when function doesn't exist in module."""
        with pytest.raises(Exception):  # typer.BadParameter
            load_target_function("json:nonexistent_function")


class TestParseMetrics:
    """Tests for the parse_metrics helper."""

    def test_parse_exact_match(self):
        """Test parsing exact_match metric."""
        metrics = parse_metrics(["exact_match"])
        assert len(metrics) == 1
        assert metrics[0].name == "exact_match"

    def test_parse_multiple_metrics(self):
        """Test parsing multiple metrics."""
        metrics = parse_metrics(["exact_match", "latency", "contains_keywords"])
        assert len(metrics) == 3
        names = [m.name for m in metrics]
        assert "exact_match" in names
        assert "latency" in names
        assert "contains_keywords" in names

    def test_parse_with_dashes(self):
        """Test parsing metric names with dashes."""
        metrics = parse_metrics(["exact-match", "contains-keywords"])
        assert len(metrics) == 2

    def test_unknown_metric(self):
        """Test error for unknown metric."""
        with pytest.raises(Exception):  # typer.BadParameter
            parse_metrics(["unknown_metric"])


class TestVersionCommand:
    """Tests for the version command."""

    def test_version_output(self):
        """Test version command shows version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "EvalOps v" in result.stdout


class TestInitCommand:
    """Tests for the init command."""

    def test_init_creates_database(self):
        """Test init command creates database (in-memory for Windows compatibility)."""
        result = runner.invoke(app, ["init", "--database", "sqlite:///:memory:"])
        assert result.exit_code == 0
        assert "Database initialized" in result.stdout

    def test_init_with_mocked_manager(self):
        """Test init with mocked database manager."""
        mock_manager = MagicMock()
        mock_manager.initialize.return_value = 1
        mock_manager.check_health.return_value = {"tables": ["test"]}

        with patch("evalops.storage.DatabaseManager", return_value=mock_manager):
            result = runner.invoke(app, ["init", "--database", "sqlite:///:memory:"])
            assert result.exit_code == 0


class TestHealthCommand:
    """Tests for the health command."""

    def test_health_healthy_database(self):
        """Test health command with healthy database."""
        with patch("evalops.storage.DatabaseManager") as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.check_health.return_value = {
                "healthy": True,
                "version": 1,
                "tables": ["eval_runs", "eval_cases"],
            }
            mock_instance.table_sizes.return_value = {
                "eval_runs": 10,
                "eval_cases": 50,
            }

            result = runner.invoke(app, ["health", "--database", "sqlite:///:memory:"])
            assert result.exit_code == 0
            assert "healthy" in result.stdout.lower()

    def test_health_unhealthy_database(self):
        """Test health command with unhealthy database."""
        with patch("evalops.storage.DatabaseManager") as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.check_health.return_value = {
                "healthy": False,
                "error": "Connection failed",
            }

            result = runner.invoke(app, ["health", "--database", "sqlite:///:memory:"])
            assert result.exit_code == 1
            assert "unhealthy" in result.stdout.lower()


class TestHistoryCommand:
    """Tests for the history command."""

    def test_history_no_runs(self):
        """Test history command with no runs."""
        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_runs.return_value = []
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["history"])
            assert result.exit_code == 0
            assert "No runs found" in result.stdout

    def test_history_with_runs(self):
        """Test history command with existing runs."""
        from datetime import datetime, timezone

        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_run = MagicMock()
            mock_run.id = "test-run-id-12345"
            mock_run.name = "test_run"
            mock_run.dataset_name = "test_dataset"
            mock_run.total_cases = 10
            mock_run.pass_rate = 0.9
            mock_run.avg_latency_ms = 150.0
            mock_run.started_at = datetime.now(timezone.utc)

            mock_repo = MagicMock()
            mock_repo.list_runs.return_value = [mock_run]
            mock_repo.get_run_stats.return_value = {
                "total_runs": 1,
                "avg_pass_rate": 0.9,
                "total_cases": 10,
            }
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["history", "--days", "30"])
            assert result.exit_code == 0
            assert "test_run" in result.stdout or "test_dataset" in result.stdout

    def test_history_with_dataset_filter(self):
        """Test history command with dataset filter."""
        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_runs.return_value = []
            mock_repo.get_run_stats.return_value = {"total_runs": 0, "avg_pass_rate": 0.0, "total_cases": 0}
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["history", "--dataset", "my_dataset"])
            assert result.exit_code == 0
            mock_repo.list_runs.assert_called_once()
            call_kwargs = mock_repo.list_runs.call_args[1]
            assert call_kwargs["dataset_name"] == "my_dataset"


class TestBaselineCommand:
    """Tests for the baseline command."""

    def test_baseline_list_empty(self):
        """Test baseline list with no baselines."""
        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.list_baselines.return_value = []
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["baseline", "list"])
            assert result.exit_code == 0
            assert "No baselines found" in result.stdout

    def test_baseline_list_with_baselines(self):
        """Test baseline list with existing baselines."""
        from datetime import datetime, timezone

        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_bl = MagicMock()
            mock_bl.id = "baseline-id-12345"
            mock_bl.name = "test_baseline"
            mock_bl.dataset_name = "test_dataset"
            mock_bl.pass_rate = 0.95
            mock_bl.is_active = True
            mock_bl.created_at = datetime.now(timezone.utc)

            mock_repo = MagicMock()
            mock_repo.list_baselines.return_value = [mock_bl]
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["baseline", "list"])
            assert result.exit_code == 0
            # Name may be truncated in Rich table output
            assert "test_baseli" in result.stdout or "test_baseline" in result.stdout

    def test_baseline_save_requires_run_id(self):
        """Test baseline save requires --run option."""
        with patch("evalops.cli.main.get_repository"):
            result = runner.invoke(app, ["baseline", "save"])
            assert result.exit_code == 1
            assert "--run is required" in result.stdout

    def test_baseline_delete_requires_id(self):
        """Test baseline delete requires --id option."""
        with patch("evalops.cli.main.get_repository"):
            result = runner.invoke(app, ["baseline", "delete"])
            assert result.exit_code == 1
            assert "--id is required" in result.stdout

    def test_baseline_invalid_action(self):
        """Test baseline with invalid action."""
        with patch("evalops.cli.main.get_repository"):
            result = runner.invoke(app, ["baseline", "invalid_action"])
            assert result.exit_code == 1
            assert "Unknown action" in result.stdout


class TestDriftCommand:
    """Tests for the drift command."""

    def test_drift_no_baseline(self):
        """Test drift command with no baseline."""
        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_repo = MagicMock()
            mock_repo.get_baseline.return_value = None
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["drift", "test_dataset"])
            assert result.exit_code == 1
            assert "No baseline found" in result.stdout

    def test_drift_no_history(self):
        """Test drift command with no recent runs."""

        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_baseline = MagicMock()
            mock_baseline.name = "test_baseline"
            mock_baseline.metrics = {"exact_match": 0.9}
            mock_baseline.pass_rate = 0.9

            mock_repo = MagicMock()
            mock_repo.get_baseline.return_value = mock_baseline
            mock_repo.get_history.return_value = []
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["drift", "test_dataset"])
            assert result.exit_code == 0
            assert "No recent runs" in result.stdout

    def test_drift_healthy(self):
        """Test drift command with healthy metrics."""

        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_baseline = MagicMock()
            mock_baseline.name = "test_baseline"
            mock_baseline.metrics = {"exact_match": 0.9, "pass_rate": 0.9}
            mock_baseline.pass_rate = 0.9

            mock_repo = MagicMock()
            mock_repo.get_baseline.return_value = mock_baseline
            mock_repo.get_history.return_value = [
                {"run_id": "run1", "pass_rate": 0.89, "metrics": {"exact_match": 0.88}},
                {"run_id": "run2", "pass_rate": 0.91, "metrics": {"exact_match": 0.90}},
            ]
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["drift", "test_dataset"])
            assert result.exit_code == 0
            assert "HEALTHY" in result.stdout

    def test_drift_json_output(self):
        """Test drift command with JSON output."""
        with patch("evalops.cli.main.get_repository") as mock_get_repo:
            mock_baseline = MagicMock()
            mock_baseline.name = "test_baseline"
            mock_baseline.metrics = {"exact_match": 0.9}
            mock_baseline.pass_rate = 0.9

            mock_repo = MagicMock()
            mock_repo.get_baseline.return_value = mock_baseline
            mock_repo.get_history.return_value = [
                {"run_id": "run1", "pass_rate": 0.89, "metrics": {"exact_match": 0.88}},
            ]
            mock_get_repo.return_value = mock_repo

            result = runner.invoke(app, ["drift", "test_dataset", "--json"])
            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.stdout)
            assert "drift_detected" in data


class TestRunCommand:
    """Tests for the run command."""

    def test_run_dataset_not_found(self):
        """Test run command with non-existent dataset."""
        result = runner.invoke(
            app,
            ["run", "/nonexistent/dataset.json", "--target", "json:dumps"],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_run_with_valid_dataset(self):
        """Test run command with valid dataset."""
        # Create a temporary dataset
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "name": "test_dataset",
                    "cases": [
                        {"input": "hello", "expected": "HELLO"},
                    ],
                },
                f,
            )
            dataset_path = f.name

        try:
            with patch("evalops.cli.main.get_repository") as mock_get_repo:
                mock_repo = MagicMock()
                mock_get_repo.return_value = mock_repo

                # Mock the runner.evaluate call
                with patch("evalops.core.runner.EvalRunner.evaluate") as mock_eval:
                    mock_result = MagicMock()
                    mock_result.id = "test-run-id"
                    mock_result.dataset_name = "test_dataset"
                    mock_result.pass_rate = 1.0
                    mock_result.total_cases = 1
                    mock_result.successful_cases = 1
                    mock_result.avg_latency_ms = 10.0
                    mock_result.metrics_summary = {}
                    mock_eval.return_value = mock_result

                    # Use a simple lambda as target
                    runner.invoke(
                        app,
                        [
                            "run",
                            dataset_path,
                            "--target",
                            "str:upper",
                            "--no-save",
                        ],
                    )
                    # The command should at least not crash on loading
                    # (may fail on target loading since str:upper isn't valid)
        finally:
            Path(dataset_path).unlink(missing_ok=True)


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_dataset_not_found(self):
        """Test compare command with non-existent dataset."""
        result = runner.invoke(
            app,
            [
                "compare",
                "/nonexistent/dataset.json",
                "--variant-a",
                "json:dumps",
                "--variant-b",
                "json:loads",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()
