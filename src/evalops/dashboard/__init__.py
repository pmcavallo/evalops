"""Streamlit dashboard for EvalOps.

This module provides a visual interface for:
- Viewing evaluation run history and statistics
- Exploring individual run results
- Comparing A/B test variants
- Monitoring drift against baselines

Usage:
    evalops-dashboard
    # or
    streamlit run src/evalops/dashboard/app.py

Note: Requires optional dashboard dependencies:
    pip install evalops[dashboard]
"""


def main():
    """Launch the Streamlit dashboard (lazy import)."""
    from evalops.dashboard.app import main as _main

    return _main()


def run_dashboard():
    """Run the dashboard via CLI (lazy import)."""
    from evalops.dashboard.cli import main as _cli_main

    return _cli_main()


__all__ = ["main", "run_dashboard"]
