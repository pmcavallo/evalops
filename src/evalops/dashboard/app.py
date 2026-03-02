"""Streamlit dashboard application for EvalOps.

This module provides a visual interface for exploring evaluation results,
comparing A/B tests, and monitoring quality drift.

Run with:
    streamlit run src/evalops/dashboard/app.py
    # or
    evalops-dashboard
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from evalops import __version__
from evalops.storage import DatabaseManager, EvalRepository


# ============================================================================
# Configuration & Session State
# ============================================================================


def init_session_state() -> None:
    """Initialize session state variables."""
    if "database_url" not in st.session_state:
        st.session_state.database_url = "sqlite:///evalops_demo.db"
    if "repository" not in st.session_state:
        st.session_state.repository = None
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None
    if "page" not in st.session_state:
        st.session_state.page = "Overview"


def get_repository() -> EvalRepository | None:
    """Get or create repository instance."""
    if st.session_state.repository is None:
        try:
            repo = EvalRepository(st.session_state.database_url)
            repo.initialize()
            st.session_state.repository = repo
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return None
    return st.session_state.repository


def reconnect_database(url: str) -> bool:
    """Reconnect to a new database."""
    try:
        repo = EvalRepository(url)
        repo.initialize()
        st.session_state.database_url = url
        st.session_state.repository = repo
        return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return False


# ============================================================================
# Styling
# ============================================================================


def apply_custom_css() -> None:
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .status-pass {
        color: #28a745;
        font-weight: bold;
    }
    .status-fail {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Overview Page
# ============================================================================


def render_overview_page(repo: EvalRepository) -> None:
    """Render the overview/home page."""
    st.title("EvalOps Dashboard")
    st.markdown("---")

    # Get overall stats
    stats = repo.get_run_stats()

    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Runs",
            value=stats.get("total_runs", 0),
        )

    with col2:
        st.metric(
            label="Total Cases",
            value=f"{stats.get('total_cases', 0):,}",
        )

    with col3:
        avg_pass_rate = stats.get("avg_pass_rate", 0)
        st.metric(
            label="Avg Pass Rate",
            value=f"{avg_pass_rate:.1%}",
        )

    with col4:
        avg_latency = stats.get("avg_latency_ms", 0)
        st.metric(
            label="Avg Latency",
            value=f"{avg_latency:.0f}ms",
        )

    st.markdown("---")

    # Recent runs and trend chart
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Recent Runs")
        recent_runs = repo.list_runs(limit=10)

        if recent_runs:
            for run in recent_runs[:5]:
                if run.pass_rate >= 0.8:
                    status_icon = "✅"
                elif run.pass_rate >= 0.5:
                    status_icon = "⚠️"
                else:
                    status_icon = "❌"
                with st.container():
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    with col_a:
                        run_name = run.name or run.dataset_name
                        st.markdown(f"**{run_name}**")
                    started_at_str = run.started_at.strftime("%Y-%m-%d %H:%M") if run.started_at else "N/A"
                    st.caption(f"{started_at_str}")
                    with col_b:
                        st.markdown(f"{status_icon} {run.pass_rate:.0%}")
                    with col_c:
                        if st.button("View", key=f"view_{run.id}"):
                            st.session_state.selected_run_id = run.id
                            st.session_state.page = "Run Detail"
                            st.rerun()
                    st.markdown("---")
        else:
            st.info("No runs found. Run some evaluations to see them here.")

    with col_right:
        st.subheader("Pass Rate Trend")
        history = repo.get_history(dataset_name=None, days=30)

        if history:
            # Convert to chart data
            dates = [h.get("timestamp", datetime.now(timezone.utc)) for h in history]
            pass_rates = [h.get("pass_rate", 0) * 100 for h in history]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=pass_rates,
                mode='lines+markers',
                name='Pass Rate',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Pass Rate (%)",
                yaxis=dict(range=[0, 105]),
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available yet.")

    # Dataset breakdown
    st.subheader("Dataset Overview")
    all_runs = repo.list_runs(limit=1000)

    if all_runs:
        # Group by dataset
        dataset_stats: dict[str, dict[str, Any]] = {}
        for run in all_runs:
            name = run.dataset_name
            if name not in dataset_stats:
                dataset_stats[name] = {
                    "runs": 0,
                    "total_pass_rate": 0,
                    "total_latency": 0,
                    "total_cases": 0,
                }
            dataset_stats[name]["runs"] += 1
            dataset_stats[name]["total_pass_rate"] += run.pass_rate
            dataset_stats[name]["total_latency"] += run.avg_latency_ms
            dataset_stats[name]["total_cases"] += run.total_cases

        # Create DataFrame-like display
        rows = []
        for name, data in dataset_stats.items():
            rows.append({
                "Dataset": name,
                "Runs": data["runs"],
                "Cases": data["total_cases"],
                "Avg Pass Rate": f"{data['total_pass_rate'] / data['runs']:.1%}",
                "Avg Latency": f"{data['total_latency'] / data['runs']:.0f}ms",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================================
# Run Explorer Page
# ============================================================================


def render_run_explorer_page(repo: EvalRepository) -> None:
    """Render the run explorer page."""
    st.title("Run Explorer")
    st.markdown("Browse and filter evaluation runs")
    st.markdown("---")

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get unique datasets
        all_runs = repo.list_runs(limit=1000)
        datasets = sorted(set(r.dataset_name for r in all_runs)) if all_runs else []
        dataset_filter = st.selectbox(
            "Dataset",
            options=["All"] + datasets,
            index=0,
        )

    with col2:
        days_filter = st.selectbox(
            "Time Range",
            options=[7, 14, 30, 60, 90, 365],
            index=2,
            format_func=lambda x: f"Last {x} days",
        )

    with col3:
        min_pass_rate = st.slider(
            "Min Pass Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            format="%.0f%%",
        )

    with col4:
        limit = st.selectbox(
            "Results",
            options=[10, 25, 50, 100],
            index=1,
        )

    # Apply filters
    start_date = datetime.now(timezone.utc) - timedelta(days=days_filter)
    runs = repo.list_runs(
        dataset_name=dataset_filter if dataset_filter != "All" else None,
        min_pass_rate=min_pass_rate if min_pass_rate > 0 else None,
        start_date=start_date,
        limit=limit,
    )

    st.markdown("---")
    st.subheader(f"Results ({len(runs)} runs)")

    if runs:
        for run in runs:
            with st.container():
                col_a, col_b, col_c, col_d, col_e = st.columns([3, 2, 1, 1, 1])

                with col_a:
                    run_name = run.name or f"Run {run.id[:8]}"
                    st.markdown(f"**{run_name}**")
                    st.caption(f"Dataset: {run.dataset_name}")

                with col_b:
                    if run.started_at:
                        st.markdown(run.started_at.strftime("%Y-%m-%d %H:%M"))
                    tags = run.tags or []
                    if tags:
                        st.caption(", ".join(tags[:3]))

                with col_c:
                    if run.pass_rate >= 0.8:
                        color = "green"
                    elif run.pass_rate >= 0.5:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(f":{color}[{run.pass_rate:.0%}]")

                with col_d:
                    st.markdown(f"{run.total_cases} cases")
                    st.caption(f"{run.avg_latency_ms:.0f}ms")

                with col_e:
                    if st.button("Details", key=f"details_{run.id}"):
                        st.session_state.selected_run_id = run.id
                        st.session_state.page = "Run Detail"
                        st.rerun()

                st.markdown("---")
    else:
        st.info("No runs found matching the filters.")


# ============================================================================
# Run Detail Page
# ============================================================================


def render_run_detail_page(repo: EvalRepository) -> None:
    """Render the run detail page."""
    run_id = st.session_state.selected_run_id

    if not run_id:
        st.warning("No run selected. Please select a run from the Explorer.")
        if st.button("Go to Explorer"):
            st.session_state.page = "Run Explorer"
            st.rerun()
        return

    run = repo.get_run(run_id)
    if not run:
        st.error(f"Run not found: {run_id}")
        return

    # Header
    st.title(run.name or f"Run {run.id[:8]}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"ID: {run.id}")
        st.caption(f"Dataset: {run.dataset_name}")
        if run.started_at:
            st.caption(f"Started: {run.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if st.button("← Back to Explorer"):
            st.session_state.page = "Run Explorer"
            st.rerun()

    st.markdown("---")

    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = "normal" if run.pass_rate >= 0.8 else "off"
        st.metric("Pass Rate", f"{run.pass_rate:.1%}", delta_color=color)

    with col2:
        st.metric("Success Rate", f"{run.success_rate:.1%}")

    with col3:
        st.metric("Total Cases", run.total_cases)

    with col4:
        st.metric("Avg Latency", f"{run.avg_latency_ms:.0f}ms")

    st.markdown("---")

    # Metrics breakdown
    if run.metrics_summary:
        st.subheader("Metrics Breakdown")

        metric_cols = st.columns(len(run.metrics_summary))
        for i, (name, summary) in enumerate(run.metrics_summary.items()):
            with metric_cols[i]:
                score = summary.get("score", 0)
                passed = summary.get("passed", False)
                status = "✅" if passed else "❌"

                if isinstance(score, float) and score <= 1:
                    score_str = f"{score:.1%}"
                else:
                    score_str = f"{score:.2f}"

                st.metric(f"{name} {status}", score_str)

        # Metrics chart
        metric_names = list(run.metrics_summary.keys())
        metric_scores = [
            run.metrics_summary[m].get("score", 0) * 100
            if run.metrics_summary[m].get("score", 0) <= 1
            else run.metrics_summary[m].get("score", 0)
            for m in metric_names
        ]

        marker_colors = [
            "#28a745" if run.metrics_summary[m].get("passed", False) else "#dc3545"
            for m in metric_names
        ]
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_scores,
                marker_color=marker_colors,
            )
        ])
        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title="Score (%)",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Case results
    st.subheader("Case Results")

    # Get cases
    cases = repo.get_cases(run_id, limit=100)

    if cases:
        # Pass/Fail distribution
        passed_count = sum(1 for c in cases if c.metrics_passed)
        failed_count = len(cases) - passed_count

        col1, col2 = st.columns([1, 2])

        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=['Passed', 'Failed'],
                values=[passed_count, failed_count],
                marker_colors=['#28a745', '#dc3545'],
                hole=0.4,
            )])
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Latency distribution
            latencies = [c.latency_ms for c in cases]
            fig = go.Figure(data=[go.Histogram(
                x=latencies,
                nbinsx=20,
                marker_color='#1f77b4',
            )])
            fig.update_layout(
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Case table
        st.subheader("Individual Cases")

        show_failed_only = st.checkbox("Show failed cases only")
        display_cases = [c for c in cases if not c.metrics_passed] if show_failed_only else cases

        for case in display_cases[:20]:
            status_icon = "✅" if case.metrics_passed else "❌"
            with st.expander(f"{status_icon} Case {case.case_id[:8]}... | {case.latency_ms:.0f}ms"):
                st.markdown("**Input:**")
                st.code(case.input[:500] + ("..." if len(case.input) > 500 else ""))

                if case.output:
                    st.markdown("**Output:**")
                    st.code(case.output[:500] + ("..." if len(case.output) > 500 else ""))

                if case.expected:
                    st.markdown("**Expected:**")
                    st.code(case.expected[:500] + ("..." if len(case.expected) > 500 else ""))

                if case.error:
                    st.markdown("**Error:**")
                    st.error(case.error)

                if case.metric_results:
                    st.markdown("**Metric Results:**")
                    st.json(case.metric_results)

        if len(display_cases) > 20:
            st.info(f"Showing 20 of {len(display_cases)} cases")
    else:
        st.info("No case data available for this run.")


# ============================================================================
# Comparison Page
# ============================================================================


def render_comparison_page(repo: EvalRepository) -> None:
    """Render the A/B comparison page."""
    st.title("A/B Comparison")
    st.markdown("Compare two evaluation runs side-by-side")
    st.markdown("---")

    # Get recent runs for selection
    runs = repo.list_runs(limit=50)

    if len(runs) < 2:
        st.warning("Need at least 2 runs to compare. Run more evaluations first.")
        return

    run_options = {f"{r.name or r.dataset_name} ({r.id[:8]})": r.id for r in runs}
    run_labels = list(run_options.keys())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variant A")
        variant_a_label = st.selectbox("Select Run A", options=run_labels, index=0, key="variant_a")

    with col2:
        st.subheader("Variant B")
        default_b = 1 if len(run_labels) > 1 else 0
        variant_b_label = st.selectbox(
            "Select Run B", options=run_labels, index=default_b, key="variant_b"
        )

    if variant_a_label == variant_b_label:
        st.warning("Please select two different runs to compare.")
        return

    run_a = repo.get_run(run_options[variant_a_label])
    run_b = repo.get_run(run_options[variant_b_label])

    if not run_a or not run_b:
        st.error("Failed to load selected runs.")
        return

    st.markdown("---")

    # Comparison metrics
    st.subheader("Overall Comparison")

    metrics_data = [
        ("Pass Rate", run_a.pass_rate, run_b.pass_rate, True),
        ("Success Rate", run_a.success_rate, run_b.success_rate, True),
        ("Avg Latency (ms)", run_a.avg_latency_ms, run_b.avg_latency_ms, False),
        ("Total Cases", run_a.total_cases, run_b.total_cases, None),
    ]

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.markdown("**Variant A**")
    with col2:
        st.markdown("**Variant B**")
    with col3:
        st.markdown("**Winner**")

    for metric_name, val_a, val_b, higher_better in metrics_data:
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            if isinstance(val_a, float) and val_a <= 1 and "Rate" in metric_name:
                st.metric(metric_name, f"{val_a:.1%}")
            else:
                st.metric(metric_name, f"{val_a:.1f}" if isinstance(val_a, float) else val_a)

        with col2:
            if isinstance(val_b, float) and val_b <= 1 and "Rate" in metric_name:
                st.metric(metric_name, f"{val_b:.1%}")
            else:
                st.metric(metric_name, f"{val_b:.1f}" if isinstance(val_b, float) else val_b)

        with col3:
            if higher_better is not None:
                if higher_better:
                    winner = "A" if val_a > val_b else "B" if val_b > val_a else "Tie"
                else:
                    winner = "A" if val_a < val_b else "B" if val_b < val_a else "Tie"

                color = "green" if winner != "Tie" else "gray"
                st.markdown(f":{color}[{winner}]")

    st.markdown("---")

    # Metrics comparison chart
    st.subheader("Metrics Comparison")

    # Collect common metrics
    all_metrics = set()
    if run_a.metrics_summary:
        all_metrics.update(run_a.metrics_summary.keys())
    if run_b.metrics_summary:
        all_metrics.update(run_b.metrics_summary.keys())

    if all_metrics:
        metric_names = list(all_metrics)
        scores_a = []
        scores_b = []

        for m in metric_names:
            score_a = 0
            if run_a.metrics_summary:
                score_a = run_a.metrics_summary.get(m, {}).get("score", 0)
            score_b = 0
            if run_b.metrics_summary:
                score_b = run_b.metrics_summary.get(m, {}).get("score", 0)

            # Convert to percentage if <= 1
            scores_a.append(score_a * 100 if score_a <= 1 else score_a)
            scores_b.append(score_b * 100 if score_b <= 1 else score_b)

        fig = go.Figure(data=[
            go.Bar(name='Variant A', x=metric_names, y=scores_a, marker_color='#1f77b4'),
            go.Bar(name='Variant B', x=metric_names, y=scores_b, marker_color='#ff7f0e'),
        ])
        fig.update_layout(
            barmode='group',
            xaxis_title="Metric",
            yaxis_title="Score (%)",
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics data available for comparison.")

    # Latency comparison
    st.subheader("Latency Distribution")

    cases_a = repo.get_cases(run_options[variant_a_label], limit=500)
    cases_b = repo.get_cases(run_options[variant_b_label], limit=500)

    if cases_a and cases_b:
        latencies_a = [c.latency_ms for c in cases_a]
        latencies_b = [c.latency_ms for c in cases_b]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=latencies_a, name="Variant A", opacity=0.7, marker_color="#1f77b4")
        )
        fig.add_trace(
            go.Histogram(x=latencies_b, name="Variant B", opacity=0.7, marker_color="#ff7f0e")
        )
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Latency (ms)",
            yaxis_title="Count",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Drift Monitor Page
# ============================================================================


def render_drift_monitor_page(repo: EvalRepository) -> None:
    """Render the drift monitoring page."""
    st.title("Drift Monitor")
    st.markdown("Monitor quality drift against baselines")
    st.markdown("---")

    # Get baselines
    baselines = repo.list_baselines(active_only=True)

    if not baselines:
        st.warning("No baselines configured. Create a baseline first to monitor drift.")
        st.info(
            "Use the CLI to create a baseline: `evalops baseline save --run RUN_ID --name NAME`"
        )
        return

    # Baseline selector
    col1, col2 = st.columns([2, 1])

    with col1:
        baseline_options = {f"{b.name} ({b.dataset_name})": b for b in baselines}
        selected_label = st.selectbox("Select Baseline", options=list(baseline_options.keys()))
        baseline = baseline_options[selected_label]

    with col2:
        days = st.selectbox(
            "Time Range", options=[7, 14, 30, 60], index=2, format_func=lambda x: f"Last {x} days"
        )

    st.markdown("---")

    # Baseline info
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Baseline Pass Rate", f"{baseline.pass_rate:.1%}")

    with col2:
        st.metric("Baseline Cases", baseline.total_cases)

    with col3:
        st.metric("Dataset", baseline.dataset_name)

    with col4:
        if baseline.created_at:
            st.metric("Created", baseline.created_at.strftime("%Y-%m-%d"))

    st.markdown("---")

    # Get history and check drift
    history = repo.get_history(dataset_name=baseline.dataset_name, days=days)

    if not history:
        st.info(f"No recent runs for dataset '{baseline.dataset_name}'")
        return

    # Drift analysis
    from evalops.comparison.drift import DriftDetector

    detector = DriftDetector(
        warning_threshold=0.05,
        critical_threshold=0.10,
    )
    detector.set_baseline_from_values(baseline.metrics, pass_rate=baseline.pass_rate)

    for entry in history:
        metrics = entry.get("metrics", {})
        metrics["pass_rate"] = entry.get("pass_rate", 0.0)
        detector.add_raw_snapshot(
            metrics=metrics,
            pass_rate=entry.get("pass_rate", 0.0),
            run_id=entry.get("run_id"),
        )

    report = detector.check()

    # Status display
    health_colors = {
        "healthy": "green",
        "improving": "blue",
        "degraded": "orange",
        "critical": "red",
        "unknown": "gray",
    }

    st.subheader("Current Status")

    col1, col2 = st.columns([1, 3])

    with col1:
        health = report.overall_health
        st.markdown(f"### :{health_colors.get(health, 'gray')}[{health.upper()}]")
        if report.drift_detected:
            st.warning(f"{len(report.alerts)} alert(s) detected")

    with col2:
        if report.alerts:
            for alert in report.alerts:
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                    alert.severity.value, "⚪"
                )
                st.markdown(f"{severity_icon} **{alert.metric_name}**: {alert.message}")

                if alert.suggested_actions:
                    with st.expander("Suggested Actions"):
                        for action in alert.suggested_actions:
                            st.markdown(f"- {action}")

    st.markdown("---")

    # Trend chart
    st.subheader("Trend Analysis")

    # Create trend data
    timestamps = [h.get("timestamp", datetime.now(timezone.utc)) for h in history]
    pass_rates = [h.get("pass_rate", 0) * 100 for h in history]

    fig = make_subplots(rows=1, cols=1)

    # Current values
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=pass_rates,
        mode='lines+markers',
        name='Current Pass Rate',
        line=dict(color='#1f77b4', width=2),
    ))

    # Baseline line
    fig.add_hline(
        y=baseline.pass_rate * 100,
        line_dash="dash",
        line_color="green",
        annotation_text="Baseline",
    )

    # Warning threshold
    warning_threshold = baseline.pass_rate * 0.95 * 100
    fig.add_hline(
        y=warning_threshold,
        line_dash="dot",
        line_color="orange",
        annotation_text="Warning (-5%)",
    )

    # Critical threshold
    critical_threshold = baseline.pass_rate * 0.90 * 100
    fig.add_hline(
        y=critical_threshold,
        line_dash="dot",
        line_color="red",
        annotation_text="Critical (-10%)",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Pass Rate (%)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metrics comparison table
    st.subheader("Metrics Comparison")

    if report.baseline_metrics and report.current_metrics:
        comparison_data = []
        for metric in report.baseline_metrics:
            baseline_val = report.baseline_metrics.get(metric, 0)
            current_val = report.current_metrics.get(metric, 0)
            trend = report.metric_trends.get(metric)

            trend_icon = {
                "improving": "📈",
                "stable": "➡️",
                "degrading": "📉",
            }.get(trend.value if trend else "stable", "➡️")

            change = ((current_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0

            comparison_data.append({
                "Metric": metric,
                "Baseline": f"{baseline_val:.3f}",
                "Current": f"{current_val:.3f}",
                "Change": f"{change:+.1f}%",
                "Trend": trend_icon,
            })

        st.dataframe(comparison_data, use_container_width=True, hide_index=True)


# ============================================================================
# Guide Page
# ============================================================================


def render_guide_page() -> None:
    """Render the guide/help page."""
    st.title("Guide")
    st.markdown("Understanding EvalOps and how to use this dashboard")
    st.markdown("---")

    # Demo notice
    st.info(
        "**📋 Demo Mode:** This dashboard is loaded with pre-prepared demo data to showcase "
        "EvalOps capabilities. The data includes 24 evaluation runs across 3 datasets "
        "(Q&A, Classification, Summarization) with simulated drift patterns and A/B test results. "
        "In production, this would connect to your own evaluation database."
    )

    st.markdown("---")

    # What is EvalOps
    st.subheader("What is EvalOps?")
    st.markdown(
        "EvalOps is a production-grade evaluation framework for LLM applications. "
        "It helps teams systematically test, monitor, and compare AI system outputs "
        "using configurable metrics, baseline comparisons, and drift detection."
    )

    st.markdown(
        "**The Problem:** Traditional software testing (unit tests, integration tests) doesn't "
        "work for LLMs because outputs are non-deterministic. A correct answer today might be "
        "phrased differently tomorrow, and simple string matching fails."
    )

    st.markdown(
        "**The Solution:** EvalOps uses semantic similarity (BERT embeddings), configurable "
        "accuracy thresholds, and statistical drift detection to evaluate LLM outputs at scale."
    )

    st.markdown("---")

    # Dashboard pages
    st.subheader("Dashboard Pages")

    pages_info = [
        ("**Overview**", "High-level summary of all evaluation runs. Shows total runs, cases, "
         "average pass rate, and trends over time. Start here to get the big picture."),
        ("**Run Explorer**", "Browse and filter individual evaluation runs. Filter by dataset, "
         "time range, or minimum pass rate. Click 'Details' to drill into any run."),
        ("**Run Detail**", "Deep dive into a single evaluation run. See metrics breakdown, "
         "pass/fail distribution, latency histogram, and individual case results."),
        ("**Comparison**", "A/B testing interface. Compare two runs side-by-side to determine "
         "which prompt, model, or configuration performs better."),
        ("**Drift Monitor**", "Track quality degradation over time. Compare current performance "
         "against saved baselines. Alerts when metrics drop below warning/critical thresholds."),
        ("**Settings**", "Configure database connection. Switch between SQLite (local) or "
         "PostgreSQL (production) databases."),
    ]

    for page_name, description in pages_info:
        st.markdown(f"{page_name}: {description}")

    st.markdown("---")

    # Key metrics
    st.subheader("Key Metrics Explained")

    metrics_info = [
        ("**Pass Rate**", "Percentage of test cases that met all metric thresholds. "
         "A case passes only if ALL configured metrics (accuracy, similarity, etc.) pass."),
        ("**Success Rate**", "Percentage of test cases that completed without errors. "
         "A case can succeed (no errors) but still fail metrics."),
        ("**Semantic Similarity**", "BERT-based comparison between actual and expected outputs. "
         "Scores 0-1 where 1.0 = identical meaning. Typically 0.8+ indicates a good match."),
        ("**Latency**", "Time taken for each evaluation case in milliseconds. "
         "Useful for identifying performance regressions."),
        ("**Drift**", "Change in metrics compared to a saved baseline. "
         "Warning at -5%, Critical at -10% by default."),
    ]

    for metric_name, description in metrics_info:
        st.markdown(f"{metric_name}: {description}")

    st.markdown("---")

    # Costs
    st.subheader("Production Costs")

    st.markdown("EvalOps itself is free and open source. Costs come from what you evaluate:")

    cost_data = [
        {"Component": "EvalOps Framework", "Cost": "Free", "Notes": "Open source, MIT license"},
        {
            "Component": "Semantic Similarity (BERT)",
            "Cost": "Free",
            "Notes": "Runs locally via sentence-transformers",
        },
        {"Component": "Database (SQLite)", "Cost": "Free", "Notes": "Local file, no server needed"},
        {
            "Component": "Database (PostgreSQL)",
            "Cost": "~$0-15/mo",
            "Notes": "Free tier available on most clouds",
        },
        {
            "Component": "LLM API (if evaluating)",
            "Cost": "Varies",
            "Notes": "Claude Haiku ~$0.25/1M tokens, GPT-4o-mini similar",
        },
    ]

    st.dataframe(cost_data, use_container_width=True, hide_index=True)

    st.markdown(
        "**Example:** Evaluating 1,000 Q&A cases with Claude Haiku (avg 500 tokens/response) "
        "costs approximately $0.13 in API calls. The evaluation framework adds zero cost."
    )

    st.markdown("---")

    # Quick start
    st.subheader("Quick Start (Production Use)")

    st.code("""
# Install
pip install evalops

# Define your test cases
dataset = Dataset.from_list([
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
])

# Run evaluation against your LLM
result = runner.run(
    dataset=dataset,
    target_fn=your_llm_function,
    metrics=[Accuracy(threshold=0.8), SemanticSimilarity(threshold=0.7)],
)

# Save results
repo = EvalRepository("sqlite:///evalops.db")
repo.save_run(result, name="nightly_eval", tags=["production"])

# Launch dashboard
# evalops-dashboard
""", language="python")

    st.markdown("---")
    st.caption("Built by Paulo Cavallo | [GitHub](https://github.com/pmcavallo/evalops)")


# ============================================================================
# Settings Page
# ============================================================================


def render_settings_page() -> None:
    """Render the settings page."""
    st.title("Settings")
    st.markdown("Configure dashboard settings")
    st.markdown("---")

    st.subheader("Database Configuration")

    current_url = st.session_state.database_url

    # Mask password if present
    display_url = current_url
    if "@" in current_url and "://" in current_url:
        parts = current_url.split("://", 1)
        if len(parts) == 2 and "@" in parts[1]:
            auth_host = parts[1].split("@", 1)
            if ":" in auth_host[0]:
                user = auth_host[0].split(":")[0]
                display_url = f"{parts[0]}://{user}:***@{auth_host[1]}"

    st.markdown(f"**Current:** `{display_url}`")

    new_url = st.text_input(
        "Database URL",
        value=current_url,
        help="SQLite: sqlite:///path/to/db.db | PostgreSQL: postgresql://user:pass@host/db",
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Connect", type="primary"):
            if reconnect_database(new_url):
                st.success("Connected successfully!")
                st.rerun()

    with col2:
        if st.button("Test Connection"):
            try:
                manager = DatabaseManager(new_url)
                health = manager.check_health()
                if health.get("healthy"):
                    tables_str = ", ".join(health.get("tables", []))
                    st.success(
                        f"Connection OK! Version: {health.get('version')}, Tables: {tables_str}"
                    )
                else:
                    st.error(f"Connection failed: {health.get('error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("---")

    st.subheader("Database Status")

    repo = get_repository()
    if repo:
        try:
            manager = DatabaseManager(st.session_state.database_url)
            health = manager.check_health()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Status", "Healthy" if health.get("healthy") else "Unhealthy")
                st.metric("Schema Version", health.get("version", "N/A"))

            with col2:
                sizes = manager.table_sizes()
                st.markdown("**Table Sizes:**")
                for table, count in sizes.items():
                    st.markdown(f"- {table}: {count:,} rows")

        except Exception as e:
            st.error(f"Failed to get database status: {e}")

    st.markdown("---")

    st.subheader("About")
    st.markdown(f"**EvalOps Dashboard** v{__version__}")
    st.markdown("Production-grade LLM evaluation and observability platform")


# ============================================================================
# Main Application
# ============================================================================


def main() -> None:
    """Main entry point for the dashboard."""
    st.set_page_config(
        page_title="EvalOps Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()
    init_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 📊 EvalOps")
        st.markdown("---")

        pages = [
            "Overview",
            "Guide",
            "Run Explorer",
            "Run Detail",
            "Comparison",
            "Drift Monitor",
            "Settings",
        ]

        for page in pages:
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()

        st.markdown("---")
        st.caption(f"v{__version__}")

    # Get repository
    repo = get_repository()

    # Render selected page
    page = st.session_state.page

    if page == "Settings":
        render_settings_page()
    elif page == "Guide":
        render_guide_page()
    elif repo is None:
        st.error("Database connection failed. Please configure the database in Settings.")
        render_settings_page()
    elif page == "Overview":
        render_overview_page(repo)
    elif page == "Run Explorer":
        render_run_explorer_page(repo)
    elif page == "Run Detail":
        render_run_detail_page(repo)
    elif page == "Comparison":
        render_comparison_page(repo)
    elif page == "Drift Monitor":
        render_drift_monitor_page(repo)
    else:
        render_overview_page(repo)


if __name__ == "__main__":
    main()
