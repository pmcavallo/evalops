"""Core evaluation components.

This module exports the fundamental building blocks for LLM evaluation:
- EvalCase: A single test case
- EvalDataset: A collection of test cases
- EvalRunner: The execution engine
- EvalResult: Result of a single evaluation
- EvalRunResult: Result of a full evaluation run
- Metric: Base class for metrics
- MetricResult: Result of applying a metric
- Built-in metrics: ExactMatch, ContainsKeywords, Latency, TokenCost
- LLMJudge: Claude-based evaluation
"""

from evalops.core.dataset import EvalCase, EvalDataset
from evalops.core.judge import JudgeScore, LLMJudge, RubricJudge
from evalops.core.metrics import (
    ContainsKeywords,
    ExactMatch,
    Latency,
    Metric,
    MetricResult,
    TokenCost,
    TokenUsage,
    PIILeakageMetric,
)
from evalops.core.runner import EvalResult, EvalRunResult, EvalRunner

__all__ = [
    # Dataset
    "EvalCase",
    "EvalDataset",
    # Runner
    "EvalResult",
    "EvalRunResult",
    "EvalRunner",
    # Metrics
    "Metric",
    "MetricResult",
    "ExactMatch",
    "ContainsKeywords",
    "Latency",
    "TokenCost",
    "TokenUsage",
    # Judge
    "LLMJudge",
    "RubricJudge",
    "JudgeScore",
]
