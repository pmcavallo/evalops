"""EvalOps - Production-grade LLM evaluation and observability platform.

EvalOps provides systematic evaluation, monitoring, and comparison capabilities
for LLM and agent systems.

Example:
    ```python
    from evalops import EvalCase, EvalDataset, EvalRunner, ExactMatch, Latency

    dataset = EvalDataset(
        name="my_eval",
        cases=[
            EvalCase(input="What is 2+2?", expected="4"),
        ]
    )

    runner = EvalRunner()
    result = await runner.evaluate(
        dataset=dataset,
        target=my_llm_function,
        metrics=[ExactMatch(), Latency(p95_target_ms=2000)],
    )
    print(f"Pass rate: {result.pass_rate:.1%}")
    ```
"""

from evalops.core.dataset import EvalCase, EvalDataset
from evalops.core.judge import JudgeScore, LLMJudge, RubricJudge
from evalops.core.metrics import (
    ContainsKeywords,
    ExactMatch,
    Latency,
    Metric,
    MetricResult,
    SemanticSimilarity,
    TokenCost,
    TokenUsage,
    PIILeakageMetric,
)
from evalops.core.runner import EvalResult, EvalRunResult, EvalRunner

__version__ = "0.1.0"

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
    "SemanticSimilarity",
    "PIILeakageMetric",
    # Judge
    "LLMJudge",
    "RubricJudge",
    "JudgeScore",
    # Version
    "__version__",
]
