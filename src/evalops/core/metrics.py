"""Metrics for evaluating LLM outputs.

This module provides the base Metric class and built-in metrics for
evaluating LLM system outputs against expected values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalops.core.runner import EvalResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass
class MetricResult:
    """Result of applying a metric to an evaluation result.

    Attributes:
        name: Name of the metric.
        score: Numeric score (interpretation depends on metric).
        passed: Whether the result meets the metric's threshold.
        details: Additional metric-specific information.
    """

    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Subclasses must implement the `compute` method to calculate
    a score from an EvalResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        ...

    @abstractmethod
    def compute(self, result: EvalResult) -> MetricResult:
        """Compute the metric for a single evaluation result.

        Args:
            result: The evaluation result to score.

        Returns:
            MetricResult with score and pass/fail status.
        """
        ...

    def compute_batch(self, results: list[EvalResult]) -> list[MetricResult]:
        """Compute the metric for multiple results.

        Args:
            results: List of evaluation results.

        Returns:
            List of MetricResults in the same order.
        """
        return [self.compute(r) for r in results]

    def aggregate(self, metric_results: list[MetricResult]) -> MetricResult:
        """Aggregate multiple metric results into a summary.

        Default implementation returns mean score and pass rate.

        Args:
            metric_results: List of individual metric results.

        Returns:
            Aggregated MetricResult.
        """
        if not metric_results:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"count": 0},
            )

        scores = [r.score for r in metric_results]
        passed_count = sum(1 for r in metric_results if r.passed)
        total = len(metric_results)

        return MetricResult(
            name=self.name,
            score=sum(scores) / total,
            passed=passed_count == total,
            details={
                "count": total,
                "passed_count": passed_count,
                "pass_rate": passed_count / total,
                "min_score": min(scores),
                "max_score": max(scores),
            },
        )


class ExactMatch(Metric):
    """Metric that checks for exact string match between output and expected.

    Attributes:
        case_sensitive: Whether comparison is case-sensitive (default True).
        strip_whitespace: Whether to strip leading/trailing whitespace (default True).
    """

    def __init__(
        self,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, result: EvalResult) -> MetricResult:
        """Check if output exactly matches expected.

        Returns score of 1.0 for match, 0.0 for no match.
        """
        if result.output is None or result.expected is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "missing output or expected"},
            )

        output = result.output
        expected = result.expected

        if self.strip_whitespace:
            output = output.strip()
            expected = expected.strip()

        if not self.case_sensitive:
            output = output.lower()
            expected = expected.lower()

        matched = output == expected

        return MetricResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            details={
                "output_length": len(result.output),
                "expected_length": len(result.expected),
            },
        )


class ContainsKeywords(Metric):
    """Metric that checks if output contains required keywords.

    Can require all keywords (default) or a minimum number.

    Attributes:
        min_matches: Minimum keywords required to pass. If None, all must match.
        case_sensitive: Whether keyword matching is case-sensitive.
    """

    def __init__(
        self,
        min_matches: int | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.min_matches = min_matches
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "contains_keywords"

    def compute(self, result: EvalResult) -> MetricResult:
        """Check if output contains expected keywords.

        Returns score as fraction of keywords found.
        """
        if result.output is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "missing output"},
            )

        keywords = result.expected_keywords or []
        if not keywords:
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=True,
                details={"reason": "no keywords to match"},
            )

        output = result.output if self.case_sensitive else result.output.lower()

        found = []
        missing = []
        for kw in keywords:
            check_kw = kw if self.case_sensitive else kw.lower()
            if check_kw in output:
                found.append(kw)
            else:
                missing.append(kw)

        score = len(found) / len(keywords)
        min_required = self.min_matches if self.min_matches is not None else len(keywords)
        passed = len(found) >= min_required

        return MetricResult(
            name=self.name,
            score=score,
            passed=passed,
            details={
                "found": found,
                "missing": missing,
                "total_keywords": len(keywords),
                "min_required": min_required,
            },
        )


class Latency(Metric):
    """Metric that evaluates response latency against thresholds.

    Attributes:
        max_ms: Maximum acceptable latency in milliseconds.
        p50_target_ms: Target p50 latency (for aggregate reporting).
        p95_target_ms: Target p95 latency (for aggregate reporting).
    """

    def __init__(
        self,
        max_ms: float | None = None,
        p50_target_ms: float | None = None,
        p95_target_ms: float | None = None,
    ) -> None:
        self.max_ms = max_ms
        self.p50_target_ms = p50_target_ms
        self.p95_target_ms = p95_target_ms

    @property
    def name(self) -> str:
        return "latency"

    def compute(self, result: EvalResult) -> MetricResult:
        """Evaluate latency for a single result.

        Score is the latency in ms (lower is better).
        Passes if under max_ms threshold (if set).
        """
        latency = result.latency_ms
        passed = True

        if self.max_ms is not None:
            passed = latency <= self.max_ms

        return MetricResult(
            name=self.name,
            score=latency,
            passed=passed,
            details={
                "latency_ms": latency,
                "max_ms": self.max_ms,
            },
        )

    def aggregate(self, metric_results: list[MetricResult]) -> MetricResult:
        """Aggregate latency results with percentile calculations."""
        if not metric_results:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"count": 0},
            )

        latencies = sorted(r.score for r in metric_results)
        count = len(latencies)

        # Calculate percentiles
        p50_idx = int(count * 0.50)
        p95_idx = min(int(count * 0.95), count - 1)

        p50 = latencies[p50_idx]
        p95 = latencies[p95_idx]
        avg = sum(latencies) / count

        # Check thresholds
        p50_ok = self.p50_target_ms is None or p50 <= self.p50_target_ms
        p95_ok = self.p95_target_ms is None or p95 <= self.p95_target_ms
        passed = p50_ok and p95_ok

        return MetricResult(
            name=self.name,
            score=avg,
            passed=passed,
            details={
                "count": count,
                "avg_ms": avg,
                "min_ms": latencies[0],
                "max_ms": latencies[-1],
                "p50_ms": p50,
                "p95_ms": p95,
                "p50_target_ms": self.p50_target_ms,
                "p95_target_ms": self.p95_target_ms,
                "p50_ok": p50_ok,
                "p95_ok": p95_ok,
            },
        )


@dataclass
class TokenUsage:
    """Token usage information for cost calculation.

    Attributes:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenCost(Metric):
    """Metric that calculates token usage and cost.

    Attributes:
        max_tokens_per_call: Maximum acceptable tokens per call.
        input_cost_per_1k: Cost per 1000 input tokens in dollars.
        output_cost_per_1k: Cost per 1000 output tokens in dollars.
    """

    # Default costs for Claude 3.5 Sonnet (as of Dec 2024)
    DEFAULT_INPUT_COST = 0.003  # $3 per 1M = $0.003 per 1K
    DEFAULT_OUTPUT_COST = 0.015  # $15 per 1M = $0.015 per 1K

    def __init__(
        self,
        max_tokens_per_call: int | None = None,
        input_cost_per_1k: float = DEFAULT_INPUT_COST,
        output_cost_per_1k: float = DEFAULT_OUTPUT_COST,
    ) -> None:
        self.max_tokens_per_call = max_tokens_per_call
        self.input_cost_per_1k = input_cost_per_1k
        self.output_cost_per_1k = output_cost_per_1k

    @property
    def name(self) -> str:
        return "token_cost"

    def compute(self, result: EvalResult) -> MetricResult:
        """Calculate token cost for a single result.

        Note: Requires token_usage in result metadata. If not present,
        estimates based on character count (rough approximation).
        """
        usage = result.metadata.get("token_usage")

        if usage is None:
            # Rough estimation: ~4 chars per token for English
            input_tokens = len(result.input) // 4
            output_tokens = len(result.output or "") // 4
            estimated = True
        else:
            if isinstance(usage, TokenUsage):
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
            else:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            estimated = False

        total_tokens = input_tokens + output_tokens
        cost = (
            (input_tokens / 1000) * self.input_cost_per_1k
            + (output_tokens / 1000) * self.output_cost_per_1k
        )

        passed = True
        if self.max_tokens_per_call is not None:
            passed = total_tokens <= self.max_tokens_per_call

        return MetricResult(
            name=self.name,
            score=cost,
            passed=passed,
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost,
                "estimated": estimated,
                "max_tokens_per_call": self.max_tokens_per_call,
            },
        )

    def aggregate(self, metric_results: list[MetricResult]) -> MetricResult:
        """Aggregate token costs with totals."""
        if not metric_results:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"count": 0},
            )

        total_cost = sum(r.score for r in metric_results)
        total_input = sum(r.details.get("input_tokens", 0) for r in metric_results)
        total_output = sum(r.details.get("output_tokens", 0) for r in metric_results)
        total_tokens = total_input + total_output
        passed_count = sum(1 for r in metric_results if r.passed)

        return MetricResult(
            name=self.name,
            score=total_cost,
            passed=passed_count == len(metric_results),
            details={
                "count": len(metric_results),
                "total_cost_usd": total_cost,
                "avg_cost_usd": total_cost / len(metric_results),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_tokens,
                "avg_tokens_per_call": total_tokens / len(metric_results),
            },
        )


class SemanticSimilarity(Metric):
    """Metric that computes semantic similarity using sentence embeddings.

    Uses sentence-transformers to compute cosine similarity between
    the output and expected text using BERT-based embeddings.

    Attributes:
        model_name: Name of the sentence-transformers model to use.
        threshold: Minimum similarity score to pass (0.0 to 1.0).
        device: Device to run the model on ('cpu', 'cuda', etc.).

    Example:
        ```python
        metric = SemanticSimilarity(threshold=0.8)
        result = metric.compute(eval_result)
        print(f"Similarity: {result.score:.2f}")
        ```
    """

    # Lazy-loaded model instance (class-level cache)
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7,
        device: str | None = None,
    ) -> None:
        """Initialize the semantic similarity metric.

        Args:
            model_name: HuggingFace model name for sentence-transformers.
                Default is 'all-MiniLM-L6-v2' (fast, good quality).
            threshold: Minimum cosine similarity to pass (default 0.7).
            device: Device for inference. None for auto-detection.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is not None:
            return self._model

        # Check class-level cache first
        cache_key = f"{self.model_name}:{self.device}"
        if cache_key in SemanticSimilarity._model_cache:
            self._model = SemanticSimilarity._model_cache[cache_key]
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SemanticSimilarity. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(self.model_name, device=self.device)
        SemanticSimilarity._model_cache[cache_key] = self._model
        return self._model

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def compute(self, result: EvalResult) -> MetricResult:
        """Compute semantic similarity between output and expected.

        Returns cosine similarity score between 0.0 and 1.0.
        """
        if result.output is None or result.expected is None:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "missing output or expected"},
            )

        if not result.output.strip() or not result.expected.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                details={"reason": "empty output or expected"},
            )

        model = self._get_model()

        # Encode both texts
        embeddings = model.encode(
            [result.output, result.expected],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Compute cosine similarity
        similarity = self._cosine_similarity(
            embeddings[0].tolist(),
            embeddings[1].tolist(),
        )

        # Ensure similarity is in valid range
        similarity = max(0.0, min(1.0, similarity))

        passed = similarity >= self.threshold

        return MetricResult(
            name=self.name,
            score=similarity,
            passed=passed,
            details={
                "similarity": similarity,
                "threshold": self.threshold,
                "model": self.model_name,
                "output_length": len(result.output),
                "expected_length": len(result.expected),
            },
        )

    def compute_batch(self, results: list[EvalResult]) -> list[MetricResult]:
        """Compute similarity for multiple results efficiently.

        Batches encoding for better performance.
        """
        metric_results = []

        # Filter valid results
        valid_indices = []
        texts = []
        for i, result in enumerate(results):
            if (
                result.output is not None
                and result.expected is not None
                and result.output.strip()
                and result.expected.strip()
            ):
                valid_indices.append(i)
                texts.extend([result.output, result.expected])

        # Handle case where no valid results
        if not valid_indices:
            return [
                MetricResult(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    details={"reason": "missing output or expected"},
                )
                for _ in results
            ]

        # Batch encode all texts
        model = self._get_model()
        all_embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Create result list with placeholders
        for i, result in enumerate(results):
            if i not in valid_indices:
                metric_results.append(
                    MetricResult(
                        name=self.name,
                        score=0.0,
                        passed=False,
                        details={"reason": "missing output or expected"},
                    )
                )
            else:
                # Find position in valid_indices
                pos = valid_indices.index(i)
                output_emb = all_embeddings[pos * 2].tolist()
                expected_emb = all_embeddings[pos * 2 + 1].tolist()

                similarity = self._cosine_similarity(output_emb, expected_emb)
                similarity = max(0.0, min(1.0, similarity))
                passed = similarity >= self.threshold

                metric_results.append(
                    MetricResult(
                        name=self.name,
                        score=similarity,
                        passed=passed,
                        details={
                            "similarity": similarity,
                            "threshold": self.threshold,
                            "model": self.model_name,
                            "output_length": len(result.output or ""),
                            "expected_length": len(result.expected or ""),
                        },
                    )
                )

        return metric_results
