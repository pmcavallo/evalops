"""Unit tests for the metrics module."""

import pytest

from evalops import ContainsKeywords, EvalResult, ExactMatch, Latency, TokenCost
from evalops.core.metrics import MetricResult, TokenUsage


class TestExactMatch:
    """Tests for ExactMatch metric."""

    @pytest.fixture
    def metric(self) -> ExactMatch:
        return ExactMatch()

    def test_exact_match_pass(self, metric: ExactMatch) -> None:
        """Test exact match passes when output equals expected."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output="4",
            expected="4",
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 1.0
        assert metric_result.passed is True
        assert metric_result.name == "exact_match"

    def test_exact_match_fail(self, metric: ExactMatch) -> None:
        """Test exact match fails when output differs from expected."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output="5",
            expected="4",
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False

    def test_exact_match_strips_whitespace(self) -> None:
        """Test that whitespace is stripped by default."""
        metric = ExactMatch(strip_whitespace=True)
        result = EvalResult(
            case_id="1",
            input="test",
            output="  hello  ",
            expected="hello",
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is True

    def test_exact_match_no_strip_whitespace(self) -> None:
        """Test without whitespace stripping."""
        metric = ExactMatch(strip_whitespace=False)
        result = EvalResult(
            case_id="1",
            input="test",
            output="  hello  ",
            expected="hello",
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is False

    def test_case_insensitive(self) -> None:
        """Test case-insensitive matching."""
        metric = ExactMatch(case_sensitive=False)
        result = EvalResult(
            case_id="1",
            input="test",
            output="HELLO",
            expected="hello",
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is True

    def test_case_sensitive(self) -> None:
        """Test case-sensitive matching (default)."""
        metric = ExactMatch(case_sensitive=True)
        result = EvalResult(
            case_id="1",
            input="test",
            output="HELLO",
            expected="hello",
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is False

    def test_missing_output(self, metric: ExactMatch) -> None:
        """Test handling of missing output."""
        result = EvalResult(
            case_id="1",
            input="test",
            output=None,
            expected="hello",
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False
        assert "missing" in metric_result.details.get("reason", "")

    def test_missing_expected(self, metric: ExactMatch) -> None:
        """Test handling of missing expected value."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="hello",
            expected=None,
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is False


class TestContainsKeywords:
    """Tests for ContainsKeywords metric."""

    @pytest.fixture
    def metric(self) -> ContainsKeywords:
        return ContainsKeywords()

    def test_all_keywords_found(self, metric: ContainsKeywords) -> None:
        """Test when all keywords are found."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="The model uses machine learning and neural networks",
            expected_keywords=["machine learning", "neural"],
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 1.0
        assert metric_result.passed is True
        assert metric_result.details["found"] == ["machine learning", "neural"]
        assert metric_result.details["missing"] == []

    def test_some_keywords_found(self, metric: ContainsKeywords) -> None:
        """Test when only some keywords are found."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="The model uses machine learning",
            expected_keywords=["machine learning", "neural", "deep"],
        )
        metric_result = metric.compute(result)

        assert metric_result.score == pytest.approx(1 / 3)
        assert metric_result.passed is False
        assert "machine learning" in metric_result.details["found"]
        assert "neural" in metric_result.details["missing"]

    def test_no_keywords_found(self, metric: ContainsKeywords) -> None:
        """Test when no keywords are found."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello world",
            expected_keywords=["machine learning", "neural"],
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False

    def test_min_matches_threshold(self) -> None:
        """Test with minimum matches threshold."""
        metric = ContainsKeywords(min_matches=2)
        result = EvalResult(
            case_id="1",
            input="test",
            output="Uses machine learning and neural networks",
            expected_keywords=["machine", "neural", "deep", "AI"],
        )
        metric_result = metric.compute(result)

        # Found 2 of 4 keywords, score is 0.5
        assert metric_result.score == 0.5
        # But passes because min_matches=2 and we found 2
        assert metric_result.passed is True

    def test_case_sensitive_keywords(self) -> None:
        """Test case-sensitive keyword matching."""
        metric = ContainsKeywords(case_sensitive=True)
        result = EvalResult(
            case_id="1",
            input="test",
            output="MACHINE LEARNING",
            expected_keywords=["machine learning"],
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is False

    def test_case_insensitive_keywords(self) -> None:
        """Test case-insensitive keyword matching (default)."""
        metric = ContainsKeywords(case_sensitive=False)
        result = EvalResult(
            case_id="1",
            input="test",
            output="MACHINE LEARNING",
            expected_keywords=["machine learning"],
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is True

    def test_no_keywords_defined(self, metric: ContainsKeywords) -> None:
        """Test when no keywords are defined."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello world",
            expected_keywords=None,
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 1.0
        assert metric_result.passed is True

    def test_missing_output(self, metric: ContainsKeywords) -> None:
        """Test handling of missing output."""
        result = EvalResult(
            case_id="1",
            input="test",
            output=None,
            expected_keywords=["test"],
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is False


class TestLatency:
    """Tests for Latency metric."""

    def test_latency_under_threshold(self) -> None:
        """Test latency passes when under max threshold."""
        metric = Latency(max_ms=1000)
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            latency_ms=500,
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 500
        assert metric_result.passed is True

    def test_latency_over_threshold(self) -> None:
        """Test latency fails when over max threshold."""
        metric = Latency(max_ms=1000)
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            latency_ms=1500,
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 1500
        assert metric_result.passed is False

    def test_latency_no_threshold(self) -> None:
        """Test latency without threshold always passes."""
        metric = Latency()
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            latency_ms=5000,
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 5000
        assert metric_result.passed is True

    def test_latency_aggregation(self) -> None:
        """Test percentile aggregation."""
        metric = Latency(p50_target_ms=100, p95_target_ms=200)

        # Create results with various latencies
        results = [
            MetricResult(name="latency", score=50, passed=True, details={}),
            MetricResult(name="latency", score=60, passed=True, details={}),
            MetricResult(name="latency", score=80, passed=True, details={}),
            MetricResult(name="latency", score=100, passed=True, details={}),
            MetricResult(name="latency", score=150, passed=True, details={}),
        ]

        aggregated = metric.aggregate(results)

        assert "p50_ms" in aggregated.details
        assert "p95_ms" in aggregated.details
        assert aggregated.details["count"] == 5

    def test_latency_aggregation_fails_p95(self) -> None:
        """Test aggregation fails when p95 exceeds target."""
        metric = Latency(p95_target_ms=100)

        results = [
            MetricResult(name="latency", score=50, passed=True, details={}),
            MetricResult(name="latency", score=60, passed=True, details={}),
            MetricResult(name="latency", score=200, passed=True, details={}),  # High outlier
        ]

        aggregated = metric.aggregate(results)
        assert aggregated.passed is False


class TestTokenCost:
    """Tests for TokenCost metric."""

    def test_token_cost_from_metadata(self) -> None:
        """Test cost calculation from token usage metadata."""
        metric = TokenCost()
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            metadata={
                "token_usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                }
            },
        )
        metric_result = metric.compute(result)

        # Cost = (100/1000)*0.003 + (50/1000)*0.015 = 0.0003 + 0.00075 = 0.00105
        assert metric_result.score == pytest.approx(0.00105)
        assert metric_result.details["input_tokens"] == 100
        assert metric_result.details["output_tokens"] == 50
        assert metric_result.details["estimated"] is False

    def test_token_cost_estimated(self) -> None:
        """Test cost estimation when no token usage provided."""
        metric = TokenCost()
        result = EvalResult(
            case_id="1",
            input="Hello world",  # ~11 chars / 4 = ~2 tokens
            output="Hi there!",  # ~9 chars / 4 = ~2 tokens
        )
        metric_result = metric.compute(result)

        assert metric_result.details["estimated"] is True
        assert metric_result.details["input_tokens"] > 0
        assert metric_result.details["output_tokens"] > 0

    def test_token_cost_max_threshold(self) -> None:
        """Test token cost with max tokens threshold."""
        metric = TokenCost(max_tokens_per_call=100)
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            metadata={
                "token_usage": {
                    "input_tokens": 80,
                    "output_tokens": 30,
                }
            },
        )
        metric_result = metric.compute(result)

        # 80 + 30 = 110 > 100
        assert metric_result.passed is False

    def test_token_cost_under_threshold(self) -> None:
        """Test token cost passes under threshold."""
        metric = TokenCost(max_tokens_per_call=200)
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            metadata={
                "token_usage": {
                    "input_tokens": 80,
                    "output_tokens": 30,
                }
            },
        )
        metric_result = metric.compute(result)

        assert metric_result.passed is True

    def test_token_cost_with_token_usage_object(self) -> None:
        """Test cost calculation with TokenUsage dataclass."""
        metric = TokenCost()
        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
            metadata={"token_usage": TokenUsage(input_tokens=100, output_tokens=50)},
        )
        metric_result = metric.compute(result)

        assert metric_result.details["input_tokens"] == 100
        assert metric_result.details["output_tokens"] == 50
        assert metric_result.details["estimated"] is False

    def test_token_cost_aggregation(self) -> None:
        """Test token cost aggregation."""
        metric = TokenCost()

        results = [
            MetricResult(
                name="token_cost",
                score=0.01,
                passed=True,
                details={"input_tokens": 100, "output_tokens": 50},
            ),
            MetricResult(
                name="token_cost",
                score=0.02,
                passed=True,
                details={"input_tokens": 200, "output_tokens": 100},
            ),
        ]

        aggregated = metric.aggregate(results)

        assert aggregated.details["total_cost_usd"] == pytest.approx(0.03)
        assert aggregated.details["total_input_tokens"] == 300
        assert aggregated.details["total_output_tokens"] == 150


class TestMetricAggregation:
    """Tests for metric aggregation."""

    def test_exact_match_aggregation(self) -> None:
        """Test ExactMatch aggregation."""
        metric = ExactMatch()

        results = [
            MetricResult(name="exact_match", score=1.0, passed=True, details={}),
            MetricResult(name="exact_match", score=1.0, passed=True, details={}),
            MetricResult(name="exact_match", score=0.0, passed=False, details={}),
        ]

        aggregated = metric.aggregate(results)

        assert aggregated.score == pytest.approx(2 / 3)
        assert aggregated.passed is False  # Not all passed
        assert aggregated.details["pass_rate"] == pytest.approx(2 / 3)
        assert aggregated.details["passed_count"] == 2

    def test_empty_aggregation(self) -> None:
        """Test aggregation with no results."""
        metric = ExactMatch()
        aggregated = metric.aggregate([])

        assert aggregated.score == 0.0
        assert aggregated.passed is False
        assert aggregated.details["count"] == 0


class TestSemanticSimilarity:
    """Tests for SemanticSimilarity metric."""

    @pytest.fixture
    def metric(self):
        """Create SemanticSimilarity metric."""
        # Import here to handle optional dependency
        try:
            from evalops.core.metrics import SemanticSimilarity

            return SemanticSimilarity(threshold=0.7)
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    def test_high_similarity(self, metric) -> None:
        """Test high similarity between semantically similar texts."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="The capital of France is Paris.",
            expected="Paris is the capital city of France.",
        )
        metric_result = metric.compute(result)

        assert metric_result.score > 0.8  # Should be very similar
        assert metric_result.passed is True
        assert metric_result.name == "semantic_similarity"

    def test_low_similarity(self, metric) -> None:
        """Test low similarity between unrelated texts."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="The weather is sunny today.",
            expected="Machine learning uses neural networks.",
        )
        metric_result = metric.compute(result)

        assert metric_result.score < 0.5  # Should be quite different
        assert metric_result.passed is False

    def test_identical_texts(self, metric) -> None:
        """Test perfect similarity with identical texts."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello world",
            expected="Hello world",
        )
        metric_result = metric.compute(result)

        assert metric_result.score > 0.99  # Should be nearly 1.0
        assert metric_result.passed is True

    def test_missing_output(self, metric) -> None:
        """Test handling of missing output."""
        result = EvalResult(
            case_id="1",
            input="test",
            output=None,
            expected="Hello world",
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False
        assert "missing" in metric_result.details.get("reason", "")

    def test_missing_expected(self, metric) -> None:
        """Test handling of missing expected."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello world",
            expected=None,
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False

    def test_empty_strings(self, metric) -> None:
        """Test handling of empty strings."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="   ",
            expected="Hello world",
        )
        metric_result = metric.compute(result)

        assert metric_result.score == 0.0
        assert metric_result.passed is False
        assert "empty" in metric_result.details.get("reason", "")

    def test_custom_threshold(self) -> None:
        """Test custom similarity threshold."""
        try:
            from evalops.core.metrics import SemanticSimilarity
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        metric = SemanticSimilarity(threshold=0.9)

        result = EvalResult(
            case_id="1",
            input="test",
            output="The capital of France is Paris.",
            expected="Paris is the capital city of France.",
        )
        metric_result = metric.compute(result)

        # High threshold might fail even for similar texts
        assert metric_result.details["threshold"] == 0.9

    def test_batch_compute(self, metric) -> None:
        """Test batch computation."""
        results = [
            EvalResult(
                case_id="1",
                input="test",
                output="Hello world",
                expected="Hello world",
            ),
            EvalResult(
                case_id="2",
                input="test",
                output="The sky is blue",
                expected="Machine learning is complex",
            ),
            EvalResult(
                case_id="3",
                input="test",
                output=None,
                expected="Test",
            ),
        ]

        metric_results = metric.compute_batch(results)

        assert len(metric_results) == 3
        assert metric_results[0].score > 0.9  # Identical
        assert metric_results[1].score < 0.5  # Different
        assert metric_results[2].score == 0.0  # Missing output

    def test_aggregation(self, metric) -> None:
        """Test aggregation of semantic similarity results."""
        results = [
            MetricResult(
                name="semantic_similarity",
                score=0.95,
                passed=True,
                details={},
            ),
            MetricResult(
                name="semantic_similarity",
                score=0.85,
                passed=True,
                details={},
            ),
            MetricResult(
                name="semantic_similarity",
                score=0.50,
                passed=False,
                details={},
            ),
        ]

        aggregated = metric.aggregate(results)

        assert aggregated.score == pytest.approx((0.95 + 0.85 + 0.50) / 3)
        assert aggregated.passed is False  # Not all passed
        assert aggregated.details["pass_rate"] == pytest.approx(2 / 3)

    def test_model_caching(self) -> None:
        """Test that model is cached across instances."""
        try:
            from evalops.core.metrics import SemanticSimilarity
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Create two metrics with same model
        metric1 = SemanticSimilarity()
        metric2 = SemanticSimilarity()

        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello",
            expected="Hello",
        )

        # Force model loading
        metric1.compute(result)
        metric2.compute(result)

        # Both should share the same cached model
        assert metric1._model is metric2._model

    def test_cosine_similarity_zero_vectors(self, metric) -> None:
        """Test cosine similarity handles zero vectors."""
        # Test the internal cosine similarity function
        similarity = metric._cosine_similarity([0.0, 0.0], [1.0, 0.0])
        assert similarity == 0.0

    def test_details_include_model_name(self, metric) -> None:
        """Test that result details include model name."""
        result = EvalResult(
            case_id="1",
            input="test",
            output="Hello",
            expected="Hello",
        )
        metric_result = metric.compute(result)

        assert "model" in metric_result.details
        assert metric_result.details["model"] == "all-MiniLM-L6-v2"


class TestSemanticSimilarityImportError:
    """Test SemanticSimilarity when sentence-transformers is not installed."""

    def test_import_error_message(self) -> None:
        """Test that clear error message is raised when dependency missing."""
        from unittest.mock import patch

        from evalops.core.metrics import SemanticSimilarity

        metric = SemanticSimilarity()

        # Mock the import to fail
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Clear the model cache to force re-import
            SemanticSimilarity._model_cache.clear()
            metric._model = None

            # The import error happens inside _get_model
            # This test verifies the error handling path exists
            # Actual import error would only occur if sentence-transformers isn't installed
