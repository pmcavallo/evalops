"""Unit tests for the LLMJudge module."""

from unittest.mock import MagicMock

import pytest

from evalops import EvalResult, LLMJudge, RubricJudge
from evalops.core.judge import JudgeScore


class TestJudgeScore:
    """Tests for JudgeScore dataclass."""

    def test_create_basic_score(self) -> None:
        """Test creating a basic judge score."""
        score = JudgeScore(overall=8.5)
        assert score.overall == 8.5
        assert score.criteria_scores == {}
        assert score.reasoning == ""

    def test_create_full_score(self) -> None:
        """Test creating a score with all fields."""
        score = JudgeScore(
            overall=8.5,
            criteria_scores={"accuracy": 9.0, "clarity": 8.0},
            reasoning="Good response overall",
            raw_response='{"overall": 8.5}',
        )
        assert score.overall == 8.5
        assert score.criteria_scores["accuracy"] == 9.0
        assert score.reasoning == "Good response overall"


class TestLLMJudge:
    """Tests for LLMJudge."""

    @pytest.fixture
    def judge(self) -> LLMJudge:
        """Create a judge instance."""
        return LLMJudge(passing_threshold=7.0)

    def test_judge_properties(self, judge: LLMJudge) -> None:
        """Test judge property values."""
        assert judge.name == "llm_judge"
        assert judge.passing_threshold == 7.0
        assert judge.model == LLMJudge.DEFAULT_MODEL

    def test_custom_criteria(self) -> None:
        """Test judge with custom criteria."""
        custom_criteria = "Rate the response 0-10 on helpfulness."
        judge = LLMJudge(criteria=custom_criteria)
        assert judge.criteria == custom_criteria

    def test_build_prompt_with_expected(self, judge: LLMJudge) -> None:
        """Test prompt building with expected answer."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output="The answer is 4.",
            expected="4",
        )
        prompt = judge._build_prompt(result)

        assert "What is 2+2?" in prompt
        assert "The answer is 4." in prompt
        assert "Reference Answer" in prompt
        assert "4" in prompt

    def test_build_prompt_without_expected(self) -> None:
        """Test prompt building without expected answer."""
        judge = LLMJudge(include_reference=False)
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output="The answer is 4.",
            expected="4",
        )
        prompt = judge._build_prompt(result)

        assert "What is 2+2?" in prompt
        assert "The answer is 4." in prompt
        # Reference should not be included
        assert "Reference Answer" not in prompt

    def test_build_prompt_no_output(self, judge: LLMJudge) -> None:
        """Test prompt building with no output."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output=None,
        )
        prompt = judge._build_prompt(result)

        assert "(No response provided)" in prompt

    def test_parse_response_valid_json(self, judge: LLMJudge) -> None:
        """Test parsing valid JSON response."""
        response = '{"overall": 8.5, "criteria_scores": {"accuracy": 9}, "reasoning": "Good"}'
        score = judge._parse_response(response)

        assert score.overall == 8.5
        assert score.criteria_scores["accuracy"] == 9.0
        assert score.reasoning == "Good"

    def test_parse_response_json_in_markdown(self, judge: LLMJudge) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        response = """Here is my evaluation:
```json
{"overall": 7.5, "criteria_scores": {}, "reasoning": "Adequate"}
```
"""
        score = judge._parse_response(response)

        assert score.overall == 7.5

    def test_parse_response_fallback_number(self, judge: LLMJudge) -> None:
        """Test fallback to extracting numeric score."""
        response = "I would rate this response a 7 out of 10."
        score = judge._parse_response(response)

        assert score.overall == 7.0
        assert "Could not parse" in score.reasoning

    def test_parse_response_complete_failure(self, judge: LLMJudge) -> None:
        """Test handling of unparseable response."""
        response = "This response cannot be parsed as a score."
        score = judge._parse_response(response)

        # No numbers, so fallback fails too
        # Actually "parsed" is in there, but let's use something with no numbers
        response = "No numeric data here whatsoever"
        score = judge._parse_response(response)

        assert score.overall == 0.0
        assert "Failed to parse" in score.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_no_output(self, judge: LLMJudge) -> None:
        """Test evaluate with no output returns zero score."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output=None,
        )
        score = await judge.evaluate(result)

        assert score.overall == 0.0
        assert "No output" in score.reasoning

    @pytest.mark.asyncio
    async def test_evaluate_with_mock(self, judge: LLMJudge) -> None:
        """Test evaluate with mocked API response."""
        result = EvalResult(
            case_id="1",
            input="What is 2+2?",
            output="4",
            expected="4",
        )

        # Mock the Anthropic client by setting _client directly
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=(
                    '{"overall": 9.5, "criteria_scores": {"accuracy": 10, "clarity": 9}, '
                    '"reasoning": "Perfect answer"}'
                )
            )
        ]
        mock_client.messages.create.return_value = mock_response
        judge._client = mock_client

        score = await judge.evaluate(result)

        assert score.overall == 9.5
        assert score.criteria_scores["accuracy"] == 10.0
        assert score.reasoning == "Perfect answer"

    def test_compute_passes_threshold(self) -> None:
        """Test compute method returns passing result."""
        judge = LLMJudge(passing_threshold=7.0)

        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"overall": 8.5}')]
        mock_client.messages.create.return_value = mock_response
        judge._client = mock_client

        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
        )
        metric_result = judge.compute(result)

        assert metric_result.score == 8.5
        assert metric_result.passed is True

    def test_compute_fails_threshold(self) -> None:
        """Test compute method returns failing result."""
        judge = LLMJudge(passing_threshold=8.0)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"overall": 6.0}')]
        mock_client.messages.create.return_value = mock_response
        judge._client = mock_client

        result = EvalResult(
            case_id="1",
            input="test",
            output="response",
        )
        metric_result = judge.compute(result)

        assert metric_result.score == 6.0
        assert metric_result.passed is False

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self) -> None:
        """Test evaluate_with_context convenience method."""
        judge = LLMJudge()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"overall": 8.0}')]
        mock_client.messages.create.return_value = mock_response
        judge._client = mock_client

        score = await judge.evaluate_with_context(
            question="What is AI?",
            response="AI is artificial intelligence.",
            reference="Artificial Intelligence",
        )

        assert score.overall == 8.0


class TestRubricJudge:
    """Tests for RubricJudge."""

    def test_predefined_rubric_qa(self) -> None:
        """Test using predefined QA accuracy rubric."""
        judge = RubricJudge(rubric="qa_accuracy")
        assert judge.name == "llm_judge_qa_accuracy"
        assert "Factual Accuracy" in judge.criteria

    def test_predefined_rubric_rag(self) -> None:
        """Test using predefined RAG quality rubric."""
        judge = RubricJudge(rubric="rag_quality")
        assert judge.name == "llm_judge_rag_quality"
        assert "Grounding" in judge.criteria

    def test_predefined_rubric_summarization(self) -> None:
        """Test using predefined summarization rubric."""
        judge = RubricJudge(rubric="summarization")
        assert judge.name == "llm_judge_summarization"
        assert "Coverage" in judge.criteria

    def test_custom_rubric(self) -> None:
        """Test using custom rubric."""
        custom = "Rate helpfulness 0-10."
        judge = RubricJudge(custom_rubric=custom)
        assert judge.name == "llm_judge_custom"
        assert judge.criteria == custom

    def test_invalid_rubric(self) -> None:
        """Test error on invalid rubric name."""
        with pytest.raises(ValueError, match="Unknown rubric"):
            RubricJudge(rubric="nonexistent")

    def test_available_rubrics(self) -> None:
        """Test all available rubrics are accessible."""
        for rubric_name in RubricJudge.RUBRICS:
            judge = RubricJudge(rubric=rubric_name)
            assert judge.rubric_name == rubric_name
