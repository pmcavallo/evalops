"""LLM-as-Judge evaluator for subjective quality assessment.

This module provides the LLMJudge class that uses Claude to evaluate
LLM outputs based on configurable criteria.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from evalops.core.metrics import Metric, MetricResult
from evalops.core.runner import EvalResult


@dataclass
class JudgeScore:
    """Structured score from LLM judge evaluation.

    Attributes:
        overall: Overall score (0-10).
        criteria_scores: Individual scores per criterion.
        reasoning: Judge's explanation for the scores.
        raw_response: The raw response from the judge LLM.
    """

    overall: float
    criteria_scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    raw_response: str = ""


class LLMJudge(Metric):
    """Uses Claude to evaluate response quality.

    The judge evaluates responses based on configurable criteria,
    returning structured scores with explanations.

    Example:
        ```python
        judge = LLMJudge(
            criteria=\"\"\"
            Evaluate on:
            1. Accuracy (0-10): Is the information correct?
            2. Completeness (0-10): Does it fully answer the question?
            3. Clarity (0-10): Is it well-written?
            \"\"\",
            passing_threshold=7.0,
        )
        result = await judge.evaluate(result)
        ```

    Attributes:
        model: Claude model to use for judging.
        criteria: Description of evaluation criteria.
        passing_threshold: Minimum overall score to pass.
        include_reference: Whether to show expected answer to judge.
        temperature: Sampling temperature for judge (lower = more consistent).
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    DEFAULT_CRITERIA = """
Evaluate the response on the following criteria, scoring each from 0-10:

1. **Accuracy** (0-10): Is the information factually correct? Does it match the reference answer if provided?
2. **Completeness** (0-10): Does the response fully address the question? Are there any missing key points?
3. **Clarity** (0-10): Is the response well-written, clear, and easy to understand?

Provide your scores and brief reasoning for each criterion.
"""

    def __init__(
        self,
        criteria: str | None = None,
        model: str = DEFAULT_MODEL,
        passing_threshold: float = 7.0,
        include_reference: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self.model = model
        self.passing_threshold = passing_threshold
        self.include_reference = include_reference
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None

    @property
    def name(self) -> str:
        return "llm_judge"

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _build_prompt(self, result: EvalResult) -> str:
        """Build the evaluation prompt for the judge."""
        parts = [
            "You are an expert evaluator assessing the quality of an AI assistant's response.",
            "",
            "## Evaluation Criteria",
            self.criteria,
            "",
            "## Input Question",
            result.input,
            "",
            "## Response to Evaluate",
            result.output or "(No response provided)",
        ]

        if self.include_reference and result.expected:
            parts.extend([
                "",
                "## Reference Answer",
                result.expected,
            ])

        parts.extend([
            "",
            "## Output Format",
            "Respond with a JSON object containing:",
            '- "overall": overall score 0-10',
            '- "criteria_scores": object with score for each criterion',
            '- "reasoning": brief explanation of your evaluation',
            "",
            "Example:",
            '{"overall": 8.5, "criteria_scores": {"accuracy": 9, "completeness": 8, "clarity": 8.5}, "reasoning": "The response is accurate and clear..."}',
            "",
            "Respond ONLY with the JSON object, no other text.",
        ])

        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> JudgeScore:
        """Parse the judge's response into a structured score."""
        # Try to extract JSON from the response
        # Handle cases where model might wrap JSON in markdown code blocks
        # Use a pattern that can handle nested braces
        json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return JudgeScore(
                    overall=float(data.get("overall", 0)),
                    criteria_scores={
                        k: float(v)
                        for k, v in data.get("criteria_scores", {}).items()
                    },
                    reasoning=data.get("reasoning", ""),
                    raw_response=response_text,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to extract a numeric score
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response_text)
        if numbers:
            # Take first number as overall score
            score = min(float(numbers[0]), 10.0)
            return JudgeScore(
                overall=score,
                reasoning="Could not parse structured response",
                raw_response=response_text,
            )

        # Complete fallback
        return JudgeScore(
            overall=0.0,
            reasoning="Failed to parse judge response",
            raw_response=response_text,
        )

    def compute(self, result: EvalResult) -> MetricResult:
        """Synchronous compute - calls the async version.

        Note: For async usage, call evaluate() directly.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.evaluate(result))
                    judge_score = future.result()
            else:
                judge_score = loop.run_until_complete(self.evaluate(result))
        except RuntimeError:
            judge_score = asyncio.run(self.evaluate(result))

        return MetricResult(
            name=self.name,
            score=judge_score.overall,
            passed=judge_score.overall >= self.passing_threshold,
            details={
                "criteria_scores": judge_score.criteria_scores,
                "reasoning": judge_score.reasoning,
                "threshold": self.passing_threshold,
                "model": self.model,
            },
        )

    async def evaluate(self, result: EvalResult) -> JudgeScore:
        """Evaluate a result using the LLM judge.

        Args:
            result: The evaluation result to judge.

        Returns:
            JudgeScore with overall score and per-criterion scores.
        """
        if result.output is None:
            return JudgeScore(
                overall=0.0,
                reasoning="No output to evaluate",
            )

        prompt = self._build_prompt(result)

        # Use sync client in async context (anthropic handles this)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text
        return self._parse_response(response_text)

    async def evaluate_with_context(
        self,
        question: str,
        response: str,
        reference: str | None = None,
        context: str | None = None,
    ) -> JudgeScore:
        """Evaluate a response with explicit parameters.

        This is a convenience method when you don't have an EvalResult.

        Args:
            question: The input question.
            response: The response to evaluate.
            reference: Optional reference/expected answer.
            context: Optional additional context.

        Returns:
            JudgeScore with evaluation results.
        """
        # Create a temporary EvalResult
        temp_result = EvalResult(
            case_id="temp",
            input=question,
            output=response,
            expected=reference,
            metadata={"context": context} if context else {},
        )
        return await self.evaluate(temp_result)


class RubricJudge(LLMJudge):
    """LLM Judge with predefined rubric scoring.

    Provides structured rubrics for common evaluation scenarios.
    """

    RUBRICS = {
        "qa_accuracy": """
Evaluate the Q&A response on these criteria (0-10 each):

1. **Factual Accuracy** (0-10): Are all stated facts correct? No hallucinations?
   - 10: All facts verified correct
   - 7-9: Minor inaccuracies that don't affect main answer
   - 4-6: Some incorrect facts but core answer is right
   - 1-3: Major factual errors
   - 0: Completely wrong or hallucinated

2. **Answer Relevance** (0-10): Does it directly answer the question?
   - 10: Directly and completely answers the question
   - 7-9: Answers the question with minor tangents
   - 4-6: Partially answers, missing key aspects
   - 1-3: Mostly off-topic
   - 0: Does not address the question

3. **Completeness** (0-10): Are all parts of the question addressed?
   - 10: All aspects fully covered
   - 7-9: Most aspects covered well
   - 4-6: Some aspects missing
   - 1-3: Very incomplete
   - 0: Empty or no meaningful content
""",
        "rag_quality": """
Evaluate the RAG response on these criteria (0-10 each):

1. **Grounding** (0-10): Is the response grounded in the provided context?
   - 10: Every claim directly supported by context
   - 7-9: Most claims supported, minimal extrapolation
   - 4-6: Mix of grounded and ungrounded claims
   - 1-3: Mostly not grounded in context
   - 0: Completely ignores context

2. **Faithfulness** (0-10): Does it avoid contradicting the context?
   - 10: No contradictions with context
   - 7-9: Minor inconsistencies
   - 4-6: Some contradictions
   - 1-3: Major contradictions
   - 0: Directly contradicts context

3. **Usefulness** (0-10): Is the response helpful for the user's query?
   - 10: Extremely helpful and actionable
   - 7-9: Helpful with good information
   - 4-6: Somewhat helpful
   - 1-3: Not very helpful
   - 0: Unhelpful or misleading
""",
        "summarization": """
Evaluate the summary on these criteria (0-10 each):

1. **Coverage** (0-10): Does it capture the key points?
   - 10: All important points included
   - 7-9: Most important points, minor omissions
   - 4-6: Some key points missing
   - 1-3: Major points missing
   - 0: Does not capture main content

2. **Conciseness** (0-10): Is it appropriately brief without losing meaning?
   - 10: Optimal length, no fluff
   - 7-9: Good length with minimal redundancy
   - 4-6: Could be more concise
   - 1-3: Too verbose or too brief
   - 0: Completely inappropriate length

3. **Coherence** (0-10): Is it well-organized and readable?
   - 10: Excellent flow and structure
   - 7-9: Good organization
   - 4-6: Adequate structure
   - 1-3: Disorganized
   - 0: Incoherent
""",
    }

    def __init__(
        self,
        rubric: str = "qa_accuracy",
        custom_rubric: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a predefined or custom rubric.

        Args:
            rubric: Name of predefined rubric (qa_accuracy, rag_quality, summarization).
            custom_rubric: Custom rubric text (overrides predefined).
            **kwargs: Additional arguments passed to LLMJudge.
        """
        if custom_rubric:
            criteria = custom_rubric
        elif rubric in self.RUBRICS:
            criteria = self.RUBRICS[rubric]
        else:
            available = ", ".join(self.RUBRICS.keys())
            raise ValueError(f"Unknown rubric '{rubric}'. Available: {available}")

        super().__init__(criteria=criteria, **kwargs)
        self.rubric_name = rubric if not custom_rubric else "custom"

    @property
    def name(self) -> str:
        return f"llm_judge_{self.rubric_name}"
