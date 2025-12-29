"""Pytest fixtures for EvalOps tests."""

import pytest

from evalops import EvalCase, EvalDataset


@pytest.fixture
def sample_case() -> EvalCase:
    """Create a sample evaluation case."""
    return EvalCase(
        input="What is 2 + 2?",
        expected="4",
        metadata={"category": "math", "difficulty": "easy"},
    )


@pytest.fixture
def sample_dataset(sample_case: EvalCase) -> EvalDataset:
    """Create a sample dataset with a few cases."""
    return EvalDataset(
        name="test_dataset",
        description="A test dataset",
        cases=[
            sample_case,
            EvalCase(
                input="What is the capital of France?",
                expected="Paris",
                metadata={"category": "geography", "difficulty": "easy"},
            ),
            EvalCase(
                input="Explain quantum computing",
                expected_keywords=["qubit", "superposition"],
                metadata={"category": "science", "difficulty": "hard"},
            ),
        ],
    )
