"""Unit tests for the Dataset module."""

import tempfile
from pathlib import Path

import pytest

from evalops import EvalCase, EvalDataset


class TestEvalCase:
    """Tests for EvalCase."""

    def test_create_with_expected(self) -> None:
        """Test creating a case with expected output."""
        case = EvalCase(input="What is 2+2?", expected="4")
        assert case.input == "What is 2+2?"
        assert case.expected == "4"
        assert case.has_expected() is True

    def test_create_with_keywords(self) -> None:
        """Test creating a case with expected keywords."""
        case = EvalCase(
            input="Explain AI",
            expected_keywords=["machine learning", "neural"],
        )
        assert case.expected is None
        assert case.expected_keywords == ["machine learning", "neural"]
        assert case.has_expected() is True

    def test_create_without_expected(self) -> None:
        """Test creating a case without expected output."""
        case = EvalCase(input="Tell me a joke")
        assert case.has_expected() is False

    def test_auto_generates_id(self) -> None:
        """Test that cases get auto-generated IDs."""
        case1 = EvalCase(input="test1")
        case2 = EvalCase(input="test2")
        assert case1.id != case2.id

    def test_metadata(self) -> None:
        """Test case metadata."""
        case = EvalCase(
            input="test",
            metadata={"category": "math", "difficulty": "hard"},
        )
        assert case.metadata["category"] == "math"
        assert case.metadata["difficulty"] == "hard"


class TestEvalDataset:
    """Tests for EvalDataset."""

    def test_create_empty(self) -> None:
        """Test creating an empty dataset."""
        ds = EvalDataset(name="empty")
        assert len(ds) == 0
        assert ds.name == "empty"

    def test_create_with_cases(self, sample_dataset: EvalDataset) -> None:
        """Test creating a dataset with cases."""
        assert len(sample_dataset) == 3
        assert sample_dataset.name == "test_dataset"

    def test_iterate(self, sample_dataset: EvalDataset) -> None:
        """Test iterating over dataset cases."""
        inputs = [case.input for case in sample_dataset]
        assert len(inputs) == 3
        assert "What is 2 + 2?" in inputs

    def test_add_case(self) -> None:
        """Test adding a case to a dataset."""
        ds = EvalDataset(name="test")
        ds.add_case(EvalCase(input="new case"))
        assert len(ds) == 1

    def test_filter_by_metadata(self, sample_dataset: EvalDataset) -> None:
        """Test filtering cases by metadata."""
        easy_cases = sample_dataset.filter_by_metadata(difficulty="easy")
        assert len(easy_cases) == 2

        math_cases = sample_dataset.filter_by_metadata(category="math")
        assert len(math_cases) == 1

    def test_save_and_load_json(self, sample_dataset: EvalDataset) -> None:
        """Test saving and loading a dataset from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.json"
            sample_dataset.to_json(path)

            loaded = EvalDataset.from_json(path)
            assert loaded.name == sample_dataset.name
            assert len(loaded) == len(sample_dataset)

    def test_from_cases(self) -> None:
        """Test creating a dataset from case dictionaries."""
        cases = [
            {"input": "q1", "expected": "a1"},
            {"input": "q2", "expected": "a2"},
        ]
        ds = EvalDataset.from_cases("test", cases)
        assert len(ds) == 2
        assert ds.cases[0].expected == "a1"
