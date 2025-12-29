"""Dataset management for LLM evaluations.

This module provides the core data structures for defining evaluation test cases
and organizing them into versioned datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single evaluation test case.

    Attributes:
        id: Unique identifier for the case.
        input: The input/question to send to the LLM system.
        expected: Expected exact output (for exact match evaluation).
        expected_keywords: Keywords that should appear in the response.
        context: Optional context to provide with the input.
        metadata: Arbitrary metadata (category, difficulty, source_doc, etc.).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    input: str
    expected: str | None = None
    expected_keywords: list[str] | None = None
    context: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def has_expected(self) -> bool:
        """Check if this case has any expected output defined."""
        return self.expected is not None or bool(self.expected_keywords)


class EvalDataset(BaseModel):
    """A collection of evaluation test cases.

    Attributes:
        id: Unique identifier for the dataset.
        name: Human-readable name.
        description: What this dataset tests.
        version: Semantic version string.
        cases: List of evaluation cases.
        created_at: When the dataset was created.
        metadata: Arbitrary metadata for the dataset.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    version: str = "1.0.0"
    cases: list[EvalCase] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of cases in the dataset."""
        return len(self.cases)

    def __iter__(self):
        """Iterate over cases in the dataset."""
        return iter(self.cases)

    def add_case(self, case: EvalCase) -> None:
        """Add a case to the dataset."""
        self.cases.append(case)

    def filter_by_metadata(self, **kwargs: Any) -> list[EvalCase]:
        """Filter cases by metadata key-value pairs.

        Args:
            **kwargs: Metadata key-value pairs to filter by.

        Returns:
            List of cases matching all provided metadata filters.
        """
        results = []
        for case in self.cases:
            if all(case.metadata.get(k) == v for k, v in kwargs.items()):
                results.append(case)
        return results

    def to_json(self, path: Path) -> None:
        """Save the dataset to a JSON file.

        Args:
            path: File path to save to.
        """
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> EvalDataset:
        """Load a dataset from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            Loaded EvalDataset instance.
        """
        data = json.loads(path.read_text())
        return cls.model_validate(data)

    @classmethod
    def from_cases(
        cls,
        name: str,
        cases: list[dict[str, Any]],
        description: str = "",
        version: str = "1.0.0",
    ) -> EvalDataset:
        """Create a dataset from a list of case dictionaries.

        Args:
            name: Dataset name.
            cases: List of case dictionaries.
            description: Dataset description.
            version: Dataset version.

        Returns:
            New EvalDataset instance.
        """
        eval_cases = [EvalCase.model_validate(c) for c in cases]
        return cls(
            name=name,
            description=description,
            version=version,
            cases=eval_cases,
        )
