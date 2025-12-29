"""Mock LLM target functions for demo purposes.

These functions simulate LLM behavior with deterministic outputs based on input hash.
This allows for reproducible evaluation results across runs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def _hash_input(input_text: str, seed: int = 0) -> int:
    """Generate a deterministic hash value from input text."""
    combined = f"{input_text}:{seed}"
    return int(hashlib.md5(combined.encode()).hexdigest(), 16)


def _should_succeed(input_text: str, success_rate: float, seed: int = 0) -> bool:
    """Deterministically decide if this input should produce correct output."""
    hash_val = _hash_input(input_text, seed)
    # Use modulo to get a value between 0 and 99
    threshold = hash_val % 100
    return threshold < (success_rate * 100)


# Load datasets for answer lookup
_DATASETS_DIR = Path(__file__).parent / "datasets"
_ANSWER_CACHE: dict[str, dict[str, str]] = {}


def _load_answers(dataset_name: str) -> dict[str, str]:
    """Load expected answers from a dataset file."""
    if dataset_name not in _ANSWER_CACHE:
        dataset_path = _DATASETS_DIR / f"{dataset_name}.json"
        if dataset_path.exists():
            with open(dataset_path) as f:
                data = json.load(f)
                _ANSWER_CACHE[dataset_name] = {
                    case["input"]: case["expected"] for case in data["cases"]
                }
        else:
            _ANSWER_CACHE[dataset_name] = {}
    return _ANSWER_CACHE[dataset_name]


# ============================================================================
# Q&A Models
# ============================================================================

def good_qa_model(input_text: str) -> str:
    """A good Q&A model that returns correct answers ~85% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("qa_eval")
    correct_answer = answers.get(input_text, "I don't know")

    if _should_succeed(input_text, success_rate=0.85, seed=42):
        return correct_answer
    else:
        # Return a plausible but wrong answer
        wrong_answers = {
            "Paris": "London",
            "William Shakespeare": "Charles Dickens",
            "H2O": "CO2",
            "1945": "1944",
            "Jupiter": "Saturn",
            "Leonardo da Vinci": "Michelangelo",
            "299792458": "300000000",
            "Tokyo": "Kyoto",
            "Alexander Fleming": "Louis Pasteur",
            "12": "14",
            "Gold": "Silver",
            "Neil Armstrong": "Buzz Aldrin",
            "Pacific Ocean": "Atlantic Ocean",
            "1776": "1775",
            "100": "212",
            "Albert Einstein": "Isaac Newton",
            "Canberra": "Sydney",
            "46": "48",
            "Diamond": "Quartz",
            "Stephen Hawking": "Carl Sagan",
        }
        return wrong_answers.get(correct_answer, "Unknown")


def bad_qa_model(input_text: str) -> str:
    """A poor Q&A model that returns correct answers ~60% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("qa_eval")
    correct_answer = answers.get(input_text, "I don't know")

    if _should_succeed(input_text, success_rate=0.60, seed=99):
        return correct_answer
    else:
        # Return wrong answers more often
        wrong_answers = {
            "Paris": "Berlin",
            "William Shakespeare": "Geoffrey Chaucer",
            "H2O": "NaCl",
            "1945": "1942",
            "Jupiter": "Mars",
            "Leonardo da Vinci": "Pablo Picasso",
            "299792458": "186000",
            "Tokyo": "Osaka",
            "Alexander Fleming": "Robert Koch",
            "12": "11",
            "Gold": "Copper",
            "Neil Armstrong": "John Glenn",
            "Pacific Ocean": "Indian Ocean",
            "1776": "1774",
            "100": "98",
            "Albert Einstein": "Niels Bohr",
            "Canberra": "Melbourne",
            "46": "44",
            "Diamond": "Graphite",
            "Stephen Hawking": "Richard Feynman",
        }
        return wrong_answers.get(correct_answer, "I'm not sure")


# ============================================================================
# Summarization Models
# ============================================================================

def good_summarizer(input_text: str) -> str:
    """A good summarizer that produces high-quality summaries ~85% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("summarization_eval")
    correct_summary = answers.get(input_text, "")

    if _should_succeed(input_text, success_rate=0.85, seed=123):
        return correct_summary
    else:
        # Return a degraded summary (truncated or slightly modified)
        words = correct_summary.split()
        if len(words) > 5:
            # Return partial summary
            return " ".join(words[:len(words)//2]) + "..."
        return correct_summary


def bad_summarizer(input_text: str) -> str:
    """A poor summarizer that produces low-quality summaries ~55% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("summarization_eval")
    correct_summary = answers.get(input_text, "")

    if _should_succeed(input_text, success_rate=0.55, seed=456):
        return correct_summary
    else:
        # Return very poor summaries
        hash_val = _hash_input(input_text, seed=456)
        poor_summaries = [
            "This text discusses various topics.",
            "The passage contains information.",
            "Summary not available.",
            "Multiple subjects are mentioned.",
            "The text is about things.",
        ]
        return poor_summaries[hash_val % len(poor_summaries)]


# ============================================================================
# Classification Models
# ============================================================================

def good_classifier(input_text: str) -> str:
    """A good classifier that returns correct intents ~88% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("classification_eval")
    correct_intent = answers.get(input_text, "unknown")

    if _should_succeed(input_text, success_rate=0.88, seed=789):
        return correct_intent
    else:
        # Return a plausible but wrong intent
        all_intents = [
            "cancellation", "account_help", "positive_feedback",
            "billing_issue", "upgrade_request", "technical_issue",
            "information_request", "refund_request", "negative_feedback",
            "escalation", "shipping_issue"
        ]
        hash_val = _hash_input(input_text, seed=789)
        wrong_intent = all_intents[hash_val % len(all_intents)]
        # Make sure we don't accidentally return the correct one
        if wrong_intent == correct_intent:
            wrong_intent = all_intents[(hash_val + 1) % len(all_intents)]
        return wrong_intent


def bad_classifier(input_text: str) -> str:
    """A poor classifier that returns correct intents ~62% of the time.

    Uses deterministic logic based on input hash for reproducibility.
    """
    answers = _load_answers("classification_eval")
    correct_intent = answers.get(input_text, "unknown")

    if _should_succeed(input_text, success_rate=0.62, seed=321):
        return correct_intent
    else:
        # Return wrong intents more often
        all_intents = [
            "cancellation", "account_help", "positive_feedback",
            "billing_issue", "upgrade_request", "technical_issue",
            "information_request", "refund_request", "negative_feedback",
            "escalation", "shipping_issue", "unknown"
        ]
        hash_val = _hash_input(input_text, seed=321)
        wrong_intent = all_intents[hash_val % len(all_intents)]
        if wrong_intent == correct_intent:
            wrong_intent = all_intents[(hash_val + 3) % len(all_intents)]
        return wrong_intent


# ============================================================================
# Degrading Models (for drift detection)
# ============================================================================

def degrading_qa_model(input_text: str, degradation: float = 0.0) -> str:
    """A Q&A model that degrades over time.

    Args:
        input_text: The question to answer
        degradation: Amount to reduce accuracy (0.0 to 0.3)

    Returns:
        Answer string
    """
    base_rate = 0.85 - degradation
    answers = _load_answers("qa_eval")
    correct_answer = answers.get(input_text, "I don't know")

    if _should_succeed(input_text, success_rate=max(0.5, base_rate), seed=42):
        return correct_answer
    else:
        wrong_answers = {
            "Paris": "London",
            "William Shakespeare": "Charles Dickens",
            "H2O": "CO2",
            "1945": "1944",
            "Jupiter": "Saturn",
            "Leonardo da Vinci": "Michelangelo",
            "299792458": "300000000",
            "Tokyo": "Kyoto",
            "Alexander Fleming": "Louis Pasteur",
            "12": "14",
            "Gold": "Silver",
            "Neil Armstrong": "Buzz Aldrin",
            "Pacific Ocean": "Atlantic Ocean",
            "1776": "1775",
            "100": "212",
            "Albert Einstein": "Isaac Newton",
            "Canberra": "Sydney",
            "46": "48",
            "Diamond": "Quartz",
            "Stephen Hawking": "Carl Sagan",
        }
        return wrong_answers.get(correct_answer, "Unknown")


def degrading_classifier(input_text: str, degradation: float = 0.0) -> str:
    """A classifier that degrades over time.

    Args:
        input_text: Text to classify
        degradation: Amount to reduce accuracy (0.0 to 0.3)

    Returns:
        Intent classification
    """
    base_rate = 0.88 - degradation
    answers = _load_answers("classification_eval")
    correct_intent = answers.get(input_text, "unknown")

    if _should_succeed(input_text, success_rate=max(0.5, base_rate), seed=789):
        return correct_intent
    else:
        all_intents = [
            "cancellation", "account_help", "positive_feedback",
            "billing_issue", "upgrade_request", "technical_issue",
            "information_request", "refund_request", "negative_feedback",
            "escalation", "shipping_issue"
        ]
        hash_val = _hash_input(input_text, seed=789)
        wrong_intent = all_intents[hash_val % len(all_intents)]
        if wrong_intent == correct_intent:
            wrong_intent = all_intents[(hash_val + 1) % len(all_intents)]
        return wrong_intent


# ============================================================================
# Latency Simulation
# ============================================================================

def add_simulated_latency(base_latency_ms: float = 50.0, variance: float = 20.0) -> float:
    """Generate a simulated latency value.

    Args:
        base_latency_ms: Base latency in milliseconds
        variance: Random variance to add

    Returns:
        Simulated latency in milliseconds
    """
    # Use a fixed seed for reproducibility within a session
    return base_latency_ms + (random.random() * variance * 2 - variance)
