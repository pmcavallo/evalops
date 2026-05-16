"""
Basic EvalOps Example

This example shows how to evaluate a simple Q&A function.
"""

from evalops.core import Dataset, EvalRunner
from evalops.core.metrics import Accuracy, SemanticSimilarity


# Simple mock LLM for demonstration
def simple_qa_bot(question: str) -> str:
    """A simple rule-based Q&A bot for demonstration."""
    responses = {
        "what is 2+2": "4",
        "what is the capital of france": "Paris",
        "who wrote hamlet": "William Shakespeare",
        "what is python": "Python is a programming language",
    }

    key = question.lower().strip("?")
    return responses.get(key, "I don't know")


def main():
    # Create test dataset
    dataset = Dataset.from_list([
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "Who wrote Hamlet?", "expected": "Shakespeare"},
        {"input": "What is Python?", "expected": "A programming language"},
        {"input": "What is the meaning of life?", "expected": "42"},  # Will fail
    ])

    # Configure metrics
    metrics = [
        Accuracy(threshold=0.8),  # 80% accuracy required
        SemanticSimilarity(threshold=0.7),  # 70% semantic match
    ]

    # Run evaluation
    runner = EvalRunner()
    result = runner.run(
        dataset=dataset,
        target_fn=simple_qa_bot,
        metrics=metrics,
    )

    # Print results
    print(f"\n{'='*50}")
    print("EvalOps Results")
    print(f"{'='*50}")
    print(f"Total Cases: {result.total_cases}")
    print(f"Passed: {result.passed_cases}")
    print(f"Failed: {result.failed_cases}")
    print(f"Pass Rate: {result.pass_rate:.1%}")
    print(f"Avg Latency: {result.avg_latency_ms:.1f}ms")
    print(f"{'='*50}")

    # Show individual results
    print("\nCase Results:")
    for case in result.cases:
        status = "✅" if case.metrics_passed else "❌"
        print(f"  {status} {case.input[:40]}...")
        if not case.metrics_passed:
            print(f"      Expected: {case.expected}")
            print(f"      Got: {case.output}")


if __name__ == "__main__":
    main()
