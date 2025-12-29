# EvalOps Documentation

## Getting Started

### Installation

```bash
pip install evalops
```

### Basic Usage

```python
from evalops.core import Dataset, EvalRunner
from evalops.core.metrics import Accuracy, SemanticSimilarity

# Load your test cases
dataset = Dataset.from_json("test_cases.json")

# Run evaluation
runner = EvalRunner()
result = runner.run(
    dataset=dataset,
    target_fn=your_llm_function,
    metrics=[Accuracy(threshold=0.8)],
)

print(f"Pass Rate: {result.pass_rate:.1%}")
```

## Guides

- [Metrics Reference](metrics.md) - Available metrics and configuration
- [CLI Usage](cli.md) - Command-line interface
- [API Reference](api.md) - REST API endpoints
- [Dashboard Guide](dashboard.md) - Visual interface

## Examples

See the `examples/` directory for:
- `basic_evaluation.py` - Simple Q&A evaluation
- More examples coming soon

## Architecture

EvalOps is organized into layers:

1. **Core** - Dataset loading, evaluation runner, metrics
2. **Comparison** - A/B testing, drift detection
3. **Storage** - Persistence to SQLite/PostgreSQL
4. **Observability** - Logging, tracing, metrics collection
5. **Interfaces** - CLI, API, Dashboard
