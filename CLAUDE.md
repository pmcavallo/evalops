# EvalOps - Project CLAUDE.md

## Project Overview
**Name:** EvalOps  
**One-liner:** Production-grade LLM evaluation and observability platform  
**Status:** Day 8 Complete - 285 tests passing

## What EvalOps Does

EvalOps helps teams systematically test, monitor, and compare LLM application outputs using:
- Semantic similarity (BERT embeddings)
- Configurable accuracy thresholds
- Statistical drift detection
- A/B comparison with significance testing
- Full observability (LangSmith, structured logging)

## Tech Stack
- **Python 3.10+**
- **sentence-transformers** - BERT embeddings for semantic similarity
- **LangSmith** - LLM tracing and observability
- **structlog** - Structured logging
- **FastAPI** - REST API
- **Typer** - CLI interface
- **Streamlit** - Dashboard
- **SQLAlchemy 2.0** - Database ORM
- **SQLite/PostgreSQL** - Storage
- **pytest** - Testing (285 tests)
- **GitHub Actions** - CI/CD

## Directory Structure
```
evalops/
├── src/evalops/          # Main package
│   ├── core/             # Dataset, Runner, Metrics (including SemanticSimilarity)
│   ├── comparison/       # A/B testing, drift detection, regression
│   ├── storage/          # SQLAlchemy models, repository, migrations
│   ├── observability/    # LangSmith, logging, metrics collection
│   ├── api/              # FastAPI REST endpoints
│   ├── cli/              # Typer CLI
│   └── dashboard/        # Streamlit visualization
├── tests/                # 285 unit tests
├── demo/                 # Demo datasets and mock targets
├── scripts/              # Utility scripts (populate_demo.py)
├── docker/               # Dockerfile
└── examples/             # Usage examples
```

## Key Commands
```bash
# Run tests
py -m pytest

# Run specific test file
py -m pytest tests/unit/test_metrics.py -v

# Run dashboard
py -m streamlit run src/evalops/dashboard/app.py

# Run API
py -m uvicorn evalops.api.app:app --reload

# CLI examples
evalops run --dataset qa_cases.json --target my_module:llm_function
evalops compare --baseline RUN_A --candidate RUN_B
evalops drift --baseline prod_v1 --dataset qa_cases
evalops-dashboard
```

## What's Implemented

### Core (Day 1-2)
- `Dataset` - Load from JSON/list, versioning, iteration
- `EvalRunner` - Execute evaluations with metrics and observability
- `EvalResult` / `EvalRunResult` - Result tracking

### Metrics (Day 2)
- `Accuracy` - Exact/fuzzy string match
- `SemanticSimilarity` - BERT embedding cosine similarity
- `ContainsKeywords` - Keyword presence
- `Latency` - Response time thresholds
- `LLMJudge` - LLM-as-judge evaluation

### Observability (Day 3)
- `LangSmithTracer` - Distributed tracing integration
- `EvalLogger` - Structured JSON logging
- `MetricsCollector` - Runtime statistics

### Comparison (Day 4)
- `ABComparison` - Statistical A/B testing with effect size
- `DriftDetector` - Baseline comparison with alerts
- `RegressionTester` - Threshold-based regression checks

### Storage (Day 5)
- SQLAlchemy models for runs, cases, baselines
- `EvalRepository` - CRUD operations
- `DatabaseManager` - Health checks, migrations

### CLI (Day 6)
- `evalops run` - Execute evaluations
- `evalops compare` - A/B comparison
- `evalops drift` - Drift detection
- `evalops baseline` - Manage baselines
- `evalops-dashboard` - Launch dashboard

### API (Day 6)
- POST `/evaluate` - Run evaluation
- GET `/runs` - List runs
- GET `/runs/{id}` - Run details
- GET `/metrics` - Available metrics
- GET `/health` - Health check

### Dashboard (Day 7-8)
- Overview with trends
- Run Explorer with filters
- Run Detail with case breakdown
- A/B Comparison interface
- Drift Monitor with alerts
- Guide page with documentation
- Settings for database config

### Demo Infrastructure (Day 8)
- 3 demo datasets (Q&A, Classification, Summarization)
- Mock LLM targets with configurable accuracy
- Degrading models for drift simulation
- 24 pre-populated runs with 470 cases

## Test Coverage
- 285 tests passing
- Coverage across all modules
- Unit tests for each component
- Dashboard tests (skipped without streamlit)

## Entry Points
```python
# Package
from evalops.core import Dataset, EvalRunner
from evalops.core.metrics import Accuracy, SemanticSimilarity
from evalops.storage import EvalRepository

# CLI
evalops --help
evalops-dashboard
evalops-api
```

## Next Steps (Future)
- AWS Lambda deployment
- Bedrock integration
- Real LLM evaluation examples
- Additional metrics (ROUGE, BLEU)
- PostgreSQL production setup
