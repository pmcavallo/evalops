# EvalOps

Production-grade evaluation framework for LLM applications.

[![CI](https://github.com/pmcavallo/evalops/actions/workflows/ci.yml/badge.svg)](https://github.com/pmcavallo/evalops/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-pmcavallo%2Fevalops-blue.svg)](https://hub.docker.com/r/pmcavallo/evalops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Problem

Traditional software testing doesn't work for LLMs. Outputs are non-deterministic - a correct answer today might be phrased differently tomorrow. Simple string matching fails, and manual review doesn't scale.

## The Solution

EvalOps provides systematic evaluation with:

- **Semantic Similarity**: BERT-based comparison that understands meaning, not just strings
- **Configurable Metrics**: Accuracy, similarity, latency thresholds you define
- **Drift Detection**: Catch quality degradation before users do
- **A/B Comparison**: Statistical testing for prompt/model changes
- **Full Observability**: LangSmith integration, structured logging, tracing

## Quick Start

### Docker (Recommended)

```bash
docker pull pmcavallo/evalops:latest
docker run -p 8501:8501 pmcavallo/evalops:latest
```

Open [http://localhost:8501](http://localhost:8501) to view the dashboard.

### Local Installation

```bash
pip install evalops
```

```python
from evalops.core import Dataset, EvalRunner
from evalops.core.metrics import Accuracy, SemanticSimilarity

# Define test cases
dataset = Dataset.from_list([
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
])

# Your LLM function
def my_llm(input_text: str) -> str:
    # Call your model here
    return "..."

# Run evaluation
runner = EvalRunner()
result = runner.run(
    dataset=dataset,
    target_fn=my_llm,
    metrics=[
        Accuracy(threshold=0.8),
        SemanticSimilarity(threshold=0.7),
    ],
)

print(f"Pass Rate: {result.pass_rate:.1%}")
```

## CLI

```bash
# Run evaluation from JSON dataset
evalops run --dataset qa_cases.json --target my_module:llm_function

# Compare two runs
evalops compare --baseline RUN_ID_A --candidate RUN_ID_B

# Check for drift against baseline
evalops drift --baseline BASELINE_NAME --dataset qa_cases

# Save a baseline
evalops baseline save --run RUN_ID --name "production_v1"

# Launch dashboard
evalops-dashboard
```

## API Server

```bash
# Start the API
evalops-api

# Or with uvicorn directly
uvicorn evalops.api.app:app --reload
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/evaluate` | Run evaluation |
| GET | `/runs` | List evaluation runs |
| GET | `/runs/{id}` | Get run details |
| GET | `/metrics` | List available metrics |
| GET | `/health` | Health check |

## Dashboard

Interactive Streamlit dashboard for exploring results:

```bash
evalops-dashboard
```

**Features:**
- Overview with pass rate trends
- Run explorer with filtering
- Detailed case-by-case analysis
- A/B comparison interface
- Drift monitoring with alerts

## Metrics

| Metric | Description |
|--------|-------------|
| `Accuracy` | Exact or fuzzy string match |
| `SemanticSimilarity` | BERT embedding cosine similarity |
| `Latency` | Response time threshold |
| `ContainsKeywords` | Required keywords present |
| `LLMJudge` | LLM-as-judge evaluation |

## Architecture

```
evalops/
├── core/           # Dataset, Runner, Metrics
├── comparison/     # A/B testing, Drift detection
├── storage/        # SQLite/PostgreSQL persistence
├── observability/  # Logging, Tracing, LangSmith
├── api/            # FastAPI REST endpoints
├── cli/            # Command-line interface
└── dashboard/      # Streamlit visualization
```

## Deployment

### Docker

Build and run locally:

```bash
docker build -t evalops .
docker run -p 8501:8501 evalops
```

Or use the pre-built image from Docker Hub:

```bash
docker pull pmcavallo/evalops:latest
docker run -d -p 8501:8501 --name evalops pmcavallo/evalops:latest
```

### AWS Deployment

EvalOps is deployed on AWS with the following architecture:

| Service | Purpose |
|---------|---------|
| EC2 (t3.micro) | Hosts Streamlit dashboard via Docker |
| DynamoDB | Stores evaluation runs, cases, and baselines |
| IAM | Least-privilege access for deployment user |

**Deploy to your own AWS:**

1. Create EC2 instance (Amazon Linux 2023, t3.micro)
2. Install Docker: `sudo dnf install docker -y && sudo systemctl start docker`
3. Pull and run: `docker run -d -p 8501:8501 pmcavallo/evalops:latest`
4. Configure Security Group to allow port 8501

## Storage

Results persist to SQLite (local) or PostgreSQL (production):

```python
from evalops.storage import EvalRepository

repo = EvalRepository("sqlite:///evalops.db")
# or
repo = EvalRepository("postgresql://user:pass@host/db")

repo.save_run(result, name="nightly_eval", tags=["production"])
```

## Observability

```python
from evalops.observability import LangSmithTracer

# Automatic tracing to LangSmith
tracer = LangSmithTracer(project_name="my-evals")
runner = EvalRunner(tracer=tracer)
```

## Development

```bash
# Clone and install
git clone https://github.com/pmcavallo/evalops.git
cd evalops
pip install -e ".[dev]"

# Run tests (285 passing)
pytest

# Run with coverage
pytest --cov=evalops
```

## Tech Stack

- **Python 3.10+** - Core language
- **sentence-transformers** - BERT embeddings for semantic similarity
- **FastAPI** - REST API
- **Streamlit** - Dashboard
- **SQLAlchemy 2.0** - Database ORM
- **Docker** - Containerization
- **AWS (EC2, DynamoDB)** - Cloud deployment

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Paulo Cavallo** - [GitHub](https://github.com/pmcavallo) | [LinkedIn](https://linkedin.com/in/paulocavallo)
