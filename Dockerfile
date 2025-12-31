# EvalOps Dashboard - Production Docker Image
# Uses CPU-only PyTorch for smaller image size

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (smaller than GPU version)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY pyproject.toml .

# Install evalops package
RUN pip install --no-cache-dir -e .

# Copy demo database (pre-populated with 24 runs, 470 cases)
COPY evalops_demo.db .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run dashboard
CMD ["streamlit", "run", "src/evalops/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
