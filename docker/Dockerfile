FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY pyproject.toml .

# Install package
RUN pip install -e .

# Default command: run dashboard
EXPOSE 8501
CMD ["streamlit", "run", "src/evalops/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
