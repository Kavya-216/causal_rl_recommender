# Minimal Dockerfile for CI-friendly runs and local testing.
# Builds a lightweight image suitable for running the API and lightweight evaluation.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps commonly needed for ML packages and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt ./
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY run_all.sh ./

# Create a non-root user for better security
RUN useradd -m appuser
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps (this keeps image small but will install everything listed)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port and set the default command to run the API server
EXPOSE 8000

USER appuser

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
