# Multi-stage Dockerfile that merges the devcontainer and CI/local images.
# - First stage builds wheels for faster installs and isolates build dependencies.
# - Final stage installs runtime deps and contains the application source.

FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build-time dependencies needed to build wheels for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Build wheels to speed up installation in the final image
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r /app/requirements.txt || true

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels from builder (if any) and install
COPY --from=builder /wheels /wheels
RUN pip install --upgrade pip && pip install --no-cache-dir /wheels/* || true

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

# Install any remaining requirements (fallback if wheels failed)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt || true

EXPOSE 8000

USER appuser

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
