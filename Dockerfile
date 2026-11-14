# syntax=docker/dockerfile:1
# Nexi Multi-Stage Production Dockerfile

FROM python:3.11-slim AS builder

WORKDIR /build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-cpu-linux.lock .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-cpu-linux.lock

# Bake models into cache directory
ENV HF_HOME="/opt/huggingface"
RUN mkdir -p /opt/huggingface

# Pre-download HuggingFace embedding and cross-encoder models
RUN python -c "\
from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', revision='c5ee2821d743d5d67484eb63eed4115ee9d765fa')"

# -------------------------------------------------------------
# Base Runtime Image
# -------------------------------------------------------------
FROM python:3.11-slim AS runtime-base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/opt/huggingface" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Copy virtualenv and baked models from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/huggingface /opt/huggingface

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/sh -m appuser && \
    mkdir -p /data /app && \
    chown -R appuser:appgroup /data /app /opt/huggingface

# Copy repository source code
COPY --chown=appuser:appgroup . /app

# Download LiveKit agent plugin dependency files (turn-detector / noise-cancellation)
USER appuser
RUN python -m agent.main download-files

# -------------------------------------------------------------
# Target: agent (LiveKit worker)
# -------------------------------------------------------------
FROM runtime-base AS agent

USER appuser
WORKDIR /app

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/healthz', timeout=2)" || exit 1

CMD ["python", "-m", "agent.main", "start", "--drain-timeout", "660"]

# -------------------------------------------------------------
# Target: token (Token and Admission API service)
# -------------------------------------------------------------
FROM runtime-base AS token

USER appuser
WORKDIR /app

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)" || exit 1

CMD ["python", "-m", "agent.token_service"]
