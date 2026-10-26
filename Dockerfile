# ==============================================================================
# Production Dockerfile for Enterprise HR Policy RAG Assistant
#
# Multi-stage minimal container with non-root execution, uv package management,
# and Cloud Run dynamic port binding.
# ==============================================================================

# Stage 1: Retrieve official uv binary
FROM ghcr.io/astral-sh/uv:0.5.24 AS uv_bin

# Stage 2: Runtime environment
FROM python:3.11-slim AS runtime

# Set non-interactive Debian frontend and secure Python runtime defaults
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    PATH="/app/.venv/bin:$PATH"

# Install minimal runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from official image
COPY --from=uv_bin /uv /bin/uv

# Create non-root system user and group (UID/GID 10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy dependency specifications first for optimal layer caching
COPY pyproject.toml README.md ./

# Create virtual environment and install dependencies via uv
RUN uv venv /app/.venv && \
    uv pip install --no-cache -e .

# Copy application directories and assets
COPY app/ ./app/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/
COPY scripts/ ./scripts/
COPY .env.example ./.env.example

# Set up runtime directories with proper ownership for non-root execution
RUN mkdir -p /app/data/processed /app/data/raw /app/data/cache /app/data/evaluation && \
    chown -R appuser:appgroup /app /home/appuser

# Switch to unprivileged non-root user
USER appuser

# Expose standard Cloud Run HTTP container port
EXPOSE 8080

# Health check using Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Launch Streamlit with dynamic $PORT binding for Cloud Run compatibility
CMD ["sh", "-c", "exec streamlit run app/main.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
