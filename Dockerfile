# ============================================================================
# Mirix Backend Dockerfile
# ============================================================================
# Multi-stage build for optimal image size and security
# ============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Default Mirix settings
    MIRIX_DIR=/app/data \
    MIRIX_IMAGES_DIR=/app/data/images \
    MIRIX_LOG_LEVEL=INFO \
    MIRIX_LOG_TO_CONSOLE=true

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash mirix

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mirix:mirix mirix/ ./mirix/
COPY --chown=mirix:mirix pyproject.toml ./
COPY --chown=mirix:mirix README.md ./

# Create data directories
RUN mkdir -p /app/data/images && chown -R mirix:mirix /app/data

# Switch to non-root user
USER mirix

# Expose port
EXPOSE 8531

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8531/health || exit 1

# Default command - run the FastAPI server
CMD ["uvicorn", "mirix.server.rest_api:app", "--host", "0.0.0.0", "--port", "8531"]

