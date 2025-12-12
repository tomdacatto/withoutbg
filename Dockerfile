# Multi-stage build for withoutbg backend
# Build context should be the repository root: docker build -f Dockerfile .

# Stage 1: Build stage with uv
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy and install the core withoutbg package
COPY packages/python/pyproject.toml packages/python/uv.lock packages/python/README.md packages/python/
COPY packages/python/src/ packages/python/src/
RUN cd packages/python && uv sync --frozen --no-dev

# Copy and install backend dependencies
COPY apps/web/backend/pyproject.toml apps/web/backend/uv.lock apps/web/backend/README.md apps/web/backend/
RUN cd apps/web/backend && uv sync --frozen --no-dev

# Stage 2: Runtime stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environments from builder
COPY --from=builder /app/packages/python/.venv /app/packages/python/.venv
COPY --from=builder /app/apps/web/backend/.venv /app/apps/web/backend/.venv

# Add both virtual environments to PATH
ENV PATH="/app/apps/web/backend/.venv/bin:/app/packages/python/.venv/bin:$PATH"

# Copy application code
COPY apps/web/backend/ ./backend/

# Copy ONNX model checkpoints
COPY models/checkpoints/ ./checkpoints/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set model paths to use local files instead of downloading from HuggingFace
ENV WITHOUTBG_DEPTH_MODEL_PATH=/app/checkpoints/depth_anything_v2_vits_slim.onnx
ENV WITHOUTBG_ISNET_MODEL_PATH=/app/checkpoints/isnet.onnx
ENV WITHOUTBG_MATTING_MODEL_PATH=/app/checkpoints/focus_matting_1.0.0.onnx
ENV WITHOUTBG_REFINER_MODEL_PATH=/app/checkpoints/focus_refiner_1.0.0.onnx

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
