# Simplified multi-stage build for withoutbg backend
# Build context: repository root

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv early
RUN pip install --no-cache-dir uv

# Copy and build withoutbg package
COPY packages/python/pyproject.toml packages/python/uv.lock packages/python/README.md packages/python/
COPY packages/python/src/ packages/python/src/
RUN cd packages/python && uv sync --frozen --no-dev

# Copy and build backend
COPY apps/web/backend/pyproject.toml apps/web/backend/uv.lock apps/web/backend/README.md apps/web/backend/
RUN cd apps/web/backend && uv sync --frozen --no-dev

# Stage 2: Runtime (use full Python image with common libraries)
FROM python:3.12

WORKDIR /app

# Copy virtual environments
COPY --from=builder /app/packages/python/.venv /app/packages/python/.venv
COPY --from=builder /app/apps/web/backend/.venv /app/apps/web/backend/.venv

# Set PATH
ENV PATH="/app/apps/web/backend/.venv/bin:/app/packages/python/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY apps/web/backend/ ./backend/

# Model environment variables
ENV WITHOUTBG_DEPTH_MODEL_PATH=/app/checkpoints/depth_anything_v2_vits_slim.onnx
ENV WITHOUTBG_ISNET_MODEL_PATH=/app/checkpoints/isnet.onnx
ENV WITHOUTBG_MATTING_MODEL_PATH=/app/checkpoints/focus_matting_1.0.0.onnx
ENV WITHOUTBG_REFINER_MODEL_PATH=/app/checkpoints/focus_refiner_1.0.0.onnx

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
