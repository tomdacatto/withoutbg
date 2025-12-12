# Simplified multi-stage build for withoutbg backend
# Build context: repository root

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv early
RUN pip install --no-cache-dir uv

# Copy and build withoutbg package (MUST be before backend)
COPY packages/python/ /app/packages/python/
RUN cd /app/packages/python && uv sync --frozen --no-dev && uv pip install -e .

# Copy and build backend (which depends on withoutbg)
COPY apps/web/backend/ /app/apps/web/backend/
RUN cd /app/apps/web/backend && uv sync --frozen --no-dev

# Stage 2: Runtime (use full Python image with common libraries)
FROM python:3.12

WORKDIR /app

# Copy virtual environments
COPY --from=builder /app/packages/python/.venv /app/packages/python/.venv
COPY --from=builder /app/apps/web/backend/.venv /app/apps/web/backend/.venv

# Set PATH - packages/python must come first so backend can import withoutbg
ENV PATH="/app/apps/web/backend/.venv/bin:/app/packages/python/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY apps/web/backend/ ./backend/

# Also make packages/python available
COPY packages/python/ ./packages/python/

# Model environment variables (for future use)
ENV WITHOUTBG_DEPTH_MODEL_PATH=/app/checkpoints/depth_anything_v2_vits_slim.onnx
ENV WITHOUTBG_ISNET_MODEL_PATH=/app/checkpoints/isnet.onnx
ENV WITHOUTBG_MATTING_MODEL_PATH=/app/checkpoints/focus_matting_1.0.0.onnx
ENV WITHOUTBG_REFINER_MODEL_PATH=/app/checkpoints/focus_refiner_1.0.0.onnx

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
