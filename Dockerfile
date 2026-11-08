# Production Dockerfile for KESSLER-SENTINEL Conjunction Triage Engine
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=9696

WORKDIR /app

# Install runtime utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and serialized model artifact
COPY src/ ./src/
COPY predict.py .
COPY model_kessler_sentinel_v1.bin .

# Service healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9696/health || exit 1

EXPOSE 9696

# Launch ASGI service
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
