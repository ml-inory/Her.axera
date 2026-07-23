FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/app/ app/
COPY frontend/static/ frontend/static/
RUN mkdir -p data

EXPOSE 8080

# Health check
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
