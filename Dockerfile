FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl && rm -rf /var/lib/apt/lists/*
COPY requirements-full.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt
COPY . .
RUN useradd -m appuser && chown -R appuser:appuser /app && mkdir -p data/benchmarks/v1.0
USER appuser
EXPOSE 8080 8501
ENV PYTHONUNBUFFERED=1 API_AUTH_REQUIRED=false
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8080/health || exit 1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
