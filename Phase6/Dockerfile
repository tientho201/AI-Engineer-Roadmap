#Demo6
# ---------- Stage 1: build ----------
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --frozen: cài ĐÚNG phiên bản trong lock -> build tái hiện được
RUN uv sync --frozen --no-dev

# ---------- Stage 2: runtime ----------
FROM python:3.12-slim

# KHÔNG chạy bằng root
RUN useradd -m -u 1000 appuser

# Chỉ cài thư viện hệ thống cần cho runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --chown=appuser:appuser src/ ./src/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]