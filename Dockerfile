# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --no-dev --no-editable
RUN uv run findex index data --out /app/data-index.bin --positions

FROM python:3.14-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" INDEX_PATH=/app/data-index.bin
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/data-index.bin /app/data-index.bin
COPY --from=builder --chown=app:app /app/data /app/data
COPY --chown=app:app src ./src
RUN mkdir -p /data && chown app:app /data
USER app
EXPOSE 8000
CMD ["findex", "serve"]
