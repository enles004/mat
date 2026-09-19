# ai/Dockerfile
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache

COPY ai/pyproject.toml ai/uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev --no-install-project --extra transformer

COPY ai/main.py ai/config.py ./
COPY ai/src ./src
COPY ai/scripts ./scripts
COPY ai/configs ./configs
COPY ai/data ./data
COPY ai/artifacts ./artifacts
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev --no-editable --extra transformer

RUN useradd --system --uid 10001 --user-group --create-home mat

RUN mkdir -p /app/artifacts/transformer/champion /home/mat/.cache/huggingface \
    && chown -R mat:mat /app/artifacts/transformer /home/mat/.cache/huggingface

USER mat
EXPOSE 8000
CMD ["uv", "run", "--frozen", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
