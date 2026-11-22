FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY contracts.py ./
COPY adapters ./adapters
COPY ares ./ares
COPY domain ./domain
COPY fusion ./fusion
COPY models ./models
COPY ui ./ui
COPY viz ./viz

RUN uvicorn --version

EXPOSE 8080

CMD ["uvicorn", "ui.nyc_live_app:app", "--host", "0.0.0.0", "--port", "8080"]
