FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV NICEWIDGETS_DEMO_HOST=0.0.0.0
ENV PORT=8080

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY examples ./examples

RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "examples.main_demo.main"]
