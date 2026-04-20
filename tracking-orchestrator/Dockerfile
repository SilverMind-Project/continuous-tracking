FROM python:3.12-slim
WORKDIR /app

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY . .
RUN uv sync --frozen

USER 65532:65532
EXPOSE 8300
CMD ["/app/.venv/bin/uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8300"]
