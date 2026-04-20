FROM python:3.12-slim AS base
WORKDIR /app

FROM base AS builder
RUN pip install --no-cache-dir pip uv
COPY pyproject.toml ./
RUN uv pip install --system --compile-bytecode -e ".[dev]"

FROM base
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
EXPOSE 8081
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8081"]
