FROM python:3.12-slim

WORKDIR /work

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy the pinned uv binary directly from the official image instead of
# installing via pip (avoids PyPI as a trust boundary for the build tool).
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /usr/local/bin/

COPY . /work/rf-ingress/

WORKDIR /work/rf-ingress
RUN uv sync --frozen --no-dev

USER 65532:65532
EXPOSE 8095
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8095/health || exit 1
CMD ["/work/rf-ingress/.venv/bin/uvicorn", \
     "rf_ingress.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8095"]
