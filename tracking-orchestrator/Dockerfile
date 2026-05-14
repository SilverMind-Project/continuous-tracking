FROM python:3.12-slim

# Build context is the monorepo root so triton-shared is reachable.
# We mirror the on-disk layout so uv.sources path "../../triton-shared"
# resolves correctly: /work/continuous-tracking/tracking-orchestrator/../../triton-shared
#                   = /work/triton-shared/

WORKDIR /work

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

COPY triton-shared/ triton-shared/
COPY continuous-tracking/tracking-orchestrator/ continuous-tracking/tracking-orchestrator/

WORKDIR /work/continuous-tracking/tracking-orchestrator
RUN uv sync --frozen --extra triton

USER 65532:65532
EXPOSE 8500
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8500/health || exit 1
CMD ["/work/continuous-tracking/tracking-orchestrator/.venv/bin/uvicorn", \
     "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8500"]
