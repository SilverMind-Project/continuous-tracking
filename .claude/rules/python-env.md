---
paths: ["**/tracking-orchestrator/**"]
---

# Python environment

- Run Python only through the project venv: `tracking-orchestrator/.venv/bin/python`, or a
  `make` target. Never system Python, bare `python`, or `pip install`.
- Sync with `cd tracking-orchestrator && uv sync --frozen --extra dev`. `uv.lock` changes only
  when you re-lock on purpose; revert any other `uv.lock` change.
- Format only the files you touched.
