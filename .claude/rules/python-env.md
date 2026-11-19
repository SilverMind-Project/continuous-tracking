---
paths: ["**/tracking-orchestrator/**", "**/rf-ingress/**"]
---

# Python environment

- Run Python only through each sub-project's venv (`tracking-orchestrator/.venv/bin/python`,
  `rf-ingress/.venv/bin/python`) or a `make` target. Never system Python, bare `python`, or
  `pip install`.
- Sync with `cd tracking-orchestrator && uv sync --frozen --extra dev` (or `cd rf-ingress && uv sync --frozen --extra dev`).
  `uv.lock` changes only when you re-lock on purpose; revert any other `uv.lock` change.
- Format only the files you touched.
