#!/usr/bin/env bash
# Check or perform coordinated dependency upgrades across Go (proto, rtsp-ingress) and
# Python (tracking-orchestrator), keeping shared pins aligned and running the full quality gate.
#
# Usage:
#   scripts/deps-upgrade.sh --check   Read-only list of outdated Go and Python packages
#   scripts/deps-upgrade.sh --apply   Upgrade Go modules + PyPI uv.lock, sync pins, run gates

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:---check}"

# shellcheck source=scripts/go-env.sh
. "$ROOT/scripts/go-env.sh"

upgrade_go_module() {
  local dir="$1"
  (
    cd "$dir"
    local pkgs
    pkgs="$(go list -m -f '{{if and (not .Main) (not .Indirect) (not .Replace)}}{{.Path}}{{end}}' all)"
    if [ -n "$pkgs" ]; then
      # shellcheck disable=SC2086
      go get -u $pkgs
    fi
    go mod tidy
  )
}

case "$MODE" in
  --check)
    echo "=== Outdated Go direct dependencies (proto) ==="
    (cd "$ROOT/proto" && go list -m -u -f '{{if and (not .Indirect) (not .Replace) .Update}}  {{.Path}}: {{.Version}} -> {{.Update.Version}}{{end}}' all)
    echo "=== Outdated Go direct dependencies (rtsp-ingress) ==="
    (cd "$ROOT/rtsp-ingress" && go list -m -u -f '{{if and (not .Indirect) (not .Replace) .Update}}  {{.Path}}: {{.Version}} -> {{.Update.Version}}{{end}}' all)
    echo "=== Outdated Python packages (tracking-orchestrator) ==="
    (cd "$ROOT/tracking-orchestrator" && uv pip list --python .venv/bin/python --outdated || true)
    ;;

  --apply)
    echo "=== 1. Upgrading Go direct dependencies in proto/ ==="
    upgrade_go_module "$ROOT/proto"

    echo "=== 2. Upgrading Go direct dependencies in rtsp-ingress/ ==="
    upgrade_go_module "$ROOT/rtsp-ingress"

    PROTOBUF_VER="$(awk '/google\.golang\.org\/protobuf / {print $NF}' "$ROOT/proto/go.mod" | head -n1)"
    if [ -n "$PROTOBUF_VER" ]; then
      echo "=== 3. Aligning google.golang.org/protobuf and protoc-gen-go to ${PROTOBUF_VER} ==="
      (cd "$ROOT/rtsp-ingress" && go get "google.golang.org/protobuf@${PROTOBUF_VER}" && go mod tidy)
      sed -i -E "s|protoc-gen-go@v[0-9]+\.[0-9]+\.[0-9]+|protoc-gen-go@${PROTOBUF_VER}|g" "$ROOT/Makefile" "$ROOT/.github/workflows/ci.yml"
      (cd "$ROOT" && make go-tools && make proto-go)
    fi

    echo "=== 4. Upgrading PyPI packages in tracking-orchestrator/uv.lock (preserving internal git pins) ==="
    (
      cd "$ROOT/tracking-orchestrator"
      mapfile -t PYPI_ARGS < <(
        .venv/bin/python - <<'PY'
import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))
for pkg in lock.get("package", []):
    src = pkg.get("source", {})
    if isinstance(src, dict) and "registry" in src:
        print(f"--upgrade-package={pkg['name']}")
PY
      )
      uv lock "${PYPI_ARGS[@]}"
      uv sync --frozen --extra dev
    )

    echo "=== 5. Running full quality gates and toolchain alignment check ==="
    (cd "$ROOT" && make check-all)
    echo "=== 6. Running Python dependency audit ==="
    (cd "$ROOT" && make audit || true)
    ;;

  *)
    echo "usage: $0 [--check|--apply]" >&2
    exit 64
    ;;
esac
