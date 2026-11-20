PY_VENV := tracking-orchestrator/.venv
PY_BIN := $(PY_VENV)/bin
PYTHON := $(PY_BIN)/python
RF_VENV := rf-ingress/.venv
RF_BIN := $(RF_VENV)/bin
UV := uv
GO_ENV := . $(CURDIR)/scripts/go-env.sh &&
GO := $(GO_ENV) go

PROTOC ?= protoc
ORCH_PROTO_OUT := tracking-orchestrator/app/proto
RF_PROTO_OUT   := rf-ingress/rf_ingress/proto
CC_PROTO_OUT   := ../cognitive-companion/backend/integrations/proto
PROTO_FILES    := \
	proto/continuoustracking/v1/frame.proto \
	proto/continuoustracking/v1/tracking.proto \
	proto/continuoustracking/v1/signals.proto \
	proto/continuoustracking/v1/scene.proto \
	proto/continuoustracking/v1/wearable.proto \
	proto/continuoustracking/v1/position_sensor.proto

.PHONY: help venv rf-venv venv-check proto proto-py proto-lint infra-up app-up docker-up docker-down docker-build lint format format-check test mypy import-lint rf-lint rf-format rf-format-check rf-mypy rf-import-lint rf-test rf-test-integration rf-check rf-audit wear-format wear-lint wear-test wear-check check check-all check-toolchain deps-outdated deps-upgrade go-install go-env-check go-tools go-lint go-test go-build go-check detector-equivalence install-hooks test-integration ci audit hooks-test check-md docs-check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create or sync the project venv at tracking-orchestrator/.venv
	cd tracking-orchestrator && $(UV) sync --frozen --extra dev

rf-venv: ## Create or sync the rf-ingress venv at rf-ingress/.venv
	cd rf-ingress && $(UV) sync --frozen --extra dev

docs-check: ## Instruction files (CLAUDE.md, skills, rules) name only files that exist
	$(PYTHON) scripts/check_doc_refs.py

check-md: ## Markdown has no em-dashes (self-test, then every tracked .md)
	bash scripts/check-no-em-dash.sh --self-test
	bash scripts/check-no-em-dash.sh

hooks-test: ## Run the Claude Code hook test table (.claude/hooks/tests/run.sh; needs jq)
	bash .claude/hooks/tests/run.sh

venv-check: ## Fail if VIRTUAL_ENV is not the project venv
	@sh scripts/require-venv.sh

# A persistent, pinned uv tool install: `uv tool run` would leave the git hook pointing at an
# interpreter in uv's cache, and a cache prune would then break every commit.
PRE_COMMIT_VERSION := 4.7.0
install-hooks: venv ## Install the git pre-commit hooks (.pre-commit-config.yaml) with a pinned pre-commit
	$(UV) tool install --quiet pre-commit==$(PRE_COMMIT_VERSION)
	$$($(UV) tool dir --bin)/pre-commit install

# pip-audit must read the locked requirements: a bare `uvx pip-audit` audits the throwaway
# environment uvx builds for pip-audit itself and always reports clean. The git-sourced
# triton-shared and cts-contracts carry no PyPI advisories and no hashes, so they are left out.
audit: ## Audit tracking-orchestrator's locked Python dependencies for published advisories
	@req=$$(mktemp); trap 'rm -f "$$req"' EXIT; \
	cd tracking-orchestrator && $(UV) export --frozen --no-emit-project \
		--no-emit-package triton-shared --no-emit-package cts-contracts \
		--extra dev --format requirements-txt > "$$req" && \
	$(UV) tool run pip-audit --desc --disable-pip -r "$$req"

proto: proto-go proto-py ## Generate protobuf code (Go + Python)

proto-go: go-env-check ## Generate Go protobuf bindings via buf
	cd proto && $(GO_ENV) buf generate

proto-py: ## Generate Python protobuf bindings (orchestrator + rf-ingress + cognitive-companion)
	@command -v $(PROTOC) >/dev/null 2>&1 || { \
	  echo "protoc not found on PATH. Install protoc >= 25 (e.g. apt install protobuf-compiler) or set PROTOC=/path/to/protoc." >&2; \
	  exit 1; \
	}
	mkdir -p $(ORCH_PROTO_OUT) $(RF_PROTO_OUT) $(CC_PROTO_OUT)
	$(PROTOC) --proto_path=proto --python_out=$(ORCH_PROTO_OUT) --pyi_out=$(ORCH_PROTO_OUT) $(PROTO_FILES)
	$(PROTOC) --proto_path=proto --python_out=$(RF_PROTO_OUT)   --pyi_out=$(RF_PROTO_OUT)   $(PROTO_FILES)
	$(PROTOC) --proto_path=proto --python_out=$(CC_PROTO_OUT)   --pyi_out=$(CC_PROTO_OUT)   $(PROTO_FILES)
	@for d in $(ORCH_PROTO_OUT) $(RF_PROTO_OUT) $(CC_PROTO_OUT) \
	  $(ORCH_PROTO_OUT)/continuoustracking $(ORCH_PROTO_OUT)/continuoustracking/v1 \
	  $(RF_PROTO_OUT)/continuoustracking $(RF_PROTO_OUT)/continuoustracking/v1 \
	  $(CC_PROTO_OUT)/continuoustracking $(CC_PROTO_OUT)/continuoustracking/v1; do \
	  test -f $$d/__init__.py || : > $$d/__init__.py; \
	done

proto-lint: go-env-check ## Lint proto files
	cd proto && $(GO_ENV) buf lint

infra-up: ## Start infrastructure services
	docker compose up --wait -d

app-up: ## Start infrastructure and app services
	docker compose --profile app up --wait -d --build

docker-up: infra-up ## Alias for infra-up

docker-down: ## Stop all services
	docker compose down -v

docker-build: ## Build all service images
	docker compose --profile app build

lint: venv ## Lint Python code
	$(PY_BIN)/ruff check tracking-orchestrator

format: venv ## Format Python code
	$(PY_BIN)/ruff format tracking-orchestrator

format-check: venv ## Check Python formatting
	$(PY_BIN)/ruff format --check tracking-orchestrator

test: venv ## Run Python tests (excludes integration marker; no Docker needed)
	cd tracking-orchestrator && ../$(PY_BIN)/pytest tests -m "not integration" -v

mypy: venv ## Type-check Python code
	$(PY_BIN)/mypy --config-file tracking-orchestrator/pyproject.toml tracking-orchestrator/app

import-lint: venv ## Enforce import layering
	cd tracking-orchestrator && ../$(PY_BIN)/lint-imports

rf-lint: rf-venv ## Lint rf-ingress Python code
	$(RF_BIN)/ruff check rf-ingress

rf-format: rf-venv ## Format rf-ingress Python code
	$(RF_BIN)/ruff format rf-ingress

rf-format-check: rf-venv ## Check rf-ingress Python formatting
	$(RF_BIN)/ruff format --check rf-ingress

rf-mypy: rf-venv ## Type-check rf-ingress Python code
	$(RF_BIN)/mypy --config-file rf-ingress/pyproject.toml rf-ingress/rf_ingress

rf-import-lint: rf-venv ## Enforce rf-ingress import layering
	cd rf-ingress && ../$(RF_BIN)/lint-imports

rf-test: rf-venv ## Run rf-ingress unit and conformance tests (no Docker needed)
	cd rf-ingress && ../$(RF_BIN)/pytest tests -m "not integration" -v

rf-test-integration: rf-venv ## Run rf-ingress integration tests (Docker Redis testcontainer required)
	cd rf-ingress && ../$(RF_BIN)/pytest -m integration tests -v

rf-check: rf-venv ## Run the rf-ingress quality gate
	$(MAKE) rf-lint
	$(MAKE) rf-format-check
	$(MAKE) rf-mypy
	$(MAKE) rf-import-lint
	$(MAKE) rf-test

rf-audit: ## Audit rf-ingress's locked Python dependencies for published advisories
	@req=$$(mktemp); trap 'rm -f "$$req"' EXIT; \
	cd rf-ingress && $(UV) export --frozen --no-emit-project \
		--no-emit-package cts-contracts \
		--extra dev --format requirements-txt > "$$req" && \
	$(UV) tool run pip-audit --desc --disable-pip -r "$$req"

wear-format: ## Format wear-companion Kotlin sources with ktlint
	cd wear-companion && ./gradlew :core:ktlintFormat --no-daemon

wear-lint: ## Run wear-companion ktlint and detekt checks
	cd wear-companion && ./gradlew :core:ktlintCheck :core:detekt --no-daemon

wear-test: ## Run wear-companion pure-Kotlin JUnit 5 tests
	cd wear-companion && ./gradlew :core:test --no-daemon

wear-check: ## Run the wear-companion quality gate (ktlint, detekt, unit tests)
	cd wear-companion && ./gradlew :core:check --no-daemon

check-toolchain: ## Verify Go/Python/tool pins stay aligned across .tool-versions, Dockerfiles, go.mod, Makefile, and CI
	@bash scripts/check-toolchain-sync.sh

deps-outdated: venv go-env-check ## List outdated Go modules and Python packages (read-only)
	@bash scripts/deps-upgrade.sh --check

deps-upgrade: venv go-env-check ## Upgrade Go modules and PyPI lockfile in lockstep, then run the full quality gate
	@bash scripts/deps-upgrade.sh --apply

check: venv ## Run the Python quality gate
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) mypy
	$(MAKE) import-lint
	$(MAKE) test
	$(MAKE) rf-check
	$(MAKE) hooks-test
	$(MAKE) check-md
	$(MAKE) docs-check
	$(MAKE) check-toolchain
	@bash scripts/check-env-var-drift.sh

test-integration: venv ## Run every integration-marked test (testcontainer Postgres/Redis required)
	cd tracking-orchestrator && ../$(PY_BIN)/pytest -m integration tests -v
	$(MAKE) rf-test-integration

ci: check-all test-integration ## Authoritative CI gate: full check + integration proofs

check-all: check go-check proto-lint wear-check ## Run the full repo quality gate

detector-equivalence: venv ## Run the detector equivalence gate (requires GPU + Triton; not part of make ci)
	# Requires: TRITON_GRPC_URL, DETECTOR_CALIB_IMAGES_DIR (path to private calibration images).
	# See triton-models/scripts/verify_detector_equivalence.py for full usage.
	cd tracking-orchestrator && ../$(PYTHON) ../triton-models/scripts/verify_detector_equivalence.py

go-install: ## Install the pinned Go toolchain into ./tools
	@sh scripts/install-go.sh

go-env-check: ## Verify the project-pinned Go toolchain is active
	@$(GO_ENV) go version | grep -q "$$(awk '/^golang / {print $$2}' .tool-versions)" || (echo "go version mismatch. Run 'make go-install'." >&2; exit 1)

go-tools: go-env-check ## Install Go-based developer tools into ./tools/go-bin
	@$(GO_ENV) go install github.com/bufbuild/buf/cmd/buf@v1.50.0
	@$(GO_ENV) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.15.0
	@$(GO_ENV) go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.12

go-lint: go-env-check ## Lint Go code
	cd rtsp-ingress && $(GO_ENV) golangci-lint run ./...

go-test: go-env-check ## Run Go tests with race detector
	cd rtsp-ingress && $(GO) test -race -cover ./...

go-build: go-env-check ## Build the Go binary
	cd rtsp-ingress && $(GO) build -o /dev/null ./cmd/server

go-check: go-env-check ## Run the Go quality gate
	$(MAKE) go-lint
	$(MAKE) go-test
	$(MAKE) go-build
