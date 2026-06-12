PY_VENV := tracking-orchestrator/.venv
PY_BIN := $(PY_VENV)/bin
PYTHON := $(PY_BIN)/python
UV := uv
GO_ENV := . $(CURDIR)/scripts/go-env.sh &&
GO := $(GO_ENV) go

PROTOC ?= protoc
ORCH_PROTO_OUT := tracking-orchestrator/app/proto
CC_PROTO_OUT   := ../cognitive-companion/backend/integrations/proto
PROTO_FILES    := \
	proto/continuoustracking/v1/frame.proto \
	proto/continuoustracking/v1/tracking.proto \
	proto/continuoustracking/v1/signals.proto \
	proto/continuoustracking/v1/scene.proto

.PHONY: help venv venv-check proto proto-py proto-lint infra-up app-up docker-up docker-down docker-build lint format format-check test mypy import-lint check check-all go-install go-env-check go-tools go-lint go-test go-build go-check detector-equivalence

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create or sync the project venv at tracking-orchestrator/.venv
	cd tracking-orchestrator && $(UV) sync --frozen --extra dev

venv-check: ## Fail if VIRTUAL_ENV is not the project venv
	@sh scripts/require-venv.sh

proto: proto-go proto-py ## Generate protobuf code (Go + Python)

proto-go: go-env-check ## Generate Go protobuf bindings via buf
	cd proto && $(GO_ENV) buf generate

proto-py: ## Generate Python protobuf bindings (orchestrator + cognitive-companion)
	@command -v $(PROTOC) >/dev/null 2>&1 || { \
	  echo "protoc not found on PATH. Install protoc >= 25 (e.g. apt install protobuf-compiler) or set PROTOC=/path/to/protoc." >&2; \
	  exit 1; \
	}
	mkdir -p $(ORCH_PROTO_OUT) $(CC_PROTO_OUT)
	$(PROTOC) --proto_path=proto --python_out=$(ORCH_PROTO_OUT) --pyi_out=$(ORCH_PROTO_OUT) $(PROTO_FILES)
	$(PROTOC) --proto_path=proto --python_out=$(CC_PROTO_OUT)   --pyi_out=$(CC_PROTO_OUT)   $(PROTO_FILES)
	@for d in $(ORCH_PROTO_OUT) $(CC_PROTO_OUT) $(ORCH_PROTO_OUT)/continuoustracking $(ORCH_PROTO_OUT)/continuoustracking/v1 $(CC_PROTO_OUT)/continuoustracking $(CC_PROTO_OUT)/continuoustracking/v1; do \
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

check: venv ## Run the Python quality gate
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) mypy
	$(MAKE) import-lint
	$(MAKE) test
	@bash scripts/check-env-var-drift.sh

test-integration: venv ## Run integration tests (testcontainer Postgres required)
	cd tracking-orchestrator && ../$(PY_BIN)/pytest -m integration tests/integration tests/contracts -v

ci: check-all test-integration ## Authoritative CI gate: full check + integration proofs

check-all: check go-check proto-lint ## Run the full repo quality gate

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
	@$(GO_ENV) go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.63.4
	@$(GO_ENV) go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.36.11

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
