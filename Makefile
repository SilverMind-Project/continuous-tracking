.PHONY: help proto proto-gen docker-up docker-down docker-build lint format test check all-check mypy go-check

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

proto: ## Generate protobuf code (Go + Python)
	cd proto && buf generate

docker-up: ## Start all services (docker-compose)
	docker compose up --wait -d

docker-down: ## Stop all services
	docker compose down -v

docker-build: ## Build all service images
	docker compose build

# -----------------------------------------------------------------------
# Python (tracking-orchestrator) — uses uv via project venv
# -----------------------------------------------------------------------

lint: ## Lint Python code
	cd tracking-orchestrator && uv run ruff check ..

format: ## Format Python code
	cd tracking-orchestrator && uv run ruff format ..

test: ## Run Python tests
	cd tracking-orchestrator && uv run pytest tests -v

mypy: ## Type-check Python code
	cd tracking-orchestrator && uv run mypy app

check: ## Run full Python quality gate (lint + format + typecheck + tests)
	cd tracking-orchestrator && uv run ruff check ..
	cd tracking-orchestrator && uv run ruff format --check ..
	cd tracking-orchestrator && uv run mypy app
	cd tracking-orchestrator && uv run pytest tests -v

all-check: check go-check proto-lint ## Run full repo quality gate (Python + Go + proto)

# -----------------------------------------------------------------------
# Go (rtsp-ingress)
# -----------------------------------------------------------------------

go-check: ## Run full Go quality gate (lint + vet + test + build)
	cd rtsp-ingress && make check

go-lint: ## Lint Go code (golangci-lint)
	cd rtsp-ingress && golangci-lint run ./...

go-test: ## Run Go tests with race detector
	cd rtsp-ingress && go test -race ./...

go-build: ## Build Go binary
	cd rtsp-ingress && go build -o /dev/null ./cmd/server
