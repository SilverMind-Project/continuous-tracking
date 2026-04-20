.PHONY: help proto proto-gen docker-up docker-down docker-build lint format test

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
# Python (tracking-orchestrator)
# -----------------------------------------------------------------------
PYTHON := uv run

lint: ## Lint Python code
	$(PYTHON) -m ruff check tracking-orchestrator

format: ## Format Python code
	$(PYTHON) -m ruff format tracking-orchestrator

test: ## Run Python tests
	$(PYTHON) -m pytest tracking-orchestrator/tests -v

mypy: ## Type-check Python code
	$(PYTHON) -m mypy tracking-orchestrator/app

# -----------------------------------------------------------------------
# Go (rtsp-ingress)
# -----------------------------------------------------------------------
.PHONY: go-lint go-test go-build

go-lint: ## Lint Go code (golangci-lint)
	cd rtsp-ingress && golangci-lint run ./...

go-test: ## Run Go tests
	cd rtsp-ingress && go test ./...

go-build: ## Build Go binary
	cd rtsp-ingress && go build -o /dev/null ./cmd/server
