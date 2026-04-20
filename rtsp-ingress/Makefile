.PHONY: help lint test build cover fmt vet

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk '{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

lint: ## Run golangci-lint
	golangci-lint run ./...

test: ## Run Go tests with race detector
	go test -race -cover -coverprofile=coverage.out ./...

cover: test ## Run tests and open coverage report
	go tool cover -html=coverage.out

fmt: ## Format Go code
	go fmt ./...

vet: ## Run go vet
	go vet ./...

build: ## Build Go binary
	CGO_ENABLED=0 go build -o /dev/null ./cmd/server

check: lint vet test build ## Run full Go quality gate
