# Go (`rtsp-ingress/`)

Read before changing Go code. Part of the `engineering-standards` skill; the rules summary is in `../SKILL.md`.

## Toolchain

- Go 1.24+, toolchain pinned in `.tool-versions` (install with `make go-install`).
- `go vet`, `staticcheck`, and `golangci-lint` are hard CI gates (`make go-lint`).
- Logging: `go.uber.org/zap` (`zap.L()` or an injected `*zap.Logger`); timestamps written to
  Redis or Postgres use `time.Now().UTC()`; metrics register in `internal/metrics/metrics.go`.

## go2rtc owns RTSP

**go2rtc owns all RTSP sessions.** Use `go2rtc.Client` -- `RegisterStream`, `DeregisterStream`, `FetchJPEG`. Never open raw RTSP connections in Go, and do not add `gortsplib` or `pion/rtp`. The ingest flow is described in `docs/systems-architecture.md`.

## Interface contracts in `internal/`

Each subsystem in `rtsp-ingress/internal/` exports an interface as its public API, not a concrete struct. Tests inject a fake; production wires the real implementation.

```go
// poll/worker.go: the consumer declares what it needs; the go2rtc Client satisfies it.
type JPEGFetcher interface {
    FetchJPEG(ctx context.Context, name string) ([]byte, error)
}
```

1. Accept interfaces, return concrete types.
2. Never pass raw Redis clients across package boundaries -- wrap them.
3. Define interfaces in the **consumer** package, not the provider.

## Error handling

1. Wrap errors with context: `fmt.Errorf("streams: register %q: %w", cfg.ID, err)`.
2. Check every error return -- `go vet` and `golangci-lint` flag unchecked returns; treat them as compile errors.
3. `context.Context` is the first parameter of every I/O function. Respect cancellation via `ctx.Err()`.

## Testing

1. Table-driven tests (`t.Run`) for multi-case functions.
2. Interfaces injected in constructors; tests replace with fakes.
3. `-race` flag mandatory: `go test -race ./...`. Hard CI gate.
4. HTTP handlers tested with `httptest.NewRecorder()`.
5. go2rtc calls tested with `httptest.Server` serving mock responses.
