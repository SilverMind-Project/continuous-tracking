package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	slog.Info("rtsp-ingress starting (M1 no-op placeholder)")

	// TODO M3: Wire RTSP client, frame capture, MinIO upload, Redis Streams publish.
	// For M1, this is a no-op service that starts and exposes a health endpoint.

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ok","service":"rtsp-ingress","version":"0.1.0"}`))
	})

	srv := &http.Server{
		Addr:         ":8082",
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("listening on :8082")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	// Wait for interrupt
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("shutdown error", "error", err)
		os.Exit(1)
	}
}
