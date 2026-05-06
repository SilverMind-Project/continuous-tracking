// Command rtsp-ingress: RTSP camera stream ingress service.
// Pulls frames from RTSP cameras (via go2rtc), gates on motion, uploads JPEGs
// to MinIO, and publishes FrameReady messages to Redis Streams.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/go2rtc"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/media"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/reconciler"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/supervisor"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "invalid config: %v\n", err)
		os.Exit(1)
	}

	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = logger.Sync() }()

	sugar := logger.Sugar()
	sugar.Infow("rtsp-ingress starting",
		"listen_addr", cfg.Server.ListenAddr,
		"go2rtc_addr", cfg.Go2RTC.Addr,
	)

	// MinIO client.
	minioClient, err := minio.New(cfg.MinIO.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.MinIO.AccessKey, cfg.MinIO.SecretKey, ""),
		Secure: cfg.MinIO.UseSSL,
	})
	if err != nil {
		logger.Fatal("minio create", zap.Error(err))
	}

	// Redis client.
	redisClient := redis.NewClient(&redis.Options{
		Addr:     cfg.Redis.Address,
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.DB,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Media publisher (MinIO + Redis).
	publisher := media.New(
		minioClient,
		cfg.MinIO.Bucket,
		redisClient,
		cfg.Redis.Stream,
		cfg.Redis.MaxLenApprox,
	)
	if err := publisher.EnsureBucket(ctx); err != nil {
		logger.Fatal("minio ensure bucket", zap.Error(err))
	}

	// go2rtc HTTP API client.
	g2r := go2rtc.New(go2rtc.Config{
		Addr:           cfg.Go2RTC.Addr,
		TimeoutSeconds: cfg.Go2RTC.TimeoutSeconds,
	})

	// Supervisor manages poll workers and go2rtc stream registrations.
	sup := supervisor.New(ctx, g2r, publisher, logger)

	// Reconciler fetches camera configs from Cognitive Companion.
	rec := reconciler.New(
		cfg.Cognitive.BaseURL,
		cfg.Cognitive.APIKey,
		cfg.AssignedCameras,
		cfg.Cognitive.ReconcileInterval,
		cfg.CameraDefaults,
		sup,
		logger,
	)

	// Start reconciler loop — camera config comes exclusively from the
	// cognitive-companion API; there is no static bootstrap list.
	go func() {
		if err := rec.Run(ctx); err != nil {
			logger.Warn("reconciler stopped", zap.Error(err))
		}
	}()

	// Health/ready state.
	var ready bool
	var readyMu sync.Mutex

	// HTTP server.
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthzHandler)
	mux.HandleFunc("/readyz", readyzHandler(&ready, &readyMu))
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		promhttp.Handler().ServeHTTP(w, r)
	})

	srv := &http.Server{
		Addr:         cfg.Server.ListenAddr,
		Handler:      mux,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	go func() {
		sugar.Infow("listening", "addr", cfg.Server.ListenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("server listen", zap.Error(err))
		}
	}()

	// Mark ready once the reconciler has fetched at least one camera and
	// the backing stores are reachable.
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				cameras := rec.LastCameras()
				isReady := len(cameras) > 0 &&
					minioClient.IsOnline() &&
					redisClient.Ping(ctx).Err() == nil
				readyMu.Lock()
				ready = isReady
				readyMu.Unlock()
			case <-ctx.Done():
				return
			}
		}
	}()

	// Wait for interrupt.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	sugar.Info("shutting down...")
	cancel()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		sugar.Errorw("server shutdown error", "error", err)
	}

	sup.Stop(cfg.Server.ShutdownTimeout)
	sugar.Info("rtsp-ingress stopped")
}

func healthzHandler(w http.ResponseWriter, r *http.Request) {
	_ = r
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok","service":"rtsp-ingress"}`))
}

func readyzHandler(ready *bool, readyMu *sync.Mutex) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_ = r
		readyMu.Lock()
		rdy := *ready
		readyMu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if !rdy {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"status":"not_ready"}`))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ready"}`))
	}
}
