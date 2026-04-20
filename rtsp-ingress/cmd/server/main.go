// Command rtsp-ingress: RTSP camera stream ingress service.
// Pulls frames off RTSP, gates on motion, uploads JPEGs to MinIO, and
// publishes FrameReady messages to Redis Streams.
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

	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/media"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/reconciler"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/rtsp"
)

func main() {
	cfg := config.Load()

	// Structured logger.
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	sugar := logger.Sugar()
	sugar.Infow("rtsp-ingress starting",
		"listen_addr", cfg.Server.ListenAddr,
		"cameras", len(cfg.Cameras),
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
	redisOpts := &redis.Options{
		Addr:     cfg.Redis.Address,
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.DB,
	}
	redisClient := redis.NewClient(redisOpts)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Health/ready state.
	var ready bool
	var readyMu sync.Mutex

	// Media publisher (MinIO + Redis).
	publisher := media.New(minioClient, cfg.MinIO.Bucket, redisClient, "frames.ready", 100000, 85)

	// Supervisor manages RTSP workers.
	supervisor := newSupervisor(publisher, logger)

	// Reconciler fetches camera configs from Cognitive Companion.
	assigned := os.Getenv("ASSIGNED_CAMERAS")
	rec := reconciler.New(
		cfg.Cognitive.BaseURL,
		cfg.Cognitive.APIKey,
		assigned,
		60*time.Second,
		supervisor,
		logger,
	)

	// Initial camera list from config.
	supervisor.Reconcile(cfg.Cameras)

	// Start reconciler loop.
	go func() {
		if err := rec.Run(ctx); err != nil {
			logger.Warn("reconciler stopped", zap.Error(err))
		}
	}()

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

	// Mark ready once reconciler succeeds and backends are reachable.
	go func() {
		time.Sleep(5 * time.Second)
		cameras := rec.LastCameras()
		if len(cameras) > 0 {
			if minioClient.IsOnline() && redisClient.Ping(ctx).Err() == nil {
				readyMu.Lock()
				ready = true
				readyMu.Unlock()
				logger.Info("service ready")
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

	supervisor.Stop()
	sugar.Info("rtsp-ingress stopped")
}

// newSupervisor creates a Supervisor that manages RTSP workers.
func newSupervisor(pub *media.Publisher, log *zap.Logger) *supervisor {
	return &supervisor{
		publisher: pub,
		log:       log,
	}
}

// supervisor manages a set of RTSP workers keyed by camera ID.
type supervisor struct {
	mu        sync.Mutex
	workers   map[string]*workerHandle
	publisher *media.Publisher
	log       *zap.Logger
}

type workerHandle struct {
	cancel context.CancelFunc
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
		if !rdy {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte(`{"status":"not_ready"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"ready"}`))
	}
}

func (s *supervisor) Reconcile(cameras []config.CameraConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()

	want := make(map[string]config.CameraConfig)
	for _, c := range cameras {
		if c.Enabled {
			want[c.ID] = c
		}
	}

	// Stop removed or disabled cameras.
	for id, h := range s.workers {
		if _, keep := want[id]; !keep {
			s.log.Info("stopping_worker", zap.String("camera_id", id))
			h.cancel()
			delete(s.workers, id)
			continue
		}
	}

	// Start new cameras.
	for id, cam := range want {
		if _, ok := s.workers[id]; !ok {
			s.log.Info("starting_worker", zap.String("camera_id", id))
			ctx, cancel := context.WithCancel(context.Background())
			h := &workerHandle{cancel: cancel}
			s.workers[id] = h
			go func(c config.CameraConfig) {
				w := rtsp.NewWorker(c, s.publisher, s.log.With(zap.String("camera_id", c.ID)))
				w.Run(ctx)
			}(cam)
		}
	}
}

func (s *supervisor) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, h := range s.workers {
		h.cancel()
	}
	s.workers = make(map[string]*workerHandle)
}
