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
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/decode"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/media"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/metrics"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/reconciler"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/rtsp"
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

	// Structured logger.
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = logger.Sync() }()

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
	publisher := media.New(
		minioClient,
		cfg.MinIO.Bucket,
		redisClient,
		cfg.Redis.Stream,
		cfg.Redis.MaxLenApprox,
		cfg.MinIO.JPEGQuality,
	)
	if err := publisher.EnsureBucket(ctx); err != nil {
		logger.Fatal("minio ensure bucket", zap.Error(err))
	}

	decoderFactory := decode.NewFactory(cfg.Decode.Preferred, cfg.Decode.FFmpegBinary)

	// Supervisor manages RTSP workers.
	supervisor := newSupervisor(ctx, publisher, decoderFactory, cfg.MinIO.JPEGQuality, logger)

	// Reconciler fetches camera configs from Cognitive Companion.
	rec := reconciler.New(
		cfg.Cognitive.BaseURL,
		cfg.Cognitive.APIKey,
		cfg.AssignedCameras,
		cfg.Cognitive.ReconcileInterval,
		cfg.CameraDefaults,
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
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				cameras := rec.LastCameras()
				if len(cameras) == 0 {
					cameras = cfg.Cameras
				}
				isReady := len(cameras) > 0 && minioClient.IsOnline() && redisClient.Ping(ctx).Err() == nil
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

	supervisor.Stop(cfg.Server.ShutdownTimeout)
	sugar.Info("rtsp-ingress stopped")
}

// newSupervisor creates a Supervisor that manages RTSP workers.
func newSupervisor(
	parent context.Context,
	pub *media.Publisher,
	decoderFactory decode.Factory,
	jpegQuality int,
	log *zap.Logger,
) *supervisor {
	return &supervisor{
		parent:         parent,
		workers:        make(map[string]*workerHandle),
		publisher:      pub,
		decoderFactory: decoderFactory,
		jpegQuality:    jpegQuality,
		log:            log,
	}
}

// supervisor manages a set of RTSP workers keyed by camera ID.
type supervisor struct {
	mu             sync.Mutex
	wg             sync.WaitGroup
	parent         context.Context
	workers        map[string]*workerHandle
	publisher      *media.Publisher
	decoderFactory decode.Factory
	jpegQuality    int
	log            *zap.Logger
}

type workerHandle struct {
	cancel context.CancelFunc
	camera config.CameraConfig
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
		next, keep := want[id]
		if !keep || next != h.camera {
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
			ctx, cancel := context.WithCancel(s.parent)
			h := &workerHandle{cancel: cancel, camera: cam}
			s.workers[id] = h
			s.wg.Add(1)
			metrics.ActiveWorkers.Inc()
			go func(c config.CameraConfig) {
				defer s.wg.Done()
				defer metrics.ActiveWorkers.Dec()
				w := rtsp.NewWorker(
					c,
					s.decoderFactory,
					s.publisher,
					s.jpegQuality,
					s.log.With(zap.String("camera_id", c.ID)),
				)
				w.Run(ctx)
			}(cam)
		}
	}
}

func (s *supervisor) Stop(timeout time.Duration) {
	s.mu.Lock()
	for _, h := range s.workers {
		h.cancel()
	}
	s.workers = make(map[string]*workerHandle)
	s.mu.Unlock()

	done := make(chan struct{})
	go func() {
		defer close(done)
		s.wg.Wait()
	}()

	select {
	case <-done:
	case <-time.After(timeout):
		s.log.Warn("worker_shutdown_timeout", zap.Duration("timeout", timeout))
	}
}
