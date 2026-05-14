// Command rtsp-ingress: RTSP camera stream ingress service.
// Pulls frames from RTSP cameras (via go2rtc), gates on motion, uploads JPEGs
// to MinIO, and publishes FrameReady messages to Redis Streams.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
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
	mux.HandleFunc("/internal/test-connection", testConnectionHandler(g2r))
	mux.HandleFunc("GET /internal/streams", listStreamsHandler(rec, g2r))
	mux.HandleFunc("GET /internal/streams/{id}/snapshot", snapshotHandler(sup))
	mux.HandleFunc("GET /internal/streams/{id}/health", streamHealthHandler(sup))
	mux.HandleFunc("POST /internal/streams/{id}/reload", reloadStreamHandler(sup))
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

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		zap.L().Error("json_encode_error", zap.Error(err))
	}
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

// testConnectionHandler probes an RTSP URL by temporarily registering it with
// go2rtc and fetching a single frame. The temporary stream is always cleaned up.
func testConnectionHandler(g2r *go2rtc.Client) http.HandlerFunc {
	type request struct {
		RTSPURL string `json:"rtsp_url"`
	}

	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_, _ = w.Write([]byte(`{"success":false,"message":"Method not allowed"}`))
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"success":false,"message":"Failed to read request body"}`))
			return
		}

		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"success":false,"message":"Invalid JSON: rtsp_url is required"}`))
			return
		}

		if req.RTSPURL == "" {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"success":false,"message":"rtsp_url is required"}`))
			return
		}

		if err := g2r.ProbeStream(r.Context(), req.RTSPURL); err != nil {
			writeJSON(w, http.StatusOK, map[string]any{
				"success": false,
				"message": fmt.Sprintf("Connection failed: %v", err),
			})
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"success": true,
			"message": "Connection successful",
		})
	}
}

// listStreamsHandler returns the set of cameras managed by this ingress
// together with their go2rtc stream state.
func listStreamsHandler(rec *reconciler.Reconciler, g2r *go2rtc.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		cameras := rec.LastCameras()
		allStreams, _ := g2r.ListStreams(r.Context())

		// go2rtc may wrap streams under a "streams" key; handle both shapes.
		streamMap := allStreams
		if wrapped, ok := allStreams["streams"].(map[string]any); ok {
			streamMap = wrapped
		}

		type cameraEntry struct {
			ID       string `json:"id"`
			RTSPURL  string `json:"rtsp_url"`
			RoomName string `json:"room_name"`
			Enabled  bool   `json:"enabled"`
			InGo2RTC bool   `json:"in_go2rtc"`
		}

		result := make([]cameraEntry, 0, len(cameras))
		for _, c := range cameras {
			_, inGo2RTC := streamMap[c.ID]
			result = append(result, cameraEntry{
				ID:       c.ID,
				RTSPURL:  c.RTSPURL,
				RoomName: c.RoomName,
				Enabled:  c.Enabled,
				InGo2RTC: inGo2RTC,
			})
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"streams": result,
		})
	}
}

// snapshotHandler returns a JPEG snapshot for the given camera stream.
func snapshotHandler(sup *supervisor.Supervisor) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cameraID := r.PathValue("id")
		if cameraID == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"error":"missing camera id"}`))
			return
		}

		jpegData, err := sup.Snapshot(r.Context(), cameraID)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			writeJSON(w, http.StatusNotFound, map[string]any{
				"error":     "snapshot failed",
				"camera_id": cameraID,
				"detail":    err.Error(),
			})
			return
		}

		w.Header().Set("Content-Type", "image/jpeg")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(jpegData)
	}
}

// streamHealthHandler returns health information for a specific camera stream.
func streamHealthHandler(sup *supervisor.Supervisor) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		cameraID := r.PathValue("id")
		if cameraID == "" {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"error":"missing camera id"}`))
			return
		}

		health := sup.StreamHealth(r.Context(), cameraID)
		writeJSON(w, http.StatusOK, health)
	}
}

// reloadStreamHandler forces go2rtc to reconnect the RTSP session and
// restarts the poll worker for the given camera.
func reloadStreamHandler(sup *supervisor.Supervisor) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		cameraID := r.PathValue("id")
		if cameraID == "" {
			w.WriteHeader(http.StatusUnprocessableEntity)
			_, _ = w.Write([]byte(`{"error":"missing camera id"}`))
			return
		}

		if err := sup.ReloadStream(r.Context(), cameraID); err != nil {
			writeJSON(w, http.StatusNotFound, map[string]any{
				"success": false,
				"message": err.Error(),
			})
			return
		}

		writeJSON(w, http.StatusOK, map[string]any{
			"success": true,
			"message": fmt.Sprintf("Stream %s reloaded", cameraID),
		})
	}
}
