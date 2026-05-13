// Package supervisor manages the lifecycle of per-camera poll workers and
// keeps go2rtc's stream registry in sync with the desired camera list.
package supervisor

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/metrics"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/motion"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/poll"
)

// Go2RTCClient is the subset of go2rtc.Client methods used by the supervisor.
// Defining it here keeps the supervisor decoupled from the concrete client type
// and makes it easy to inject fakes in tests.
type Go2RTCClient interface {
	RegisterStream(ctx context.Context, name, rtspURL string) error
	DeregisterStream(ctx context.Context, name string) error
	FetchJPEG(ctx context.Context, name string) ([]byte, error)
}

// Supervisor manages a set of per-camera poll workers and the corresponding
// go2rtc stream registrations. Reconcile is called each time the desired
// camera list changes; it registers all current streams (idempotent — heals
// go2rtc restarts) and starts/stops workers as needed.
type Supervisor struct {
	mu        sync.Mutex
	wg        sync.WaitGroup
	parent    context.Context
	workers   map[string]*workerHandle
	go2rtc    Go2RTCClient
	publisher poll.Publisher
	log       *zap.Logger
}

type workerHandle struct {
	cancel context.CancelFunc
	camera config.CameraConfig
}

// New creates a Supervisor.
func New(
	parent context.Context,
	client Go2RTCClient,
	publisher poll.Publisher,
	log *zap.Logger,
) *Supervisor {
	return &Supervisor{
		parent:    parent,
		workers:   make(map[string]*workerHandle),
		go2rtc:    client,
		publisher: publisher,
		log:       log,
	}
}

// Reconcile synchronises running workers with the given camera list.
//
// For each enabled camera:
//   - RegisterStream is called unconditionally (idempotent PUT — heals go2rtc
//     restarts within one reconcile interval without any extra bookkeeping).
//   - A new poll.Worker is started if no worker is running for this camera, or
//     if the camera config has changed.
//
// Workers for cameras that were removed or disabled are stopped and their
// go2rtc streams are deregistered.
func (s *Supervisor) Reconcile(cameras []config.CameraConfig) {
	s.mu.Lock()
	defer s.mu.Unlock()

	want := make(map[string]config.CameraConfig, len(cameras))
	for _, c := range cameras {
		if c.Enabled {
			want[c.ID] = c
		}
	}

	// Re-register ALL desired streams — idempotent PUT heals go2rtc restarts.
	for id, cam := range want {
		if err := s.go2rtc.RegisterStream(s.parent, id, cam.RTSPURL); err != nil {
			metrics.Go2RTCRegistrationErrorsTotal.WithLabelValues(id).Inc()
			s.log.Warn("register_stream_failed",
				zap.String("camera_id", id),
				zap.Error(err),
			)
		}
	}

	// Stop removed or config-changed cameras.
	for id, h := range s.workers {
		next, keep := want[id]
		if !keep || next != h.camera {
			if err := s.go2rtc.DeregisterStream(s.parent, id); err != nil {
				s.log.Warn("deregister_stream_failed",
					zap.String("camera_id", id),
					zap.Error(err),
				)
			}
			s.log.Info("stopping_worker", zap.String("camera_id", id))
			h.cancel()
			delete(s.workers, id)
		}
	}

	// Start workers for new cameras.
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
				gate := motion.New(c.MotionThreshold)
				w := poll.NewWorker(
					c,
					s.go2rtc,
					gate,
					s.publisher,
					s.log.With(zap.String("camera_id", c.ID)),
				)
				w.Run(ctx)
			}(cam)
		}
	}
}

// Snapshot fetches a JPEG frame for the given camera from go2rtc.
func (s *Supervisor) Snapshot(ctx context.Context, cameraID string) ([]byte, error) {
	return s.go2rtc.FetchJPEG(ctx, cameraID)
}

// StreamHealth probes whether a stream is healthy by fetching a single JPEG
// frame via go2rtc and measuring the round-trip latency.
func (s *Supervisor) StreamHealth(ctx context.Context, cameraID string) map[string]any {
	start := time.Now()
	_, err := s.go2rtc.FetchJPEG(ctx, cameraID)
	elapsed := time.Since(start)

	result := map[string]any{
		"camera_id":  cameraID,
		"healthy":    err == nil,
		"latency_ms": elapsed.Milliseconds(),
	}
	s.mu.Lock()
	_, running := s.workers[cameraID]
	s.mu.Unlock()
	result["worker_running"] = running

	if err != nil {
		result["error"] = err.Error()
	}
	return result
}

// ReloadStream forces go2rtc to reconnect to the RTSP source and eagerly
// restarts the poll worker for the given camera. Returns an error when the
// camera is not managed by this ingress instance.
func (s *Supervisor) ReloadStream(ctx context.Context, cameraID string) error {
	s.mu.Lock()
	h, ok := s.workers[cameraID]
	if !ok {
		s.mu.Unlock()
		return fmt.Errorf("camera %q is not managed by this ingress instance", cameraID)
	}
	cam := h.camera
	s.mu.Unlock()

	// Force a fresh go2rtc RTSP session.
	_ = s.go2rtc.DeregisterStream(ctx, cameraID)
	if err := s.go2rtc.RegisterStream(ctx, cameraID, cam.RTSPURL); err != nil {
		return fmt.Errorf("re-register stream: %w", err)
	}

	s.log.Info("reloading_stream", zap.String("camera_id", cameraID))

	// Stop the existing worker.
	s.mu.Lock()
	if h, ok := s.workers[cameraID]; ok {
		h.cancel()
		delete(s.workers, cameraID)
	}

	// Start a fresh worker immediately — don't wait for the next reconcile.
	workerCtx, cancel := context.WithCancel(s.parent)
	s.workers[cameraID] = &workerHandle{cancel: cancel, camera: cam}
	s.mu.Unlock()

	s.wg.Add(1)
	metrics.ActiveWorkers.Inc()
	go func(c config.CameraConfig) {
		defer s.wg.Done()
		defer metrics.ActiveWorkers.Dec()
		gate := motion.New(c.MotionThreshold)
		w := poll.NewWorker(
			c,
			s.go2rtc,
			gate,
			s.publisher,
			s.log.With(zap.String("camera_id", c.ID)),
		)
		w.Run(workerCtx)
	}(cam)

	return nil
}

// Stop cancels all running workers and waits for them to finish, up to timeout.
func (s *Supervisor) Stop(timeout time.Duration) {
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
