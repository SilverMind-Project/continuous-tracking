// Package poll implements a tick-based JPEG polling worker.
//
// Instead of managing RTSP sessions directly, it polls go2rtc's HTTP API for
// pre-decoded JPEG frames. This eliminates H264 NAL assembly, GOP buffering,
// and the ffmpeg subprocess used by the previous approach.
package poll

import (
	"bytes"
	"context"
	"image/jpeg"
	"time"

	"go.uber.org/zap"

	pb "github.com/SilverMind-Project/continuous-tracking/proto/continuoustracking/v1"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/imageops"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/metrics"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/motion"
)

// JPEGFetcher retrieves raw JPEG bytes for a named stream.
// The go2rtc Client satisfies this interface.
type JPEGFetcher interface {
	FetchJPEG(ctx context.Context, name string) ([]byte, error)
}

// Publisher uploads frames and publishes stream metadata.
// media.Publisher satisfies this interface.
type Publisher interface {
	Publish(ctx context.Context, meta *pb.FrameReady, jpegIn []byte) error
}

// Worker polls go2rtc at a fixed interval, gates on motion, and publishes
// motion-bearing frames to MinIO + Redis Streams.
//
// Every fetch makes go2rtc open (or reuse) an RTSP session to the camera, so a camera that keeps
// failing is retried at a widening interval rather than on every tick: the first retry waits
// ReconnectBackoffSeconds, each further consecutive failure doubles the wait, capped at
// ReconnectBackoffMaxSeconds, and one good frame resets it. Without this a dead or refusing
// camera receives a new session attempt every frame interval, indefinitely.
type Worker struct {
	cam         config.CameraConfig
	fetcher     JPEGFetcher
	gate        *motion.Gate
	pub         Publisher
	log         *zap.Logger
	now         func() time.Time
	seq         int64
	lastPublish time.Time
	failures    int       // consecutive failed fetch or decode attempts
	retryAt     time.Time // no fetch before this instant while failures > 0
}

// Option customizes a Worker.
type Option func(*Worker)

// WithClock replaces the wall clock the backoff schedule reads (tests advance a fake clock).
func WithClock(now func() time.Time) Option {
	return func(w *Worker) { w.now = now }
}

// NewWorker creates a Worker for the given camera.
func NewWorker(
	cam config.CameraConfig,
	fetcher JPEGFetcher,
	gate *motion.Gate,
	pub Publisher,
	log *zap.Logger,
	opts ...Option,
) *Worker {
	w := &Worker{
		cam:     cam,
		fetcher: fetcher,
		gate:    gate,
		pub:     pub,
		log:     log,
		now:     time.Now,
	}
	for _, opt := range opts {
		opt(w)
	}
	return w
}

// backoffDelay is the wait after the current run of consecutive failures.
func (w *Worker) backoffDelay() time.Duration {
	initial := time.Duration(w.cam.ReconnectBackoffSeconds * float64(time.Second))
	maxDelay := time.Duration(w.cam.ReconnectBackoffMaxSeconds * float64(time.Second))
	if maxDelay < initial {
		maxDelay = initial
	}
	delay := initial
	for i := 1; i < w.failures && delay < maxDelay; i++ {
		delay *= 2
	}
	if delay > maxDelay {
		delay = maxDelay
	}
	return delay
}

// recordFailure schedules the next attempt and logs once per attempt; while backing off,
// attempts (and so log lines) are at most one per backoff interval.
func (w *Worker) recordFailure(event string, err error) {
	metrics.FetchErrorsTotal.WithLabelValues(w.cam.ID).Inc()
	w.failures++
	delay := w.backoffDelay()
	w.retryAt = w.now().Add(delay)
	metrics.CameraBackoffSeconds.WithLabelValues(w.cam.ID).Set(delay.Seconds())
	w.log.Warn(event,
		zap.String("camera_id", w.cam.ID),
		zap.Error(err),
		zap.Int("consecutive_failures", w.failures),
		zap.Duration("retry_in", delay),
	)
}

func (w *Worker) recordSuccess() {
	if w.failures == 0 {
		return
	}
	w.log.Info("fetch_recovered",
		zap.String("camera_id", w.cam.ID),
		zap.Int("failed_attempts", w.failures),
	)
	w.failures = 0
	w.retryAt = time.Time{}
	metrics.CameraBackoffSeconds.WithLabelValues(w.cam.ID).Set(0)
}

// Run polls go2rtc until ctx is cancelled.
func (w *Worker) Run(ctx context.Context) {
	interval := time.Duration(w.cam.FrameIntervalMs) * time.Millisecond
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			w.poll(ctx)
		}
	}
}

func (w *Worker) poll(ctx context.Context) {
	if w.failures > 0 && w.now().Before(w.retryAt) {
		return
	}

	jpegBytes, err := w.fetcher.FetchJPEG(ctx, w.cam.ID)
	if err != nil {
		w.recordFailure("fetch_failed", err)
		return
	}

	// Decode JPEG once for the motion gate. go2rtc answers a frame request with a truncated body
	// when the camera drops the session, so a decode failure is a failed fetch too.
	img, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		w.recordFailure("jpeg_decode_failed", err)
		return
	}
	w.recordSuccess()

	// Apply per-camera rotation between decode and motion gate so every
	// downstream consumer sees a single canonical orientation.  The
	// rotated JPEG replaces the original bytes for publish.
	if w.cam.RotationDegrees != 0 {
		rotStart := time.Now()
		rotated, err := imageops.RotateCW(img, w.cam.RotationDegrees)
		if err != nil {
			metrics.RotationErrorsTotal.WithLabelValues(w.cam.ID).Inc()
			w.log.Warn("rotation_failed",
				zap.String("camera_id", w.cam.ID),
				zap.Int("degrees", w.cam.RotationDegrees),
				zap.Error(err),
			)
		} else {
			reencoded, err := imageops.EncodeJPEG(rotated, 90)
			if err != nil {
				metrics.RotationErrorsTotal.WithLabelValues(w.cam.ID).Inc()
				w.log.Warn("rotation_reencode_failed",
					zap.String("camera_id", w.cam.ID),
					zap.Error(err),
				)
			} else {
				jpegBytes = reencoded
				img = rotated
				metrics.RotationLatencySeconds.WithLabelValues(w.cam.ID).Observe(time.Since(rotStart).Seconds())
			}
		}
	}

	if w.gate.IsStatic(img) {
		if w.cam.StaticSampleIntervalS <= 0 || time.Since(w.lastPublish) < time.Duration(w.cam.StaticSampleIntervalS)*time.Second {
			metrics.FramesFilteredTotal.WithLabelValues(w.cam.ID, "motion").Inc()
			return
		}
	}

	w.seq++
	bounds := img.Bounds()
	now := time.Now()
	meta := &pb.FrameReady{
		CameraId:           w.cam.ID,
		FrameIndex:         w.seq,
		CaptureTimeUnixNs:  uint64(now.UnixNano()), //nolint:gosec
		ReceivedTimeUnixNs: uint64(now.UnixNano()), //nolint:gosec
		Width:              int32(bounds.Dx()),     //nolint:gosec
		Height:             int32(bounds.Dy()),     //nolint:gosec
	}
	if err := w.pub.Publish(ctx, meta, jpegBytes); err != nil {
		w.log.Warn("publish_failed", zap.String("camera_id", w.cam.ID), zap.Error(err))
		return
	}
	w.lastPublish = now
}
