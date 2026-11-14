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
	"sync"
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
	cam          config.CameraConfig
	fetcher      JPEGFetcher
	gate         *motion.Gate
	pub          Publisher
	log          *zap.Logger
	now          func() time.Time
	seq          int64
	stateMu      sync.Mutex
	lastPublish  time.Time
	lastMotionAt time.Time
	failures     int           // consecutive failed fetch or decode attempts
	backoff      time.Duration // current retry delay while failures > 0
	retryAt      time.Time     // no fetch before this instant while failures > 0
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
	w.stateMu.Lock()
	w.failures++
	delay := w.backoffDelay()
	w.backoff = delay
	w.retryAt = w.now().Add(delay)
	failures := w.failures
	w.stateMu.Unlock()
	metrics.CameraBackoffSeconds.WithLabelValues(w.cam.ID).Set(delay.Seconds())
	w.log.Warn(event,
		zap.String("camera_id", w.cam.ID),
		zap.Error(err),
		zap.Int("consecutive_failures", failures),
		zap.Duration("retry_in", delay),
	)
}

func (w *Worker) recordSuccess() {
	w.stateMu.Lock()
	if w.failures == 0 {
		w.stateMu.Unlock()
		return
	}
	failedAttempts := w.failures
	w.failures = 0
	w.backoff = 0
	w.retryAt = time.Time{}
	w.stateMu.Unlock()
	w.log.Info("fetch_recovered",
		zap.String("camera_id", w.cam.ID),
		zap.Int("failed_attempts", failedAttempts),
	)
	metrics.CameraBackoffSeconds.WithLabelValues(w.cam.ID).Set(0)
}

// LastFrameAgeSeconds returns the elapsed seconds since the last published
// frame, and false when no frame has been published yet.
func (w *Worker) LastFrameAgeSeconds(now time.Time) (float64, bool) {
	w.stateMu.Lock()
	last := w.lastPublish
	w.stateMu.Unlock()
	if last.IsZero() {
		return 0, false
	}
	age := now.Sub(last).Seconds()
	if age < 0 {
		age = 0
	}
	return age, true
}

// CurrentBackoffSeconds returns the active retry backoff in seconds (0 when healthy).
func (w *Worker) CurrentBackoffSeconds() float64 {
	w.stateMu.Lock()
	defer w.stateMu.Unlock()
	return w.backoff.Seconds()
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
	w.stateMu.Lock()
	inBackoff := w.failures > 0 && w.now().Before(w.retryAt)
	w.stateMu.Unlock()
	if inBackoff {
		return
	}

	fetchStart := w.now()
	jpegBytes, err := w.fetcher.FetchJPEG(ctx, w.cam.ID)
	fetchDuration := w.now().Sub(fetchStart)
	metrics.FetchDurationSeconds.WithLabelValues(w.cam.ID).Observe(fetchDuration.Seconds())
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
	fetchDoneAt := w.now()
	metrics.LastFetchSuccessTimestampSeconds.WithLabelValues(w.cam.ID).Set(
		float64(fetchDoneAt.UnixNano()) / 1e9,
	)

	// Apply per-camera rotation between decode and motion gate so every
	// downstream consumer sees a single canonical orientation.  The
	// rotated JPEG replaces the original bytes for publish.
	if w.cam.RotationDegrees != 0 {
		rotStart := w.now()
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
				metrics.RotationLatencySeconds.WithLabelValues(w.cam.ID).Observe(
					w.now().Sub(rotStart).Seconds(),
				)
			}
		}
	}

	receivedAt := w.now()
	isStatic := w.gate.IsStatic(img)
	metrics.MotionActiveCells.WithLabelValues(w.cam.ID).Observe(float64(w.gate.LastActiveCells()))
	if !isStatic {
		w.lastMotionAt = receivedAt
	} else {
		inHoldWindow := w.cam.MotionHoldS > 0 &&
			!w.lastMotionAt.IsZero() &&
			receivedAt.Sub(w.lastMotionAt) <= time.Duration(w.cam.MotionHoldS*float64(time.Second))
		w.stateMu.Lock()
		lastPub := w.lastPublish
		w.stateMu.Unlock()
		inStaticSample := w.cam.StaticSampleIntervalS > 0 &&
			receivedAt.Sub(lastPub) >= time.Duration(w.cam.StaticSampleIntervalS)*time.Second
		if !inHoldWindow && !inStaticSample {
			metrics.FramesFilteredTotal.WithLabelValues(w.cam.ID, "motion").Inc()
			return
		}
	}

	latencyOffset := time.Duration(w.cam.CaptureLatencyMs) * time.Millisecond
	if latencyOffset < 0 {
		latencyOffset = 0
	}
	captureTime := fetchStart.Add(-latencyOffset)
	if captureTime.After(fetchStart) {
		captureTime = fetchStart
	}

	w.seq++
	bounds := img.Bounds()
	meta := &pb.FrameReady{
		CameraId:           w.cam.ID,
		FrameIndex:         w.seq,
		CaptureTimeUnixNs:  uint64(captureTime.UnixNano()), //nolint:gosec
		ReceivedTimeUnixNs: uint64(receivedAt.UnixNano()),  //nolint:gosec
		Width:              int32(bounds.Dx()),             //nolint:gosec
		Height:             int32(bounds.Dy()),             //nolint:gosec
	}
	if err := w.pub.Publish(ctx, meta, jpegBytes); err != nil {
		w.log.Warn("publish_failed", zap.String("camera_id", w.cam.ID), zap.Error(err))
		return
	}
	w.gate.MarkPublished()
	w.stateMu.Lock()
	w.lastPublish = receivedAt
	w.stateMu.Unlock()
	metrics.LastFramePublishedTimestampSeconds.WithLabelValues(w.cam.ID).Set(
		float64(receivedAt.UnixNano()) / 1e9,
	)
}
