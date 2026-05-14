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
type Worker struct {
	cam     config.CameraConfig
	fetcher JPEGFetcher
	gate    *motion.Gate
	pub     Publisher
	log     *zap.Logger
	seq     int64
}

// NewWorker creates a Worker for the given camera.
func NewWorker(
	cam config.CameraConfig,
	fetcher JPEGFetcher,
	gate *motion.Gate,
	pub Publisher,
	log *zap.Logger,
) *Worker {
	return &Worker{
		cam:     cam,
		fetcher: fetcher,
		gate:    gate,
		pub:     pub,
		log:     log,
	}
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
	jpegBytes, err := w.fetcher.FetchJPEG(ctx, w.cam.ID)
	if err != nil {
		metrics.FetchErrorsTotal.WithLabelValues(w.cam.ID).Inc()
		w.log.Warn("fetch_failed", zap.String("camera_id", w.cam.ID), zap.Error(err))
		return
	}

	// Decode JPEG once for the motion gate.
	img, err := jpeg.Decode(bytes.NewReader(jpegBytes))
	if err != nil {
		metrics.FetchErrorsTotal.WithLabelValues(w.cam.ID).Inc()
		w.log.Warn("jpeg_decode_failed", zap.String("camera_id", w.cam.ID), zap.Error(err))
		return
	}

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
		metrics.FramesFilteredTotal.WithLabelValues(w.cam.ID, "motion").Inc()
		return
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
	}
}
