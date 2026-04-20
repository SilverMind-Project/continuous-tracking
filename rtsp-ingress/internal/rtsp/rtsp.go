// Package rtsp handles RTSP connection management and frame capture.
package rtsp

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"time"

	"github.com/bluenviron/gortsplib/v4"
	"github.com/bluenviron/gortsplib/v4/pkg/base"
	"github.com/bluenviron/gortsplib/v4/pkg/format"
	"github.com/pion/rtp"
	"go.uber.org/zap"

	pb "github.com/khoofia/continuous-tracking/proto/continuoustracking/v1"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/decode"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/media"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/metrics"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/motion"
)

// FramePublisher is the interface for publishing encoded frames.
// Implemented by the media package (MinIO + Redis Streams).
type FramePublisher interface {
	Publish(ctx context.Context, meta *pb.FrameReady, jpeg []byte) error
}

// Worker processes a single RTSP camera stream, decoding frames, gating on
// motion, and publishing keyframes via the FramePublisher.
type Worker struct {
	camera         config.CameraConfig
	decoderFactory decode.Factory
	publisher      FramePublisher
	jpegQuality    int
	log            *zap.Logger
	seq            atomic.Uint64
}

// NewWorker creates a new Worker for the given camera config.
func NewWorker(
	cam config.CameraConfig,
	decoderFactory decode.Factory,
	pub FramePublisher,
	jpegQuality int,
	log *zap.Logger,
) *Worker {
	return &Worker{
		camera:         cam,
		decoderFactory: decoderFactory,
		publisher:      pub,
		jpegQuality:    jpegQuality,
		log:            log,
	}
}

// Run starts the RTSP session loop with exponential backoff on reconnect.
func (w *Worker) Run(ctx context.Context) {
	backoff := w.initialBackoff()
	for ctx.Err() == nil {
		err := w.session(ctx)
		if err != nil && ctx.Err() == nil {
			metrics.RTSPReconnectsTotal.WithLabelValues(w.camera.ID).Inc()
			w.log.Warn("rtsp_session_ended",
				zap.String("camera_id", w.camera.ID),
				zap.Error(err),
				zap.Float64("backoff_s", backoff.Seconds()),
			)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return
			}
			backoff = min(backoff*2, 60*time.Second)
			continue
		}
		backoff = w.initialBackoff()
	}
}

func (w *Worker) session(ctx context.Context) error {
	client := &gortsplib.Client{
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	u, err := base.ParseURL(w.camera.RTSPURL)
	if err != nil {
		return fmt.Errorf("parse url: %w", err)
	}

	if err := client.Start(u.Scheme, u.Host); err != nil {
		return fmt.Errorf("start: %w", err)
	}
	defer client.Close()

	desc, _, err := client.Describe(u)
	if err != nil {
		return fmt.Errorf("describe: %w", err)
	}

	var h264 *format.H264
	rtspMedia := desc.FindFormat(&h264)
	if rtspMedia == nil {
		return fmt.Errorf("camera %s: no H264 track", w.camera.ID)
	}

	decoder, err := w.decoderFactory(h264)
	if err != nil {
		return fmt.Errorf("decoder init: %w", err)
	}
	defer func() { _ = decoder.Close() }()

	if _, err := client.Setup(desc.BaseURL, rtspMedia, 0, 0); err != nil {
		return fmt.Errorf("setup: %w", err)
	}

	gate := motion.New(w.camera.MotionThreshold)
	interval := time.Duration(w.camera.FrameIntervalMs) * time.Millisecond
	last := time.Time{}

	// gortsplib invokes RTP callbacks serially for a given stream.
	client.OnPacketRTP(rtspMedia, h264, func(pkt *rtp.Packet) {
		img, derr := decoder.DecodePacket(pkt)
		if derr != nil {
			metrics.DecodeErrorsTotal.WithLabelValues(w.camera.ID).Inc()
			return
		}
		if img == nil {
			return
		}

		now := time.Now().UTC()
		if !last.IsZero() && now.Sub(last) < interval {
			metrics.FramesFilteredTotal.WithLabelValues(w.camera.ID, "interval").Inc()
			return
		}
		last = now

		if gate.IsStatic(img) {
			metrics.FramesFilteredTotal.WithLabelValues(w.camera.ID, "motion").Inc()
			return
		}

		jpegBuf, err := media.EncodeJPEG(img, w.jpegQuality)
		if err != nil {
			return
		}

		seq := w.seq.Add(1) - 1
		captureTime := now.UnixNano()
		if captureTime < 0 {
			return
		}
		width := img.Bounds().Dx()
		height := img.Bounds().Dy()
		if seq > math.MaxInt64 || width > math.MaxInt32 || height > math.MaxInt32 {
			return
		}
		//nolint:gosec // Bounds are checked immediately above.
		frameIndex := int64(seq)
		//nolint:gosec // Bounds are checked immediately above.
		frameWidth := int32(width)
		//nolint:gosec // Bounds are checked immediately above.
		frameHeight := int32(height)

		meta := &pb.FrameReady{
			CameraId:           w.camera.ID,
			FrameIndex:         frameIndex,
			CaptureTimeUnixNs:  uint64(captureTime),
			ReceivedTimeUnixNs: uint64(captureTime),
			Width:              frameWidth,
			Height:             frameHeight,
			SampleFps:          0,
		}

		if err := w.publisher.Publish(ctx, meta, jpegBuf.Bytes()); err != nil {
			w.log.Warn("publish_failed",
				zap.String("camera_id", w.camera.ID),
				zap.Uint64("seq", seq),
				zap.Error(err),
			)
			return
		}
	})

	if _, err := client.Play(nil); err != nil {
		return fmt.Errorf("play: %w", err)
	}
	<-ctx.Done()
	return nil
}

func (w *Worker) initialBackoff() time.Duration {
	backoff := time.Duration(w.camera.ReconnectBackoffSeconds * float64(time.Second))
	if backoff <= 0 {
		return 2 * time.Second
	}
	return backoff
}
