// Package rtsp handles RTSP connection management and frame capture.
package rtsp

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"sync/atomic"
	"time"

	"github.com/bluenviron/gortsplib/v4"
	"github.com/bluenviron/gortsplib/v4/pkg/base"
	"github.com/bluenviron/gortsplib/v4/pkg/description"
	"github.com/bluenviron/gortsplib/v4/pkg/format"
	"github.com/pion/rtp"
	"go.uber.org/zap"

	pb "github.com/khoofia/continuous-tracking/proto/continuoustracking/v1"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
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
	publisher      FramePublisher
	log            *zap.Logger
	seq            atomic.Uint64
}

// NewWorker creates a new Worker for the given camera config.
func NewWorker(cam config.CameraConfig, pub FramePublisher, log *zap.Logger) *Worker {
	return &Worker{
		camera:  cam,
		publisher: pub,
		log:     log,
	}
}

// Run starts the RTSP session loop with exponential backoff on reconnect.
func (w *Worker) Run(ctx context.Context) {
	backoff := time.Duration(w.camera.ReconnectBackoffSeconds * float64(time.Second))
	for ctx.Err() == nil {
		err := w.session(ctx)
		if err != nil && ctx.Err() == nil {
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
		backoff = time.Duration(w.camera.ReconnectBackoffSeconds * float64(time.Second))
	}
}

func (w *Worker) session(ctx context.Context) error {
	client := &gortsplib.Client{}

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
	media := desc.FindFormat(h264)
	if media == nil {
		return fmt.Errorf("camera %s: no H264 track", w.camera.ID)
	}

	if _, err := client.Setup(desc.BaseURL, media, 0, 0); err != nil {
		return fmt.Errorf("setup: %w", err)
	}

	gate := motion.New(w.camera.MotionThreshold)
	interval := time.Duration(w.camera.FrameIntervalMs) * time.Millisecond
	last := time.Time{}

	client.OnPacketRTP(media, h264, func(pkt *rtp.Packet) {
		img, derr := decodeRTPToImage(pkt, media, h264)
		if derr != nil {
			return
		}
		if img == nil {
			return
		}

		now := time.Now().UTC()
		if !last.IsZero() && now.Sub(last) < interval {
			return
		}
		last = now

		if gate.IsStatic(img) {
			return
		}

		// Encode to JPEG.
		jpegBuf := bytes.NewBuffer(make([]byte, 0, 128*1024))
		if err := jpeg.Encode(jpegBuf, img, &jpeg.Options{Quality: 85}); err != nil {
			return
		}

		seq := w.seq.Add(1) - 1
		captureTime := now.UnixNano()

		meta := &pb.FrameReady{
			CameraId:           w.camera.ID,
			FrameIndex:         int64(seq),
			CaptureTimeUnixNs:  uint64(captureTime),
			ReceivedTimeUnixNs: uint64(now.UnixNano()),
			Width:              int32(img.Bounds().Dx()),
			Height:             int32(img.Bounds().Dy()),
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

// decodeRTPToImage is a minimal H.264 NALU assembler that reassembles complete
// frames from RTP packets. In production this feeds packets into a decoder
// (software or NVDEC). This stub returns nil (no frame).
func decodeRTPToImage(pkt *rtp.Packet, media *description.Media, h264 *format.H264) (image.Image, error) {
	_ = pkt
	_ = media
	_ = h264
	// TODO M2: Implement H.264 decoder integration.
	return nil, nil
}
