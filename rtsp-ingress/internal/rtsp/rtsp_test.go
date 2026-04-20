package rtsp

import (
	"context"
	"testing"

	"go.uber.org/zap"

	pb "github.com/khoofia/continuous-tracking/proto/continuoustracking/v1"
	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
)

type nopPublisher struct{}

func (p *nopPublisher) Publish(_ context.Context, _ *pb.FrameReady, _ []byte) error {
	return nil
}

func TestNewWorker(t *testing.T) {
	log := zap.NewNop()

	cam := config.CameraConfig{
		ID:   "test-cam",
		RTSPURL: "rtsp://localhost/test",
	}
	w := NewWorker(cam, &nopPublisher{}, log.With(zap.String("camera_id", "test-cam")))
	if w == nil {
		t.Fatal("expected non-nil worker")
	}
	if w.camera.ID != "test-cam" {
		t.Errorf("camera id: got %q", w.camera.ID)
	}
}

func TestWorkerSeqIncrements(t *testing.T) {
	log := zap.NewNop()

	cam := config.CameraConfig{ID: "test-cam"}
	w := NewWorker(cam, &nopPublisher{}, log)

	seq1 := w.seq.Add(1) - 1
	seq2 := w.seq.Add(1) - 1
	seq3 := w.seq.Add(1) - 1

	if seq1 != 0 || seq2 != 1 || seq3 != 2 {
		t.Errorf("seq: got %d, %d, %d; want 0, 1, 2", seq1, seq2, seq3)
	}
}
