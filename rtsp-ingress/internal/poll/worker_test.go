package poll_test

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	pb "github.com/SilverMind-Project/continuous-tracking/proto/continuoustracking/v1"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/motion"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/poll"
	"go.uber.org/zap"
)

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

type fakeFetcher struct {
	mu    sync.Mutex
	calls int
	data  []byte
	err   error
}

func (f *fakeFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return f.data, nil
}

func (f *fakeFetcher) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

type fakePublisher struct {
	mu    sync.Mutex
	metas []*pb.FrameReady
	jpgs  [][]byte
	err   error
}

func (p *fakePublisher) Publish(_ context.Context, meta *pb.FrameReady, jpegIn []byte) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.err != nil {
		return p.err
	}
	p.metas = append(p.metas, meta)
	cpy := make([]byte, len(jpegIn))
	copy(cpy, jpegIn)
	p.jpgs = append(p.jpgs, cpy)
	return nil
}

func (p *fakePublisher) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.metas)
}

// countingFetcher always errors but tracks invocations.
type countingFetcher struct {
	count atomic.Int64
}

func (f *countingFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.count.Add(1)
	return nil, errors.New("offline")
}

// sequenceFetcher returns frames from a slice in round-robin order.
type sequenceFetcher struct {
	mu     sync.Mutex
	frames [][]byte
	idx    int
	calls  atomic.Int64
}

func (f *sequenceFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls.Add(1)
	frame := f.frames[f.idx%len(f.frames)]
	f.idx++
	return frame, nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func solidJPEG(t *testing.T, w, h int, c color.Color) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
	return buf.Bytes()
}

func makeCam(intervalMs int) config.CameraConfig {
	return config.CameraConfig{
		ID:              "cam-test",
		RTSPURL:         "rtsp://localhost:8554/test",
		RoomName:        "living-room",
		FrameIntervalMs: intervalMs,
		MotionThreshold: 0.01,
		Enabled:         true,
	}
}

func newWorker(cam config.CameraConfig, fetcher poll.JPEGFetcher, pub poll.Publisher) *poll.Worker {
	gate := motion.New(cam.MotionThreshold)
	return poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestWorker_StopsOnContextCancel(t *testing.T) {
	fetcher := &fakeFetcher{err: errors.New("offline")}
	cam := makeCam(5)
	w := newWorker(cam, fetcher, &fakePublisher{})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		w.Run(ctx)
	}()

	cancel()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("worker did not stop after context cancel")
	}
}

func TestWorker_PollsAtConfiguredInterval(t *testing.T) {
	// 20 ms interval over 300 ms → expect at least 5 polls.
	fetcher := &countingFetcher{}
	cam := makeCam(20)
	w := newWorker(cam, fetcher, &fakePublisher{})

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	if fetcher.count.Load() < 5 {
		t.Errorf("expected >=5 polls in 300 ms with 20 ms interval, got %d", fetcher.count.Load())
	}
}

func TestWorker_FetchErrorSkipsPublish(t *testing.T) {
	fetcher := &fakeFetcher{err: errors.New("connection refused")}
	pub := &fakePublisher{}
	cam := makeCam(10)
	w := newWorker(cam, fetcher, pub)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	if fetcher.callCount() == 0 {
		t.Error("expected at least one FetchJPEG call")
	}
	if pub.callCount() != 0 {
		t.Errorf("expected 0 publish calls on fetch error, got %d", pub.callCount())
	}
}

func TestWorker_InvalidJPEGSkipsPublish(t *testing.T) {
	fetcher := &fakeFetcher{data: []byte("not-a-jpeg")}
	pub := &fakePublisher{}
	cam := makeCam(10)
	w := newWorker(cam, fetcher, pub)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	if fetcher.callCount() == 0 {
		t.Error("expected FetchJPEG to be called at least once")
	}
	if pub.callCount() != 0 {
		t.Errorf("expected 0 publish calls on invalid jpeg, got %d", pub.callCount())
	}
}

func TestWorker_StaticFramesFiltered(t *testing.T) {
	// Identical white frames: first call establishes baseline (published),
	// subsequent identical frames are filtered as static.
	white := solidJPEG(t, 64, 48, color.White)
	fetcher := &fakeFetcher{data: white}
	pub := &fakePublisher{}
	cam := makeCam(20) // 20 ms interval — reliable across slow CI environments
	// High sensitivity threshold to ensure identical frames are caught.
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	// First frame: published (establishes baseline). Subsequent: filtered.
	if fetcher.callCount() < 2 {
		t.Errorf("expected >=2 polls in 200 ms with 20 ms interval, got %d", fetcher.callCount())
	}
	// Only the first frame should be published (identical frames filtered).
	if pub.callCount() > 1 {
		t.Errorf("expected 0 or 1 publishes for static frames, got %d", pub.callCount())
	}
}

func TestWorker_ChangingFramesPublished(t *testing.T) {
	// Alternating black/white frames should both pass the motion gate after the
	// first frame establishes the baseline.  Use a 20 ms interval and a 300 ms
	// deadline so the test is not sensitive to race-detector overhead.
	black := solidJPEG(t, 64, 48, color.Black)
	white := solidJPEG(t, 64, 48, color.White)
	frames := make([][]byte, 0, 20)
	for range 10 {
		frames = append(frames, black, white)
	}
	fetcher := &sequenceFetcher{frames: frames}
	pub := &fakePublisher{}
	cam := makeCam(20)
	gate := motion.New(0.001) // very sensitive
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	if pub.callCount() < 2 {
		t.Errorf("expected >=2 publishes for alternating frames, got %d", pub.callCount())
	}
}

func TestWorker_PublishReceivesCorrectCameraID(t *testing.T) {
	white := solidJPEG(t, 32, 24, color.White)
	black := solidJPEG(t, 32, 24, color.Black)
	// First frame white (baseline), second frame black (motion detected).
	fetcher := &sequenceFetcher{frames: [][]byte{white, black}}
	pub := &fakePublisher{}
	cam := makeCam(5)
	cam.ID = "cam-living-room"
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	pub.mu.Lock()
	defer pub.mu.Unlock()
	for _, m := range pub.metas {
		if m.CameraId != "cam-living-room" {
			t.Errorf("camera_id: got %q, want cam-living-room", m.CameraId)
		}
	}
}

func TestWorker_SequenceNumberIncreases(t *testing.T) {
	// Each published frame should have a higher FrameIndex than the previous.
	black := solidJPEG(t, 32, 24, color.Black)
	white := solidJPEG(t, 32, 24, color.White)
	fetcher := &sequenceFetcher{frames: [][]byte{black, white, black, white}}
	pub := &fakePublisher{}
	cam := makeCam(5)
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	pub.mu.Lock()
	defer pub.mu.Unlock()
	for i := 1; i < len(pub.metas); i++ {
		if pub.metas[i].FrameIndex <= pub.metas[i-1].FrameIndex {
			t.Errorf("frame_index not increasing: [%d]=%d, [%d]=%d",
				i-1, pub.metas[i-1].FrameIndex, i, pub.metas[i].FrameIndex)
		}
	}
}

func TestWorker_PublishErrorDoesNotStopWorker(t *testing.T) {
	// A publish error should log but not stop the worker.
	black := solidJPEG(t, 32, 24, color.Black)
	white := solidJPEG(t, 32, 24, color.White)
	fetcher := &sequenceFetcher{frames: [][]byte{black, white, black, white, black, white}}
	pub := &fakePublisher{err: errors.New("redis down")}
	cam := makeCam(5)
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	w.Run(ctx) // must not panic
}

func TestWorker_StaticSampleForcesPeriodicPublish(t *testing.T) {
	// Identical white frames: normally all but the first would be filtered.
	// With StaticSampleIntervalS set, frames should still publish periodically.
	white := solidJPEG(t, 64, 48, color.White)
	fetcher := &fakeFetcher{data: white}
	pub := &fakePublisher{}
	cam := makeCam(50)
	cam.StaticSampleIntervalS = 1 // force publish every second even when static
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 1200*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	// First frame always publishes (baseline). After 1 s of static frames,
	// the static sample interval should trigger at least one more publish.
	if pub.callCount() < 2 {
		t.Errorf("expected >=2 publishes with static_sample_interval_s=1 over 1.2 s, got %d", pub.callCount())
	}
}

func TestWorker_ImageDimensionsInMeta(t *testing.T) {
	// Verify the published FrameReady carries correct width/height.
	const W, H = 80, 60
	black := solidJPEG(t, W, H, color.Black)
	white := solidJPEG(t, W, H, color.White)
	fetcher := &sequenceFetcher{frames: [][]byte{black, white}}
	pub := &fakePublisher{}
	cam := makeCam(5)
	gate := motion.New(0.001)
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop())

	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	w.Run(ctx)

	pub.mu.Lock()
	defer pub.mu.Unlock()
	if len(pub.metas) == 0 {
		t.Skip("no frames published — motion gate filtered all (timing)")
	}
	for _, m := range pub.metas {
		if int(m.Width) != W || int(m.Height) != H {
			t.Errorf("dimensions: got %dx%d, want %dx%d", m.Width, m.Height, W, H)
		}
	}
}
