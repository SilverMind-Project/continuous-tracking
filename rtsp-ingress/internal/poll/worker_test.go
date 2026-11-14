package poll_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
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
	"github.com/prometheus/client_golang/prometheus"
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

// ---------------------------------------------------------------------------
// Backoff after consecutive failures
// ---------------------------------------------------------------------------

// fakeClock is a manually advanced clock, safe to read from the worker goroutine.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// switchableFetcher fails until healthy is set, then serves a valid JPEG.
type switchableFetcher struct {
	count   atomic.Int64
	healthy atomic.Bool
	data    []byte
}

func (f *switchableFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.count.Add(1)
	if f.healthy.Load() {
		return f.data, nil
	}
	return nil, errors.New("connection reset by peer")
}

// backoffCam gives each test its own camera id so the per-camera gauge starts clean.
func backoffCam(id string) config.CameraConfig {
	cam := makeCam(2)
	cam.ID = id
	cam.ReconnectBackoffSeconds = 2
	cam.ReconnectBackoffMaxSeconds = 60
	return cam
}

// waitForBackoff waits until the worker has scheduled its retry after a failure: it sets the
// gauge only after computing the retry time, so advancing the clock earlier would race it.
func waitForBackoff(t *testing.T, cameraID string, seconds float64) {
	t.Helper()
	waitFor(t, "the retry to be scheduled", func() bool {
		return backoffGauge(t, cameraID) == seconds
	})
}

// backoffGauge reads rtsp_camera_backoff_seconds for one camera from the default registry.
func backoffGauge(t *testing.T, cameraID string) float64 {
	t.Helper()
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		t.Fatalf("gather metrics: %v", err)
	}
	for _, family := range families {
		if family.GetName() != "rtsp_camera_backoff_seconds" {
			continue
		}
		for _, metric := range family.GetMetric() {
			for _, label := range metric.GetLabel() {
				if label.GetName() == "camera_id" && label.GetValue() == cameraID {
					return metric.GetGauge().GetValue()
				}
			}
		}
	}
	return 0
}

// waitFor polls cond for up to three seconds (polls are slow under -race).
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestWorker_BacksOffAfterAFailedFetch(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_700_000_000, 0)}
	fetcher := &switchableFetcher{}
	cam := backoffCam("cam-backoff")
	w := poll.NewWorker(cam, fetcher, motion.New(cam.MotionThreshold), &fakePublisher{}, zap.NewNop(),
		poll.WithClock(clock.Now))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	// The first failure schedules a retry 2 s later; with the clock frozen no tick may fetch.
	waitForBackoff(t, cam.ID, 2)
	time.Sleep(50 * time.Millisecond) // about 25 ticks
	if got := fetcher.count.Load(); got != 1 {
		t.Fatalf("fetched %d times while backing off, want 1", got)
	}

	// After 2 s it retries once, and the second failure doubles the wait to 4 s.
	clock.Advance(2 * time.Second)
	waitForBackoff(t, cam.ID, 4)
	if got := fetcher.count.Load(); got != 2 {
		t.Fatalf("fetched %d times by the first retry, want 2", got)
	}
	clock.Advance(3 * time.Second)
	time.Sleep(50 * time.Millisecond)
	if got := fetcher.count.Load(); got != 2 {
		t.Fatalf("retried after 3 s of a 4 s backoff (%d fetches)", got)
	}
	clock.Advance(time.Second)
	waitFor(t, "the second retry", func() bool { return fetcher.count.Load() == 3 })
}

func TestWorker_RecoveryRestoresTheFrameRate(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_700_000_000, 0)}
	fetcher := &switchableFetcher{data: solidJPEG(t, 64, 48, color.Gray{Y: 100})}
	cam := backoffCam("cam-recovery")
	w := poll.NewWorker(cam, fetcher, motion.New(cam.MotionThreshold), &fakePublisher{}, zap.NewNop(),
		poll.WithClock(clock.Now))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	waitForBackoff(t, cam.ID, 2)
	fetcher.healthy.Store(true)
	clock.Advance(2 * time.Second)

	// One good frame resets the backoff, so polling continues with the clock still frozen; a
	// worker still backing off would stop at 2 fetches. (Decoding is slow under -race, so the
	// bar is a few fetches, not the full tick rate.)
	waitFor(t, "polling to resume", func() bool { return fetcher.count.Load() >= 5 })
	waitForBackoff(t, cam.ID, 0)
}

func TestWorker_TruncatedFramesBackOffLikeFetchErrors(t *testing.T) {
	clock := &fakeClock{now: time.Unix(1_700_000_000, 0)}
	fetcher := &fakeFetcher{data: []byte{0xff, 0xd8, 0xff}} // a JPEG cut off mid-header
	cam := backoffCam("cam-truncated")
	w := poll.NewWorker(cam, fetcher, motion.New(cam.MotionThreshold), &fakePublisher{}, zap.NewNop(),
		poll.WithClock(clock.Now))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	waitForBackoff(t, cam.ID, 2)
	time.Sleep(50 * time.Millisecond)
	if got := fetcher.callCount(); got != 1 {
		t.Fatalf("fetched %d times after a truncated frame, want 1", got)
	}
}

type clockAdvancingFetcher struct {
	clock   *fakeClock
	advance time.Duration
	data    []byte
	calls   atomic.Int64
}

func (f *clockAdvancingFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.calls.Add(1)
	f.clock.Advance(f.advance)
	return f.data, nil
}

func TestWorker_CaptureAndReceivedTimestampsAreDistinct(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	clock := &fakeClock{now: t0}
	fetcher := &clockAdvancingFetcher{
		clock:   clock,
		advance: 120 * time.Millisecond,
		data:    solidJPEG(t, 64, 48, color.Gray{Y: 200}),
	}
	pub := &fakePublisher{}
	cam := makeCam(5)
	cam.ID = "cam-capture-ts"
	cam.CaptureLatencyMs = 40
	w := poll.NewWorker(cam, fetcher, motion.New(cam.MotionThreshold), pub, zap.NewNop(),
		poll.WithClock(clock.Now))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	waitFor(t, "first frame publish", func() bool { return pub.callCount() >= 1 })
	cancel()

	pub.mu.Lock()
	meta := pub.metas[0]
	pub.mu.Unlock()

	wantCaptureNs := t0.Add(-40 * time.Millisecond).UnixNano()
	wantReceivedNs := t0.Add(120 * time.Millisecond).UnixNano()
	if fmt.Sprintf("%d", meta.CaptureTimeUnixNs) != fmt.Sprintf("%d", wantCaptureNs) {
		t.Fatalf("CaptureTimeUnixNs = %d, want %d", meta.CaptureTimeUnixNs, wantCaptureNs)
	}
	if fmt.Sprintf("%d", meta.ReceivedTimeUnixNs) != fmt.Sprintf("%d", wantReceivedNs) {
		t.Fatalf("ReceivedTimeUnixNs = %d, want %d", meta.ReceivedTimeUnixNs, wantReceivedNs)
	}
	diffMs := float64(meta.ReceivedTimeUnixNs-meta.CaptureTimeUnixNs) / 1e6
	if diffMs < 120.0 {
		t.Fatalf("expected ReceivedTimeUnixNs - CaptureTimeUnixNs >= 120ms, got %.2fms", diffMs)
	}

	age, ok := w.LastFrameAgeSeconds(t0.Add(5120 * time.Millisecond))
	if !ok || age < 4.9 || age > 5.1 {
		t.Fatalf("LastFrameAgeSeconds = (%v, %v), want (~5.0, true)", age, ok)
	}
}

type clampingSequenceFetcher struct {
	mu     sync.Mutex
	frames [][]byte
	idx    int
	calls  atomic.Int64
}

func (f *clampingSequenceFetcher) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls.Add(1)
	i := f.idx
	if i >= len(f.frames) {
		i = len(f.frames) - 1
	}
	f.idx++
	return f.frames[i], nil
}

func TestWorker_MotionHoldWindowPublishesStaticFramesUntilExpiry(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	clock := &fakeClock{now: t0}
	black := solidJPEG(t, 64, 48, color.Black)
	white := solidJPEG(t, 64, 48, color.White)
	// Frame 1: black (baseline -> not static). Frame 2: white (motion). Frame 3+: white (static).
	fetcher := &clampingSequenceFetcher{frames: [][]byte{black, white}}
	pub := &fakePublisher{}
	cam := makeCam(5)
	cam.ID = "cam-motion-hold"
	cam.MotionHoldS = 3.0
	gate := motion.NewWithOptions(motion.Options{
		Mode:          motion.ModeCells,
		Threshold:     0.02,
		CellThreshold: 0.04,
		MinCells:      1,
	})
	w := poll.NewWorker(cam, fetcher, gate, pub, zap.NewNop(), poll.WithClock(clock.Now))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.Run(ctx)

	// Wait for >= 4 frames to be published while clock is within the 3.0s hold window
	// (frames 3 and 4 are identical white frames published only because of MotionHoldS).
	waitFor(t, "frames published inside hold window", func() bool { return pub.callCount() >= 4 })
	// Advance clock past the 3.0s hold window; subsequent static white frames must be filtered.
	clock.Advance(4 * time.Second)
	beforeFetches := fetcher.calls.Load()
	waitFor(t, "one tick to observe advanced clock", func() bool {
		return fetcher.calls.Load() >= beforeFetches+1
	})
	beforeCount := pub.callCount()
	afterAdvanceFetches := fetcher.calls.Load()
	waitFor(t, "additional static fetches after hold window", func() bool {
		return fetcher.calls.Load() >= afterAdvanceFetches+4
	})
	afterCount := pub.callCount()
	if afterCount != beforeCount {
		t.Fatalf("expected static frames after hold window expiry to be filtered, before=%d after=%d", beforeCount, afterCount)
	}
}
