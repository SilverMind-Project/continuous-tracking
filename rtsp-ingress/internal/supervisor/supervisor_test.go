package supervisor_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap"

	pb "github.com/SilverMind-Project/continuous-tracking/proto/continuoustracking/v1"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/supervisor"
)

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

type fakeGo2RTC struct {
	mu              sync.Mutex
	registered      map[string]string // name → rtspURL
	registerErr     error
	deregisterErr   error
	fetchErr        error
	fetchData       []byte
	registerCalls   int
	deregisterCalls int
	fetchCalls      int
}

func newFakeGo2RTC() *fakeGo2RTC {
	return &fakeGo2RTC{registered: make(map[string]string)}
}

func (f *fakeGo2RTC) RegisterStream(_ context.Context, name, rtspURL string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.registerCalls++
	if f.registerErr != nil {
		return f.registerErr
	}
	f.registered[name] = rtspURL
	return nil
}

func (f *fakeGo2RTC) DeregisterStream(_ context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.deregisterCalls++
	if f.deregisterErr != nil {
		return f.deregisterErr
	}
	delete(f.registered, name)
	return nil
}

func (f *fakeGo2RTC) FetchJPEG(_ context.Context, _ string) ([]byte, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fetchCalls++
	return f.fetchData, f.fetchErr
}

func (f *fakeGo2RTC) getRegistered() map[string]string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make(map[string]string, len(f.registered))
	for k, v := range f.registered {
		out[k] = v
	}
	return out
}

func (f *fakeGo2RTC) getRegisterCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.registerCalls
}

func (f *fakeGo2RTC) getDeregisterCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.deregisterCalls
}

type fakePublisher struct {
	calls atomic.Int64
}

func (p *fakePublisher) Publish(_ context.Context, _ *pb.FrameReady, _ []byte) error {
	p.calls.Add(1)
	return nil
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func cam(id, rtspURL string) config.CameraConfig {
	return config.CameraConfig{
		ID:              id,
		RTSPURL:         rtspURL,
		FrameIntervalMs: 5000, // slow interval — workers won't fire in unit tests
		MotionThreshold: 0.02,
		Enabled:         true,
	}
}

func newSupervisor(ctx context.Context, g2r *fakeGo2RTC) *supervisor.Supervisor {
	return supervisor.New(ctx, g2r, &fakePublisher{}, zap.NewNop())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestSupervisor_ReconcileRegistersStreams(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame") // workers won't publish
	sup := newSupervisor(ctx, g2r)

	cameras := []config.CameraConfig{
		cam("cam-1", "rtsp://192.168.1.1:554/s1"),
		cam("cam-2", "rtsp://192.168.1.2:554/s1"),
	}
	sup.Reconcile(cameras)

	registered := g2r.getRegistered()
	if len(registered) != 2 {
		t.Fatalf("expected 2 registered streams, got %d: %v", len(registered), registered)
	}
	if registered["cam-1"] != "rtsp://192.168.1.1:554/s1" {
		t.Errorf("cam-1 URL: got %q", registered["cam-1"])
	}
}

func TestSupervisor_ReconcileStartsWorkers(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	sup.Reconcile([]config.CameraConfig{cam("cam-1", "rtsp://host/s1")})
	// Give goroutine a moment to start and attempt a poll.
	time.Sleep(20 * time.Millisecond)

	// RegisterStream must have been called for cam-1.
	if g2r.getRegisterCalls() < 1 {
		t.Error("expected at least 1 RegisterStream call")
	}
}

func TestSupervisor_ReconcileRemovesStoppedCameras(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	sup.Reconcile([]config.CameraConfig{
		cam("cam-1", "rtsp://host/s1"),
		cam("cam-2", "rtsp://host/s2"),
	})

	// Second reconcile removes cam-1.
	sup.Reconcile([]config.CameraConfig{
		cam("cam-2", "rtsp://host/s2"),
	})

	registered := g2r.getRegistered()
	if _, ok := registered["cam-1"]; ok {
		t.Error("cam-1 should have been deregistered")
	}
	if _, ok := registered["cam-2"]; !ok {
		t.Error("cam-2 should still be registered")
	}
}

func TestSupervisor_ReconcileReregistersAllOnEveryCall(t *testing.T) {
	// Every Reconcile call must re-register ALL current streams (for go2rtc
	// state recovery after a restart).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	cameras := []config.CameraConfig{cam("cam-1", "rtsp://host/s1")}
	sup.Reconcile(cameras)
	calls1 := g2r.getRegisterCalls()
	sup.Reconcile(cameras)
	calls2 := g2r.getRegisterCalls()

	if calls2 <= calls1 {
		t.Errorf("expected additional RegisterStream calls on second reconcile: got %d then %d", calls1, calls2)
	}
}

func TestSupervisor_DisabledCameraNotStarted(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	disabled := cam("cam-off", "rtsp://host/s1")
	disabled.Enabled = false
	sup.Reconcile([]config.CameraConfig{disabled})

	// Disabled camera must not appear in go2rtc.
	if g2r.getRegisterCalls() != 0 {
		t.Errorf("disabled camera should not be registered, got %d calls", g2r.getRegisterCalls())
	}
}

func TestSupervisor_ConfigChangedRestartsWorker(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	c1 := cam("cam-1", "rtsp://host/s1")
	sup.Reconcile([]config.CameraConfig{c1})

	// Change config — different RTSP URL.
	c2 := cam("cam-1", "rtsp://newhost/s1")
	sup.Reconcile([]config.CameraConfig{c2})

	// DeregisterStream should have been called for the old config.
	if g2r.getDeregisterCalls() < 1 {
		t.Error("expected DeregisterStream call when config changes")
	}
}

func TestSupervisor_Stop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	sup.Reconcile([]config.CameraConfig{cam("cam-1", "rtsp://host/s1")})

	done := make(chan struct{})
	go func() {
		defer close(done)
		sup.Stop(500 * time.Millisecond)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Stop did not return within 2 s")
	}
}

func TestSupervisor_EmptyReconcileNoop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	sup := newSupervisor(ctx, g2r)
	sup.Reconcile(nil) // must not panic
}

func TestSupervisor_RegisterErrorLogged(t *testing.T) {
	// A registration error must not stop the reconcile loop or panic.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.registerErr = errors.New("go2rtc unavailable")
	g2r.fetchErr = errors.New("no frame")
	sup := newSupervisor(ctx, g2r)

	sup.Reconcile([]config.CameraConfig{cam("cam-1", "rtsp://host/s1")})
	// No panic is the assertion.
}

func TestSupervisor_MultipleReconcilesConcurrent(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	g2r := newFakeGo2RTC()
	g2r.fetchErr = errors.New("offline")
	sup := newSupervisor(ctx, g2r)

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			sup.Reconcile([]config.CameraConfig{cam("cam-1", "rtsp://host/s1")})
		}()
	}
	wg.Wait()
	// No race conditions, no panic.
}
