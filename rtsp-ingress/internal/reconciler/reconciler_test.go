package reconciler

import (
	"testing"

	"go.uber.org/zap"

	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
)

func makeCameras(ids ...string) []config.CameraConfig {
	cameras := make([]config.CameraConfig, 0, len(ids))
	for _, id := range ids {
		cameras = append(cameras, config.CameraConfig{ID: id, Enabled: true})
	}
	return cameras
}

func TestFilterAll(t *testing.T) {
	cameras := makeCameras("cam1", "cam2", "cam3")
	r := &Reconciler{assigned: "ALL"}
	result := r.filterByShard(cameras)
	if len(result) != 3 {
		t.Errorf("expected 3 cameras, got %d", len(result))
	}
}

func TestFilterEmptyShard(t *testing.T) {
	cameras := makeCameras("cam1", "cam2")
	r := &Reconciler{assigned: ""}
	result := r.filterByShard(cameras)
	if len(result) != 2 {
		t.Errorf("empty shard should return all cameras, got %d", len(result))
	}
}

func TestFilterCSV(t *testing.T) {
	cameras := makeCameras("cam1", "cam2", "cam3")
	r := &Reconciler{assigned: "cam1, cam3"}
	result := r.filterByShard(cameras)
	if len(result) != 2 {
		t.Fatalf("expected 2 cameras, got %d", len(result))
	}
	ids := map[string]bool{result[0].ID: true, result[1].ID: true}
	if !ids["cam1"] || !ids["cam3"] {
		t.Errorf("expected cam1 and cam3, got %v", ids)
	}
}

func TestFilterCSVNoMatch(t *testing.T) {
	cameras := makeCameras("cam1", "cam2")
	r := &Reconciler{assigned: "cam99"}
	result := r.filterByShard(cameras)
	if len(result) != 0 {
		t.Errorf("expected 0 cameras, got %d", len(result))
	}
}

func TestFilterCaseInsensitiveAll(t *testing.T) {
	cameras := makeCameras("cam1")
	r := &Reconciler{assigned: "all"}
	result := r.filterByShard(cameras)
	if len(result) != 1 {
		t.Errorf("'all' (lowercase) should return all cameras, got %d", len(result))
	}
}

type noOpSupervisor struct{}

func (s *noOpSupervisor) Reconcile([]config.CameraConfig) {}

func TestNewReconciler(t *testing.T) {
	log := zap.NewNop()

	r := New("http://example.com", "api-key", "ALL", 60, &noOpSupervisor{}, log)
	if r == nil {
		t.Fatal("expected non-nil reconciler")
	}
	if r.ccBaseURL != "http://example.com" {
		t.Errorf("base url: got %q", r.ccBaseURL)
	}
	if r.ccAPIKey != "api-key" {
		t.Errorf("api key: got %q", r.ccAPIKey)
	}
}
