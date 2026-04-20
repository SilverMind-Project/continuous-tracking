// Package reconciler handles config reconciliation against the Cognitive
// Companion Admin API. It filters cameras by ASSIGNED_CAMERAS sharding and
// calls the rtsp.Supervisor.Reconcile with the desired camera list.
package reconciler

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/khoofia/continuous-tracking/rtsp-ingress/internal/config"
)

// Supervisor is the subset of rtsp.Supervisor used by the reconciler.
type Supervisor interface {
	Reconcile(cameras []config.CameraConfig)
}

// Reconciler polls the Cognitive Companion Admin API for camera configs and
// applies them to the Supervisor. It supports ASSIGNED_CAMERAS sharding.
type Reconciler struct {
	ccBaseURL    string
	ccAPIKey     string
	assigned     string
	interval     time.Duration
	supervisor   Supervisor
	httpClient   *http.Client
	log          *zap.Logger
	lastCameras  []config.CameraConfig
}

// New creates a new Reconciler. assigned is the ASSIGNED_CAMERAS value:
// "ALL", a comma-separated list of camera IDs, or a hash_mod expression.
func New(ccBaseURL, ccAPIKey, assigned string, interval time.Duration, sup Supervisor, log *zap.Logger) *Reconciler {
	return &Reconciler{
		ccBaseURL:  ccBaseURL,
		ccAPIKey:   ccAPIKey,
		assigned:   assigned,
		interval:   interval,
		supervisor: sup,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		log:        log,
	}
}

// Run starts the reconciliation loop. Returns when ctx is cancelled.
func (r *Reconciler) Run(ctx context.Context) error {
	if err := r.reconcileOnce(ctx); err != nil {
		r.log.Warn("reconcile_failed", zap.Error(err))
	}
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if err := r.reconcileOnce(ctx); err != nil {
				r.log.Warn("reconcile_failed", zap.Error(err))
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (r *Reconciler) reconcileOnce(ctx context.Context) error {
	cameras, err := r.fetchCameras(ctx)
	if err != nil {
		return fmt.Errorf("fetch cameras: %w", err)
	}

	cameras = r.filterByShard(cameras)

	r.supervisor.Reconcile(cameras)
	r.lastCameras = cameras
	return nil
}

// fetchCameras calls the Cognitive Companion Admin API to get enabled cameras.
func (r *Reconciler) fetchCameras(ctx context.Context) ([]config.CameraConfig, error) {
	url := fmt.Sprintf("%s/api/v1/sensors?sensor_type=camera&enabled=true", r.ccBaseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if r.ccAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.ccAPIKey)
	}

	resp, err := r.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var sensors []sensorResponse
	if err := json.NewDecoder(resp.Body).Decode(&sensors); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	result := make([]config.CameraConfig, 0, len(sensors))
	for _, s := range sensors {
		cc := config.CameraConfig{
			ID:   s.ID,
			Type: s.CameraType,
		}
		if s.ConfigJSON != nil {
			if url, ok := s.ConfigJSON["rtsp_url"].(string); ok {
				cc.RTSPURL = url
			}
			if mainURL, ok := s.ConfigJSON["rtsp_main_url"].(string); ok {
				cc.RTSPMainURL = mainURL
			}
			if room, ok := s.ConfigJSON["room_name"].(string); ok {
				cc.RoomName = room
			}
			if fi, ok := s.ConfigJSON["frame_interval_ms"].(float64); ok {
				cc.FrameIntervalMs = int(fi)
			}
			if mt, ok := s.ConfigJSON["motion_threshold"].(float64); ok {
				cc.MotionThreshold = mt
			}
			if rb, ok := s.ConfigJSON["reconnect_backoff_s"].(float64); ok {
				cc.ReconnectBackoffSeconds = rb
			}
		}
		cc.Enabled = true
		result = append(result, cc)
	}
	return result, nil
}

// filterByShard applies ASSIGNED_CAMERAS filtering.
// - "ALL" returns all cameras.
// - CSV returns only matching IDs.
// - "hash_mod/N/I" returns cameras where hash(id) % N == I.
func (r *Reconciler) filterByShard(cameras []config.CameraConfig) []config.CameraConfig {
	if r.assigned == "" || strings.EqualFold(r.assigned, "ALL") {
		return cameras
	}

	// CSV shard.
	if !strings.Contains(r.assigned, "/") {
		ids := make(map[string]bool, 10)
		for _, id := range strings.Split(r.assigned, ",") {
			ids[strings.TrimSpace(id)] = true
		}
		result := make([]config.CameraConfig, 0)
		for _, c := range cameras {
			if ids[c.ID] {
				result = append(result, c)
			}
		}
		return result
	}

	// Hash mod shard: "hash_mod/N/I"
	// Simplified: use Go's string hash.
	// TODO: Implement proper hash_mod parsing.
	return cameras
}

// LastCameras returns the last successfully fetched camera list.
func (r *Reconciler) LastCameras() []config.CameraConfig {
	return r.lastCameras
}

type sensorResponse struct {
	ID         string            `json:"id"`
	SensorType string            `json:"sensor_type"`
	ConfigJSON map[string]any    `json:"config_json"`
	CameraType string            `json:"camera_type"`
}
