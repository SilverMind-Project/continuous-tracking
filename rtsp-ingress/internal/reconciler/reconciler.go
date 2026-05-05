// Package reconciler handles config reconciliation against the Cognitive
// Companion Admin API. It filters cameras by ASSIGNED_CAMERAS sharding and
// calls the rtsp.Supervisor.Reconcile with the desired camera list.
package reconciler

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/metrics"
)

// Supervisor is the subset of rtsp.Supervisor used by the reconciler.
type Supervisor interface {
	Reconcile(cameras []config.CameraConfig)
}

// Reconciler polls the Cognitive Companion Admin API for camera configs and
// applies them to the Supervisor. It supports ASSIGNED_CAMERAS sharding.
type Reconciler struct {
	ccBaseURL   string
	ccAPIKey    string
	assigned    string
	interval    time.Duration
	defaults    config.CameraDefaults
	supervisor  Supervisor
	httpClient  *http.Client
	log         *zap.Logger
	mu          sync.RWMutex
	lastCameras []config.CameraConfig
}

// New creates a new Reconciler. assigned is the ASSIGNED_CAMERAS value:
// "ALL", a comma-separated list of camera IDs, or a hash_mod expression.
func New(
	ccBaseURL, ccAPIKey, assigned string,
	interval time.Duration,
	defaults config.CameraDefaults,
	sup Supervisor,
	log *zap.Logger,
) *Reconciler {
	return &Reconciler{
		ccBaseURL:  ccBaseURL,
		ccAPIKey:   ccAPIKey,
		assigned:   assigned,
		interval:   interval,
		defaults:   defaults,
		supervisor: sup,
		httpClient: &http.Client{Timeout: 10 * time.Second},
		log:        log,
	}
}

// Run starts the reconciliation loop. Returns when ctx is cancelled.
func (r *Reconciler) Run(ctx context.Context) error {
	if err := r.reconcileOnce(ctx); err != nil {
		metrics.ReconcileErrorsTotal.Inc()
		r.log.Warn("reconcile_failed", zap.Error(err))
	}
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			if err := r.reconcileOnce(ctx); err != nil {
				metrics.ReconcileErrorsTotal.Inc()
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
	r.mu.Lock()
	r.lastCameras = cloneCameras(cameras)
	r.mu.Unlock()
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
	defer func() { _ = resp.Body.Close() }()

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
			ID:      s.ID,
			Type:    s.CameraType,
			Enabled: true,
		}
		if len(s.ConfigJSON) > 0 {
			var cj cameraConfigJSON
			if err := json.Unmarshal(s.ConfigJSON, &cj); err != nil {
				r.log.Warn("config_json_parse_error",
					zap.String("camera_id", s.ID),
					zap.Error(err),
				)
			} else {
				cc.RTSPURL = cj.RTSPURL
				cc.RTSPMainURL = cj.RTSPMainURL
				cc.Host = cj.Host
				cc.Port = cj.Port
				cc.Username = cj.Username
				cc.Password = cj.Password
				cc.StreamPath = cj.StreamPath
				cc.RoomName = cj.RoomName
				cc.FrameIntervalMs = cj.FrameIntervalMs
				cc.MotionThreshold = cj.MotionThreshold
				cc.ReconnectBackoffSeconds = cj.ReconnectBackoffSeconds
			}
		}
		cc.BuildRTSPURL()
		if cc.FrameIntervalMs <= 0 {
			cc.FrameIntervalMs = r.defaults.FrameIntervalMs
		}
		if cc.MotionThreshold <= 0 {
			cc.MotionThreshold = r.defaults.MotionThreshold
		}
		if cc.ReconnectBackoffSeconds <= 0 {
			cc.ReconnectBackoffSeconds = r.defaults.ReconnectBackoffSeconds
		}
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
	parts := strings.Split(r.assigned, "/")
	if len(parts) != 3 || parts[0] != "hash_mod" {
		return cameras
	}

	modulus, err := strconv.Atoi(parts[1])
	if err != nil || modulus <= 0 {
		return cameras
	}
	if modulus > math.MaxUint32 {
		return cameras
	}
	index, err := strconv.Atoi(parts[2])
	if err != nil || index < 0 || index >= modulus {
		return cameras
	}

	result := make([]config.CameraConfig, 0, len(cameras))
	for _, c := range cameras {
		if shardIndex(c.ID, modulus) == index {
			result = append(result, c)
		}
	}
	return result
}

// LastCameras returns the last successfully fetched camera list.
func (r *Reconciler) LastCameras() []config.CameraConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneCameras(r.lastCameras)
}

type sensorResponse struct {
	ID         string          `json:"id"`
	SensorType string          `json:"sensor_type"`
	ConfigJSON json.RawMessage `json:"config_json"`
	CameraType string          `json:"camera_type"`
}

// cameraConfigJSON mirrors the JSON keys accepted in sensor config_json.
type cameraConfigJSON struct {
	RTSPURL                 string  `json:"rtsp_url"`
	RTSPMainURL             string  `json:"rtsp_main_url"`
	Host                    string  `json:"host"`
	Port                    int     `json:"port"`
	Username                string  `json:"username"`
	Password                string  `json:"password"`
	StreamPath              string  `json:"stream_path"`
	RoomName                string  `json:"room_name"`
	FrameIntervalMs         int     `json:"frame_interval_ms"`
	MotionThreshold         float64 `json:"motion_threshold"`
	ReconnectBackoffSeconds float64 `json:"reconnect_backoff_s"`
}

func shardIndex(cameraID string, modulus int) int {
	hasher := fnv.New32a()
	_, _ = hasher.Write([]byte(cameraID))
	//nolint:gosec // modulus is validated to fit in uint32 before calling shardIndex.
	modulus32 := uint32(modulus)
	return int(hasher.Sum32() % modulus32)
}

func cloneCameras(cameras []config.CameraConfig) []config.CameraConfig {
	cloned := make([]config.CameraConfig, len(cameras))
	copy(cloned, cameras)
	return cloned
}
