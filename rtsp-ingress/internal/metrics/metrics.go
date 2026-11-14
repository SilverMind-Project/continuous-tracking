package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	FramesPublishedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{Name: "rtsp_frames_published_total"},
		[]string{"camera_id"},
	)
	FramesFilteredTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{Name: "rtsp_frames_filtered_total"},
		[]string{"camera_id", "cause"},
	)
	FetchErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{Name: "rtsp_fetch_errors_total"},
		[]string{"camera_id"},
	)
	CameraBackoffSeconds = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "rtsp_camera_backoff_seconds",
			Help: "Current retry delay after consecutive frame-fetch failures; 0 while healthy.",
		},
		[]string{"camera_id"},
	)
	Go2RTCRegistrationErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{Name: "rtsp_go2rtc_registration_errors_total"},
		[]string{"camera_id"},
	)
	PublishErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{Name: "rtsp_publish_errors_total"},
		[]string{"camera_id", "cause"},
	)
	ActiveWorkers = promauto.NewGauge(
		prometheus.GaugeOpts{Name: "rtsp_active_workers"},
	)
	FramePayloadBytes = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rtsp_frame_payload_bytes",
			Buckets: prometheus.ExponentialBuckets(4096, 2, 10),
		},
		[]string{"camera_id"},
	)
	ReconcileErrorsTotal = promauto.NewCounter(
		prometheus.CounterOpts{Name: "rtsp_reconcile_errors_total"},
	)
	RotationLatencySeconds = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "cts_ingress_rotation_latency_seconds",
			Help:    "Time spent rotating and re-encoding frames at ingest.",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 8),
		},
		[]string{"camera_id"},
	)
	RotationErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cts_ingress_rotation_errors_total",
			Help: "Rotation failures at ingest (invalid config or encode errors).",
		},
		[]string{"camera_id"},
	)
	FetchDurationSeconds = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rtsp_fetch_duration_seconds",
			Help:    "Round-trip duration of go2rtc JPEG frame fetches.",
			Buckets: []float64{0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5},
		},
		[]string{"camera_id"},
	)
	MotionActiveCells = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "rtsp_motion_active_cells",
			Help:    "Number of active 20x20 grid cells on the 320x180 motion evaluation frame.",
			Buckets: []float64{0, 1, 2, 4, 8, 16, 32, 64, 144},
		},
		[]string{"camera_id"},
	)
	LastFramePublishedTimestampSeconds = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "rtsp_last_frame_published_timestamp_seconds",
			Help: "Unix timestamp (seconds) of the last frame published for a camera.",
		},
		[]string{"camera_id"},
	)
	LastFetchSuccessTimestampSeconds = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "rtsp_last_fetch_success_timestamp_seconds",
			Help: "Unix timestamp (seconds) of the last successful go2rtc JPEG fetch for a camera.",
		},
		[]string{"camera_id"},
	)
)
