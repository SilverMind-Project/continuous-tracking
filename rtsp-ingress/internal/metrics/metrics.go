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
)
