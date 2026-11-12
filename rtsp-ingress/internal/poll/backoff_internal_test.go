package poll

import (
	"testing"
	"time"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/config"
)

func TestBackoffDelayDoublesFromTheInitialDelayUpToTheCap(t *testing.T) {
	cases := []struct {
		name     string
		initial  float64
		max      float64
		failures int
		want     time.Duration
	}{
		{"first failure waits the initial delay", 2, 60, 1, 2 * time.Second},
		{"second failure doubles it", 2, 60, 2, 4 * time.Second},
		{"fifth failure", 2, 60, 5, 32 * time.Second},
		{"sixth failure reaches the cap", 2, 60, 6, 60 * time.Second},
		{"later failures stay at the cap", 2, 60, 40, 60 * time.Second},
		{"a cap below the initial delay is raised to it", 5, 1, 3, 5 * time.Second},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := &Worker{
				cam: config.CameraConfig{
					ReconnectBackoffSeconds:    tc.initial,
					ReconnectBackoffMaxSeconds: tc.max,
				},
				failures: tc.failures,
			}
			if got := w.backoffDelay(); got != tc.want {
				t.Errorf("backoffDelay() = %v, want %v", got, tc.want)
			}
		})
	}
}
