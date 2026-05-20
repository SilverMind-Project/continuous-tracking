package config

import (
	"os"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Server.ListenAddr != ":8090" {
		t.Errorf("listen addr: got %q, want %q", cfg.Server.ListenAddr, ":8090")
	}
	if cfg.Server.ReadTimeout != 5*time.Second {
		t.Errorf("read timeout: got %v", cfg.Server.ReadTimeout)
	}
	if cfg.Redis.Address != "localhost:6379" {
		t.Errorf("redis addr: got %q, want %q", cfg.Redis.Address, "localhost:6379")
	}
	if cfg.MinIO.Endpoint != "localhost:9000" {
		t.Errorf("minio endpoint: got %q, want %q", cfg.MinIO.Endpoint, "localhost:9000")
	}
	if cfg.MinIO.Bucket != "cts-frames" {
		t.Errorf("minio bucket: got %q, want %q", cfg.MinIO.Bucket, "cts-frames")
	}
	if cfg.Cognitive.BaseURL != "http://localhost:8000" {
		t.Errorf("cognitive base url: got %q, want %q", cfg.Cognitive.BaseURL, "http://localhost:8000")
	}
	if cfg.Redis.Stream != "frames.ready" {
		t.Errorf("redis stream: got %q", cfg.Redis.Stream)
	}
	if cfg.Go2RTC.Addr != "http://localhost:1984" {
		t.Errorf("go2rtc addr: got %q, want %q", cfg.Go2RTC.Addr, "http://localhost:1984")
	}
	if cfg.Go2RTC.TimeoutSeconds != 10 {
		t.Errorf("go2rtc timeout: got %d, want 10", cfg.Go2RTC.TimeoutSeconds)
	}
	if cfg.CameraDefaults.StaticSampleIntervalS != 0 {
		t.Errorf("static_sample_interval_s: got %d, want 0", cfg.CameraDefaults.StaticSampleIntervalS)
	}
}

func TestGo2RTCConfigFromYAML(t *testing.T) {
	yaml := `
go2rtc:
  addr: http://go2rtc:1984
  timeout_s: 5
`
	cfg, err := loadFromYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if cfg.Go2RTC.Addr != "http://go2rtc:1984" {
		t.Errorf("addr: got %q", cfg.Go2RTC.Addr)
	}
	if cfg.Go2RTC.TimeoutSeconds != 5 {
		t.Errorf("timeout_s: got %d", cfg.Go2RTC.TimeoutSeconds)
	}
}

func TestGo2RTCAddrEnvOverride(t *testing.T) {
	t.Setenv("GO2RTC_ADDR", "http://custom-go2rtc:1984")
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Go2RTC.Addr != "http://custom-go2rtc:1984" {
		t.Errorf("go2rtc addr: got %q", cfg.Go2RTC.Addr)
	}
}

func TestLoadEnvOverrides(t *testing.T) {
	t.Setenv("SERVER_LISTEN_ADDR", ":9999")
	t.Setenv("REDIS_ADDRESS", "redis:6379")
	t.Setenv("MINIO_ENDPOINT", "minio:9000")
	t.Setenv("MINIO_BUCKET", "test-bucket")
	t.Setenv("COGNITIVE_BASE_URL", "http://cc:8080")
	t.Setenv("COGNITIVE_API_KEY", "test-key")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cfg.Server.ListenAddr != ":9999" {
		t.Errorf("listen addr: got %q, want %q", cfg.Server.ListenAddr, ":9999")
	}
	if cfg.Redis.Address != "redis:6379" {
		t.Errorf("redis addr: got %q, want %q", cfg.Redis.Address, "redis:6379")
	}
	if cfg.MinIO.Endpoint != "minio:9000" {
		t.Errorf("minio endpoint: got %q, want %q", cfg.MinIO.Endpoint, "minio:9000")
	}
	if cfg.MinIO.Bucket != "test-bucket" {
		t.Errorf("minio bucket: got %q, want %q", cfg.MinIO.Bucket, "test-bucket")
	}
	if cfg.Cognitive.BaseURL != "http://cc:8080" {
		t.Errorf("cognitive base url: got %q, want %q", cfg.Cognitive.BaseURL, "http://cc:8080")
	}
	if cfg.Cognitive.APIKey != "test-key" {
		t.Errorf("cognitive api key: got %q, want %q", cfg.Cognitive.APIKey, "test-key")
	}
}

func TestLoadPreservesDefaultsWhenNoEnv(t *testing.T) {
	envs := []string{
		"SERVER_LISTEN_ADDR", "REDIS_ADDRESS", "REDIS_PASSWORD", "REDIS_DB",
		"MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET",
		"COGNITIVE_BASE_URL", "COGNITIVE_API_KEY", "COGNITIVE_JWT_SECRET",
	}
	for _, e := range envs {
		if err := os.Unsetenv(e); err != nil {
			t.Fatalf("Unsetenv(%q) failed: %v", e, err)
		}
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defaults := DefaultConfig()

	if cfg.Server.ListenAddr != defaults.Server.ListenAddr {
		t.Errorf("listen addr: got %q, want default %q", cfg.Server.ListenAddr, defaults.Server.ListenAddr)
	}
	if cfg.MinIO.AccessKey != "minioadmin" {
		t.Errorf("minio access key: got %q, want %q", cfg.MinIO.AccessKey, "minioadmin")
	}
}

func TestLoadFromYAMLAndValidate(t *testing.T) {
	cfg, err := LoadFromYAML("../../config/settings.yaml")
	if err != nil {
		t.Fatalf("LoadFromYAML failed: %v", err)
	}
	if cfg.MinIO.Bucket != "cts-frames" {
		t.Fatalf("bucket: got %q", cfg.MinIO.Bucket)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate failed: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Helpers (package-internal)
// ---------------------------------------------------------------------------

func loadFromYAMLBytes(data []byte) (Config, error) {
	cfg := DefaultConfig()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, err
	}
	cfg.normalize()
	return cfg, nil
}
