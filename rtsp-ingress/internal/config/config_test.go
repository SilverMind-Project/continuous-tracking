package config

import (
	"os"
	"testing"
	"time"
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
	// Clear all relevant env vars.
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
