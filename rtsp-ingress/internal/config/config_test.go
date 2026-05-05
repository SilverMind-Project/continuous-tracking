package config

import (
	"os"
	"path/filepath"
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
	if cfg.Go2RTC.Addr != "http://localhost:1984" {
		t.Errorf("go2rtc addr: got %q, want %q", cfg.Go2RTC.Addr, "http://localhost:1984")
	}
	if cfg.Go2RTC.TimeoutSeconds != 10 {
		t.Errorf("go2rtc timeout: got %d, want 10", cfg.Go2RTC.TimeoutSeconds)
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

// ---------------------------------------------------------------------------
// Camera YAML parsing
// ---------------------------------------------------------------------------

func TestCameraYAMLParsingWithHostPort(t *testing.T) {
	yaml := `
cameras:
  - id: cam-test
    host: 192.168.1.100
    port: 554
    username: admin
    password: secret
    stream_path: /stream1
    type: overhead
    room_name: bedroom
    enabled: true
`
	cfg, err := loadFromYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(cfg.Cameras) != 1 {
		t.Fatalf("want 1 camera, got %d", len(cfg.Cameras))
	}
	c := cfg.Cameras[0]
	if c.ID != "cam-test" {
		t.Errorf("id: got %q", c.ID)
	}
	if c.Host != "192.168.1.100" {
		t.Errorf("host: got %q", c.Host)
	}
	if c.Port != 554 {
		t.Errorf("port: got %d", c.Port)
	}
	if c.Username != "admin" {
		t.Errorf("username: got %q", c.Username)
	}
	if c.RoomName != "bedroom" {
		t.Errorf("room_name: got %q", c.RoomName)
	}
	if c.Type != "overhead" {
		t.Errorf("type: got %q", c.Type)
	}
	if !c.Enabled {
		t.Error("enabled: want true")
	}
}

func TestBuildRTSPURLFromComponents(t *testing.T) {
	tests := []struct {
		name    string
		cam     CameraConfig
		wantURL string
	}{
		{
			name:    "with credentials",
			cam:     CameraConfig{Host: "192.168.1.100", Port: 554, Username: "admin", Password: "s3cr3t", StreamPath: "/live"},
			wantURL: "rtsp://admin:s3cr3t@192.168.1.100:554/live",
		},
		{
			name:    "no credentials",
			cam:     CameraConfig{Host: "10.0.0.5", Port: 8554, StreamPath: "/stream"},
			wantURL: "rtsp://10.0.0.5:8554/stream",
		},
		{
			name:    "default port 554 when zero",
			cam:     CameraConfig{Host: "cam.local", StreamPath: "/ch0"},
			wantURL: "rtsp://cam.local:554/ch0",
		},
		{
			name:    "stream_path slash prefix added",
			cam:     CameraConfig{Host: "192.168.0.1", Port: 554, StreamPath: "stream1"},
			wantURL: "rtsp://192.168.0.1:554/stream1",
		},
		{
			name:    "explicit rtsp_url wins",
			cam:     CameraConfig{RTSPURL: "rtsp://existing:554/foo", Host: "other.host"},
			wantURL: "rtsp://existing:554/foo",
		},
		{
			name:    "no host no-op",
			cam:     CameraConfig{},
			wantURL: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cam.BuildRTSPURL()
			if tt.cam.RTSPURL != tt.wantURL {
				t.Errorf("RTSPURL: got %q, want %q", tt.cam.RTSPURL, tt.wantURL)
			}
		})
	}
}

func TestCameraDefaultsAppliedAndURLBuilt(t *testing.T) {
	yaml := `
defaults:
  frame_interval_ms: 1000
  motion_threshold: 0.05
  reconnect_backoff_s: 3.0
cameras:
  - id: cam-1
    host: 192.168.1.1
    port: 554
    username: user
    password: pass
    stream_path: /stream
    enabled: true
`
	cfg, err := loadFromYAMLBytes([]byte(yaml))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(cfg.Cameras) != 1 {
		t.Fatalf("want 1 camera, got %d", len(cfg.Cameras))
	}
	c := cfg.Cameras[0]
	if c.FrameIntervalMs != 1000 {
		t.Errorf("frame_interval_ms: got %d", c.FrameIntervalMs)
	}
	if c.RTSPURL != "rtsp://user:pass@192.168.1.1:554/stream" {
		t.Errorf("RTSPURL: got %q", c.RTSPURL)
	}
}

// ---------------------------------------------------------------------------
// .env loading
// ---------------------------------------------------------------------------

func TestLoadDotEnv(t *testing.T) {
	dir := t.TempDir()
	dotenv := filepath.Join(dir, ".env")
	if err := os.WriteFile(dotenv, []byte("TEST_SECRET=hello\nOTHER=world\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Unsetenv("TEST_SECRET"); err != nil {
		t.Fatal(err)
	}
	if err := loadDotEnv(dotenv); err != nil {
		t.Fatalf("loadDotEnv: %v", err)
	}
	if got := os.Getenv("TEST_SECRET"); got != "hello" {
		t.Errorf("TEST_SECRET: got %q, want %q", got, "hello")
	}
}

func TestLoadDotEnvMissingFileIsOK(t *testing.T) {
	if err := loadDotEnv("/nonexistent/.env"); err != nil {
		t.Errorf("missing .env should not error, got: %v", err)
	}
}

func TestLoadDotEnvDoesNotOverrideExistingEnv(t *testing.T) {
	t.Setenv("ALREADY_SET", "original")
	dir := t.TempDir()
	dotenv := filepath.Join(dir, ".env")
	if err := os.WriteFile(dotenv, []byte("ALREADY_SET=overridden\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := loadDotEnv(dotenv); err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv("ALREADY_SET"); got != "original" {
		t.Errorf("env var overridden: got %q", got)
	}
}

func TestLoadDotEnvQuotedValues(t *testing.T) {
	dir := t.TempDir()
	dotenv := filepath.Join(dir, ".env")
	content := `DOUBLE_QUOTED="value with spaces"
SINGLE_QUOTED='another value'
EXPORT_PREFIX=export_val
export EXPORTED_KEY=exported_value
`
	if err := os.WriteFile(dotenv, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"DOUBLE_QUOTED", "SINGLE_QUOTED", "EXPORT_PREFIX", "EXPORTED_KEY"} {
		if err := os.Unsetenv(k); err != nil {
			t.Fatal(err)
		}
	}
	if err := loadDotEnv(dotenv); err != nil {
		t.Fatalf("loadDotEnv: %v", err)
	}
	if got := os.Getenv("DOUBLE_QUOTED"); got != "value with spaces" {
		t.Errorf("DOUBLE_QUOTED: got %q", got)
	}
	if got := os.Getenv("SINGLE_QUOTED"); got != "another value" {
		t.Errorf("SINGLE_QUOTED: got %q", got)
	}
	if got := os.Getenv("EXPORTED_KEY"); got != "exported_value" {
		t.Errorf("EXPORTED_KEY: got %q", got)
	}
}

// ---------------------------------------------------------------------------
// Placeholder expansion
// ---------------------------------------------------------------------------

func TestExpandPlaceholdersResolvesVars(t *testing.T) {
	t.Setenv("MY_HOST", "192.168.1.50")
	t.Setenv("MY_PASS", "topsecret")
	input := "host: ${MY_HOST}\npassword: ${MY_PASS}"
	got, err := expandPlaceholders(input)
	if err != nil {
		t.Fatalf("expandPlaceholders: %v", err)
	}
	want := "host: 192.168.1.50\npassword: topsecret"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestExpandPlaceholdersErrorOnMissing(t *testing.T) {
	if err := os.Unsetenv("MISSING_VAR"); err != nil {
		t.Fatal(err)
	}
	_, err := expandPlaceholders("key: ${MISSING_VAR}")
	if err == nil {
		t.Error("want error for unresolved placeholder, got nil")
	}
}

func TestExpandPlaceholdersNoOp(t *testing.T) {
	input := "key: plainvalue\nother: no_dollars_here"
	got, err := expandPlaceholders(input)
	if err != nil {
		t.Fatalf("expandPlaceholders: %v", err)
	}
	if got != input {
		t.Errorf("got %q, want %q", got, input)
	}
}

// ---------------------------------------------------------------------------
// End-to-end: YAML with .env placeholders
// ---------------------------------------------------------------------------

func TestLoadFromYAMLWithDotEnvPlaceholders(t *testing.T) {
	dir := t.TempDir()

	dotenv := filepath.Join(dir, ".env")
	if err := os.WriteFile(dotenv, []byte("CAM_PASSWORD=hunter2\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Unsetenv("CAM_PASSWORD"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RTSP_DOTENV_PATH", dotenv)

	yamlContent := `
cameras:
  - id: cam-e2e
    host: 10.0.0.1
    port: 554
    username: admin
    password: ${CAM_PASSWORD}
    stream_path: /live
    enabled: true
`
	cfgFile := filepath.Join(dir, "settings.yaml")
	if err := os.WriteFile(cfgFile, []byte(yamlContent), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadFromYAML(cfgFile)
	if err != nil {
		t.Fatalf("LoadFromYAML: %v", err)
	}
	if len(cfg.Cameras) != 1 {
		t.Fatalf("want 1 camera, got %d", len(cfg.Cameras))
	}
	c := cfg.Cameras[0]
	if c.Password != "hunter2" {
		t.Errorf("password not expanded: got %q", c.Password)
	}
	if c.RTSPURL != "rtsp://admin:hunter2@10.0.0.1:554/live" {
		t.Errorf("RTSPURL: got %q", c.RTSPURL)
	}
}

// ---------------------------------------------------------------------------
// Helpers (package-internal)
// ---------------------------------------------------------------------------

// loadFromYAMLBytes unmarshals YAML content without loading a .env file,
// useful for unit tests that control environment variables directly.
func loadFromYAMLBytes(data []byte) (Config, error) {
	cfg := DefaultConfig()
	if err := unmarshalYAML(data, &cfg); err != nil {
		return Config{}, err
	}
	cfg.normalize()
	cfg.applyCameraDefaults()
	return cfg, nil
}
