package config

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"
)

// Config holds all rtsp-ingress configuration, populated from env vars.
type Config struct {
	Server          ServerConfig    `yaml:"server"`
	Redis           RedisConfig     `yaml:"redis"`
	MinIO           MinIOConfig     `yaml:"minio"`
	Cognitive       CognitiveConfig `yaml:"cognitive_companion"`
	Go2RTC          Go2RTCConfig    `yaml:"go2rtc"`
	CameraDefaults  CameraDefaults  `yaml:"defaults"`
	AssignedCameras string          `yaml:"assigned_cameras"`
}

type ServerConfig struct {
	// TODO(phase-0 0.27): add the separate mTLS admin listener on :8510.
	ListenAddr           string        `yaml:"health_addr"`
	ReadTimeout          time.Duration `yaml:"-"`
	WriteTimeout         time.Duration `yaml:"-"`
	ShutdownGraceSeconds int           `yaml:"shutdown_grace_s"`
	ShutdownTimeout      time.Duration `yaml:"-"`
}

type RedisConfig struct {
	Address      string `yaml:"addr"`
	Password     string `yaml:"password"`
	DB           int    `yaml:"db"`
	Stream       string `yaml:"stream"`
	MaxLenApprox int64  `yaml:"maxlen_approx"`
}

type MinIOConfig struct {
	Endpoint    string `yaml:"endpoint"`
	AccessKey   string `yaml:"access_key"`
	SecretKey   string `yaml:"secret_key"`
	UseSSL      bool   `yaml:"secure"`
	Bucket      string `yaml:"bucket"`
	JPEGQuality int    `yaml:"jpeg_quality"`
}

type CognitiveConfig struct {
	BaseURL                  string        `yaml:"base_url"`
	APIKey                   string        `yaml:"api_key"`
	ReconcileIntervalSeconds int           `yaml:"reconcile_interval_s"`
	ReconcileInterval        time.Duration `yaml:"-"`
}

// Go2RTCConfig configures the go2rtc sidecar HTTP API connection.
type Go2RTCConfig struct {
	Addr           string `yaml:"addr"`
	TimeoutSeconds int    `yaml:"timeout_s"`
}

type CameraDefaults struct {
	FrameIntervalMs         int     `yaml:"frame_interval_ms"`
	MotionThreshold         float64 `yaml:"motion_threshold"`
	ReconnectBackoffSeconds float64 `yaml:"reconnect_backoff_s"`
	StaticSampleIntervalS   int     `yaml:"static_sample_interval_s"`
}

// CameraConfig describes a single RTSP camera stream as sourced from the
// cognitive-companion API. RTSPURL must be set; the reconciler populates it
// directly from the API response.
type CameraConfig struct {
	ID                      string
	RTSPURL                 string
	RoomName                string
	FrameIntervalMs         int
	MotionThreshold         float64
	ReconnectBackoffSeconds float64
	StaticSampleIntervalS   int
	Enabled                 bool
	RotationDegrees         int
}

// ValidateRotation checks that degrees is one of the allowed rotation values.
func ValidateRotation(degrees int) error {
	switch degrees {
	case 0, 90, 180, 270:
		return nil
	default:
		return fmt.Errorf("rotation_degrees must be 0, 90, 180, or 270, got %d", degrees)
	}
}

// DefaultConfig returns Config with sensible defaults, overridden by env vars.
func DefaultConfig() Config {
	return Config{
		Server: ServerConfig{
			ListenAddr:           ":8090",
			ReadTimeout:          5 * time.Second,
			WriteTimeout:         10 * time.Second,
			ShutdownGraceSeconds: 15,
			ShutdownTimeout:      15 * time.Second,
		},
		Redis: RedisConfig{
			Address:      "localhost:6379",
			Password:     "",
			DB:           0,
			Stream:       "frames.ready",
			MaxLenApprox: 100000,
		},
		MinIO: MinIOConfig{
			Endpoint:    "localhost:9000",
			AccessKey:   "minioadmin",
			SecretKey:   "minioadmin",
			UseSSL:      false,
			Bucket:      "cts-frames",
			JPEGQuality: 85,
		},
		Cognitive: CognitiveConfig{
			BaseURL:                  "http://localhost:8000",
			APIKey:                   "",
			ReconcileIntervalSeconds: 60,
			ReconcileInterval:        60 * time.Second,
		},
		Go2RTC: Go2RTCConfig{
			Addr:           "http://localhost:1984",
			TimeoutSeconds: 10,
		},
		CameraDefaults: CameraDefaults{
			FrameIntervalMs:         500,
			MotionThreshold:         0.02,
			ReconnectBackoffSeconds: 2.0,
			StaticSampleIntervalS:   0,
		},
		AssignedCameras: "ALL",
	}
}

// LoadFromYAML reads service settings (Redis, MinIO, go2rtc, etc.) from YAML.
// Camera configuration is sourced exclusively from the cognitive-companion API
// via the reconciler; there is no static camera list in the YAML.
func LoadFromYAML(path string) (Config, error) {
	cfg := DefaultConfig()
	if path == "" {
		cfg.normalize()
		return cfg, nil
	}

	//nolint:gosec // The operator controls the config path via RTSP_CONFIG_PATH.
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}

	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("unmarshal %s: %w", path, err)
	}
	cfg.normalize()
	return cfg, nil
}

// Load reads config from YAML and environment variables.
func Load() (Config, error) {
	path := getenv("RTSP_CONFIG_PATH", "config/settings.yaml")

	cfg, err := LoadFromYAML(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return Config{}, err
		}
		cfg = DefaultConfig()
	}

	if v := os.Getenv("SERVER_LISTEN_ADDR"); v != "" {
		cfg.Server.ListenAddr = v
	}
	if v := os.Getenv("ASSIGNED_CAMERAS"); v != "" {
		cfg.AssignedCameras = v
	}
	if v := os.Getenv("REDIS_ADDRESS"); v != "" {
		cfg.Redis.Address = v
	}
	if v := os.Getenv("REDIS_PASSWORD"); v != "" {
		cfg.Redis.Password = v
	}
	if v := os.Getenv("REDIS_DB"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Redis.DB = n
		}
	}
	if v := os.Getenv("REDIS_STREAM"); v != "" {
		cfg.Redis.Stream = v
	}
	if v := os.Getenv("REDIS_MAXLEN_APPROX"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			cfg.Redis.MaxLenApprox = n
		}
	}
	if v := os.Getenv("MINIO_ENDPOINT"); v != "" {
		cfg.MinIO.Endpoint = v
	}
	if v := os.Getenv("MINIO_ACCESS_KEY"); v != "" {
		cfg.MinIO.AccessKey = v
	}
	if v := os.Getenv("MINIO_SECRET_KEY"); v != "" {
		cfg.MinIO.SecretKey = v
	}
	if v := os.Getenv("MINIO_USE_SSL"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.MinIO.UseSSL = b
		}
	}
	if v := os.Getenv("MINIO_BUCKET"); v != "" {
		cfg.MinIO.Bucket = v
	}
	if v := os.Getenv("MINIO_JPEG_QUALITY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.MinIO.JPEGQuality = n
		}
	}
	if v := os.Getenv("COGNITIVE_BASE_URL"); v != "" {
		cfg.Cognitive.BaseURL = v
	}
	if v := os.Getenv("COGNITIVE_API_KEY"); v != "" {
		cfg.Cognitive.APIKey = v
	}
	if v := os.Getenv("COGNITIVE_RECONCILE_INTERVAL_S"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.Cognitive.ReconcileIntervalSeconds = n
		}
	}
	if v := os.Getenv("GO2RTC_ADDR"); v != "" {
		cfg.Go2RTC.Addr = v
	}
	if v := os.Getenv("DEFAULT_FRAME_INTERVAL_MS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.CameraDefaults.FrameIntervalMs = n
		}
	}
	if v := os.Getenv("DEFAULT_MOTION_THRESHOLD"); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.CameraDefaults.MotionThreshold = n
		}
	}
	if v := os.Getenv("DEFAULT_RECONNECT_BACKOFF_S"); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.CameraDefaults.ReconnectBackoffSeconds = n
		}
	}
	if v := os.Getenv("DEFAULT_STATIC_SAMPLE_INTERVAL_S"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			cfg.CameraDefaults.StaticSampleIntervalS = n
		}
	}

	cfg.normalize()
	return cfg, nil
}

// Validate checks required runtime values.
func (c Config) Validate() error {
	switch {
	case c.Server.ListenAddr == "":
		return fmt.Errorf("server.health_addr must not be empty")
	case c.Redis.Address == "":
		return fmt.Errorf("redis.addr must not be empty")
	case c.Redis.Stream == "":
		return fmt.Errorf("redis.stream must not be empty")
	case c.Redis.MaxLenApprox <= 0:
		return fmt.Errorf("redis.maxlen_approx must be > 0")
	case c.MinIO.Endpoint == "":
		return fmt.Errorf("minio.endpoint must not be empty")
	case c.MinIO.Bucket == "":
		return fmt.Errorf("minio.bucket must not be empty")
	case c.MinIO.JPEGQuality < 1 || c.MinIO.JPEGQuality > 100:
		return fmt.Errorf("minio.jpeg_quality must be between 1 and 100")
	case c.Cognitive.BaseURL == "":
		return fmt.Errorf("cognitive_companion.base_url must not be empty")
	case c.CameraDefaults.FrameIntervalMs <= 0:
		return fmt.Errorf("defaults.frame_interval_ms must be > 0")
	case c.CameraDefaults.ReconnectBackoffSeconds <= 0:
		return fmt.Errorf("defaults.reconnect_backoff_s must be > 0")
	default:
		return nil
	}
}

func (c *Config) normalize() {
	if c.Server.ListenAddr == "" {
		c.Server.ListenAddr = ":8090"
	}
	if c.Server.ReadTimeout == 0 {
		c.Server.ReadTimeout = 5 * time.Second
	}
	if c.Server.WriteTimeout == 0 {
		c.Server.WriteTimeout = 10 * time.Second
	}
	if c.Server.ShutdownGraceSeconds <= 0 {
		c.Server.ShutdownGraceSeconds = 15
	}
	c.Server.ShutdownTimeout = time.Duration(c.Server.ShutdownGraceSeconds) * time.Second

	if c.Cognitive.ReconcileIntervalSeconds <= 0 {
		c.Cognitive.ReconcileIntervalSeconds = 60
	}
	c.Cognitive.ReconcileInterval = time.Duration(c.Cognitive.ReconcileIntervalSeconds) * time.Second

	if c.Go2RTC.Addr == "" {
		c.Go2RTC.Addr = "http://localhost:1984"
	}
	if c.Go2RTC.TimeoutSeconds <= 0 {
		c.Go2RTC.TimeoutSeconds = 10
	}
	if c.AssignedCameras == "" {
		c.AssignedCameras = "ALL"
	}
}

func getenv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
