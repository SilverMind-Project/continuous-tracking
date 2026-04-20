package config

import (
	"os"
	"time"
)

// Config holds all rtsp-ingress configuration, populated from env vars.
type Config struct {
	Server     ServerConfig
	Redis      RedisConfig
	MinIO      MinIOConfig
	Cognitive  CognitiveConfig
	Cameras    []CameraConfig
	ListenAddr string
}

type ServerConfig struct {
	ListenAddr string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	ShutdownTimeout time.Duration
}

type RedisConfig struct {
	Address  string
	Password string
	DB       int
}

type MinIOConfig struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	UseSSL    bool
	Bucket    string
}

type CognitiveConfig struct {
	BaseURL   string
	APIKey    string
	JWTSecret string
}

type CameraConfig struct {
	ID                      string
	RTSPURL                 string
	RTSPMainURL             string
	Type                    string // overhead, eye_level, doorway
	RoomName                string
	FrameIntervalMs         int
	MotionThreshold         float64
	ReconnectBackoffSeconds float64
	Enabled                 bool
}

// DefaultConfig returns Config with sensible defaults, overridden by env vars.
func DefaultConfig() Config {
	return Config{
		Server: ServerConfig{
			ListenAddr:     ":8090",
			ReadTimeout:    5 * time.Second,
			WriteTimeout:   10 * time.Second,
			ShutdownTimeout: 10 * time.Second,
		},
		Redis: RedisConfig{
			Address:  "localhost:6379",
			Password: "",
			DB:       0,
		},
		MinIO: MinIOConfig{
			Endpoint:  "localhost:9000",
			AccessKey: "minioadmin",
			SecretKey: "minioadmin",
			UseSSL:    false,
			Bucket:    "continuous-tracking",
		},
		Cognitive: CognitiveConfig{
			BaseURL:   "http://localhost:8000",
			APIKey:    "",
			JWTSecret: "",
		},
		ListenAddr: ":8090",
	}
}

// Load reads config from environment variables.
func Load() Config {
	cfg := DefaultConfig()

	if v := os.Getenv("SERVER_LISTEN_ADDR"); v != "" {
		cfg.Server.ListenAddr = v
	}
	if v := os.Getenv("REDIS_ADDRESS"); v != "" {
		cfg.Redis.Address = v
	}
	if v := os.Getenv("REDIS_PASSWORD"); v != "" {
		cfg.Redis.Password = v
	}
	if v := os.Getenv("REDIS_DB"); v != "" {
		// parse later
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
	if v := os.Getenv("MINIO_BUCKET"); v != "" {
		cfg.MinIO.Bucket = v
	}
	if v := os.Getenv("COGNITIVE_BASE_URL"); v != "" {
		cfg.Cognitive.BaseURL = v
	}
	if v := os.Getenv("COGNITIVE_API_KEY"); v != "" {
		cfg.Cognitive.APIKey = v
	}
	if v := os.Getenv("COGNITIVE_JWT_SECRET"); v != "" {
		cfg.Cognitive.JWTSecret = v
	}

	return cfg
}
