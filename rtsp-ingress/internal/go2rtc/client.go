// Package go2rtc is an HTTP API client for the go2rtc RTSP proxy server.
//
// go2rtc (https://github.com/AlexxIT/go2rtc) multiplexes RTSP camera streams
// and exposes a lightweight HTTP API for stream management and JPEG capture.
// rtsp-ingress uses it as a sidecar: go2rtc owns the RTSP sessions; this
// service polls go2rtc for JPEG frames instead of managing RTSP directly.
package go2rtc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"
)

// Client communicates with a go2rtc instance over its HTTP API.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// Config carries the parameters needed to construct a Client.
type Config struct {
	Addr           string // base URL, e.g. "http://localhost:1984"
	TimeoutSeconds int    // per-request HTTP timeout; 0 → 10 s
}

// New returns a Client for the go2rtc server described by cfg.
func New(cfg Config) *Client {
	to := time.Duration(cfg.TimeoutSeconds) * time.Second
	if to <= 0 {
		to = 10 * time.Second
	}
	return &Client{
		baseURL:    cfg.Addr,
		httpClient: &http.Client{Timeout: to},
	}
}

// APIError wraps a non-2xx response from go2rtc.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("go2rtc: HTTP %d: %s", e.StatusCode, e.Body)
}

// RegisterStream registers (or replaces) a stream in go2rtc.
//
// go2rtc PUT uses replace semantics — re-registering an existing stream with
// the same URL is safe and used for state-recovery after go2rtc restarts.
func (c *Client) RegisterStream(ctx context.Context, name, rtspURL string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, c.baseURL+"/api/streams", nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	q := url.Values{}
	q.Set("src", rtspURL)
	q.Set("name", name)
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("put stream: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(resp.Body)
		return &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	}
	return nil
}

// DeregisterStream removes a stream from go2rtc by name.
//
// go2rtc DELETE returns 200 even if the stream does not exist (silent delete).
// Note: go2rtc DELETE /api/streams uses the "src" query parameter for the
// stream name, not "name" — this matches the go2rtc v1.9.x API.
func (c *Client) DeregisterStream(ctx context.Context, name string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.baseURL+"/api/streams", nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	q := url.Values{}
	q.Set("src", name)
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("delete stream: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(resp.Body)
		return &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	}
	return nil
}

// ProbeStream tests whether an RTSP URL is reachable by temporarily
// registering it with go2rtc and attempting to fetch a single JPEG frame.
// The temporary stream is always deregistered; callers can rely on
// ProbeStream having no lasting side effects.
//
// Returns nil when a frame was successfully retrieved. Returns an error
// (possibly wrapping *APIError) when registration or frame fetch fails.
func (c *Client) ProbeStream(ctx context.Context, rtspURL string) error {
	name := fmt.Sprintf("probe-%d", time.Now().UnixNano())

	if err := c.RegisterStream(ctx, name, rtspURL); err != nil {
		return fmt.Errorf("register stream: %w", err)
	}

	// Always clean up the temporary stream. Use a background context so
	// the deregister isn't cancelled when the caller's context expires.
	defer func() { _ = c.DeregisterStream(context.Background(), name) }()

	// Give go2rtc a moment to establish the RTSP session before
	// requesting a frame; the probeCtx deadline bounds the total wait.
	probeCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()

	select {
	case <-probeCtx.Done():
		return probeCtx.Err()
	case <-time.After(2 * time.Second):
	}

	if _, err := c.FetchJPEG(probeCtx, name); err != nil {
		return fmt.Errorf("fetch frame: %w", err)
	}
	return nil
}

// FetchJPEG fetches a JPEG frame from go2rtc for the named stream.
//
// Returns the raw JPEG bytes (caller owns the slice).
// Returns an *APIError with StatusCode 404 if the stream is not registered.
func (c *Client) FetchJPEG(ctx context.Context, name string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/frame.jpeg", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	q := url.Values{}
	q.Set("src", name)
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("get frame: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read frame body: %w", err)
	}
	return data, nil
}

// ListStreams returns the set of streams currently registered in go2rtc.
// The returned map keys are stream names; each value is the raw stream
// info object from go2rtc (producers, consumers, state, etc.).
func (c *Client) ListStreams(ctx context.Context) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/streams", nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list streams: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, &APIError{StatusCode: resp.StatusCode, Body: string(body)}
	}

	var streams map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&streams); err != nil {
		return nil, fmt.Errorf("decode streams response: %w", err)
	}
	return streams, nil
}

// ReloadStream forces go2rtc to reconnect an existing RTSP stream by
// deregistering it and re-registering with the same URL.
func (c *Client) ReloadStream(ctx context.Context, name, rtspURL string) error {
	// Deregister first (ignore "not found" — the goal is a fresh session).
	_ = c.DeregisterStream(ctx, name)
	return c.RegisterStream(ctx, name, rtspURL)
}
