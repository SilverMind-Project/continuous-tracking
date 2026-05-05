// Package go2rtc is an HTTP API client for the go2rtc RTSP proxy server.
//
// go2rtc (https://github.com/AlexxIT/go2rtc) multiplexes RTSP camera streams
// and exposes a lightweight HTTP API for stream management and JPEG capture.
// rtsp-ingress uses it as a sidecar: go2rtc owns the RTSP sessions; this
// service polls go2rtc for JPEG frames instead of managing RTSP directly.
package go2rtc

import (
	"context"
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
