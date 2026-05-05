package go2rtc_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/go2rtc"
)

// newTestServer returns a test server that records the last request and
// responds with the configured status code and body.
type testServer struct {
	*httptest.Server
	lastMethod string
	lastPath   string
	lastQuery  string
	statusCode int
	body       string
}

func newTestServer(status int, body string) *testServer {
	ts := &testServer{statusCode: status, body: body}
	ts.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ts.lastMethod = r.Method
		ts.lastPath = r.URL.Path
		ts.lastQuery = r.URL.RawQuery
		w.WriteHeader(ts.statusCode)
		_, _ = w.Write([]byte(ts.body))
	}))
	return ts
}

func newClient(ts *testServer) *go2rtc.Client {
	return go2rtc.New(go2rtc.Config{Addr: ts.URL, TimeoutSeconds: 5})
}

func TestRegisterStream_Success(t *testing.T) {
	ts := newTestServer(http.StatusOK, "")
	defer ts.Close()

	client := newClient(ts)
	err := client.RegisterStream(context.Background(), "cam-1", "rtsp://192.168.1.1:554/stream")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ts.lastMethod != http.MethodPut {
		t.Errorf("method: got %q, want PUT", ts.lastMethod)
	}
	if ts.lastPath != "/api/streams" {
		t.Errorf("path: got %q, want /api/streams", ts.lastPath)
	}
	if !strings.Contains(ts.lastQuery, "name=cam-1") {
		t.Errorf("query missing name=cam-1: %q", ts.lastQuery)
	}
	if !strings.Contains(ts.lastQuery, "src=rtsp") {
		t.Errorf("query missing src=rtsp: %q", ts.lastQuery)
	}
}

func TestRegisterStream_Error(t *testing.T) {
	ts := newTestServer(http.StatusInternalServerError, "server error")
	defer ts.Close()

	client := newClient(ts)
	err := client.RegisterStream(context.Background(), "cam-1", "rtsp://192.168.1.1:554/stream")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var apiErr *go2rtc.APIError
	if !asAPIError(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("status code: got %d, want 500", apiErr.StatusCode)
	}
}

func TestDeregisterStream_Success(t *testing.T) {
	ts := newTestServer(http.StatusOK, "")
	defer ts.Close()

	client := newClient(ts)
	err := client.DeregisterStream(context.Background(), "cam-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ts.lastMethod != http.MethodDelete {
		t.Errorf("method: got %q, want DELETE", ts.lastMethod)
	}
	// go2rtc DELETE uses "src" param for stream name (not "name")
	if !strings.Contains(ts.lastQuery, "src=cam-1") {
		t.Errorf("query missing src=cam-1: %q", ts.lastQuery)
	}
	if strings.Contains(ts.lastQuery, "name=") {
		t.Errorf("query should not contain name= param: %q", ts.lastQuery)
	}
}

func TestDeregisterStream_404IsOK(t *testing.T) {
	// go2rtc returns 200 on delete for non-existent streams (tested here via
	// mock that returns 200 — the point is we accept 2xx).
	ts := newTestServer(http.StatusOK, "")
	defer ts.Close()

	client := newClient(ts)
	if err := client.DeregisterStream(context.Background(), "nonexistent"); err != nil {
		t.Errorf("unexpected error for silent delete: %v", err)
	}
}

func TestFetchJPEG_Success(t *testing.T) {
	jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0} // JPEG magic bytes
	ts := newTestServer(http.StatusOK, string(jpegData))
	defer ts.Close()

	client := newClient(ts)
	data, err := client.FetchJPEG(context.Background(), "cam-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ts.lastMethod != http.MethodGet {
		t.Errorf("method: got %q, want GET", ts.lastMethod)
	}
	if ts.lastPath != "/api/frame.jpeg" {
		t.Errorf("path: got %q, want /api/frame.jpeg", ts.lastPath)
	}
	if !strings.Contains(ts.lastQuery, "src=cam-1") {
		t.Errorf("query missing src=cam-1: %q", ts.lastQuery)
	}
	if string(data) != string(jpegData) {
		t.Errorf("data mismatch: got %v, want %v", data, jpegData)
	}
}

func TestFetchJPEG_NotFound(t *testing.T) {
	ts := newTestServer(http.StatusNotFound, "stream not found")
	defer ts.Close()

	client := newClient(ts)
	_, err := client.FetchJPEG(context.Background(), "cam-missing")
	if err == nil {
		t.Fatal("expected error for 404, got nil")
	}
	var apiErr *go2rtc.APIError
	if !asAPIError(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusNotFound {
		t.Errorf("status code: got %d, want 404", apiErr.StatusCode)
	}
}

func TestFetchJPEG_ServerError(t *testing.T) {
	ts := newTestServer(http.StatusServiceUnavailable, "busy")
	defer ts.Close()

	client := newClient(ts)
	_, err := client.FetchJPEG(context.Background(), "cam-1")
	if err == nil {
		t.Fatal("expected error for 503, got nil")
	}
}

func TestNewClient_DefaultTimeout(t *testing.T) {
	// Zero timeout should not panic; client uses 10 s default.
	c := go2rtc.New(go2rtc.Config{Addr: "http://localhost:1984", TimeoutSeconds: 0})
	if c == nil {
		t.Fatal("New returned nil")
	}
}

func TestAPIError_Message(t *testing.T) {
	err := &go2rtc.APIError{StatusCode: 503, Body: "overloaded"}
	msg := err.Error()
	if !strings.Contains(msg, "503") {
		t.Errorf("error message missing status code: %q", msg)
	}
	if !strings.Contains(msg, "overloaded") {
		t.Errorf("error message missing body: %q", msg)
	}
}

func TestContextCancellation(t *testing.T) {
	ts := newTestServer(http.StatusOK, "")
	defer ts.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	client := newClient(ts)
	err := client.RegisterStream(ctx, "cam-1", "rtsp://host/stream")
	if err == nil {
		t.Error("expected error for cancelled context, got nil")
	}
}

// asAPIError unwraps err into *go2rtc.APIError via errors.As.
func asAPIError(err error, target **go2rtc.APIError) bool {
	return errors.As(err, target)
}
