package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/go2rtc"
)

func TestHealthzHandler(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()

	healthzHandler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusOK)
	}

	var body map[string]string
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	if body["status"] != "ok" {
		t.Errorf("status: got %q, want %q", body["status"], "ok")
	}
	if body["service"] != "rtsp-ingress" {
		t.Errorf("service: got %q, want %q", body["service"], "rtsp-ingress")
	}
}

func TestReadyzHandler_NotReady(t *testing.T) {
	var ready bool
	var mu sync.Mutex
	handler := readyzHandler(&ready, &mu)

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusServiceUnavailable)
	}
}

func TestReadyzHandler_Ready(t *testing.T) {
	var ready bool
	var mu sync.Mutex
	handler := readyzHandler(&ready, &mu)

	// Mark ready.
	mu.Lock()
	ready = true
	mu.Unlock()

	req := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusOK)
	}
}

// go2rtcProxy returns a test server that mimics go2rtc's HTTP API just enough
// for ProbeStream (PUT /api/streams, GET /api/frame.jpeg, DELETE /api/streams).
// Every request gets a 200 OK; the frame endpoint returns valid JPEG magic bytes.
func go2rtcProxy() *httptest.Server {
	jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/jpeg")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(jpegData)
	}))
}

func TestTestConnectionHandler_Success(t *testing.T) {
	proxy := go2rtcProxy()
	defer proxy.Close()

	g2r := go2rtc.New(go2rtc.Config{Addr: proxy.URL, TimeoutSeconds: 5})
	handler := testConnectionHandler(g2r)

	body := strings.NewReader(`{"rtsp_url":"rtsp://192.168.1.1:554/stream"}`)
	req := httptest.NewRequest(http.MethodPost, "/internal/test-connection", body)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusOK)
	}

	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if resp["success"] != true {
		t.Errorf("success: got %v, want true. body: %s", resp["success"], w.Body.String())
	}
}

func TestTestConnectionHandler_MissingRTSPURL(t *testing.T) {
	proxy := go2rtcProxy()
	defer proxy.Close()

	g2r := go2rtc.New(go2rtc.Config{Addr: proxy.URL, TimeoutSeconds: 5})
	handler := testConnectionHandler(g2r)

	body := strings.NewReader(`{}`)
	req := httptest.NewRequest(http.MethodPost, "/internal/test-connection", body)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusUnprocessableEntity)
	}
}

func TestTestConnectionHandler_WrongMethod(t *testing.T) {
	proxy := go2rtcProxy()
	defer proxy.Close()

	g2r := go2rtc.New(go2rtc.Config{Addr: proxy.URL, TimeoutSeconds: 5})
	handler := testConnectionHandler(g2r)

	req := httptest.NewRequest(http.MethodGet, "/internal/test-connection", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusMethodNotAllowed)
	}
}

func TestTestConnectionHandler_InvalidJSON(t *testing.T) {
	proxy := go2rtcProxy()
	defer proxy.Close()

	g2r := go2rtc.New(go2rtc.Config{Addr: proxy.URL, TimeoutSeconds: 5})
	handler := testConnectionHandler(g2r)

	body := strings.NewReader(`not json`)
	req := httptest.NewRequest(http.MethodPost, "/internal/test-connection", body)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusUnprocessableEntity {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusUnprocessableEntity)
	}
}

func TestTestConnectionHandler_ProbeFails(t *testing.T) {
	// go2rtc returns 500 for everything, so registration fails.
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("internal error"))
	}))
	defer proxy.Close()

	g2r := go2rtc.New(go2rtc.Config{Addr: proxy.URL, TimeoutSeconds: 5})
	handler := testConnectionHandler(g2r)

	body := strings.NewReader(`{"rtsp_url":"rtsp://host/stream"}`)
	req := httptest.NewRequest(http.MethodPost, "/internal/test-connection", body)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status: got %d, want %d", w.Code, http.StatusOK)
	}
	var resp map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if resp["success"] != false {
		t.Errorf("success: got %v, want false. body: %s", resp["success"], w.Body.String())
	}
	if msg, _ := resp["message"].(string); !strings.Contains(msg, "Connection failed") {
		t.Errorf("message should contain 'Connection failed': %q", msg)
	}
}
