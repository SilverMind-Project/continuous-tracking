package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
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
