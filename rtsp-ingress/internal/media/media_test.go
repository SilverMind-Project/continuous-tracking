package media

import (
	"image"
	"testing"

	"github.com/minio/minio-go/v7/pkg/lifecycle"
)

func solidImage(w, h int, v uint8) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = v
	}
	return img
}

func TestEncodeJPEG(t *testing.T) {
	img := solidImage(100, 100, 128)
	buf, err := EncodeJPEG(img, 85)
	if err != nil {
		t.Fatalf("EncodeJPEG failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("EncodeJPEG returned empty buffer")
	}

	// Verify it starts with JPEG SOI marker.
	if buf.Len() < 2 {
		t.Fatal("buffer too short")
	}
	if buf.Bytes()[0] != 0xFF || buf.Bytes()[1] != 0xD8 {
		t.Error("buffer does not start with JPEG SOI marker")
	}
}

func TestFramesLifecycle(t *testing.T) {
	cfg := framesLifecycle()
	if len(cfg.Rules) != 1 {
		t.Fatalf("expected 1 lifecycle rule, got %d", len(cfg.Rules))
	}
	rule := cfg.Rules[0]
	if rule.Status != "Enabled" {
		t.Errorf("rule Status = %q, want Enabled", rule.Status)
	}
	if rule.RuleFilter.Prefix != "frames/" {
		t.Errorf("rule Prefix = %q, want frames/", rule.RuleFilter.Prefix)
	}
	if rule.Expiration.Days != lifecycle.ExpirationDays(1) {
		t.Errorf("rule Expiration.Days = %d, want 1", rule.Expiration.Days)
	}
}

func TestEncodeJPEGQuality(t *testing.T) {
	img := solidImage(100, 100, 64)
	buf100, err := EncodeJPEG(img, 100)
	if err != nil {
		t.Fatalf("EncodeJPEG(100) failed: %v", err)
	}
	buf50, err := EncodeJPEG(img, 50)
	if err != nil {
		t.Fatalf("EncodeJPEG(50) failed: %v", err)
	}
	// Higher quality should produce larger (or equal) output.
	if buf50.Len() > buf100.Len() {
		t.Errorf("quality 50 produced larger output (%d) than quality 100 (%d)",
			buf50.Len(), buf100.Len())
	}
}
