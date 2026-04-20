package media

import (
	"image"
	"testing"
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
