package imageops

import (
	"image"
	"image/color"
	"testing"
)

// A 4×4 test pattern with known corners.
func testImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	// Top-left red
	img.Set(0, 0, color.RGBA{255, 0, 0, 255})
	// Top-right green
	img.Set(3, 0, color.RGBA{0, 255, 0, 255})
	// Bottom-left blue
	img.Set(0, 3, color.RGBA{0, 0, 255, 255})
	// Bottom-right white
	img.Set(3, 3, color.RGBA{255, 255, 255, 255})
	return img
}

func mustRotate(t *testing.T, img image.Image, degrees int) image.Image {
	t.Helper()
	rot, err := RotateCW(img, degrees)
	if err != nil {
		t.Fatalf("RotateCW(%d): %v", degrees, err)
	}
	return rot
}

func TestRotateCW_0(t *testing.T) {
	img := testImage()
	rot := mustRotate(t, img, 0)
	b := rot.Bounds()
	if b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("size: want 4x4, got %dx%d", b.Dx(), b.Dy())
	}
}

func TestRotateCW_90(t *testing.T) {
	img := testImage()
	rot := mustRotate(t, img, 90)
	b := rot.Bounds()
	if b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("size: want 4x4, got %dx%d", b.Dx(), b.Dy())
	}
	// After 90° CW: top-left pixel was bottom-left.
	r, g, b2, _ := rot.At(0, 0).RGBA()
	if r>>8 != 0 || g>>8 != 0 || b2>>8 != 255 {
		t.Errorf("top-left after 90CW: want blue, got R=%d G=%d B=%d", r>>8, g>>8, b2>>8)
	}
}

func TestRotateCW_180(t *testing.T) {
	img := testImage()
	rot := mustRotate(t, img, 180)
	b := rot.Bounds()
	if b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("size: want 4x4, got %dx%d", b.Dx(), b.Dy())
	}
	// After 180°: top-left was bottom-right (white).
	r, g, b2, _ := rot.At(0, 0).RGBA()
	if r>>8 != 255 || g>>8 != 255 || b2>>8 != 255 {
		t.Errorf("top-left after 180: want white, got R=%d G=%d B=%d", r>>8, g>>8, b2>>8)
	}
}

func TestRotateCW_270(t *testing.T) {
	img := testImage()
	rot := mustRotate(t, img, 270)
	b := rot.Bounds()
	if b.Dx() != 4 || b.Dy() != 4 {
		t.Errorf("size: want 4x4, got %dx%d", b.Dx(), b.Dy())
	}
	// After 270° CW: top-left was top-right (green).
	r, g, b2, _ := rot.At(0, 0).RGBA()
	if r>>8 != 0 || g>>8 != 255 || b2>>8 != 0 {
		t.Errorf("top-left after 270CW: want green, got R=%d G=%d B=%d", r>>8, g>>8, b2>>8)
	}
}

func TestRotateCW_Invalid(t *testing.T) {
	_, err := RotateCW(testImage(), 45)
	if err == nil {
		t.Error("wanted error for 45 degrees")
	}
}

func TestRotateCW_NonSquare(t *testing.T) {
	// 6×4 rectangle: rotate 90 should become 4×6.
	img := image.NewRGBA(image.Rect(0, 0, 6, 4))
	rot := mustRotate(t, img, 90)
	b := rot.Bounds()
	if b.Dx() != 4 || b.Dy() != 6 {
		t.Errorf("size: want 4x6, got %dx%d", b.Dx(), b.Dy())
	}
}

func TestEncodeJPEGRoundTrip(t *testing.T) {
	img := testImage()
	jpg, err := EncodeJPEG(img, 90)
	if err != nil {
		t.Fatalf("EncodeJPEG: %v", err)
	}
	if len(jpg) == 0 {
		t.Fatal("empty jpeg")
	}
}
