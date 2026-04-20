package motion

import (
	"image"
	"testing"
)

func grayImage(w, h int, fill uint8) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = fill
	}
	return img
}

func TestFirstCallReturnsFalse(t *testing.T) {
	g := New(0.05)
	img := grayImage(640, 480, 128)
	if g.IsStatic(img) {
		t.Fatal("first call should return false (no motion)")
	}
}

func TestIdenticalFramesAreStatic(t *testing.T) {
	g := New(0.05)
	img := grayImage(640, 480, 128)
	_ = g.IsStatic(img) // baseline
	if !g.IsStatic(img) {
		t.Fatal("identical frames should be static")
	}
}

func TestDifferentFramesTriggerMotion(t *testing.T) {
	g := New(0.05)
	baseline := grayImage(640, 480, 128)
	_ = g.IsStatic(baseline) // baseline

	high := grayImage(640, 480, 200)
	if g.IsStatic(high) {
		t.Fatal("significantly different frames should trigger motion")
	}
}

func TestResolutionOverlap(t *testing.T) {
	g := New(0.05)
	small := grayImage(320, 240, 128)
	_ = g.IsStatic(small) // baseline

	large := grayImage(640, 480, 128)
	// Different resolution: implementation computes overlap.
	// Both uniform gray => delta is 0 => static.
	if !g.IsStatic(large) {
		t.Fatal("same-color frames at different resolutions should be static")
	}
}

func TestThresholdSensitivity(t *testing.T) {
	// 5-level delta normalized: 5/255 ≈ 0.0196
	slight := grayImage(640, 480, 133)

	// Sensitive threshold (0.01) should detect the motion.
	g := New(0.01)
	_ = g.IsStatic(grayImage(640, 480, 128))
	if g.IsStatic(slight) {
		t.Fatal("sensitive gate should detect 5-level motion at 0.01 threshold")
	}

	// Conservative threshold (0.15) should consider it static.
	g2 := New(0.15)
	_ = g2.IsStatic(grayImage(640, 480, 128))
	if !g2.IsStatic(slight) {
		t.Fatal("conservative gate should consider 5-level delta static at 0.15")
	}
}

func TestSubtleMotionBelowThreshold(t *testing.T) {
	g := New(0.05)
	baseline := grayImage(640, 480, 128)
	_ = g.IsStatic(baseline)

	// 1-level delta normalized: 1/255 ≈ 0.004, well below 0.05.
	subtle := grayImage(640, 480, 129)
	if !g.IsStatic(subtle) {
		t.Fatal("subtle changes below threshold should be static")
	}
}

func TestConsecutiveMotion(t *testing.T) {
	g := New(0.05)
	baseline := grayImage(640, 480, 100)
	_ = g.IsStatic(baseline) // baseline

	// Each frame differs by ~14 levels (14/255 ≈ 0.055 > 0.05).
	for i := uint8(114); i <= 200; i += 14 {
		frame := grayImage(640, 480, i)
		if g.IsStatic(frame) {
			t.Fatalf("frame with value %d should show motion", i)
		}
	}
}
