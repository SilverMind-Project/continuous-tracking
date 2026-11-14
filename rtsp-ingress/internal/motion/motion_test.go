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

func grayImageWithPatch(w, h int, bg uint8, px0, py0, pw, ph int, patchVal uint8) *image.Gray {
	img := grayImage(w, h, bg)
	for y := py0; y < py0+ph && y < h; y++ {
		for x := px0; x < px0+pw && x < w; x++ {
			img.Pix[y*img.Stride+x] = patchVal
		}
	}
	return img
}

func TestCellsModeSyntheticScenarios(t *testing.T) {
	// 1. All-black / uniform baseline frame -> IsStatic == false.
	cellsGate := NewWithOptions(Options{
		Mode:          ModeCells,
		Threshold:     0.02,
		CellThreshold: 0.04,
		MinCells:      1,
	})
	baseline := grayImage(640, 360, 0)
	if cellsGate.IsStatic(baseline) {
		t.Fatal("first baseline frame must return IsStatic == false")
	}
	cellsGate.MarkPublished()

	// 2. Identical second frame -> IsStatic == true.
	if !cellsGate.IsStatic(baseline) {
		t.Fatal("identical second frame must return IsStatic == true")
	}

	// 3. Localized 30x30 px patch (~1% of a 640x360 frame) changed by +25 grey levels ->
	// frame_mean mode returns IsStatic == true, cells mode returns IsStatic == false.
	frameMeanGate := NewWithOptions(Options{
		Mode:          ModeFrameMean,
		Threshold:     0.02,
		CellThreshold: 0.04,
		MinCells:      1,
	})
	_ = frameMeanGate.IsStatic(baseline)
	frameMeanGate.MarkPublished()

	patchFrame := grayImageWithPatch(640, 360, 0, 80, 80, 30, 30, 25)
	if !frameMeanGate.IsStatic(patchFrame) {
		t.Fatal("frame_mean mode should treat localized 30x30 +25 patch as static (whole-frame mean ~0.001 < 0.02)")
	}
	if cellsGate.IsStatic(patchFrame) {
		t.Fatal("cells mode must detect localized 30x30 +25 patch as motion (IsStatic == false)")
	}
	if cellsGate.LastActiveCells() < 1 {
		t.Fatalf("expected LastActiveCells >= 1, got %d", cellsGate.LastActiveCells())
	}
	cellsGate.MarkPublished()

	// 4. Whole-frame uniform +2 grey-level jitter (codec noise) -> both modes return IsStatic == true.
	base100 := grayImage(640, 360, 100)
	jitter102 := grayImage(640, 360, 102)
	gMeanNoise := NewWithOptions(Options{Mode: ModeFrameMean, Threshold: 0.02, CellThreshold: 0.04, MinCells: 1})
	gCellsNoise := NewWithOptions(Options{Mode: ModeCells, Threshold: 0.02, CellThreshold: 0.04, MinCells: 1})
	_ = gMeanNoise.IsStatic(base100)
	gMeanNoise.MarkPublished()
	_ = gCellsNoise.IsStatic(base100)
	gCellsNoise.MarkPublished()
	if !gMeanNoise.IsStatic(jitter102) {
		t.Fatal("frame_mean mode must ignore +2 grey-level jitter")
	}
	if !gCellsNoise.IsStatic(jitter102) {
		t.Fatal("cells mode must ignore +2 grey-level jitter")
	}

	// 5. Slow sub-threshold drift across 10 frames where each step is below motion_cell_threshold
	// against prevGray (+2 levels/step = 0.0078 < 0.04) but cumulative delta against lastPublishedGray
	// exceeds 0.04 (>= 11 levels = 0.0431) -> cells mode eventually returns IsStatic == false,
	// and after MarkPublished() returns IsStatic == true on the next identical frame.
	driftGate := NewWithOptions(Options{
		Mode:          ModeCells,
		Threshold:     0.02,
		CellThreshold: 0.04,
		MinCells:      1,
	})
	_ = driftGate.IsStatic(grayImage(640, 360, 50))
	driftGate.MarkPublished()

	triggered := false
	var lastFrame *image.Gray
	for step := uint8(1); step <= 10; step++ {
		val := 50 + step*2 // +2 per step; by step 6 cumulative delta is +12 (12/255 = 0.047 >= 0.04)
		lastFrame = grayImageWithPatch(640, 360, 50, 0, 0, 40, 40, val)
		if !driftGate.IsStatic(lastFrame) {
			triggered = true
			driftGate.MarkPublished()
			break
		}
	}
	if !triggered {
		t.Fatal("cells mode must eventually trigger on cumulative slow drift against lastPublishedGray")
	}
	if !driftGate.IsStatic(lastFrame) {
		t.Fatal("after MarkPublished(), the next identical frame must return IsStatic == true")
	}
}
