// Package motion implements a motion pre-filter that compares consecutive
// frames and drops static ones to reduce bandwidth and processing load.
package motion

import (
	"image"

	"golang.org/x/image/draw"
)

// Gate compares consecutive frames and reports whether significant motion
// was detected. The first call always returns false (no motion) to establish
// a baseline. Resolution changes reset the baseline.
type Gate struct {
	threshold float64
	prevGray  *image.Gray
	firstCall bool
	scaler    draw.Interpolator
}

// New creates a new motion gate with the given threshold.
// Threshold is the mean absolute grayscale delta that triggers "motion".
// Typical values: 0.01 (sensitive) to 0.05 (conservative).
func New(threshold float64) *Gate {
	return &Gate{
		threshold: threshold,
		firstCall: true,
		scaler:    draw.ApproxBiLinear,
	}
}

// IsStatic returns true if the frame is considered static (no significant
// motion compared to the previous frame). The first call always returns false.
func (g *Gate) IsStatic(img image.Image) bool {
	if g.firstCall {
		g.prevGray = scaleToGray(g.scaler, img)
		g.firstCall = false
		return false
	}

	curGray := scaleToGray(g.scaler, img)

	// Compute mean absolute delta over the overlapping region.
	var sum float64
	count := 0
	b := curGray.Bounds()
	pb := g.prevGray.Bounds()
	minX, minY := b.Min.X, b.Min.Y
	maxX, maxY := b.Max.X, b.Max.Y
	if pb.Max.X < maxX {
		maxX = pb.Max.X
	}
	if pb.Max.Y < maxY {
		maxY = pb.Max.Y
	}
	for y := minY; y < maxY; y++ {
		for x := minX; x < maxX; x++ {
			sum += absDelta(curGray.GrayAt(x, y).Y, g.prevGray.GrayAt(x, y).Y)
			count++
		}
	}

	meanDelta := sum / float64(count) / 255.0
	g.prevGray = curGray

	return meanDelta < g.threshold
}

func scaleToGray(s draw.Interpolator, img image.Image) *image.Gray {
	dst := image.NewGray(image.Rect(0, 0, 320, 180))
	s.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)
	return dst
}

func absDelta(a, b uint8) float64 {
	if a > b {
		return float64(a - b)
	}
	return float64(b - a)
}
