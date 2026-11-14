// Package motion implements a motion pre-filter that compares consecutive
// frames and drops static ones to reduce bandwidth and processing load.
package motion

import (
	"image"

	"golang.org/x/image/draw"
)

const (
	// ModeFrameMean compares only the whole-frame mean delta against Threshold.
	ModeFrameMean = "frame_mean"
	// ModeCells evaluates a 16x9 grid of 20x20 cells on the 320x180 frame.
	ModeCells = "cells"

	GridCols   = 16
	GridRows   = 9
	CellWidth  = 20
	CellHeight = 20
	TotalCells = GridCols * GridRows
)

// Options configures a motion Gate.
type Options struct {
	Mode          string
	Threshold     float64
	CellThreshold float64
	MinCells      int
}

// Gate compares consecutive frames and reports whether significant motion
// was detected. The first call always returns false (no motion) to establish
// a baseline.
type Gate struct {
	mode              string
	threshold         float64
	cellThreshold     float64
	minCells          int
	prevGray          *image.Gray
	lastPublishedGray *image.Gray
	lastActiveCells   int
	firstCall         bool
	scaler            draw.Interpolator
}

// New creates a new motion gate with the given threshold in frame_mean mode.
// Threshold is the mean absolute grayscale delta that triggers "motion".
// Typical values: 0.01 (sensitive) to 0.05 (conservative).
func New(threshold float64) *Gate {
	return NewWithOptions(Options{
		Mode:          ModeFrameMean,
		Threshold:     threshold,
		CellThreshold: 0.04,
		MinCells:      1,
	})
}

// NewWithOptions creates a motion gate configured with Options.
func NewWithOptions(opts Options) *Gate {
	mode := opts.Mode
	if mode != ModeCells {
		mode = ModeFrameMean
	}
	cellThresh := opts.CellThreshold
	if cellThresh <= 0 {
		cellThresh = 0.04
	}
	minCells := opts.MinCells
	if minCells < 1 {
		minCells = 1
	}
	return &Gate{
		mode:          mode,
		threshold:     opts.Threshold,
		cellThreshold: cellThresh,
		minCells:      minCells,
		firstCall:     true,
		scaler:        draw.ApproxBiLinear,
	}
}

// MarkPublished records the most recently evaluated frame as the last
// published reference so slow sub-threshold drift across many frames is
// compared against the last published frame in addition to prevGray.
func (g *Gate) MarkPublished() {
	if g.prevGray != nil {
		g.lastPublishedGray = cloneGray(g.prevGray)
	}
}

// LastActiveCells returns the number of cells that exceeded cellThreshold on
// the most recent IsStatic evaluation.
func (g *Gate) LastActiveCells() int {
	return g.lastActiveCells
}

// IsStatic returns true if the frame is considered static (no significant
// motion compared to the previous frame). The first call always returns false.
func (g *Gate) IsStatic(img image.Image) bool {
	curGray := scaleToGray(g.scaler, img)
	if g.firstCall || g.prevGray == nil {
		g.prevGray = curGray
		g.lastPublishedGray = cloneGray(curGray)
		g.firstCall = false
		g.lastActiveCells = 0
		return false
	}

	meanPrev, cellsPrev := ComputeCellDeltas(curGray, g.prevGray)
	var cellsPub [TotalCells]float64
	hasPub := g.lastPublishedGray != nil
	if hasPub {
		_, cellsPub = ComputeCellDeltas(curGray, g.lastPublishedGray)
	}

	activeCells := 0
	for i := 0; i < TotalCells; i++ {
		if cellsPrev[i] >= g.cellThreshold || (hasPub && cellsPub[i] >= g.cellThreshold) {
			activeCells++
		}
	}
	g.lastActiveCells = activeCells
	g.prevGray = curGray

	if g.mode == ModeCells {
		if activeCells >= g.minCells || meanPrev >= g.threshold {
			return false
		}
		return true
	}

	return meanPrev < g.threshold
}

// ScaleToGray downsamples img to a 320x180 8-bit grayscale image.
func ScaleToGray(img image.Image) *image.Gray {
	return scaleToGray(draw.ApproxBiLinear, img)
}

// ComputeCellDeltas computes the normalized [0,1] whole-frame mean absolute
// delta and per-cell mean absolute deltas over the 16x9 grid of 20x20 cells.
func ComputeCellDeltas(curGray, refGray *image.Gray) (float64, [TotalCells]float64) {
	var cellMeans [TotalCells]float64
	if curGray == nil || refGray == nil {
		return 0, cellMeans
	}
	var totalSum float64
	const cellPixels = float64(CellWidth * CellHeight)
	for cy := 0; cy < GridRows; cy++ {
		y0 := cy * CellHeight
		for cx := 0; cx < GridCols; cx++ {
			x0 := cx * CellWidth
			var cellSum float64
			for y := y0; y < y0+CellHeight; y++ {
				rowOff := y * curGray.Stride
				refOff := y * refGray.Stride
				for x := x0; x < x0+CellWidth; x++ {
					cellSum += absDelta(curGray.Pix[rowOff+x], refGray.Pix[refOff+x])
				}
			}
			totalSum += cellSum
			cellMeans[cy*GridCols+cx] = (cellSum / cellPixels) / 255.0
		}
	}
	frameMean := (totalSum / float64(320*180)) / 255.0
	return frameMean, cellMeans
}

func cloneGray(src *image.Gray) *image.Gray {
	if src == nil {
		return nil
	}
	dst := image.NewGray(src.Bounds())
	copy(dst.Pix, src.Pix)
	return dst
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
