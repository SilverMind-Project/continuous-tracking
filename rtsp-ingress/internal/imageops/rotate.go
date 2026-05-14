// Package imageops provides in-process image transforms.
//
// Rotation runs at ingest, between JPEG decode and motion gate, so
// every downstream consumer (detector, ReID, face id, tracker, identity
// resolver, live view) sees a single canonical orientation.  No cgo, no
// external dependencies — just the Go stdlib image package.
package imageops

import (
	"fmt"
	"image"
	"image/jpeg"
	"io"
)

// RotateCW returns a new image rotated clockwise by degrees.
// degrees must be 0, 90, 180, or 270.
func RotateCW(src image.Image, degrees int) (image.Image, error) {
	switch degrees {
	case 0:
		return src, nil
	case 90:
		return rotate90CW(src), nil
	case 180:
		return rotate180(src), nil
	case 270:
		return rotate270CW(src), nil
	default:
		return nil, fmt.Errorf("unsupported rotation: %d", degrees)
	}
}

// EncodeJPEG encodes img as JPEG at the given quality (1–100).
func EncodeJPEG(img image.Image, quality int) ([]byte, error) {
	var buf []byte
	w := &writer{buf: &buf}
	err := jpeg.Encode(w, img, &jpeg.Options{Quality: quality})
	if err != nil {
		return nil, fmt.Errorf("jpeg encode: %w", err)
	}
	return *w.buf, nil
}

// writer wraps a []byte pointer so jpeg.Encode can write to it.
type writer struct {
	buf *[]byte
}

func (w *writer) Write(p []byte) (int, error) {
	*w.buf = append(*w.buf, p...)
	return len(p), nil
}

// ---------------------------------------------------------------------------
// Rotation primitives (stdlib-only, no cgo)
// ---------------------------------------------------------------------------

// newRGBA allocates an RGBA image with the given dimensions.
func newRGBA(w, h int) *image.RGBA {
	return image.NewRGBA(image.Rect(0, 0, w, h))
}

// rotate90CW rotates clockwise: (x, y) → (h-1-y, x).
func rotate90CW(src image.Image) *image.RGBA {
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	dst := newRGBA(sh, sw)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			dst.Set(sh-1-y, x, src.At(x, y))
		}
	}
	return dst
}

// rotate180 rotates 180°: (x, y) → (w-1-x, h-1-y).
func rotate180(src image.Image) *image.RGBA {
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	dst := newRGBA(sw, sh)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			dst.Set(sw-1-x, sh-1-y, src.At(x, y))
		}
	}
	return dst
}

// rotate270CW rotates 270° clockwise (or 90° CCW): (x, y) → (y, w-1-x).
func rotate270CW(src image.Image) *image.RGBA {
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	dst := newRGBA(sh, sw)
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			dst.Set(y, sw-1-x, src.At(x, y))
		}
	}
	return dst
}

// DecodeJPEG decodes JPEG bytes into an image.Image.
func DecodeJPEG(r io.Reader) (image.Image, error) {
	return jpeg.Decode(r)
}
