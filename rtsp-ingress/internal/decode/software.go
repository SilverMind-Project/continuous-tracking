package decode

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/png"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/bluenviron/gortsplib/v4/pkg/format"
	"github.com/bluenviron/gortsplib/v4/pkg/format/rtph264"
	"github.com/pion/rtp"
)

var ErrNoFrameDecoded = errors.New("no frame decoded")

type Decoder interface {
	DecodePacket(pkt *rtp.Packet) (image.Image, error)
	Close() error
}

type Factory func(h264 *format.H264) (Decoder, error)

// NewFactory creates a decoder factory. NVDEC remains a future milestone;
// software decoding is always available via ffmpeg/libavcodec.
func NewFactory(preferred, ffmpegBinary string) Factory {
	_ = preferred
	return func(h264 *format.H264) (Decoder, error) {
		return NewSoftware(h264, ffmpegBinary)
	}
}

func NewSoftware(h264Fmt *format.H264, ffmpegBinary string) (Decoder, error) {
	if h264Fmt == nil {
		return nil, fmt.Errorf("h264 format is required")
	}
	if ffmpegBinary == "" {
		ffmpegBinary = "ffmpeg"
	}
	path, err := exec.LookPath(ffmpegBinary)
	if err != nil {
		return nil, fmt.Errorf("find ffmpeg: %w", err)
	}

	rtpDecoder, err := h264Fmt.CreateDecoder()
	if err != nil {
		return nil, fmt.Errorf("create RTP/H264 decoder: %w", err)
	}

	sps, pps := h264Fmt.SafeParams()
	return &softwareDecoder{
		ffmpegPath:     path,
		rtpDecoder:     rtpDecoder,
		sps:            cloneBytes(sps),
		pps:            cloneBytes(pps),
		maxAccessUnits: 60,
		decodeTimeout:  2 * time.Second,
	}, nil
}

type softwareDecoder struct {
	ffmpegPath     string
	rtpDecoder     *rtph264.Decoder
	sps            []byte
	pps            []byte
	gop            [][]byte
	keyframeSeen   bool
	maxAccessUnits int
	decodeTimeout  time.Duration
}

func (d *softwareDecoder) DecodePacket(pkt *rtp.Packet) (image.Image, error) {
	au, err := d.rtpDecoder.Decode(pkt)
	if err != nil {
		if errors.Is(err, rtph264.ErrMorePacketsNeeded) {
			return nil, nil
		}
		return nil, err
	}
	if len(au) == 0 {
		return nil, nil
	}

	hasIDR := false
	for _, nalu := range au {
		switch naluType(nalu) {
		case 5:
			hasIDR = true
		case 7:
			d.sps = cloneBytes(nalu)
		case 8:
			d.pps = cloneBytes(nalu)
		}
	}

	if hasIDR {
		d.gop = d.gop[:0]
		d.keyframeSeen = true
		if len(d.sps) > 0 {
			d.gop = append(d.gop, annexBMarshal([][]byte{d.sps}))
		}
		if len(d.pps) > 0 {
			d.gop = append(d.gop, annexBMarshal([][]byte{d.pps}))
		}
	}
	if !d.keyframeSeen {
		return nil, nil
	}

	d.gop = append(d.gop, annexBMarshal(au))
	if len(d.gop) > d.maxAccessUnits {
		d.gop = d.gop[len(d.gop)-d.maxAccessUnits:]
	}

	img, err := d.decodeGOP()
	if err != nil && !hasIDR {
		return nil, nil
	}
	return img, err
}

func (d *softwareDecoder) Close() error {
	return nil
}

func (d *softwareDecoder) decodeGOP() (image.Image, error) {
	data := bytes.Join(d.gop, nil)
	if len(data) == 0 {
		return nil, ErrNoFrameDecoded
	}

	ctx, cancel := context.WithTimeout(context.Background(), d.decodeTimeout)
	defer cancel()

	//nolint:gosec // ffmpeg is configured by the operator and required for decoding.
	cmd := exec.CommandContext(
		ctx,
		d.ffmpegPath,
		"-hide_banner",
		"-loglevel", "error",
		"-f", "h264",
		"-i", "pipe:0",
		"-vsync", "0",
		"-f", "image2pipe",
		"-vcodec", "png",
		"pipe:1",
	)
	cmd.Stdin = bytes.NewReader(data)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg decode: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	img, err := decodeLastPNG(stdout.Bytes())
	if err != nil {
		return nil, err
	}
	return img, nil
}

func decodeLastPNG(payload []byte) (image.Image, error) {
	reader := bytes.NewReader(payload)
	var last image.Image
	for reader.Len() > 0 {
		img, err := png.Decode(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("decode png stream: %w", err)
		}
		last = img
	}
	if last == nil {
		return nil, ErrNoFrameDecoded
	}
	return last, nil
}

func annexBMarshal(nalus [][]byte) []byte {
	var buf bytes.Buffer
	for _, nalu := range nalus {
		if len(nalu) == 0 {
			continue
		}
		_, _ = buf.Write([]byte{0x00, 0x00, 0x00, 0x01})
		_, _ = buf.Write(nalu)
	}
	return buf.Bytes()
}

func naluType(nalu []byte) uint8 {
	if len(nalu) == 0 {
		return 0
	}
	return nalu[0] & 0x1F
}

func cloneBytes(src []byte) []byte {
	if len(src) == 0 {
		return nil
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}
