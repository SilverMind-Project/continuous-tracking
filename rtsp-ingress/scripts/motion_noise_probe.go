// Command motion_noise_probe is a read-only diagnostic that fetches consecutive
// JPEG frames from go2rtc for one or more streams and prints p50 / p95 / p99 /
// max of whole-frame and 16x9 per-cell grayscale deltas.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/motion"
)

type streamStats struct {
	CameraID     string
	Frames       int
	Pairs        int
	FrameDeltas  []float64
	MaxCellDelta []float64
	AllCellDelta []float64
}

func main() {
	go2rtcAddr := flag.String("go2rtc", "http://localhost:1984", "go2rtc base URL")
	camerasFlag := flag.String("cameras", "", "comma-separated camera IDs (default: auto-discover from go2rtc /api/streams)")
	samples := flag.Int("samples", 600, "number of consecutive frames per camera")
	interval := flag.Duration("interval", 500*time.Millisecond, "poll interval between frames")
	timeout := flag.Duration("timeout", 5*time.Second, "per-request HTTP timeout")
	flag.Parse()

	client := &http.Client{Timeout: *timeout}
	cameras := parseCameras(*camerasFlag)
	if len(cameras) == 0 {
		discovered, err := discoverStreams(context.Background(), client, *go2rtcAddr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "discover streams from %s: %v\n", *go2rtcAddr, err)
			os.Exit(1)
		}
		cameras = discovered
	}
	if len(cameras) == 0 {
		fmt.Fprintln(os.Stderr, "no cameras found to probe")
		os.Exit(1)
	}

	fmt.Printf("Probing %d camera(s) at %s: samples=%d interval=%s\n\n",
		len(cameras), *go2rtcAddr, *samples, *interval)

	prevByCam := make(map[string]*image.Gray, len(cameras))
	statsByCam := make(map[string]*streamStats, len(cameras))
	for _, id := range cameras {
		statsByCam[id] = &streamStats{CameraID: id}
	}

	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	for i := 0; i < *samples; i++ {
		if i > 0 {
			<-ticker.C
		}
		for _, id := range cameras {
			gray, err := fetchGrayFrame(context.Background(), client, *go2rtcAddr, id)
			if err != nil {
				fmt.Fprintf(os.Stderr, "[sample %d] %s fetch error: %v\n", i+1, id, err)
				continue
			}
			st := statsByCam[id]
			st.Frames++
			if prev, ok := prevByCam[id]; ok {
				frameMean, cellMeans := motion.ComputeCellDeltas(gray, prev)
				st.Pairs++
				st.FrameDeltas = append(st.FrameDeltas, frameMean)
				maxCell := 0.0
				for _, c := range cellMeans {
					st.AllCellDelta = append(st.AllCellDelta, c)
					if c > maxCell {
						maxCell = c
					}
				}
				st.MaxCellDelta = append(st.MaxCellDelta, maxCell)
			}
			prevByCam[id] = gray
		}
	}

	fmt.Printf("%-24s %6s %10s %10s %10s %10s | %10s %10s %10s %10s %10s\n",
		"camera_id", "pairs",
		"frm_p50", "frm_p95", "frm_p99", "frm_max",
		"cell_p50", "cell_p95", "cell_p99", "cell_p999", "cell_max")
	fmt.Println(strings.Repeat("-", 133))
	for _, id := range cameras {
		st := statsByCam[id]
		fmt.Printf("%-24s %6d %10.5f %10.5f %10.5f %10.5f | %10.5f %10.5f %10.5f %10.5f %10.5f\n",
			st.CameraID,
			st.Pairs,
			percentile(st.FrameDeltas, 0.50),
			percentile(st.FrameDeltas, 0.95),
			percentile(st.FrameDeltas, 0.99),
			maxVal(st.FrameDeltas),
			percentile(st.AllCellDelta, 0.50),
			percentile(st.AllCellDelta, 0.95),
			percentile(st.AllCellDelta, 0.99),
			percentile(st.AllCellDelta, 0.999),
			maxVal(st.AllCellDelta),
		)
	}
}

func parseCameras(raw string) []string {
	var out []string
	for _, part := range strings.Split(raw, ",") {
		p := strings.TrimSpace(part)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func discoverStreams(ctx context.Context, client *http.Client, baseAddr string) ([]string, error) {
	u := strings.TrimRight(baseAddr, "/") + "/api/streams"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(payload))
	for name := range payload {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

func fetchGrayFrame(ctx context.Context, client *http.Client, baseAddr, stream string) (*image.Gray, error) {
	u := fmt.Sprintf("%s/api/frame.jpeg?src=%s", strings.TrimRight(baseAddr, "/"), url.QueryEscape(stream))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return motion.ScaleToGray(img), nil
}

func percentile(vals []float64, p float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	cp := append([]float64(nil), vals...)
	sort.Float64s(cp)
	idx := int(math.Ceil(p*float64(len(cp)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}

func maxVal(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	m := vals[0]
	for _, v := range vals[1:] {
		if v > m {
			m = v
		}
	}
	return m
}
