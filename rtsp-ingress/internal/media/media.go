// Package media handles frame JPEG encoding and MinIO upload.
// It also publishes FrameReady messages to Redis Streams.
package media

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"math"
	"math/rand"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"

	pb "github.com/SilverMind-Project/continuous-tracking/proto/continuoustracking/v1"
	"github.com/SilverMind-Project/continuous-tracking/rtsp-ingress/internal/metrics"
)

var ErrNoImageData = errors.New("no image data provided")

// Publisher implements the rtsp.FramePublisher interface: it encodes JPEG,
// uploads to MinIO, and publishes FrameReady to Redis Streams.
type Publisher struct {
	minio  *minio.Client
	bucket string
	redis  *redis.Client
	stream string
	maxlen int64
}

// New creates a new Publisher with the given MinIO client, bucket, Redis
// client, stream name, and approximate MAXLEN trim.
func New(minioClient *minio.Client, bucket string, redisClient *redis.Client, stream string, maxlen int64) *Publisher {
	return &Publisher{
		minio:  minioClient,
		bucket: bucket,
		redis:  redisClient,
		stream: stream,
		maxlen: maxlen,
	}
}

// EnsureBucket creates the target bucket if it does not exist.
func (p *Publisher) EnsureBucket(ctx context.Context) error {
	exists, err := p.minio.BucketExists(ctx, p.bucket)
	if err != nil {
		return fmt.Errorf("bucket exists: %w", err)
	}
	if exists {
		return nil
	}

	err = p.minio.MakeBucket(ctx, p.bucket, minio.MakeBucketOptions{})
	if err == nil {
		return nil
	}
	resp := minio.ToErrorResponse(err)
	if resp.Code == "BucketAlreadyOwnedByYou" || resp.Code == "BucketAlreadyExists" {
		return nil
	}
	return fmt.Errorf("make bucket: %w", err)
}

// Publish uploads to MinIO, updates the FrameReady message with the object key,
// and XADDs the metadata to Redis Streams.
//
// Key format: frames/{camera_id}/{YYYY/MM/DD/HH}/{frame_index}-{capture_time}.jpg
func (p *Publisher) Publish(ctx context.Context, meta *pb.FrameReady, jpegIn []byte) error {
	if jpegIn == nil {
		metrics.PublishErrorsTotal.WithLabelValues(meta.GetCameraId(), "input").Inc()
		return ErrNoImageData
	}

	captureTimeNS := meta.CaptureTimeUnixNs
	if captureTimeNS == 0 || captureTimeNS > math.MaxInt64 {
		//nolint:gosec // Current wall-clock nanoseconds are non-negative here.
		captureTimeNS = uint64(time.Now().UnixNano())
	}
	//nolint:gosec // captureTimeNS is bounded to MaxInt64 above.
	captureTime := time.Unix(0, int64(captureTimeNS)).UTC()
	key := fmt.Sprintf("frames/%s/%s/%020d-%d.jpg",
		meta.CameraId,
		captureTime.Format("2006/01/02/15"),
		meta.FrameIndex,
		captureTimeNS,
	)

	if err := retry(ctx, 3, func() error {
		_, err := p.minio.PutObject(ctx, p.bucket, key, bytes.NewReader(jpegIn), int64(len(jpegIn)), minio.PutObjectOptions{
			ContentType: "image/jpeg",
			UserMetadata: map[string]string{
				"camera-id": meta.CameraId,
				"captured":  captureTime.Format(time.RFC3339Nano),
			},
		})
		return err
	}); err != nil {
		metrics.PublishErrorsTotal.WithLabelValues(meta.GetCameraId(), "minio").Inc()
		return fmt.Errorf("minio put: %w", err)
	}
	meta.MinioKey = key
	meta.CaptureTimeUnixNs = captureTimeNS

	payload, err := proto.Marshal(meta)
	if err != nil {
		metrics.PublishErrorsTotal.WithLabelValues(meta.GetCameraId(), "marshal").Inc()
		return fmt.Errorf("proto marshal: %w", err)
	}

	if err := retry(ctx, 3, func() error {
		return p.redis.XAdd(ctx, &redis.XAddArgs{
			Stream: p.stream,
			MaxLen: p.maxlen,
			Approx: true,
			Values: map[string]any{"frame": payload},
		}).Err()
	}); err != nil {
		metrics.PublishErrorsTotal.WithLabelValues(meta.GetCameraId(), "redis").Inc()
		return fmt.Errorf("redis xadd: %w", err)
	}

	metrics.FramesPublishedTotal.WithLabelValues(meta.GetCameraId()).Inc()
	metrics.FramePayloadBytes.WithLabelValues(meta.GetCameraId()).Observe(float64(len(jpegIn)))
	return nil
}

// EncodeJPEG encodes an image to JPEG with the publisher's configured quality.
func EncodeJPEG(img image.Image, quality int) (*bytes.Buffer, error) {
	buf := bytes.NewBuffer(make([]byte, 0, 128*1024))
	if err := jpeg.Encode(buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, fmt.Errorf("jpeg encode: %w", err)
	}
	return buf, nil
}

func retry(ctx context.Context, attempts int, fn func() error) error {
	var err error
	backoff := 100 * time.Millisecond
	for i := 0; i < attempts; i++ {
		err = fn()
		if err == nil {
			return nil
		}
		if i == attempts-1 {
			break
		}
		jitter := time.Duration(rand.Int63n(int64(backoff / 2)))
		select {
		case <-time.After(backoff + jitter):
		case <-ctx.Done():
			return ctx.Err()
		}
		backoff *= 2
	}
	return err
}
