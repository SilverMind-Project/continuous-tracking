// Package media handles frame JPEG encoding and MinIO upload.
// It also publishes FrameReady messages to Redis Streams.
package media

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"

	pb "github.com/khoofia/continuous-tracking/proto/continuoustracking/v1"
)

// Publisher implements the rtsp.FramePublisher interface: it encodes JPEG,
// uploads to MinIO, and publishes FrameReady to Redis Streams.
type Publisher struct {
	minio  *minio.Client
	bucket string
	redis  *redis.Client
	stream string
	maxlen int64
	quality int
}

// New creates a new Publisher with the given MinIO client, bucket, Redis
// client, stream name, approximate MAXLEN trim, and JPEG quality (1-100).
func New(minioClient *minio.Client, bucket string, redisClient *redis.Client, stream string, maxlen int64, quality int) *Publisher {
	return &Publisher{
		minio:   minioClient,
		bucket:  bucket,
		redis:   redisClient,
		stream:  stream,
		maxlen:  maxlen,
		quality: quality,
	}
}

// Publish encodes the image as JPEG (if provided), uploads to MinIO, updates
// the FrameReady message with the object key, and XADDs to Redis Streams.
//
// Key format: frames/{camera_id}/{YYYY/MM/DD/HH}/{frame_index}-{timestamp}.jpg
func (p *Publisher) Publish(ctx context.Context, meta *pb.FrameReady, jpegIn []byte) error {
	var buf *bytes.Buffer

	// Encode JPEG if raw image provided, otherwise use caller-provided bytes.
	if jpegIn == nil {
		// This case shouldn't happen in practice (rtsp worker always has img),
		// but handle it gracefully.
		return fmt.Errorf("no image data provided")
	}
	buf = bytes.NewBuffer(jpegIn)

	// Upload to MinIO.
	now := time.Now().UTC()
	key := fmt.Sprintf("frames/%s/%s/%020d-%d.jpg",
		meta.CameraId,
		now.Format("2006/01/02/15"),
		meta.FrameIndex,
		now.UnixNano(),
	)

	_, err := p.minio.PutObject(ctx, p.bucket, key, bytes.NewReader(buf.Bytes()), int64(buf.Len()), minio.PutObjectOptions{
		ContentType: "image/jpeg",
		UserMetadata: map[string]string{
			"camera-id": meta.CameraId,
			"captured":  now.Format(time.RFC3339Nano),
		},
	})
	if err != nil {
		return fmt.Errorf("minio put: %w", err)
	}
	meta.MinioKey = key

	// Publish to Redis Streams.
	payload, err := proto.Marshal(meta)
	if err != nil {
		return fmt.Errorf("proto marshal: %w", err)
	}

	err = p.redis.XAdd(ctx, &redis.XAddArgs{
		Stream: p.stream,
		MaxLen: p.maxlen,
		Approx: true,
		Values: map[string]any{"frame": string(payload)},
	}).Err()
	if err != nil {
		return fmt.Errorf("redis xadd: %w", err)
	}
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
