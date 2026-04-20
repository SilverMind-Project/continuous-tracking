// Package streams handles Redis Streams FrameReady publishing.
package streams

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
	"google.golang.org/protobuf/proto"

	pb "github.com/khoofia/continuous-tracking/proto/continuoustracking/v1"
)

// Publisher sends FrameReady messages to a Redis Stream.
type Publisher struct {
	client *redis.Client
	stream string
	maxlen int64
}

// New creates a new Publisher. maxlen with Approx trims the stream to roughly
// that many entries to bound Redis memory usage.
func New(client *redis.Client, stream string, maxlen int64) *Publisher {
	return &Publisher{
		client: client,
		stream: stream,
		maxlen: maxlen,
	}
}

// Publish marshals the FrameReady proto and XADDs it to the stream.
func (p *Publisher) Publish(ctx context.Context, meta *pb.FrameReady) error {
	payload, err := proto.Marshal(meta)
	if err != nil {
		return fmt.Errorf("proto marshal: %w", err)
	}

	err = p.client.XAdd(ctx, &redis.XAddArgs{
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
