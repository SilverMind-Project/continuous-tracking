# person-detector-dynamic throughput benchmark

Run `make detector-equivalence` first to confirm output parity, then collect
these numbers with `perf_analyzer` on the DGX before enabling in production.

## How to run

```bash
# Triton must be running with both person-detector and person-detector-dynamic loaded.
export TRITON_URL=localhost:8700

for BATCH in 1 2 4 8; do
  echo "=== static batch=$BATCH ==="
  perf_analyzer -m person-detector \
    --shape images:${BATCH},3,640,640 \
    -u $TRITON_URL --protocol grpc \
    --concurrency-range 1:4 \
    --measurement-interval 10000

  echo "=== dynamic batch=$BATCH ==="
  perf_analyzer -m person-detector-dynamic \
    --shape images:${BATCH},3,640,640 \
    -u $TRITON_URL --protocol grpc \
    --concurrency-range 1:4 \
    --measurement-interval 10000
done
```

## Results

<!-- Populate after running the benchmark on DGX. -->

| Model | Batch | Throughput (infer/s) | p50 latency (ms) | p95 latency (ms) | GPU util (%) |
| --- | --- | --- | --- | --- | --- |
| person-detector (static) | 1 | — | — | — | — |
| person-detector (static) | 2 | — | — | — | — |
| person-detector (static) | 4 | — | — | — | — |
| person-detector (static) | 8 | — | — | — | — |
| person-detector-dynamic | 1 | — | — | — | — |
| person-detector-dynamic | 2 | — | — | — | — |
| person-detector-dynamic | 4 | — | — | — | — |
| person-detector-dynamic | 8 | — | — | — | — |

## Occupancy analysis

Under the single-resident DGX workload (detector requests ≈ consumed frames,
batch occupancy ≈ 1 at 4 cameras), the static model wastes 7/8 slots per call.
The dynamic model sends only the real frame count, eliminating that padding cost.

Expected outcome: at occupancy 1 (1 real frame), dynamic should match or beat
static on latency because TRT does not pad; at occupancy 8, parity is expected.
