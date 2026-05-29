# CTS Triton Models for Jetson Orin Nano Super

This is a Jetson-specific Triton model repository for the Continuous Tracking
System. It serves only the four CTS runtime models:

- `person-detector` - YOLO26L
- `pose-rtmpose` - RTMPose-m
- `reid-solider` - SOLIDER-REID
- `depth-anything-v2` - Depth-Anything-V2-Metric-Indoor-Small-hf

The DGX deployment continues to use `triton-models/` and the main
`docker-compose.yml`. This directory is separate so Jetson tuning does not
change the DGX model repository.

## Runtime Strategy

Jetson Orin Nano Super is an Ampere Jetson with 8 GB unified memory. The most
reliable high-performance path for these exported ONNX models is:

1. Build TensorRT FP16 plan files on the Jetson itself.
2. Serve the generated `.plan` files with Triton's TensorRT backend.
3. Keep model names and tensor names identical to the DGX repository so CTS
   clients do not need code changes.

INT8 can be faster, but it should only be enabled after calibrated/QDQ ONNX
exports are available for these exact models. The checked-in script defaults to
FP16 because it is hardware-accelerated, calibration-free, and preserves model
quality.

## Build Plans on the Jetson

Copy or mount `continuous-tracking/` on the Jetson, then run:

```bash
cd continuous-tracking
python triton-models/scripts/export_yolo.py --weights yolo26l.pt \
  --batch 8 --out triton-models-jetson/person-detector/1/model.onnx
bash triton-models-jetson/scripts/build_tensorrt_plans.sh
```

The script reads ONNX files from `triton-models/` and writes:

```text
triton-models-jetson/<model>/1/model.plan
```

TensorRT engines are hardware-, CUDA-, and TensorRT-version specific, so do not
build these on the DGX and copy them to the Jetson.

If the Depth Anything source ONNX was exported with PyTorch 2.12's dynamo
exporter, re-export it before building plans:

```bash
uv run --with torch --with transformers --with onnx \
  python triton-models/scripts/export_depth_anything_v2.py \
    --output triton-models/depth-anything-v2/1/model.onnx \
    --opset 17
```

The default export is static 1x3x518x518 for TensorRT compatibility.

## Start Triton on the Jetson

```bash
cd continuous-tracking
docker compose -f docker-compose.jetson-triton.yml up -d
```

Verify readiness from any host on the LAN:

```bash
curl -f http://nano.int.khoofia.com:8700/v2/health/ready
curl -f http://nano.int.khoofia.com:8700/v2/models/person-detector/ready
curl -f http://nano.int.khoofia.com:8700/v2/models/pose-rtmpose/ready
curl -f http://nano.int.khoofia.com:8700/v2/models/reid-solider/ready
curl -f http://nano.int.khoofia.com:8700/v2/models/depth-anything-v2/ready
```

Point CTS at the Nano:

```bash
TRITON_GRPC_URL=nano.int.khoofia.com:8701
TRITON_DETECTOR_STATIC_BATCH_SIZE=8
```

## Notes

- The compose file uses `nvcr.io/nvidia/tritonserver:24.08-py3-igpu` by
  default because that line matches JetPack 6.2's CUDA 12.6 and TensorRT 10.3
  stack. Override with `TRITON_JETSON_TAG` only after confirming the Jetson
  JetPack/Triton compatibility matrix.
- Keep the Jetson in MAXN SUPER or an equivalent custom power mode when
  benchmarking.
- Generated TensorRT plans are ignored by git.
