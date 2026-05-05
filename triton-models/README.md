# triton-models

Triton Inference Server model repository for the Continuous Tracking System.

All three models use ONNX format, which runs on both NVIDIA and Intel Arc GPUs
via Triton's ONNX Runtime backend.

## Directory layout

```text
triton-models/
├── person-detector/          YOLO26L person detector (NMS-Free)
│   ├── config.pbtxt          Triton config (ONNX Runtime, NVIDIA default)
│   ├── config.pbtxt.intel    Triton config (ONNX Runtime + OpenVINO EP, Intel Arc)
│   └── 1/
│       └── model.onnx        ONNX model — NOT in git (export_yolo.py)
├── reid-solider/             SOLIDER-REID 768-dim body embedder
│   ├── config.pbtxt          Triton config (ONNX Runtime, NVIDIA default)
│   ├── config.pbtxt.intel    Triton config (ONNX Runtime + OpenVINO EP, Intel Arc)
│   └── 1/
│       └── model.onnx        ONNX model — NOT in git (export_reid.py)
├── pose-rtmpose/             RTMPose-m 2D pose estimator
│   ├── config.pbtxt          Triton config (ONNX Runtime, NVIDIA default)
│   ├── config.pbtxt.intel    Triton config (ONNX Runtime + OpenVINO EP, Intel Arc)
│   └── 1/
│       └── model.onnx        ONNX model — NOT in git (export_pose.py)
└── scripts/
    ├── configure_gpu.py      Activate NVIDIA or Intel Arc configs (run once)
    ├── export_yolo.py        YOLO26L → ONNX
    ├── export_reid.py        SOLIDER-REID → ONNX
    └── export_pose.py        RTMPose-m → ONNX
```

Model binary files (`.onnx`) are excluded from git — they are several hundred
MB each. Generate them on the target machine with the export scripts.

## GPU vendor support

| Model           | Format | NVIDIA                  | Intel Arc          |
| --------------- | ------ | ----------------------- | ------------------ |
| person-detector | ONNX   | TensorRT EP (auto)      | OpenVINO EP        |
| reid-solider    | ONNX   | CUDA EP (auto)          | OpenVINO EP        |
| pose-rtmpose    | ONNX   | CUDA EP (auto)          | OpenVINO EP        |

**NVIDIA**: Triton's ONNX Runtime auto-selects `TensorrtExecutionProvider`
then `CUDAExecutionProvider`. On first load it builds and caches a TRT engine
from the ONNX graph; subsequent loads use the cache.

**Intel Arc**: requires a Triton image with the OpenVINO backend and Intel
Compute Runtime drivers. Run `configure_gpu.py --vendor intel` to activate.

## Setup

### Step 1 — select GPU vendor config

Run once on each target machine before starting Triton:

```bash
# NVIDIA (default — skip if config.pbtxt files are already correct)
python triton-models/scripts/configure_gpu.py --vendor nvidia

# Intel Arc
python triton-models/scripts/configure_gpu.py --vendor intel
```

This is idempotent: re-running with the same vendor is safe.

### Step 2 — export model weights

#### person-detector (YOLO26L → ONNX)

```bash
pip install ultralytics>=8.4.0

# Download or fine-tune weights:
#   yolo26l.pt       — Ultralytics pretrained (released Jan 2026)
#   yolo26l_cts.pt   — Fine-tuned on overhead indoor footage (preferred)

# NVIDIA
python triton-models/scripts/export_yolo.py \
    --weights yolo26l_cts.pt \
    --out triton-models/person-detector/1/model.onnx

# Intel Arc (requires Intel Extension for PyTorch: pip install intel-extension-for-pytorch)
python triton-models/scripts/export_yolo.py \
    --weights yolo26l_cts.pt \
    --device xpu \
    --out triton-models/person-detector/1/model.onnx
```

YOLO26L is NMS-Free: NMS is baked into the model graph and preserved in the
ONNX export. **Verify the output shape before deploying:**

```bash
python -c "
import onnx
m = onnx.load('triton-models/person-detector/1/model.onnx')
print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])
"
# Expected: [0, 300, 6]  (0 = dynamic batch)
# If you see [0, 84, 8400] the NMS head was not preserved — upgrade
# ultralytics and retry.
```

#### reid-solider (SOLIDER-REID → ONNX)

```bash
git clone https://github.com/tinyvision/SOLIDER-REID
cd SOLIDER-REID
pip install -r requirements.txt

# Download weights: https://github.com/tinyvision/SOLIDER-REID#model-zoo

# Same ONNX file works on both NVIDIA and Intel Arc
python ../triton-models/scripts/export_reid.py \
    --config configs/MSMT17/swin_tiny.yml \
    --weights /path/to/solider_swin_tiny_msmt17.pth \
    --out ../triton-models/reid-solider/1/model.onnx
```

#### pose-rtmpose (RTMPose-m → ONNX)

```bash
pip install mmpose torch>=2.0 onnx

# Download config + weights from MMPose model zoo:
#   https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html
# Model: RTMPose-m, 256x192, COCO pretrained

# NVIDIA
python triton-models/scripts/export_pose.py \
    --config rtmpose-m_8xb256-420e_coco-256x192.py \
    --weights rtmpose-m_simcc-aic-coco_420e-256x192.pth \
    --out triton-models/pose-rtmpose/1/model.onnx \
    --device cuda:0

# Intel Arc (requires Intel Extension for PyTorch: pip install intel-extension-for-pytorch)
python triton-models/scripts/export_pose.py \
    --config rtmpose-m_8xb256-420e_coco-256x192.py \
    --weights rtmpose-m_simcc-aic-coco_420e-256x192.pth \
    --out triton-models/pose-rtmpose/1/model.onnx \
    --device xpu
```

### Step 3 — start Triton and verify

```bash
# NVIDIA
docker compose up triton

# Intel Arc — use a Triton image with the OpenVINO backend
# See "Intel Arc container" section below
```

Wait for all three models to report READY:

```bash
curl -s http://localhost:8000/v2/models/person-detector/ready
curl -s http://localhost:8000/v2/models/reid-solider/ready
curl -s http://localhost:8000/v2/models/pose-rtmpose/ready
```

Run the benchmark to verify latency targets:

```bash
python tracking-orchestrator/scripts/benchmark_triton.py
```

## Performance targets

| Model           | Batch | p99 latency | NVIDIA (RTX 4060) | Intel Arc (A770) |
| --------------- | ----- | ----------- | ----------------- | ---------------- |
| person-detector | 8     | ≤ 12 ms     | ~10 ms            | ~14 ms           |
| reid-solider    | 8     | ≤ 8 ms      | ~6 ms             | ~9 ms            |
| pose-rtmpose    | 8     | ≤ 8 ms      | ~6 ms             | ~9 ms            |

NVIDIA numbers use the cached TRT engine (after first-load compilation).
Intel Arc numbers use OpenVINO EP on Arc A770 16 GB.

## Intel Arc: Triton container image

The standard `nvcr.io/nvidia/tritonserver` image does not include the OpenVINO
backend or Intel Compute Runtime drivers. Use a custom image:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:24.12-py3
RUN apt-get update && apt-get install -y intel-opencl-icd intel-level-zero-gpu
RUN pip install openvino>=2024.2
```

Refer to the [Triton OpenVINO backend docs](https://github.com/triton-inference-server/openvino_backend)
for the current recommended base image and driver versions.
