# triton-models

Triton Inference Server model repository shared by the Continuous Tracking
System (CTS) and scene-analysis-service (SAS).

All models use **INT8-quantized ONNX** format, which runs on both NVIDIA and
Intel Arc GPUs via Triton's ONNX Runtime backend.

## Directory layout

```text
triton-models/
├── person-detector/          YOLO26L person detector (ONNX Runtime)
│   ├── config.pbtxt          Triton config (NVIDIA: TensorRT/CUDA EP)
│   ├── config.pbtxt.intel    Triton config (Intel Arc: OpenVINO EP)
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 95 MB
│       └── model_int8.onnx   INT8 ONNX — 24 MB  ← active
├── clip-vision/              CLIP ViT-L/14 vision encoder (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 1,160 MB
│       └── model_int8.onnx   INT8 ONNX — 293 MB  ← active
├── florence-2/               Florence-2-large scene description (Python backend)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.py              Python backend script
│       ├── vision_encoder_int8.onnx   350 MB
│       ├── encoder_model_int8.onnx    146 MB
│       ├── decoder_model_merged_int8.onnx  246 MB
│       ├── embed_tokens_int8.onnx      51 MB
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── reid-solider/             Swin-Tiny body re-identification (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 107 MB
│       └── model_int8.onnx   INT8 ONNX — 29 MB  ← active
├── pose-rtmpose/             RTMPose-m 2D pose estimation (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 52 MB
│       └── model_int8.onnx   INT8 ONNX — 13 MB  ← active
└── scripts/
    ├── configure_gpu.py      Activate NVIDIA or Intel Arc configs
    ├── export_yolo.py        YOLO26L .pt → ONNX
    ├── export_clip.py        CLIP ViT-L/14 vision encoder → ONNX
    ├── export_reid.py        SOLIDER-REID → ONNX (requires SOLIDER-REID repo)
    ├── export_pose.py        RTMPose-m → ONNX (requires mmpose)
    ├── download_florence.py  Download Florence-2 INT8 ONNX from onnx-community
    ├── download_models.py    One-shot download of all onnx-community models
    └── quantize_int8.py      Dynamic INT8 quantization for any ONNX model
```

Model binary files (`.onnx`) are excluded from git — they are several hundred
MB each. Generate them with the export/download/quantize scripts.

## Model inventory

| Model | Triton name | Format | FP32 | INT8 | Input | Output |
|-------|-------------|--------|------|------|-------|--------|
| YOLO26L | `person-detector` | ONNX Runtime | 95 MB | **24 MB** | `images` [N,3,640,640] | `output0` [N,300,6] NMS-free |
| CLIP ViT-L/14 | `clip-vision` | ONNX Runtime | 1,160 MB | **293 MB** | `input` [N,3,224,224] | `output` [N,768] |
| Florence-2-large | `florence-2` | Python (ORT) | — | **794 MB** | `pixel_values` [1,3,H,W] + `input_ids` [1,seq] | `output_ids` [1,max_len] |
| Swin-Tiny ReID | `reid-solider` | ONNX Runtime | 107 MB | **29 MB** | `input` [N,3,256,128] | `output` [N,768] |
| RTMPose-m | `pose-rtmpose` | ONNX Runtime | 52 MB | **13 MB** | `input` [N,3,256,192] | `simcc_x` [N,17,384], `simcc_y` [N,17,512] |

**Total model repo: ~1.15 GB** (INT8 active + FP32 backups).

## GPU vendor support

| Model | NVIDIA | Intel Arc |
|-------|--------|-----------|
| person-detector | TensorRT EP (auto) | OpenVINO EP |
| clip-vision | TensorRT EP (auto) | OpenVINO EP |
| florence-2 | CUDAExecutionProvider | OpenVINOExecutionProvider |
| reid-solider | TensorRT EP (auto) | OpenVINO EP |
| pose-rtmpose | TensorRT EP (auto) | OpenVINO EP |

**NVIDIA**: Triton's ONNX Runtime auto-selects `TensorrtExecutionProvider`
then `CUDAExecutionProvider`. On first load it builds and caches a TRT engine.

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

### Step 2 — generate model files

All models use INT8 quantization for production. Each model has an FP32
`.onnx` file and an INT8 `.model_int8.onnx` file. Triton loads the INT8
version via `default_model_filename: "model_int8.onnx"` in each config.

#### person-detector (YOLO26L)

```bash
# Export FP32 ONNX from Ultralytics weights
uv run --with ultralytics --with torch --with onnx \
    python triton-models/scripts/export_yolo.py --weights yolo26l.pt

# Quantize to INT8
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/person-detector/1/model.onnx \
    --output triton-models/person-detector/1/model_int8.onnx

# Verify output shape: [batch, 300, 6]
python -c "import onnx; m=onnx.load('triton-models/person-detector/1/model_int8.onnx'); \
  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"
# Expected: [16, 300, 6]
```

YOLO26L is NMS-Free: NMS is baked into the ONNX graph. No post-processing
NMS needed at inference time.

#### clip-vision (CLIP ViT-L/14)

```bash
# Export FP32 ONNX from OpenCLIP
uv run --with open_clip_torch --with torch --with onnx \
    python triton-models/scripts/export_clip.py

# Quantize to INT8
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/clip-vision/1/model.onnx \
    --output triton-models/clip-vision/1/model_int8.onnx

# Verify: [batch, 768]
python -c "import onnx; m=onnx.load('triton-models/clip-vision/1/model_int8.onnx'); \
  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"
# Expected: [0, 768]
```

Exports the vision encoder (`model.visual`) from OpenCLIP ViT-L-14.
Client-side: L2-normalize the 768-dim output.

#### florence-2 (Florence-2-large)

```bash
# Download INT8 ONNX files + tokenizer from onnx-community
uv run --with huggingface_hub \
    python triton-models/scripts/download_florence.py

# Ensure ONNX files are in triton-models/florence-2/1/ (not a subdirectory)
```

Pre-quantized INT8 from `onnx-community/Florence-2-large`. No separate
quantization step needed. Uses Triton's Python backend — the `model.py`
script orchestrates the autoregressive generation loop.

#### reid-solider (Swin-Tiny ReID)

```bash
# Export FP32 ONNX via timm
uv run --with torch --with onnx --with timm \
    python triton-models/scripts/export_reid.py

# Quantize to INT8
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/reid-solider/1/model.onnx \
    --output triton-models/reid-solider/1/model_int8.onnx

# Verify: [batch, 768]
python -c "import onnx; m=onnx.load('triton-models/reid-solider/1/model_int8.onnx'); \
  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"
# Expected: [0, 768]
```

Note: This uses a Swin-Tiny backbone from timm without the SOLIDER semantic
controller or MSMT17 fine-tuning. For production ReID accuracy, train on the
full SOLIDER-REID pipeline with proper weights.

#### pose-rtmpose (RTMPose-m)

```bash
# Export FP32 ONNX from official MMPose checkpoint
# Requires Python 3.11 venv with mmpose stack — see script header for setup.
python triton-models/scripts/export_pose.py

# Quantize to INT8
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/pose-rtmpose/1/model.onnx \
    --output triton-models/pose-rtmpose/1/model_int8.onnx

# Verify outputs: simcc_x, simcc_y
python -c "import onnx; m=onnx.load('triton-models/pose-rtmpose/1/model_int8.onnx'); \
  print([o.name for o in m.graph.output])"
# Expected: ['simcc_x', 'simcc_y']
```

SimCC head with split_ratio=2.0: argmax over 384 x-bins (192×2) and
512 y-bins (256×2) gives pixel coordinates.

### Step 3 — start Triton and verify

```bash
# NVIDIA
docker compose up triton

# Intel Arc — use a Triton image with the OpenVINO backend
# See "Intel Arc container" section below
```

Wait for all five models to report READY:

```bash
curl -s http://localhost:8000/v2/models/person-detector/ready
curl -s http://localhost:8000/v2/models/clip-vision/ready
curl -s http://localhost:8000/v2/models/florence-2/ready
curl -s http://localhost:8000/v2/models/reid-solider/ready
curl -s http://localhost:8000/v2/models/pose-rtmpose/ready
```

Run the benchmark to verify latency targets:

```bash
python tracking-orchestrator/scripts/benchmark_triton.py
```

## Performance targets

| Model | Batch | p99 latency | NVIDIA (RTX 4060) | Intel Arc (A770) |
|-------|-------|-------------|-------------------|-------------------|
| person-detector | 8 | ≤ 12 ms | ~10 ms | ~14 ms |
| clip-vision | 8 | ≤ 10 ms | ~8 ms | ~12 ms |
| reid-solider | 8 | ≤ 8 ms | ~6 ms | ~9 ms |
| pose-rtmpose | 8 | ≤ 8 ms | ~6 ms | ~9 ms |
| florence-2 | 1 | ≤ 2 s | ~1.5 s | ~2 s |

NVIDIA numbers use the cached TRT engine (after first-load compilation).
Intel Arc numbers use OpenVINO EP on Arc A770 16 GB.
Florence-2 latency depends on generated token count (typically 50–200 tokens).

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

## Shared with scene-analysis-service

The `person-detector`, `clip-vision`, and `florence-2` models are shared with
`scene-analysis-service` (SAS). SAS uses the same Triton instance and the
shared `triton-shared/` client library for inference.

Both services use identical gRPC client code — GPU vendor differences are
handled entirely by Triton configs.
