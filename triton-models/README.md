# triton-models

Triton Inference Server model repository shared by the Continuous Tracking
System (CTS) and scene-analysis-service (SAS).

All models use **QDQ-quantized ONNX** format (QuantizeLinear → Op → DequantizeLinear).
This uses standard FP32 ops internally, making it portable across NVIDIA
(CUDAExecutionProvider) and Intel Arc (OpenVINOExecutionProvider) GPUs without
requiring specialized INT8 CUDA kernels.

## Directory layout

```text
triton-models/
├── person-detector/          YOLO26L person detector (ONNX Runtime)
│   ├── config.pbtxt          Triton config (NVIDIA: CUDA EP)
│   ├── config.pbtxt.intel    Triton config (Intel Arc: OpenVINO EP)
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 95 MB
│       └── model_qdq.onnx    QDQ ONNX — ~24 MB  ← active
├── clip-vision/              CLIP ViT-L/14 vision encoder (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 1,160 MB
│       └── model_qdq.onnx    QDQ ONNX — ~293 MB  ← active
├── florence-2/               Florence-2-large scene description (Python BLS orchestrator)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.py                          Python BLS orchestrator
│       ├── vision_encoder_int8.onnx          ~349 MB  (linked by florence-2-vision)
│       ├── encoder_model_int8.onnx           ~146 MB  (linked by florence-2-encoder)
│       ├── decoder_model_merged_int8.onnx    ~246 MB  (linked by florence-2-decoder)
│       ├── embed_tokens_int8.onnx            ~50 MB   (linked by florence-2-embed)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── generation_config.json
├── florence-2-vision/         Florence-2 vision encoder (ONNX Runtime)
├── florence-2-embed/          Florence-2 token embedding (ONNX Runtime)
├── florence-2-encoder/        Florence-2 text encoder (ONNX Runtime)
├── florence-2-decoder/        Florence-2 autoregressive decoder (ONNX Runtime)
├── reid-solider/             Swin-Tiny body re-identification (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 107 MB
│       └── model_qdq.onnx    QDQ ONNX — ~29 MB  ← active
├── pose-rtmpose/             RTMPose-m 2D pose estimation (ONNX Runtime)
│   ├── config.pbtxt
│   ├── config.pbtxt.intel
│   └── 1/
│       ├── model.onnx        FP32 ONNX — 52 MB
│       └── model_qdq.onnx    QDQ ONNX — ~13 MB  ← active
├── embeddinggemma-300m/      Gemma 3 300M sentence embeddings (ONNX Runtime)
│   ├── config.pbtxt
│   └── 1/
│       ├── model.onnx             ONNX graph (FP32)
│       ├── model.onnx_data        external weights (FP32, ~1.23 GB)
│       └── tokenizer.json         tokenizer (20 MB)
└── scripts/
    ├── configure_gpu.py      Activate NVIDIA or Intel Arc configs
    ├── export_yolo.py        YOLO26L .pt → ONNX
    ├── export_clip.py        CLIP ViT-L/14 vision encoder → ONNX
    ├── export_reid.py        SOLIDER-REID → ONNX (requires SOLIDER-REID repo)
    ├── export_pose.py        RTMPose-m → ONNX (requires mmpose)
    ├── download_florence.py  Download Florence-2 QDQ ONNX from onnx-community
    ├── download_models.py    One-shot download of all onnx-community models
    └── quantize_int8.py      Dynamic QDQ quantization for any ONNX model
```

Model binary files (`.onnx`) are excluded from git — they are several hundred
MB each. Generate them with the export/download/quantize scripts.

## Model inventory

| Model | Triton name | Format | FP32 | QDQ | Input | Output |
|-------|-------------|--------|------|-----|-------|--------|
| YOLO26L | `person-detector` | ONNX Runtime | 95 MB | **~24 MB** | `images` [N,3,640,640] | `output0` [N,300,6] NMS-free |
| CLIP ViT-L/14 | `clip-vision` | ONNX Runtime | 1,160 MB | **~293 MB** | `input` [N,3,224,224] | `output` [N,768] |
| Florence-2-large | `florence-2` | Python (BLS) | — | **~794 MB** | `pixel_values` [1,3,H,W] + `input_ids` [1,seq] | `output_ids` [1,max_len] |
| Swin-Tiny ReID | `reid-solider` | ONNX Runtime | 107 MB | **~29 MB** | `input` [N,3,256,128] | `output` [N,768] |
| RTMPose-m | `pose-rtmpose` | ONNX Runtime | 52 MB | **~13 MB** | `input` [N,3,256,192] | `simcc_x` [N,17,384], `simcc_y` [N,17,512] |
| embeddinggemma-300m | `embeddinggemma-300m` | ONNX Runtime | **1,230 MB** | — | `input_ids` [N,2048], `attention_mask` [N,2048] | `sentence_embedding` [N,768] |

**Total model repo: ~1.65 GB** (QDQ active + FP32 backups + embeddinggemma FP32).

## GPU vendor support

| Model | NVIDIA | Intel Arc |
|-------|--------|-----------|
| person-detector | CUDAExecutionProvider | OpenVINOExecutionProvider |
| clip-vision | CUDAExecutionProvider | OpenVINOExecutionProvider |
| florence-2 | CUDAExecutionProvider | OpenVINOExecutionProvider |
| reid-solider | CUDAExecutionProvider | OpenVINOExecutionProvider |
| pose-rtmpose | CUDAExecutionProvider | OpenVINOExecutionProvider |
| embeddinggemma-300m | CUDAExecutionProvider | OpenVINOExecutionProvider |

**NVIDIA**: Triton's ONNX Runtime selects `CUDAExecutionProvider`. QDQ format
uses standard FP32 ops so no specialized INT8 CUDA kernels are needed.

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

All locally-exported models use QDQ quantization for production. Each model has
an FP32 `.onnx` file and a QDQ `model_qdq.onnx` file. Triton loads the QDQ
version via `default_model_filename: "model_qdq.onnx"` in each config.

#### person-detector (YOLO26L)

```bash
# Export FP32 ONNX from Ultralytics weights
uv run --with ultralytics --with torch --with onnx \
    python triton-models/scripts/export_yolo.py --weights yolo26l.pt

# Quantize to QDQ
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/person-detector/1/model.onnx \
    --output triton-models/person-detector/1/model_qdq.onnx

# Verify output shape: [batch, 300, 6]
python -c "import onnx; m=onnx.load('triton-models/person-detector/1/model_qdq.onnx'); \
  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"
# Expected: [8, 300, 6]
```

YOLO26L is NMS-Free: NMS is baked into the ONNX graph. No post-processing
NMS needed at inference time.

#### clip-vision (CLIP ViT-L/14)

```bash
# Export FP32 ONNX from OpenCLIP
uv run --with open_clip_torch --with torch --with onnx \
    python triton-models/scripts/export_clip.py

# Quantize to QDQ
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/clip-vision/1/model.onnx \
    --output triton-models/clip-vision/1/model_qdq.onnx

# Verify: [batch, 768]
python -c "import onnx; m=onnx.load('triton-models/clip-vision/1/model_qdq.onnx'); \
  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"
# Expected: [0, 768]
```

Exports the vision encoder (`model.visual`) from OpenCLIP ViT-L-14.
Client-side: L2-normalize the 768-dim output.

#### florence-2 (Florence-2-large)

```bash
# Download INT8 QDQ ONNX files + tokenizer from onnx-community
uv run --with huggingface_hub \
    python triton-models/scripts/download_florence.py

# Ensure ONNX files are in triton-models/florence-2/1/ and symlinks exist:
#   florence-2-vision/1/model.onnx  → ../../florence-2/1/vision_encoder_int8.onnx
#   florence-2-embed/1/model.onnx   → ../../florence-2/1/embed_tokens_int8.onnx
#   florence-2-encoder/1/model.onnx → ../../florence-2/1/encoder_model_int8.onnx
#   florence-2-decoder/1/model.onnx → ../../florence-2/1/decoder_model_merged_int8.onnx
```

Pre-quantized INT8 QDQ from `onnx-community/Florence-2-large`. No separate
quantization step needed. Uses Business Logic Scripting (BLS): the Python
`model.py` orchestrates four native ONNX Runtime sub-models (florence-2-vision,
florence-2-embed, florence-2-encoder, florence-2-decoder) for the autoregressive
generation loop. All heavy computation runs on GPU via Triton's ONNX Runtime
backend (CUDA EP for NVIDIA, OpenVINO EP for Intel Arc).

Sub-model directories with `config.pbtxt` files are required alongside the
main `florence-2` model directory — Triton registers them as independent models.
The NVIDIA/Intel config swap (`configure_gpu.py`) applies to all five
Florence-2 directories.

#### reid-solider (Swin-Tiny ReID)

```bash
# Export FP32 ONNX via timm
uv run --with torch --with onnx --with timm \
    python triton-models/scripts/export_reid.py

# Quantize to QDQ
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/reid-solider/1/model.onnx \
    --output triton-models/reid-solider/1/model_qdq.onnx

# Verify: [batch, 768]
python -c "import onnx; m=onnx.load('triton-models/reid-solider/1/model_qdq.onnx'); \
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

# Quantize to QDQ
uv run --with onnxruntime --with onnx --with sympy \
    python triton-models/scripts/quantize_int8.py \
    --input triton-models/pose-rtmpose/1/model.onnx \
    --output triton-models/pose-rtmpose/1/model_qdq.onnx

# Verify outputs: simcc_x, simcc_y
python -c "import onnx; m=onnx.load('triton-models/pose-rtmpose/1/model_qdq.onnx'); \
  print([o.name for o in m.graph.output])"
# Expected: ['simcc_x', 'simcc_y']
```

SimCC head with split_ratio=2.0: argmax over 384 x-bins (192×2) and
512 y-bins (256×2) gives pixel coordinates.

#### embeddinggemma-300m (Gemma 3 300M sentence embeddings)

```bash
# Download FP32 ONNX model + tokenizer from onnx-community
uv run --with huggingface_hub \
    python triton-models/scripts/download_embeddinggemma.py
```

FP32 model from `onnx-community/embeddinggemma-300m-ONNX`. Uses FP32 for
maximum portability across ONNX Runtime versions and GPU vendors. No export
or quantization step needed — the model is downloaded directly. The ONNX model
uses external data (`.onnx` graph + `.onnx_data` weights); both files must be
present in the model version directory.

Client-side: tokenization + L2 normalization via
`triton-shared/triton_shared/inference/text_embedding.py` and
`triton-shared/triton_shared/models/embedder.py`.

Used by the **Knowledge Repository** RAG pipeline in cognitive-companion for
document chunk embedding and senior query vector search.

### Step 3 — start Triton and verify

```bash
# Build and start the custom Triton image
docker compose up triton --build
```

Wait for all models to report READY:

```bash
curl -s http://localhost:8700/v2/models/person-detector/ready
curl -s http://localhost:8700/v2/models/clip-vision/ready
curl -s http://localhost:8700/v2/models/florence-2/ready
curl -s http://localhost:8700/v2/models/florence-2-vision/ready
curl -s http://localhost:8700/v2/models/florence-2-embed/ready
curl -s http://localhost:8700/v2/models/florence-2-encoder/ready
curl -s http://localhost:8700/v2/models/florence-2-decoder/ready
curl -s http://localhost:8700/v2/models/reid-solider/ready
curl -s http://localhost:8700/v2/models/pose-rtmpose/ready
curl -s http://localhost:8700/v2/models/embeddinggemma-300m/ready
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
| embeddinggemma-300m | 16 | ≤ 100 ms | ~80 ms | ~120 ms |

Florence-2 latency depends on generated token count (typically 50–200 tokens).

## Intel Arc: Triton container image

The standard `nvcr.io/nvidia/tritonserver` image does not include the OpenVINO
backend or Intel Compute Runtime drivers. Use a custom image:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:26.04-py3
RUN apt-get update && apt-get install -y intel-opencl-icd intel-level-zero-gpu
RUN pip install openvino>=2024.2 onnxruntime-gpu
```

Refer to the [Triton OpenVINO backend docs](https://github.com/triton-inference-server/openvino_backend)
for the current recommended base image and driver versions.

## Shared with cognitive-companion and scene-analysis-service

The `person-detector`, `clip-vision`, and `florence-2` models are shared with
`scene-analysis-service` (SAS). The `embeddinggemma-300m` model is used by
`cognitive-companion` for the Knowledge Repository RAG pipeline (document
chunk embedding and senior query vector search).

All services use the same Triton instance and the shared `triton-shared/`
client library for inference. GPU vendor differences are handled entirely by
Triton configs.
