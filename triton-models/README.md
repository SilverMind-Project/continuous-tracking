# triton-models

Triton Inference Server model repository for the Continuous Tracking System.

## Directory layout

```text
triton-models/
├── person-detector/          YOLO26L person detector (NMS-Free)
│   ├── config.pbtxt          Triton model config (TensorRT)
│   └── 1/
│       └── model.plan        TensorRT engine — NOT in git (generate with scripts/)
├── reid-solider/             SOLIDER-REID 768-dim body embedder
│   ├── config.pbtxt          Triton model config (ONNX Runtime)
│   └── 1/
│       └── model.onnx        ONNX model — NOT in git (generate with scripts/)
├── pose-rtmpose/             RTMPose-m 2D pose estimator
│   ├── config.pbtxt          Triton model config (ONNX Runtime)
│   └── 1/
│       └── model.onnx        ONNX model — NOT in git (generate with scripts/)
└── scripts/
    ├── export_yolo.py        YOLO26L → TensorRT .plan
    ├── export_reid.py        SOLIDER-REID → ONNX
    └── export_pose.py        RTMPose-m → ONNX
```

Model binary files (`.plan`, `.onnx`) are excluded from git — they are GPU-specific
and several hundred MB each. Generate them on the target machine with the scripts.

## Materialising model files

All three scripts must be run on a machine with an NVIDIA GPU.

### 1. person-detector (YOLO26L → TensorRT)

```bash
pip install ultralytics>=8.4.0

# Download or fine-tune weights first:
#   yolo26l.pt  — Ultralytics pretrained (released Jan 2026)
#   yolo26l_cts.pt — fine-tuned on overhead indoor footage (preferred)

python triton-models/scripts/export_yolo.py \
    --weights yolo26l_cts.pt \
    --out triton-models/person-detector/1/model.plan \
    --batch 16 --imgsz 640 --device 0
```

YOLO26L uses a NMS-Free (end-to-end) architecture: the export bakes NMS into
the TensorRT engine. Output tensor `output0` is `[batch, 300, 6]` where the 6
columns are `x1, y1, x2, y2` (letterbox pixel space), `confidence`, `class_id`.

Re-run this script if you move to a different GPU model.

### 2. reid-solider (SOLIDER-REID → ONNX)

```bash
git clone https://github.com/tinyvision/SOLIDER-REID
cd SOLIDER-REID
pip install -r requirements.txt

# Download weights: https://github.com/tinyvision/SOLIDER-REID#model-zoo
# Example: solider_swin_tiny_msmt17.pth

python ../triton-models/scripts/export_reid.py \
    --config configs/MSMT17/swin_tiny.yml \
    --weights /path/to/solider_swin_tiny_msmt17.pth \
    --out ../triton-models/reid-solider/1/model.onnx
```

### 3. pose-rtmpose (RTMPose-m → ONNX)

```bash
pip install mmpose mmdeploy onnx onnxruntime-gpu

# Download config + weights from MMPose model zoo:
#   https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html
# Model: RTMPose-m, 256×192, COCO pretrained

python triton-models/scripts/export_pose.py \
    --config rtmpose-m_8xb256-420e_coco-256x192.py \
    --weights rtmpose-m_simcc-aic-coco_420e-256x192.pth \
    --out triton-models/pose-rtmpose/1/model.onnx
```

## Verifying Triton loads all models

```bash
# Start Triton with the model repository mounted:
docker compose up triton

# Wait for model readiness (all three must return "READY"):
curl -s http://localhost:8000/v2/models/person-detector/ready
curl -s http://localhost:8000/v2/models/reid-solider/ready
curl -s http://localhost:8000/v2/models/pose-rtmpose/ready
```

## Performance target

| Model           | Batch | p99 latency | Hardware            |
|-----------------|-------|-------------|---------------------|
| person-detector | 8     | ≤ 12 ms     | RTX 4060 or equiv.  |
| reid-solider    | 8     | ≤ 8 ms      | RTX 4060 or equiv.  |
| pose-rtmpose    | 8     | ≤ 8 ms      | RTX 4060 or equiv.  |

Run `python tracking-orchestrator/scripts/benchmark_triton.py` after loading models.

## Intel Arc (OpenVINO) path

Replace `platform: "tensorrt_plan"` / `"onnxruntime_onnx"` in each `config.pbtxt`
with `"openvino"`, and convert models via OpenVINO Model Optimizer:

```bash
mo --input_model reid-solider/1/model.onnx \
   --output_dir reid-solider/1/ \
   --input_shape "[1,3,256,128]"
# Produces reid-solider/1/model.xml + reid-solider/1/model.bin
```
