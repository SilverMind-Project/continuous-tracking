# CTS INT8 Models for Jetson Orin Nano Super

This repository serves the eight inference graphs selected for the Jetson CTS
deployment:

- `person-detector`: YOLO26L, fixed internal batch 1; QAT INT8 backbone
- `pose-rtmpose`: RTMPose-m, dynamic batch 1-8; INT8 stem
- `reid-solider`: SOLIDER body ReID, dynamic batch 1-8; INT8 Conv/MatMul
- `face-detector-scrfd`: `buffalo_l/det_10g.onnx`, batch 1
- `face-recognition-arcface`: `buffalo_l/w600k_r50.onnx`, batch 1; INT8
  early backbone
- `face-landmark-2d106`: `buffalo_l/2d106det.onnx`, batch 1; selective INT8
- `face-landmark-3d68`: `buffalo_l/1k3d68.onnx`, batch 1; selective INT8
- `face-attribute-genderage`: `buffalo_l/genderage.onnx`, batch 1; selective INT8

Depth Anything is intentionally not loaded. All five Buffalo_L graphs are
present so the person-identification service uses the same inference contract
against the DGX and Jetson endpoints.

## INT8 Contract

The deployment uses explicit-Q/DQ ONNX models produced by NVIDIA ModelOpt.
Calibration uses representative household camera frames, not random tensors.
TensorRT plans are built on the Jetson and the build emits per-layer reports.

These are mixed-precision INT8 engines. Accuracy-sensitive residual, neck,
late-backbone, and prediction-head operations remain FP32 where quantization
did not meet the semantic regression gates. The acceptance check requires each
engine to contain INT8 tensor formats; it does not claim that every graph
operation is INT8.

The generated Q/DQ graphs retain FP32 fallback operations. This also keeps
quantization scale tensors standards-compliant; ModelOpt 0.43's FP16 autocast
can otherwise cast QLinear scales to FP16 and create graphs rejected by ONNX
Runtime.

Generated camera data, Q/DQ ONNX files, TensorRT plans, and reports are ignored
by git.

## 1. Export Calibration Images

Run where the deployed orchestrator and MinIO are reachable:

```bash
cd continuous-tracking
set -a
source .env
set +a
export RESIDENT_ID=resident-id

person-identification-service/.venv/bin/python \
  triton-models-jetson/scripts/export_keyframe_calibration_data.py \
  --orchestrator-url http://127.0.0.1:8500 \
  --person-id "$RESIDENT_ID" \
  --output-dir calibration-data/jetson \
  --max-samples 128 \
  --max-per-camera 40 \
  --min-separation-seconds 20
```

The exporter balances cameras and time, prefers caregiver-overridden boxes,
then stored keyframe boxes. The output contains identifiable household images;
keep it private.

## 2. Prepare Exact Model Inputs

```bash
PID_ROOT=person-identification-service
MPLCONFIGDIR=/tmp/matplotlib "$PID_ROOT/.venv/bin/python" \
  triton-models-jetson/scripts/prepare_calibration_tensors.py \
  --data-dir calibration-data/jetson \
  --buffalo-dir "$PID_ROOT/data/models/buffalo_l"
```

This creates full-frame YOLO tensors, person-crop pose/ReID tensors, SCRFD
inputs, ArcFace-aligned 112x112 faces, 192x192 landmark crops, and 96x96
gender/age crops.

## 3. Export and QAT YOLO26L

The supplied checkpoint has Ultralytics `8.3.222` metadata, while the deployed
ONNX was exported with `8.4.57`. Despite that metadata difference, all 332
learned ONNX initializers are byte-identical and a runtime comparison produced
zero output difference. Export it separately rather than overwriting the DGX
model:

```bash
uv run --with 'ultralytics==8.4.57' \
  python triton-models/scripts/export_yolo.py \
  --weights calibration-data/jetson/checkpoints/yolo26l.pt \
  --out calibration-data/jetson/candidates/person-detector/model.onnx \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

PTQ alone does not pass the detector gate. Run selective backbone QAT:

```bash
uv run \
  --with 'ultralytics==8.4.57' \
  --with 'nvidia-modelopt' \
  --with 'huggingface-hub==1.16.1' \
  --with 'onnx<2' \
  python triton-models-jetson/scripts/qat_person_detector.py \
  --weights calibration-data/jetson/checkpoints/yolo26l.pt \
  --calibration-tensor calibration-data/jetson/tensors/person-detector.npy \
  --output triton-models-jetson/person-detector/1/model_int8.onnx \
  --checkpoint-output calibration-data/jetson/checkpoints/yolo26l-qat-modelopt.pt
```

The QAT defaults reproduce the accepted run: 112 training samples, 12 epochs,
learning rate `3e-6`, and seed `23`.

The QAT run uses training batches of eight, but the 8 GB Jetson deployment
must not use the resulting fixed batch-8 ONNX graph. Restore the accepted
ModelOpt checkpoint and re-export it at fixed batch 1:

```bash
uv run \
  --with 'ultralytics==8.4.57' \
  --with 'nvidia-modelopt' \
  --with 'huggingface-hub==1.16.1' \
  --with 'onnx<2' \
  python triton-models-jetson/scripts/export_qat_person_detector.py \
  --weights calibration-data/jetson/checkpoints/yolo26l.pt \
  --checkpoint calibration-data/jetson/checkpoints/yolo26l-qat-modelopt.pt \
  --output triton-models-jetson/person-detector/1/model_int8.onnx \
  --batch-size 1 \
  --device cpu
```

Export a matching FP32 batch-1 graph for validation:

```bash
uv run --with 'ultralytics==8.4.57' \
  python triton-models/scripts/export_yolo.py \
  --weights calibration-data/jetson/checkpoints/yolo26l.pt \
  --out calibration-data/jetson/candidates/person-detector/model-batch1.onnx \
  --batch 1 \
  --imgsz 640 \
  --device cpu
```

## 4. Quantize the Remaining Models

```bash
uv run --with 'nvidia-modelopt[onnx]' \
  python triton-models-jetson/scripts/quantize_int8_models.py \
  --buffalo-dir "$PID_ROOT/data/models/buffalo_l" \
  --calibration-dir calibration-data/jetson/tensors
```

The default PTQ set excludes `person-detector` so it cannot overwrite the QAT
graph. Each deployment output is written to:

```text
triton-models-jetson/<model>/1/model_int8.onnx
```

Validate output drift before copying models to the Jetson:

```bash
uv run --with 'nvidia-modelopt[onnx]' \
  python triton-models-jetson/scripts/validate_int8_onnx.py \
  --buffalo-dir "$PID_ROOT/data/models/buffalo_l" \
  --calibration-dir calibration-data/jetson/tensors \
  --person-detector-source \
    calibration-data/jetson/candidates/person-detector/model-batch1.onnx \
  --detector-candidate-threshold 0.69
```

ReID and ArcFace are checked by embedding cosine similarity. Pose is checked by
decoded keypoint displacement. YOLO is checked by matched detection recall,
IoU, and confidence. SCRFD is checked on active face anchors, boxes, and
keypoints. The 2D and 3D landmark graphs are checked in pixel space, and the
attribute graph is checked for gender agreement and age error.

### Current validation result

The representative set contains 128 resident keyframes balanced across four
cameras and 49 detected/aligned faces. Full-set validation found:

- PASS YOLO26L batch-1 QAT export: recall `0.954`, precision agreement `0.937`,
  median IoU `0.993`, confidence MAE `0.0395`, using INT8 threshold `0.69`
  against the FP32 `0.70` baseline
- PASS RTMPose: mean keypoint error `1.96 px`, p95 `7.00 px`
- PASS SOLIDER: minimum embedding cosine `0.9722`, median `0.9826`
- PASS SCRFD: threshold agreement `0.999981`, box MAE `0.0270`
- PASS ArcFace: minimum embedding cosine `0.9883`, median `0.9973`
- PASS 2D landmarks: mean error `0.10 px`, p95 `0.32 px`
- PASS 3D landmarks: mean XY error `0.26 px`, p95 `0.70 px`
- PASS gender/age: gender agreement `1.0000`, age MAE `0.19 years`

RTMPose only quantizes its three stem convolutions. Quantizing stage 1 raised
p95 keypoint error to `10.5 px`; broader Conv quantization was worse. Treat it
as INT8-compatible but expect limited speedup compared with the other graphs.

The Jetson orchestrator must use `pipeline.detector_confidence: 0.69`. Keep the
DGX setting at `0.70`; this is hardware/model-specific confidence calibration.

## 5. Build TensorRT Plans on the Jetson

TensorRT engines are tied to the Jetson's GPU, JetPack, CUDA, and TensorRT
versions. Copy the Q/DQ ONNX model repository to the Jetson, then use one
Triton iGPU image compatible with the installed JetPack to build and serve the
plans.

From the workstation where the Q/DQ ONNX files were produced, set the SSH
target and copy the Jetson model repository:

```bash
cd continuous-tracking

export JETSON_HOST=<jetson-hostname-or-ip>

ssh "$JETSON_HOST" 'mkdir -p ~/continuous-tracking'
rsync -av \
  --exclude 'reports/' \
  --exclude '**/__pycache__/' \
  --exclude '*/1/model.plan' \
  triton-models-jetson/ \
  "$JETSON_HOST:~/continuous-tracking/triton-models-jetson/"
scp docker-compose.jetson-triton.yml \
  "$JETSON_HOST:~/continuous-tracking/"
```

For first-time setup, use the full `rsync` command above. The build script
requires all eight `model_int8.onnx` files before it starts, while the Compose
file must be copied separately because it lives at the repository root. If the
full model repository is already present on the Jetson and only the person
detector was regenerated, copy the batch-1 detector artifact, config, build
script, and current Compose file:

```bash
export JETSON_HOST=<jetson-hostname-or-ip>

ssh "$JETSON_HOST" \
  'mkdir -p ~/continuous-tracking/triton-models-jetson/person-detector/1 \
    ~/continuous-tracking/triton-models-jetson/scripts'
scp triton-models-jetson/person-detector/1/model_int8.onnx \
  "$JETSON_HOST:~/continuous-tracking/triton-models-jetson/person-detector/1/"
scp triton-models-jetson/person-detector/config.pbtxt \
  "$JETSON_HOST:~/continuous-tracking/triton-models-jetson/person-detector/"
scp triton-models-jetson/scripts/build_tensorrt_plans.sh \
  "$JETSON_HOST:~/continuous-tracking/triton-models-jetson/scripts/"
scp docker-compose.jetson-triton.yml \
  "$JETSON_HOST:~/continuous-tracking/"
```

On the Jetson, verify the deployment files before building:

```bash
cd ~/continuous-tracking

test -f docker-compose.jetson-triton.yml
test -f triton-models-jetson/scripts/build_tensorrt_plans.sh
sha256sum triton-models-jetson/person-detector/1/model_int8.onnx
```

The current accepted batch-1 detector checksum is:

```text
d01275712b754fb677c02b67b218d50cf70860183a399475ce54143f302391b8
```

Verify that every required Q/DQ model is present:

```bash
for model in \
  person-detector pose-rtmpose reid-solider \
  face-detector-scrfd face-recognition-arcface \
  face-landmark-2d106 face-landmark-3d68 face-attribute-genderage
do
  test -f "triton-models-jetson/$model/config.pbtxt" || {
    echo "missing config: $model"
    exit 1
  }
  test -f "triton-models-jetson/$model/1/model_int8.onnx" || {
    echo "missing ONNX: $model"
    exit 1
  }
done

test "$(find triton-models-jetson -type f -name model_int8.onnx | wc -l)" -eq 8
echo "All Jetson deployment inputs are present."
```

Set the build environment on the Jetson:

```bash
export TRITON_JETSON_IMAGE=nvcr.io/nvidia/tritonserver:24.08-py3-igpu
# Set this only after the small-model compatibility test described below.
export NVIDIA_DISABLE_REQUIRE=1
export TRT_WORKSPACE_MB=256
export TRT_BUILDER_OPT_LEVEL=2
export TRT_MAX_AUX_STREAMS=0
export TRT_OPT_BATCH=1
export TRT_MAX_BATCH=4
export TRT_SPARSITY=disable
```

JetPack 6.2.x may report the Jetson BSP driver as `540.4.0` while a Triton
iGPU container applies the discrete-GPU `560.28` requirement. Do not bypass
that check until the exact `-igpu` image has passed a TensorRT build test. If
the small-model test in the public deployment guide succeeds, explicitly set:

```bash
export NVIDIA_DISABLE_REQUIRE=1
```

The build script and Compose file propagate this opt-in setting. It defaults
to disabled.

Remove stale plan files from previous failed builds, then build:

```bash
chmod +x triton-models-jetson/scripts/build_tensorrt_plans.sh
rm -f triton-models-jetson/*/1/model.plan
rm -f triton-models-jetson/reports/*.build.log
rm -f triton-models-jetson/reports/*.layers.json
bash triton-models-jetson/scripts/build_tensorrt_plans.sh
```

The script:

1. Rejects missing Q/DQ models.
2. Builds strongly typed TensorRT plans.
3. Writes detailed layer reports under `triton-models-jetson/reports/`.
4. Fails if any model report contains no INT8 tensor format.

On an 8 GB Jetson, stop unrelated GPU and memory-heavy processes before
building. Use the fixed batch-1 detector export; the batch-8 graph can exhaust
builder memory even with conservative tactic settings. The default build
profile uses `TRT_WORKSPACE_MB=256`, `TRT_BUILDER_OPT_LEVEL=2`, and
`TRT_MAX_AUX_STREAMS=0`.

Start Triton:

```bash
test "$(find triton-models-jetson -type f -name model.plan | wc -l)" -eq 8
docker compose -f docker-compose.jetson-triton.yml config
docker compose -f docker-compose.jetson-triton.yml up -d
docker compose -f docker-compose.jetson-triton.yml ps
```

The Jetson CTS configuration must set
`triton.detector_static_batch_size: 1`. Keep the DGX configuration at `8`.
Disable the depth slow path in the Jetson deployment because that model is not
served.

The person-identification service uses Triton as its sole inference backend.
Set `TRITON_GRPC_URL` to this server and
`PERSON_ID_MODEL_PROFILE=int8`. Startup fails unless all five Buffalo_L models
are ready.

```bash
TRITON_GRPC_URL=jetson-hostname-or-ip:8701
PERSON_ID_MODEL_PROFILE=int8
```

Verify all face models before starting the API:

```bash
JETSON_HOST=jetson-hostname-or-ip
  for model in \
    person-detector pose-rtmpose reid-solider \
    face-detector-scrfd face-recognition-arcface \
    face-landmark-2d106 face-landmark-3d68 face-attribute-genderage
  do
    code=$(curl -sS -o /dev/null -w '%{http_code}' \
      "http://${JETSON_HOST}:8700/v2/models/${model}/ready")
    printf '%-30s %s\n' "$model" "$code"
  done

  #All models should report 200. Also verify server health:

  curl -sS -o /dev/null -w 'live: %{http_code}\n' \
    "http://${JETSON_HOST}:8700/v2/health/live"

  curl -sS -o /dev/null -w 'ready: %{http_code}\n' \
    "http://${JETSON_HOST}:8700/v2/health/ready"
```

## Structured Sparse INT8

The advertised 67 TOPS is the sparse INT8 figure; NVIDIA lists about 33 dense
INT8 TOPS for Super mode. The current pretrained weights are dense.
`--sparsity=enable` only selects sparse tactics when weights already satisfy the
2:4 pattern; it does not make a dense model sparse.

Use this only after pruning and fine-tuning the source checkpoint:

```bash
TRT_SPARSITY=enable \
  bash triton-models-jetson/scripts/build_tensorrt_plans.sh
```

The build logs report how many layers were eligible and how many actually used
sparse tactics. `sparsity=force` is intentionally unsupported because it
rewrites weights without preserving accuracy.

Dense, mixed-precision INT8 is the production baseline. Sparse INT8 requires
model-specific 2:4 pruning, fine-tuning or QAT, and the same regression/soak
tests before use.

## Production Gate

Before moving all cameras:

1. Pass ONNX drift validation.
2. Pass TensorRT INT8 layer-report validation on the Jetson.
3. Keep at least 1.5 GB unified-memory headroom under sustained load.
4. At eight fully active cameras, ingress can publish 40 frames/s
   (`200 ms` polling). A fixed batch-1 detector may therefore require up to
   40 executions/s. Measure queue latency and reduce polling frequency before
   increasing beyond six cameras.
5. Soak six cameras for 24 hours, then eight cameras for 24 hours.
6. Compare detection recall, pose quality, ReID continuity, ArcFace similarity,
   dropped frames, queue latency, and thermal throttling against the DGX
   baseline.

Start production qualification with six cameras. Eight is conditional on the
latency, memory, drop-rate, and thermal gates above; reduce active-camera
polling to `333-500 ms` if the detector queue grows. Do not count the advertised
sparse INT8 TOPS unless TensorRT confirms sparse tactics were selected.

All eight ONNX Q/DQ graphs now pass their regression gates. Production approval
still depends on TensorRT layer reports, Jetson latency/memory measurements,
and the six-camera then eight-camera soak tests. RTMPose remains a likely
latency constraint because most of that graph must stay in higher precision.

For file transfer, Jetson software checks, the Triton iGPU driver warning,
unified-memory fragmentation recovery, service switching, Git LFS, and
troubleshooting, see the
[Jetson INT8 deployment runbook](../docs/jetson-int8-deployment.md).
