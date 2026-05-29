#!/usr/bin/env bash
set -euo pipefail

# Build Jetson-native TensorRT plan files for the CTS Triton models.
# Run this on the Jetson, from the continuous-tracking directory.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_REPO="$ROOT_DIR/triton-models"
JETSON_REPO="$ROOT_DIR/triton-models-jetson"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.08-py3-igpu"
WORKSPACE_MB="${TRT_WORKSPACE_MB:-1024}"
FP_MODE="${TRT_PRECISION:-fp16}"
MAX_BATCH="${TRT_MAX_BATCH:-8}"
OPT_BATCH="${TRT_OPT_BATCH:-4}"

if [[ "$FP_MODE" != "fp16" && "$FP_MODE" != "fp32" ]]; then
  echo "TRT_PRECISION must be fp16 or fp32; got: $FP_MODE" >&2
  exit 2
fi

precision_args=()
if [[ "$FP_MODE" == "fp16" ]]; then
  precision_args+=(--fp16)
fi

required_models=(
  pose-rtmpose
  reid-solider
  depth-anything-v2
)

if [[ ! -f "$JETSON_REPO/person-detector/1/model.onnx" ]]; then
  echo "Missing $JETSON_REPO/person-detector/1/model.onnx" >&2
  echo "Export a Jetson batch-8 detector first:" >&2
  echo "  python triton-models/scripts/export_yolo.py --weights yolo26l.pt --batch 8 --out triton-models-jetson/person-detector/1/model.onnx" >&2
  exit 1
fi

for model in "${required_models[@]}"; do
  if [[ ! -f "$SOURCE_REPO/$model/1/model.onnx" ]]; then
    echo "Missing $SOURCE_REPO/$model/1/model.onnx" >&2
    exit 1
  fi
  mkdir -p "$JETSON_REPO/$model/1"
done

run_trtexec() {
  local model_name="$1"
  shift

  echo "==> Building $model_name TensorRT plan"
  docker run --rm --runtime nvidia \
    --network none \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$SOURCE_REPO:/source:ro" \
    -v "$JETSON_REPO:/models:rw" \
    "$TRITON_IMAGE" \
    bash -lc 'set -euo pipefail
      if command -v trtexec >/dev/null 2>&1; then
        exec trtexec "$@"
      fi
      if [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
        exec /usr/src/tensorrt/bin/trtexec "$@"
      fi
      echo "trtexec not found in Triton image" >&2
      exit 127
    ' bash "$@"
}

common_args=(
  --memPoolSize="workspace:${WORKSPACE_MB}"
  --builderOptimizationLevel=5
)

run_trtexec person-detector \
  --onnx=/models/person-detector/1/model.onnx \
  --saveEngine=/models/person-detector/1/model.plan \
  "${precision_args[@]}" \
  "${common_args[@]}"

run_trtexec pose-rtmpose \
  --onnx=/source/pose-rtmpose/1/model.onnx \
  --saveEngine=/models/pose-rtmpose/1/model.plan \
  --minShapes=input:1x3x256x192 \
  --optShapes=input:${OPT_BATCH}x3x256x192 \
  --maxShapes=input:${MAX_BATCH}x3x256x192 \
  "${precision_args[@]}" \
  "${common_args[@]}"

run_trtexec reid-solider \
  --onnx=/source/reid-solider/1/model.onnx \
  --saveEngine=/models/reid-solider/1/model.plan \
  --minShapes=input:1x3x384x128 \
  --optShapes=input:${MAX_BATCH}x3x384x128 \
  --maxShapes=input:${MAX_BATCH}x3x384x128 \
  "${precision_args[@]}" \
  "${common_args[@]}"

run_trtexec depth-anything-v2 \
  --onnx=/source/depth-anything-v2/1/model.onnx \
  --saveEngine=/models/depth-anything-v2/1/model.plan \
  "${precision_args[@]}" \
  "${common_args[@]}"

echo "TensorRT plans written under $JETSON_REPO"
