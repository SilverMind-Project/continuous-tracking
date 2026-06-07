#!/usr/bin/env bash
set -euo pipefail

# Build Jetson-native TensorRT plans from explicit-Q/DQ INT8 ONNX models.
# Run this on the Jetson from the continuous-tracking repository root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JETSON_REPO="$ROOT_DIR/triton-models-jetson"
TRITON_IMAGE="${TRITON_JETSON_IMAGE:?Set a JetPack-compatible Triton iGPU image}"
WORKSPACE_MB="${TRT_WORKSPACE_MB:-256}"
BUILDER_OPT_LEVEL="${TRT_BUILDER_OPT_LEVEL:-2}"
MAX_AUX_STREAMS="${TRT_MAX_AUX_STREAMS:-0}"
MAX_BATCH="${TRT_MAX_BATCH:-8}"
OPT_BATCH="${TRT_OPT_BATCH:-4}"
SPARSITY="${TRT_SPARSITY:-disable}"
DISABLE_DRIVER_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-0}"
REPORT_DIR="$JETSON_REPO/reports"

if [[ "$SPARSITY" != "disable" && "$SPARSITY" != "enable" ]]; then
  echo "TRT_SPARSITY must be disable or enable; got: $SPARSITY" >&2
  echo "TensorRT sparsity=force is intentionally unsupported because it changes weights." >&2
  exit 2
fi
if [[ "$DISABLE_DRIVER_REQUIRE" != "0" && "$DISABLE_DRIVER_REQUIRE" != "1" ]]; then
  echo "NVIDIA_DISABLE_REQUIRE must be 0 or 1; got: $DISABLE_DRIVER_REQUIRE" >&2
  exit 2
fi
if (( BUILDER_OPT_LEVEL < 0 || BUILDER_OPT_LEVEL > 5 )); then
  echo "Require 0 <= TRT_BUILDER_OPT_LEVEL <= 5" >&2
  exit 2
fi
if (( MAX_AUX_STREAMS < 0 )); then
  echo "TRT_MAX_AUX_STREAMS must be non-negative" >&2
  exit 2
fi
if (( OPT_BATCH < 1 || MAX_BATCH < OPT_BATCH || MAX_BATCH > 8 )); then
  echo "Require 1 <= TRT_OPT_BATCH <= TRT_MAX_BATCH <= 8" >&2
  exit 2
fi

models=(
  person-detector
  pose-rtmpose
  reid-solider
  face-detector-scrfd
  face-recognition-arcface
  face-landmark-2d106
  face-landmark-3d68
  face-attribute-genderage
)

for model in "${models[@]}"; do
  qdq_model="$JETSON_REPO/$model/1/model_int8.onnx"
  if [[ ! -f "$qdq_model" ]]; then
    echo "Missing explicit-Q/DQ model: $qdq_model" >&2
    echo "Run quantize_int8_models.py with representative calibration data first." >&2
    exit 1
  fi
done

mkdir -p "$REPORT_DIR"

run_trtexec() {
  local model_name="$1"
  shift
  local log_file="$REPORT_DIR/$model_name.build.log"
  local docker_args=(
    --rm
    --runtime nvidia
    --network none
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -v "$JETSON_REPO:/models:rw"
  )

  if [[ "$DISABLE_DRIVER_REQUIRE" == "1" ]]; then
    docker_args+=(-e NVIDIA_DISABLE_REQUIRE=1)
  fi

  echo "==> Building $model_name TensorRT plan"
  docker run "${docker_args[@]}" \
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
    ' bash "$@" 2>&1 | tee "$log_file"
}

common_args=(
  --memPoolSize="workspace:${WORKSPACE_MB}"
  --builderOptimizationLevel="${BUILDER_OPT_LEVEL}"
  --maxAuxStreams="${MAX_AUX_STREAMS}"
  --stronglyTyped
  --skipInference
  --profilingVerbosity=detailed
  --dumpLayerInfo
)

if [[ "$SPARSITY" == "enable" ]]; then
  common_args+=(--sparsity=enable --verbose)
fi

run_trtexec person-detector \
  --onnx=/models/person-detector/1/model_int8.onnx \
  --saveEngine=/models/person-detector/1/model.plan \
  --exportLayerInfo=/models/reports/person-detector.layers.json \
  "${common_args[@]}"

run_trtexec pose-rtmpose \
  --onnx=/models/pose-rtmpose/1/model_int8.onnx \
  --saveEngine=/models/pose-rtmpose/1/model.plan \
  --minShapes=input:1x3x256x192 \
  --optShapes=input:"${OPT_BATCH}"x3x256x192 \
  --maxShapes=input:"${MAX_BATCH}"x3x256x192 \
  --exportLayerInfo=/models/reports/pose-rtmpose.layers.json \
  "${common_args[@]}"

run_trtexec reid-solider \
  --onnx=/models/reid-solider/1/model_int8.onnx \
  --saveEngine=/models/reid-solider/1/model.plan \
  --minShapes=input:1x3x384x128 \
  --optShapes=input:"${OPT_BATCH}"x3x384x128 \
  --maxShapes=input:"${MAX_BATCH}"x3x384x128 \
  --exportLayerInfo=/models/reports/reid-solider.layers.json \
  "${common_args[@]}"

run_trtexec face-detector-scrfd \
  --onnx=/models/face-detector-scrfd/1/model_int8.onnx \
  --saveEngine=/models/face-detector-scrfd/1/model.plan \
  --minShapes=input.1:1x3x640x640 \
  --optShapes=input.1:1x3x640x640 \
  --maxShapes=input.1:1x3x640x640 \
  --exportLayerInfo=/models/reports/face-detector-scrfd.layers.json \
  "${common_args[@]}"

run_trtexec face-recognition-arcface \
  --onnx=/models/face-recognition-arcface/1/model_int8.onnx \
  --saveEngine=/models/face-recognition-arcface/1/model.plan \
  --minShapes=input.1:1x3x112x112 \
  --optShapes=input.1:1x3x112x112 \
  --maxShapes=input.1:1x3x112x112 \
  --exportLayerInfo=/models/reports/face-recognition-arcface.layers.json \
  "${common_args[@]}"

run_trtexec face-landmark-2d106 \
  --onnx=/models/face-landmark-2d106/1/model_int8.onnx \
  --saveEngine=/models/face-landmark-2d106/1/model.plan \
  --minShapes=data:1x3x192x192 \
  --optShapes=data:1x3x192x192 \
  --maxShapes=data:1x3x192x192 \
  --exportLayerInfo=/models/reports/face-landmark-2d106.layers.json \
  "${common_args[@]}"

run_trtexec face-landmark-3d68 \
  --onnx=/models/face-landmark-3d68/1/model_int8.onnx \
  --saveEngine=/models/face-landmark-3d68/1/model.plan \
  --minShapes=data:1x3x192x192 \
  --optShapes=data:1x3x192x192 \
  --maxShapes=data:1x3x192x192 \
  --exportLayerInfo=/models/reports/face-landmark-3d68.layers.json \
  "${common_args[@]}"

run_trtexec face-attribute-genderage \
  --onnx=/models/face-attribute-genderage/1/model_int8.onnx \
  --saveEngine=/models/face-attribute-genderage/1/model.plan \
  --minShapes=data:1x3x96x96 \
  --optShapes=data:1x3x96x96 \
  --maxShapes=data:1x3x96x96 \
  --exportLayerInfo=/models/reports/face-attribute-genderage.layers.json \
  "${common_args[@]}"

reports=()
for model in "${models[@]}"; do
  reports+=("$REPORT_DIR/$model.layers.json")
done
python3 "$JETSON_REPO/scripts/verify_tensorrt_precision.py" "${reports[@]}"

if [[ "$SPARSITY" == "enable" ]]; then
  echo
  echo "Structured sparsity summaries:"
  for model in "${models[@]}"; do
    echo "--- $model ---"
    grep -E "\\(Sparsity\\).*eligible|\\(Sparsity\\).*using sparse" \
      "$REPORT_DIR/$model.build.log" || echo "No sparse tactic selected."
  done
fi

echo "TensorRT INT8 plans and layer reports written under $JETSON_REPO"
