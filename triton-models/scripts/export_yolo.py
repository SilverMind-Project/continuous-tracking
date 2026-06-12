"""Export YOLO26L to ONNX for the Triton person-detector model.

The ONNX file works on both NVIDIA and Intel Arc GPUs:
  - NVIDIA: Triton's ONNX Runtime auto-selects TensorRT EP then CUDA EP.
  - Intel Arc: run configure_gpu.py --vendor intel to activate OpenVINO EP.

Usage (on the target GPU machine):
    pip install ultralytics>=8.4.0

    # Static batch-8 (default, used by person-detector and Jetson)
    python triton-models/scripts/export_yolo.py \\
        --weights yolo26l.pt \\
        --out triton-models/person-detector/1/model.onnx

    # Dynamic batch (used by person-detector-dynamic, DGX/FP32 only)
    python triton-models/scripts/export_yolo.py \\
        --weights yolo26l.pt \\
        --dynamic-batch \\
        --out triton-models/person-detector-dynamic/1/model.onnx

    # Intel Arc (requires Intel Extension for PyTorch)
    python triton-models/scripts/export_yolo.py \\
        --weights yolo26l.pt \\
        --device xpu \\
        --out triton-models/person-detector/1/model.onnx

YOLO26L uses a NMS-Free (end-to-end) architecture. The ONNX export preserves
the baked-in NMS; no post-processing NMS is needed at inference time.

Output tensor "output0" shape: [batch, 300, 6]
  300 = maximum detections per image
  6   = x1, y1, x2, y2 (letterbox pixel space), confidence, class_id

IMPORTANT — verify output shape before deploying to Triton:
    python -c "
    import onnx
    m = onnx.load('triton-models/person-detector/1/model.onnx')
    dims = [d.dim_param or d.dim_value
            for d in m.graph.output[0].type.tensor_type.shape.dim]
    print('output0 dims:', dims)
    # Static export: [8, 300, 6]
    # Dynamic export: ['batch', 300, 6]  (batch is a symbolic dim_param)
    "
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(
    weights: Path,
    out: Path,
    batch: int,
    imgsz: int,
    device: str,
    dynamic_batch: bool,
) -> None:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.4.0") from exc

    model = YOLO(str(weights))
    result: Path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            # dynamic=True exports the batch dim as a symbolic dim_param so the
            # ONNX graph accepts any batch size 1..N.  batch=1 here is the
            # reference size used during tracing; it does not cap the runtime batch.
            batch=1 if dynamic_batch else batch,
            device=device,
            simplify=True,
            dynamic=dynamic_batch,
            opset=17,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(result, out)

    if dynamic_batch:
        print(f"Exported dynamic-batch ONNX → {out}")
        print(f"Input:  images  [batch (symbolic), 3, {imgsz}, {imgsz}]")
        print(f"Output: output0 [batch (symbolic), 300, 6]  (NMS-free)")
        print()
        print("Verify batch dimension is symbolic before deploying:")
        print(
            f"  python -c \"import onnx; m=onnx.load('{out}'); "
            "print([d.dim_param or d.dim_value "
            "for d in m.graph.output[0].type.tensor_type.shape.dim])\""
        )
        print("  Expected: ['batch', 300, 6]  (first element is a non-empty string)")
    else:
        print(f"Exported static-batch ONNX → {out}")
        print(f"Input:  images  [batch={batch}, 3, {imgsz}, {imgsz}]")
        print(f"Output: output0 [batch={batch}, 300, 6]  (NMS-free)")
        print()
        print("Verify output shape before deploying:")
        print(
            f"  python -c \"import onnx; m=onnx.load('{out}'); "
            'print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])"'
        )
        print(f"  Expected: [{batch}, 300, 6]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo26l.pt"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/person-detector/1/model.onnx"),
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device: '0' for NVIDIA CUDA, 'xpu' for Intel Arc",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        default=False,
        help=(
            "Export with a symbolic batch dimension (for person-detector-dynamic). "
            "Output path defaults to triton-models/person-detector/1/model.onnx; "
            "pass --out triton-models/person-detector-dynamic/1/model.onnx explicitly."
        ),
    )
    args = parser.parse_args()
    export(args.weights, args.out, args.batch, args.imgsz, args.device, args.dynamic_batch)


if __name__ == "__main__":
    main()
