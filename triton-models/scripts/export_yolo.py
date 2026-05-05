"""Export YOLO26L to ONNX for the Triton person-detector model.

The ONNX file works on both NVIDIA and Intel Arc GPUs:
  - NVIDIA: Triton's ONNX Runtime auto-selects TensorRT EP then CUDA EP.
  - Intel Arc: run configure_gpu.py --vendor intel to activate OpenVINO EP.

Usage (on the target GPU machine):
    pip install ultralytics>=8.4.0

    # NVIDIA
    python triton-models/scripts/export_yolo.py \\
        --weights yolo26l.pt \\
        --out triton-models/person-detector/1/model.onnx

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
    dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print('output0 dims:', dims)
    # Expected: [0 or batch_size, 300, 6]
    # If you see [batch, 84, 8400] the NMS head was not preserved — upgrade
    # ultralytics and retry, or open an issue against the model export.
    "
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(weights: Path, out: Path, batch: int, imgsz: int, device: str) -> None:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.4.0") from exc

    model = YOLO(str(weights))
    result: Path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            batch=batch,
            device=device,
            simplify=True,
            dynamic=False,
            opset=17,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(result, out)
    print(f"Exported ONNX → {out}")
    print(f"Input:  images  [batch={batch}, 3, {imgsz}, {imgsz}]")
    print(f"Output: output0 [batch={batch}, 300, 6]  (NMS-free)")
    print()
    print("Verify output shape before deploying:")
    print(f"  python -c \"import onnx; m=onnx.load('{out}'); "
          "print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])\"")
    print("  Expected: [0, 300, 6]  (0 = dynamic batch)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo26l.pt"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/person-detector/1/model.onnx"),
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device: '0' for NVIDIA CUDA, 'xpu' for Intel Arc",
    )
    args = parser.parse_args()
    export(args.weights, args.out, args.batch, args.imgsz, args.device)


if __name__ == "__main__":
    main()
