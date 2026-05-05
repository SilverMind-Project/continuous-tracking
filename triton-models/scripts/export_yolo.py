"""Export YOLO26L to TensorRT engine for Triton person-detector model.

Usage (on the target GPU machine):
    pip install ultralytics>=8.4.0
    python triton-models/scripts/export_yolo.py \
        --weights yolo26l.pt \
        --out triton-models/person-detector/1/model.plan

YOLO26L uses a NMS-Free (end-to-end) architecture. The default export
produces a TensorRT engine with baked-in NMS; no post-processing NMS is
needed at inference time.

Output tensor "output0" shape: [batch, 300, 6]
  300 = maximum detections per image
  6   = x1, y1, x2, y2 (letterbox pixel space), confidence, class_id
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(weights: Path, out: Path, batch: int, imgsz: int, device: int) -> None:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.4.0") from exc

    model = YOLO(str(weights))
    result: Path = Path(
        model.export(
            format="engine",
            imgsz=imgsz,
            batch=batch,
            device=device,
            simplify=True,
            dynamic=False,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(result, out)
    print(f"Exported TensorRT engine → {out}")
    print(f"Input:  images  [batch={batch}, 3, {imgsz}, {imgsz}]")
    print(f"Output: output0 [batch={batch}, 300, 6]  (NMS-free)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo26l.pt"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/person-detector/1/model.plan"),
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    export(args.weights, args.out, args.batch, args.imgsz, args.device)


if __name__ == "__main__":
    main()
