"""Export YOLO11m to TensorRT engine for Triton person-detector model.

Usage (on the target GPU machine):
    pip install ultralytics
    python triton-models/scripts/export_yolo.py \
        --weights yolo11m.pt \
        --out triton-models/person-detector/1/model.plan

The Ultralytics export produces a TensorRT engine (.engine) which must be
renamed to model.plan for Triton. Fine-tune the weights on indoor overhead
footage before exporting (see Phase 0 section 0.4.1).

Output tensor "output0" shape: [batch, 84, 8400]
  84 = 4 (cx,cy,w,h in input-pixel space) + 80 COCO class scores
  8400 = 6400 + 1600 + 400 (three detection scales: 80×80, 40×40, 20×20)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export(weights: Path, out: Path, batch: int, imgsz: int, device: int) -> None:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install ultralytics>=8.3.0") from exc

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
    print(f"Output: output0 [batch={batch}, 84, 8400]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11m.pt"))
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
