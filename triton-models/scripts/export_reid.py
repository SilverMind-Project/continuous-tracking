"""Export SOLIDER-REID to ONNX for Triton reid-solider model.

Usage:
    git clone https://github.com/tinyvision/SOLIDER-REID
    cd SOLIDER-REID
    pip install -r requirements.txt
    python ../triton-models/scripts/export_reid.py \
        --config configs/MSMT17/swin_tiny.yml \
        --weights /path/to/solider_swin_tiny_msmt17.pth \
        --out ../triton-models/reid-solider/1/model.onnx

Output tensor "output" shape: [batch, 768]
  768-dim L2-normalised appearance embedding (Swin-Tiny backbone).
  The model includes L2 normalisation in the final layer — do NOT
  normalise again in the inference client.

Verify exported tensor names:
    python -c "
    import onnx
    m = onnx.load('triton-models/reid-solider/1/model.onnx')
    print('inputs:', [i.name for i in m.graph.input])
    print('outputs:', [o.name for o in m.graph.output])
    "
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(config: Path, weights: Path, out: Path, batch: int) -> None:
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install torch>=2.0") from exc

    # SOLIDER-REID must be on sys.path (clone the repo first)
    try:
        from config import cfg  # type: ignore[import-untyped]
        from model import make_model  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "Run from inside the SOLIDER-REID repo clone, or add it to PYTHONPATH."
        ) from exc

    cfg.merge_from_file(str(config))
    cfg.freeze()

    model = make_model(cfg, num_class=1, camera_num=0, view_num=0, semantic_weight=1.0)
    model.load_param(str(weights))
    model.eval()

    dummy = torch.zeros(batch, 3, 256, 128)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported SOLIDER-REID ONNX → {out}")
    print(f"Input:  input  [batch, 3, 256, 128]")
    print(f"Output: output [batch, 768]  (L2-normalised)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/reid-solider/1/model.onnx"),
    )
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    export(args.config, args.weights, args.out, args.batch)


if __name__ == "__main__":
    main()
