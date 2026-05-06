"""Export CLIP ViT-L/14 vision encoder to ONNX for Triton.

Extracts the vision encoder (``model.visual``) from OpenCLIP ViT-L-14 and
exports it as a standalone ONNX model. This is a one-time export — the same
ONNX file works on both NVIDIA and Intel Arc GPUs.

Usage (on a machine with OpenCLIP installed):
    pip install open_clip_torch torch
    python triton-models/scripts/export_clip.py

Output:
    triton-models/clip-vision/1/model.onnx

Input:  "input"  [batch, 3, 224, 224]  FP32  (CLIP-preprocessed: resize, center-crop, normalize)
Output: "output" [batch, 768]          FP32  (image features, L2-normalize client-side)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export(out: Path) -> None:
    try:
        import open_clip  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install open_clip_torch torch") from exc

    print("Loading OpenCLIP ViT-L-14 (pretrained=openai)...")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval()
    vision = model.visual  # ViT-L/14 vision transformer only

    # Probe the output dimension.
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = vision(dummy)
    dim = output.shape[-1]
    print(f"Vision encoder output dim: {dim}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        vision,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX → {out}")
    print(f"Input:  input   [batch, 3, 224, 224]  FP32")
    print(f"Output: output  [batch, {dim}]          FP32")
    print()
    print("Verification:")
    print(f"  python -c \"import onnx; m=onnx.load('{out}'); ")
    print("  print([d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim])\"")
    print(f"  Expected: [0, {dim}]  (0 = dynamic batch)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/clip-vision/1/model.onnx"),
        help="Output path for the ONNX model",
    )
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
