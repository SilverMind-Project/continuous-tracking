"""Export Swin-Tiny ReID feature extractor to ONNX.

Creates a Swin-Tiny backbone (256x128 input, 768-dim output) using the
``timm`` library and exports to ONNX. This is a simplified ReID model
without SOLIDER semantic controller or MSMT17 fine-tuning.

For production ReID accuracy, train on SOLIDER-REID with proper weights.
For development, this standalone export is sufficient.

Usage:
    uv run --with torch --with onnx --with timm \\
        python triton-models/scripts/export_reid.py

Output:
    triton-models/reid-solider/1/model.onnx  (FP32, ~107 MB)

Then quantize with quantize_int8.py.
"""

from __future__ import annotations

import os
from pathlib import Path


def export(out: Path) -> None:
    import timm
    import torch

    print("Creating Swin-Tiny feature extractor (256x128, 768-dim)...")
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=0,
        img_size=(256, 128),
    )
    model.eval()

    dummy = torch.zeros(1, 3, 256, 128)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("Input:  input   [batch, 3, 256, 128]")
    print("Output: output  [batch, 768]")

    # Verify
    import onnx

    m = onnx.load(str(out))
    for i in m.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        print(f"  Verify input:  {i.name} {dims}")
    for o in m.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  Verify output: {o.name} {dims}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # triton-models/
    out = repo_root / "reid-solider" / "1" / "model.onnx"
    export(out)


if __name__ == "__main__":
    main()
