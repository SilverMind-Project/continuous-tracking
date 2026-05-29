#!/usr/bin/env python3
"""Export Depth Anything v2 ViT-S Metric Indoor (HF Transformers) to ONNX for Triton.

Downloads ``depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`` from
HuggingFace Hub, loads it via ``transformers.DepthAnythingForDepthEstimation``,
and exports to ONNX with input ``pixel_values`` / output ``predicted_depth``.

Usage::

    uv run --with torch --with transformers --with onnx \
        python triton-models/scripts/export_depth_anything_v2.py

    # With a pre-downloaded HF cache:
    HF_HUB_CACHE=/path/to/cache uv run --with torch --with transformers --with onnx \
        python triton-models/scripts/export_depth_anything_v2.py

Output
------
Writes ``triton-models/depth-anything-v2/1/model.onnx``.

Verification
------------
After exporting::

    uv run --with onnx --with onnxruntime \
        python -c "
    import onnx
    m = onnx.load('triton-models/depth-anything-v2/1/model.onnx')
    print('inputs:', [i.name for i in m.graph.input])
    print('outputs:', [o.name for o in m.graph.output])
    "

Expected::

    inputs:  ['pixel_values']
    outputs: ['predicted_depth']
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path


def export_onnx(
    checkpoint: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    output: str | Path = "",
    height: int = 518,
    width: int = 518,
    opset_version: int = 17,
    dynamic_axes: bool = False,
) -> None:
    import torch
    from transformers import DepthAnythingForDepthEstimation

    output_path = Path(output) if output else (
        Path(__file__).resolve().parent.parent / "depth-anything-v2" / "1" / "model.onnx"
    )

    print(f"Loading model from {checkpoint} ...")
    model = DepthAnythingForDepthEstimation.from_pretrained(checkpoint)
    model.eval()

    # DepthAnythingForDepthEstimation.forward returns a DepthEstimatorOutput
    # with .predicted_depth and .hidden_states.  We wrap it so ONNX export
    # only captures the predicted_depth tensor.
    class _ExportWrapper(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values).predicted_depth

    wrapped = _ExportWrapper(model)

    dummy = torch.zeros(1, 3, height, width)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX to {output_path} ...")
    export_kwargs = {
        "input_names": ["pixel_values"],
        "output_names": ["predicted_depth"],
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    # PyTorch 2.12's dynamo exporter emits symbolic shape-runtime math that
    # TensorRT 10.3 rejects (integer POW). The legacy exporter with static
    # 518x518 shapes produces a TensorRT-friendly graph. Older torch releases
    # do not expose this kwarg, so gate it by signature.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    if dynamic_axes:
        export_kwargs["dynamic_axes"] = {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(wrapped, dummy, str(output_path), **export_kwargs)

    file_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {output_path} ({file_mb:.1f} MB)")

    # Verify the ONNX graph.
    import onnx

    m = onnx.load(str(output_path))
    onnx.checker.check_model(m)
    inputs = [i.name for i in m.graph.input]
    outputs = [o.name for o in m.graph.output]
    input_shape = [
        dim.dim_value if dim.dim_value else str(dim.dim_param)
        for dim in m.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dim.dim_value if dim.dim_value else str(dim.dim_param)
        for dim in m.graph.output[0].type.tensor_type.shape.dim
    ]
    print(f"ONNX check passed — inputs: {inputs} {input_shape}, outputs: {outputs} {output_shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        help="HF repo ID or local path",
    )
    parser.add_argument("--output", default="", help="Output ONNX path (default: auto)")
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--width", type=int, default=518)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Export dynamic batch/height/width axes. Default is static TensorRT-friendly export.",
    )
    args = parser.parse_args()
    export_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        height=args.height,
        width=args.width,
        opset_version=args.opset,
        dynamic_axes=args.dynamic_axes,
    )


if __name__ == "__main__":
    main()
