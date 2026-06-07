#!/usr/bin/env python3
"""Restore a ModelOpt YOLO26L QAT checkpoint and export fixed-batch Q/DQ ONNX."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _configure_export(model: object) -> None:
    head = model.model[-1]
    head.dynamic = False
    head.export = True
    head.format = "onnx"
    head.max_det = 300
    head.shape = None


def _verify_export(path: Path, batch_size: int) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    quantize_count = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dequantize_count = sum(
        node.op_type == "DequantizeLinear" for node in model.graph.node
    )
    if quantize_count == 0 or dequantize_count == 0:
        raise RuntimeError(f"{path} does not contain explicit Q/DQ nodes")

    input_dims = [
        dimension.dim_value
        for dimension in model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_dims = [
        dimension.dim_value
        for dimension in model.graph.output[0].type.tensor_type.shape.dim
    ]
    expected_input = [batch_size, 3, 640, 640]
    expected_output = [batch_size, 300, 6]
    if input_dims != expected_input:
        raise RuntimeError(f"Expected input shape {expected_input}; got {input_dims}")
    if output_dims != expected_output:
        raise RuntimeError(f"Expected output shape {expected_output}; got {output_dims}")

    print(
        f"Verified {path}: input={input_dims} output={output_dims} "
        f"Q={quantize_count} DQ={dequantize_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not args.weights.is_file():
        raise SystemExit(f"Missing YOLO weights: {args.weights}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing ModelOpt checkpoint: {args.checkpoint}")

    import modelopt.torch.opt as mto
    from ultralytics import YOLO

    base = YOLO(str(args.weights)).model.fuse().float().to(args.device).eval()
    _configure_export(base)
    quantized = mto.restore(base, args.checkpoint)
    quantized.to(args.device).eval()
    _configure_export(quantized)

    wrapper = YOLO(str(args.weights))
    wrapper.model = quantized
    exported = Path(
        wrapper.export(
            format="onnx",
            imgsz=640,
            batch=args.batch_size,
            device=args.device,
            simplify=False,
            dynamic=False,
            opset=17,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)
    _verify_export(args.output, args.batch_size)


if __name__ == "__main__":
    main()
