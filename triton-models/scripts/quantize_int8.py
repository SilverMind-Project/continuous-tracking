"""Quantize ONNX models to INT8 using ONNX Runtime dynamic quantization.

Dynamic quantization quantizes weights to INT8 while keeping activations
in FP32. Good balance of size reduction (~4x) and accuracy preservation.

Usage:
    uv run --with onnxruntime --with onnx --with sympy \\
        python triton-models/scripts/quantize_int8.py --input model.onnx --output model_int8.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    size_mb = os.path.getsize(src) / (1024**2)
    print(f"Quantizing: {src}")
    print(f"  FP32 size: {size_mb:.0f} MB")

    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True},
    )

    new_mb = os.path.getsize(dst) / (1024**2)
    print(f"  INT8 size: {new_mb:.0f} MB")
    reduction = 100 * (1 - os.path.getsize(dst) / os.path.getsize(src))
    print(f"  Reduction: {reduction:.0f}%")
    print(f"  Done: {dst}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
