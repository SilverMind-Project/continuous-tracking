"""Quantize ONNX models to INT8 QDQ format using ONNX Runtime static quantization.

QDQ (QuantizeLinear → Op → DequantizeLinear) format wraps standard FP32
ops with quantize/dequantize nodes. The actual compute runs in FP32, so no
specialized INT8 CUDA kernels are needed. This makes the quantized model
portable across NVIDIA (CUDA EP), Intel Arc (OpenVINO EP), and CPU (CPU EP).

Uses MinMax calibration with random input data — sufficient for models that
were previously running with dynamic INT8 quantization. The model structure
is preserved; only weight representation changes.

Good balance of size reduction (~4x) and portability with minimal accuracy
loss compared to QOperator format.

Usage:
    uv run --with onnxruntime --with onnx --with sympy \
        python triton-models/scripts/quantize_int8.py --input model.onnx --output model_qdq.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


class RandomCalibrationReader:
    """Generates random calibration data matching the model input shape."""

    def __init__(self, input_name: str, input_shape: tuple[int, ...], num_samples: int = 100):
        self._name = input_name
        self._shape = input_shape
        self._num_samples = num_samples
        self._iter = 0

    def get_next(self) -> dict[str, np.ndarray]:
        if self._iter >= self._num_samples:
            return {}
        self._iter += 1
        return {self._name: np.random.randn(*self._shape).astype(np.float32)}

    def rewind(self) -> None:
        self._iter = 0

    def set_range(self, start_index: int, end_index: int) -> None:
        self._iter = start_index


def _get_first_input_info(model_path: Path) -> tuple[str, tuple[int, ...]]:
    """Extract the name and shape of the first model input."""
    import onnx

    model = onnx.load(str(model_path))
    inp = model.graph.input[0]
    name = inp.name
    shape = []
    for d in inp.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else 1)
    return name, tuple(shape)


def quantize(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.calibrate import CalibrationMethod

    size_mb = os.path.getsize(src) / (1024**2)
    print(f"Quantizing: {src}")
    print(f"  FP32 size: {size_mb:.0f} MB")

    input_name, input_shape = _get_first_input_info(src)
    print(f"  Input: {input_name} {input_shape}")

    dr = RandomCalibrationReader(input_name, input_shape, num_samples=100)

    quantize_static(
        str(src),
        str(dst),
        calibration_data_reader=dr,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True},
    )

    new_mb = os.path.getsize(dst) / (1024**2)
    print(f"  QDQ size: {new_mb:.0f} MB")
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
