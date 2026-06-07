#!/usr/bin/env python3
"""Create TensorRT-ready explicit-Q/DQ INT8 ONNX models with NVIDIA ModelOpt."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    source: Path
    input_name: str
    calibration_shape: str
    op_types_to_quantize: tuple[str, ...]
    nodes_to_quantize: tuple[str, ...] = ()
    nodes_to_exclude: tuple[str, ...] = ()


def _verify_qdq(path: Path) -> tuple[int, int]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    quantize_count = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    dequantize_count = sum(
        node.op_type == "DequantizeLinear" for node in model.graph.node
    )
    if quantize_count == 0 or dequantize_count == 0:
        raise RuntimeError(
            f"{path} does not contain explicit QuantizeLinear/DequantizeLinear nodes"
        )
    onnx.checker.check_model(model)
    return quantize_count, dequantize_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--buffalo-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--person-detector-source",
        type=Path,
        help="Override the deployed person-detector ONNX with a candidate export",
    )
    parser.add_argument("--calibration-ep", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()

    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as exc:
        raise SystemExit(
            "Install NVIDIA ModelOpt with the ONNX extra: nvidia-modelopt[onnx]"
        ) from exc

    jetson_repo = args.repo_root / "triton-models-jetson"
    source_repo = args.repo_root / "triton-models"
    person_detector_source = args.person_detector_source or (
        source_repo / "person-detector/1/model.onnx"
    )
    specs = {
        "person-detector": ModelSpec(
            person_detector_source,
            "images",
            "images:8x3x640x640",
            ("Conv",),
            (r"/model\.(0|1|2|3|4|5|6|7|8|9|10)/.*/Conv(_\d+)?$",),
            (r"/model\.23/.*",),
        ),
        "pose-rtmpose": ModelSpec(
            source_repo / "pose-rtmpose/1/model.onnx",
            "input",
            "input:1x3x256x192",
            ("Conv",),
            (r"/model/backbone/stem/.*Conv$",),
        ),
        "reid-solider": ModelSpec(
            source_repo / "reid-solider/1/model.onnx",
            "input",
            "input:1x3x384x128",
            ("Conv", "MatMul"),
        ),
        "face-detector-scrfd": ModelSpec(
            args.buffalo_dir / "det_10g.onnx",
            "input.1",
            "input.1:1x3x640x640",
            ("Conv",),
            (),
            (
                r"Conv_(164|165|167)",
                r"Conv_(187|188|190)",
                r"Conv_(210|211|213)",
            ),
        ),
        "face-recognition-arcface": ModelSpec(
            args.buffalo_dir / "w600k_r50.onnx",
            "input.1",
            "input.1:1x3x112x112",
            ("Conv",),
            (r"Conv_(0|3|5|6|9|11|14|16|19|21|22)$",),
            (r"Gemm_128",),
        ),
        "face-landmark-3d68": ModelSpec(
            args.buffalo_dir / "1k3d68.onnx",
            "data",
            "data:1x3x192x192",
            ("Conv",),
            (r"conv0$",),
            (r"conv2", r"fc1"),
        ),
        "face-landmark-2d106": ModelSpec(
            args.buffalo_dir / "2d106det.onnx",
            "data",
            "data:1x3x192x192",
            ("Conv",),
            (r"conv_1_conv2d$",),
            (r"fc1",),
        ),
        "face-attribute-genderage": ModelSpec(
            args.buffalo_dir / "genderage.onnx",
            "data",
            "data:1x3x96x96",
            ("Conv",),
            (r"conv_1_conv2d$",),
            (r"fc1",),
        ),
    }

    default_models = (
        "pose-rtmpose",
        "reid-solider",
        "face-detector-scrfd",
        "face-recognition-arcface",
        "face-landmark-2d106",
        "face-landmark-3d68",
        "face-attribute-genderage",
    )
    selected = args.models or list(default_models)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}")

    for model_name in selected:
        spec = specs[model_name]
        calibration_path = args.calibration_dir / f"{model_name}.npy"
        output_path = jetson_repo / model_name / "1/model_int8.onnx"
        if not spec.source.is_file():
            raise SystemExit(f"Missing source ONNX for {model_name}: {spec.source}")
        if not calibration_path.is_file():
            raise SystemExit(
                f"Missing calibration tensor for {model_name}: {calibration_path}"
            )

        data = np.load(calibration_path, mmap_mode="r")
        if data.ndim != 4 or data.shape[0] < 32:
            raise SystemExit(
                f"{model_name} needs at least 32 NCHW calibration samples; got {data.shape}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"==> Quantizing {model_name} with {data.shape[0]} representative samples"
        )
        with tempfile.TemporaryDirectory(
            prefix=f"{model_name}-modelopt-",
            dir=output_path.parent,
        ) as temporary_directory:
            temporary_source = Path(temporary_directory) / spec.source.name
            shutil.copy2(spec.source, temporary_source)
            quantize(
                onnx_path=str(temporary_source),
                quantize_mode="int8",
                calibration_data={spec.input_name: data},
                calibration_method=args.method,
                calibration_shapes=spec.calibration_shape,
                calibration_eps=[args.calibration_ep],
                op_types_to_quantize=list(spec.op_types_to_quantize),
                nodes_to_quantize=list(spec.nodes_to_quantize) or None,
                nodes_to_exclude=list(spec.nodes_to_exclude),
                output_path=str(output_path),
                # ModelOpt 0.43 can cast QLinear scale tensors to FP16 when this
                # is fp16, producing graphs rejected by ONNX Runtime. Keep
                # scales and non-quantized fallback operations in FP32.
                high_precision_dtype="fp32",
                use_external_data_format=False,
                direct_io_types=False,
            )
        q_count, dq_count = _verify_qdq(output_path)
        print(f"    wrote {output_path} ({q_count} Q nodes, {dq_count} DQ nodes)")


if __name__ == "__main__":
    main()
