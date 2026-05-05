"""Export RTMPose-m to ONNX for Triton pose-rtmpose model.

Usage:
    pip install mmpose torch>=2.0 onnx

    # NVIDIA
    python triton-models/scripts/export_pose.py \\
        --config rtmpose-m_8xb256-420e_coco-256x192.py \\
        --weights rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63be025b_20230126.pth \\
        --out triton-models/pose-rtmpose/1/model.onnx

    # Intel Arc (requires Intel Extension for PyTorch: pip install intel-extension-for-pytorch)
    python triton-models/scripts/export_pose.py \\
        --config rtmpose-m_8xb256-420e_coco-256x192.py \\
        --weights rtmpose-m_simcc-aic-coco_420e-256x192.pth \\
        --out triton-models/pose-rtmpose/1/model.onnx \\
        --device xpu

Config + weights: https://mmpose.readthedocs.io/en/latest/model_zoo/body_2d_keypoint.html
  Model: RTMPose-m, 256×192, COCO pretrained

Output tensors:
    simcc_x: [batch, 17, 384]   x-axis SimCC logits (192 px × split_ratio 2.0)
    simcc_y: [batch, 17, 512]   y-axis SimCC logits (256 px × split_ratio 2.0)

Decoding keypoint k:
    x_pixel = argmax(simcc_x[k]) / 2.0
    y_pixel = argmax(simcc_y[k]) / 2.0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(config: Path, weights: Path, out: Path, device: str) -> None:
    try:
        import torch  # type: ignore[import-untyped]
        from mmpose.apis import init_model  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install mmpose torch>=2.0") from exc

    model = init_model(str(config), str(weights), device=device)
    model.eval()

    dummy = torch.zeros(1, 3, 256, 192, device=device)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        dynamic_axes={
            "input": {0: "batch"},
            "simcc_x": {0: "batch"},
            "simcc_y": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported RTMPose-m ONNX → {out}")
    print("Input:  input   [batch, 3, 256, 192]")
    print("Output: simcc_x [batch, 17, 384]")
    print("        simcc_y [batch, 17, 512]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/pose-rtmpose/1/model.onnx"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for export: 'cuda:0' for NVIDIA, 'xpu' for Intel Arc",
    )
    args = parser.parse_args()
    export(args.config, args.weights, args.out, args.device)


if __name__ == "__main__":
    main()
