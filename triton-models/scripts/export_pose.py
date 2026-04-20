"""Export RTMPose-m to ONNX for Triton pose-rtmpose model.

Usage:
    pip install mmpose mmdeploy onnx onnxruntime
    python triton-models/scripts/export_pose.py \
        --config rtmpose-m_8xb256-420e_coco-256x192.py \
        --weights rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63be025b_20230126.pth \
        --out triton-models/pose-rtmpose/1/model.onnx

Config file: download from mmpose model zoo or use the mmdeploy-converted one.
Weights: https://download.openmmlab.com/mmpose/v1/projects/rtmpose/

Output tensors:
    simcc_x: [batch, 17, 384]   x-axis SimCC logits (192 px × split_ratio 2.0)
    simcc_y: [batch, 17, 512]   y-axis SimCC logits (256 px × split_ratio 2.0)

Decoding keypoint k:
    x_pixel = argmax(simcc_x[k]) / 2.0
    y_pixel = argmax(simcc_y[k]) / 2.0
    (coordinates are in the 256×192 crop space)

Verify exported tensor names:
    python -c "
    import onnx
    m = onnx.load('triton-models/pose-rtmpose/1/model.onnx')
    print('inputs:', [i.name for i in m.graph.input])
    print('outputs:', [o.name for o in m.graph.output])
    "
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(config: Path, weights: Path, out: Path) -> None:
    try:
        from mmdeploy.apis import torch2onnx  # type: ignore[import-untyped]
        from mmdeploy.backend.onnxruntime import ORTWrapper  # type: ignore[import-untyped]
    except ImportError:
        # Fallback: direct torch.onnx.export via mmpose
        _export_via_mmpose(config, weights, out)
        return

    deploy_cfg = "configs/mmpose/pose-detection_onnxruntime_static.py"
    torch2onnx(
        img="demo/resources/human-pose.jpg",
        work_dir=str(out.parent),
        save_file="model.onnx",
        deploy_cfg=deploy_cfg,
        model_cfg=str(config),
        model_checkpoint=str(weights),
        device="cuda:0",
    )
    print(f"Exported RTMPose-m ONNX → {out}")
    print("Input:  input   [batch, 3, 256, 192]")
    print("Output: simcc_x [batch, 17, 384]")
    print("        simcc_y [batch, 17, 512]")


def _export_via_mmpose(config: Path, weights: Path, out: Path) -> None:
    """Direct ONNX export via mmpose (without mmdeploy)."""
    try:
        import torch  # type: ignore[import-untyped]
        from mmpose.apis import init_model  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("pip install mmpose torch>=2.0") from exc

    model = init_model(str(config), str(weights), device="cuda:0")
    model.eval()

    dummy = torch.zeros(1, 3, 256, 192, device="cuda:0")
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model.backbone,
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
    print(f"Exported RTMPose-m ONNX (via mmpose) → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/pose-rtmpose/1/model.onnx"),
    )
    args = parser.parse_args()
    export(args.config, args.weights, args.out)


if __name__ == "__main__":
    main()
