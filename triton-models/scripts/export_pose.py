"""Export RTMPose-m to ONNX from official MMPose checkpoint.

Downloads the official RTMPose-m AIC+COCO 256x192 weights from OpenMMLab
and exports to ONNX using mmpose/mmdet.

Requires a Python 3.11 venv with the OpenMMLab stack and torch.
Setup (one-time):
    uv venv --python 3.11 .venv-pose
    source .venv-pose/bin/activate
    uv pip install numpy cython setuptools wheel
    uv pip install xtcocotools --no-build-isolation
    uv pip install mmcv mmdet mmpose mmengine --no-build-isolation
    uv pip install torch onnx opencv-python-headless

Usage:
    source .venv-pose/bin/activate
    python triton-models/scripts/export_pose.py

Output:
    triton-models/pose-rtmpose/1/model.onnx  (FP32, ~52 MB)

Then quantize with quantize_int8.py.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
)


def _patch_mmdet() -> None:
    """Patch mmdet version check for mmcv 2.2.0 compatibility."""
    import site

    sitepkg = site.getsitepackages()[0]
    init_file = Path(sitepkg) / "mmdet" / "__init__.py"
    content = init_file.read_text()
    # Replace the mmcv version assert with a no-op.
    lines = content.split("\n")
    new_lines = []
    skip = False
    for line in lines:
        if "assert (mmcv_version >= digit_version(mmcv_minimum_version)" in line:
            new_lines.append("assert True  # patched for mmcv 2.2.0 compat")
            skip = True
            continue
        if skip:
            if line.rstrip().endswith("'.") or line.rstrip().endswith("'") or line.rstrip().endswith(')"'):
                # Last line of the multi-line assert
                skip = False
            continue
        new_lines.append(line)
    init_file.write_text("\n".join(new_lines))
    print("Patched mmdet version check")


def download_checkpoint(dest: Path) -> None:
    """Download RTMPose-m checkpoint if not present."""
    if dest.exists():
        print(f"Checkpoint already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading RTMPose-m checkpoint...")
    urllib.request.urlretrieve(_CHECKPOINT_URL, str(dest))
    print(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def export(checkpoint_path: Path, onnx_path: Path) -> None:
    """Build RTMPose-m model and export to ONNX."""
    import torch
    from mmpose.apis import init_model

    print("Building model from checkpoint...")
    # We need a config file.  Write the config from the checkpoint metadata.
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    cfg_text = ckpt["meta"]["cfg"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(cfg_text)
        cfg_path = f.name

    try:
        model = init_model(cfg_path, str(checkpoint_path), device="cpu")
    finally:
        os.unlink(cfg_path)

    model.eval()
    print("Model built successfully")

    # Export to ONNX.
    dummy = torch.zeros(1, 3, 256, 192)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        dynamic_axes={
            "input": {0: "batch"},
            "simcc_x": {0: "batch"},
            "simcc_y": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX → {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify.
    import onnx

    m = onnx.load(str(onnx_path))
    print("Input:")
    for i in m.graph.input:
        print(f"  {i.name}: {[d.dim_value for d in i.type.tensor_type.shape.dim]}")
    print("Output:")
    for o in m.graph.output:
        print(f"  {o.name}: {[d.dim_value for d in o.type.tensor_type.shape.dim]}")


def main() -> None:
    # Must run from repo root or set paths relative to this script.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # triton-models/scripts -> triton-models -> continuous-tracking
    checkpoint_dir = Path("/tmp/rtmpose_export")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "rtmpose-m-aic-coco.pth"
    onnx_path = repo_root / "triton-models" / "pose-rtmpose" / "1" / "model.onnx"

    _patch_mmdet()
    download_checkpoint(checkpoint_path)
    export(checkpoint_path, onnx_path)


if __name__ == "__main__":
    main()
