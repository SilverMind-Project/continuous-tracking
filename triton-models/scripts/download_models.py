"""One-shot download of all onnx-community models for the Triton model repository.

Downloads pre-quantized INT8 models from HuggingFace onnx-community repos.
Covers models that don't need local export (Florence-2).

For models needing local export, see the individual export scripts:
    export_yolo.py   — YOLO26L from Ultralytics
    export_clip.py   — CLIP ViT-L/14 from OpenCLIP
    export_pose.py   — RTMPose-m from official MMPose checkpoint
    export_reid.py   — Swin-Tiny ReID via timm
    quantize_int8.py — Dynamic INT8 quantization for any ONNX model

Usage:
    uv run --with huggingface_hub python triton-models/scripts/download_models.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MODELS = [
    {
        "name": "florence-2",
        "repo": "onnx-community/Florence-2-large",
        "script": "download_florence.py",
        "description": "Florence-2-large scene description (INT8, ~794 MB)",
    },
]


def main() -> None:
    scripts_dir = _REPO_ROOT / "scripts"
    print("Downloading onnx-community models...")
    print()

    for model in _MODELS:
        print(f"--- {model['name']}: {model['description']} ---")
        script = scripts_dir / model["script"]
        if not script.exists():
            print(f"  ERROR: script not found: {script}")
            sys.exit(1)

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(_REPO_ROOT.parent),
        )
        if result.returncode != 0:
            print(f"  ERROR: {model['name']} download failed")
            sys.exit(1)
        print()

    print("All downloads complete.")
    print()
    print("Next steps:")
    print("  1. Export models needing local conversion:")
    print("     uv run --with ultralytics --with torch --with onnx \\")
    print("         python triton-models/scripts/export_yolo.py")
    print("     uv run --with open_clip_torch --with torch --with onnx \\")
    print("         python triton-models/scripts/export_clip.py")
    print("     # pose requires Python 3.11 venv with mmpose stack")
    print("     python triton-models/scripts/export_pose.py")
    print("     uv run --with torch --with onnx --with timm \\")
    print("         python triton-models/scripts/export_reid.py")
    print("  2. Quantize all FP32 models to INT8:")
    print("     uv run --with onnxruntime --with onnx --with sympy \\")
    print("         python triton-models/scripts/quantize_int8.py --input ... --output ...")
    print("  3. Select GPU vendor:")
    print("     python triton-models/scripts/configure_gpu.py --vendor nvidia|intel")


if __name__ == "__main__":
    main()
