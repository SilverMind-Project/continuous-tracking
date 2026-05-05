"""Configure Triton model configs for the target GPU vendor.

Run once on the target machine before starting Triton. Copies the correct
config.pbtxt variant into each model directory so Triton loads the right
backend and execution provider.

Usage:
    # NVIDIA GPU (default — TensorRT for detector, ONNX Runtime/CUDA for ReID + Pose)
    python triton-models/scripts/configure_gpu.py --vendor nvidia

    # Intel Arc GPU (ONNX Runtime + OpenVINO EP for all three models)
    python triton-models/scripts/configure_gpu.py --vendor intel

Vendor requirements:
    nvidia — standard nvcr.io/nvidia/tritonserver image
    intel  — Triton image with OpenVINO backend + Intel Compute Runtime drivers
             (see triton-models/README.md for the correct container image)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent  # triton-models/

_MODELS = ["person-detector", "reid-solider", "pose-rtmpose"]


def configure(vendor: str, repo: Path, dry_run: bool) -> None:
    for model in _MODELS:
        model_dir = repo / model
        dest = model_dir / "config.pbtxt"

        if vendor == "nvidia":
            src = model_dir / "config.pbtxt.nvidia"
            if not src.exists():
                # No .nvidia variant means the default config.pbtxt IS the NVIDIA config.
                print(f"  {model}: already configured for NVIDIA (no .nvidia variant needed)")
                continue
        else:
            src = model_dir / "config.pbtxt.intel"
            if not src.exists():
                raise FileNotFoundError(
                    f"Intel Arc config not found: {src}\n"
                    "This file should be in the repository. Check your git state."
                )

        if dry_run:
            print(f"  [dry-run] {model}: would copy {src.name} → config.pbtxt")
        else:
            shutil.copy(src, dest)
            print(f"  {model}: {src.name} → config.pbtxt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor",
        choices=["nvidia", "intel"],
        required=True,
        help="GPU vendor: 'nvidia' for NVIDIA GPUs, 'intel' for Intel Arc",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=_REPO_ROOT,
        help="Path to the triton-models directory (default: auto-detected)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()

    print(f"Configuring Triton models for {args.vendor.upper()} GPU...")
    configure(args.vendor, args.repo, args.dry_run)
    print("Done. Restart Triton for changes to take effect.")


if __name__ == "__main__":
    main()
