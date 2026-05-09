"""Download embeddinggemma-300m ONNX model from onnx-community for Triton.

Downloads the FP32 ONNX model, external weight data, and tokenizer
from ``onnx-community/embeddinggemma-300m-ONNX`` on HuggingFace Hub.

The FP32 variant (``model.onnx`` + ``model.onnx_data``) is ~1.23 GB and uses
full precision — the most portable format that works across all GPU vendors
and ONNX Runtime versions. QDQ-quantized and FP16 variants are also available
but not downloaded by default.

Usage:
    uv run --with huggingface_hub python triton-models/scripts/download_embeddinggemma.py

Output (written to triton-models/embeddinggemma-300m/1/):
    model.onnx              — ONNX graph (FP32)
    model.onnx_data         — external weights (FP32, ~1.23 GB)
    tokenizer.json          — HuggingFace tokenizer (20 MB)
    tokenizer_config.json   — tokenizer metadata
    special_tokens_map.json — special token mapping
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_REPO = "onnx-community/embeddinggemma-300m-ONNX"

# ONNX model files (graph + external data). Both must be present for Triton.
_ONNX_FILES = [
    "onnx/model.onnx",
    "onnx/model.onnx_data",
]

# Tokenizer files needed for client-side tokenization (triton-shared).
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def download(out_dir: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ONNX model files from {_REPO}...")
    for filename in _ONNX_FILES:
        local_path = hf_hub_download(
            repo_id=_REPO,
            filename=filename,
        )
        # Copy to flat output directory (strip onnx/ prefix).
        dest = out_dir / Path(filename).name
        if not dest.exists() or dest.stat().st_size != Path(local_path).stat().st_size:
            size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            print(f"  {filename} ({size_mb:.0f} MB) → {dest}")
            shutil.copy(local_path, dest)
        else:
            print(f"  {filename} → {dest} (already up to date)")

    print(f"Downloading tokenizer files from {_REPO}...")
    for filename in _TOKENIZER_FILES:
        try:
            local_path = hf_hub_download(
                repo_id=_REPO,
                filename=filename,
            )
            dest = out_dir / filename
            size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            print(f"  {filename} ({size_mb:.0f} MB) → {dest}")
            shutil.copy(local_path, dest)
        except Exception:
            print(f"  {filename} — skipped (not found in repo)")

    print()
    print("Done. Model directory ready for Triton:")
    print(f"  {out_dir}")
    print()
    _verify(out_dir)


def _verify(out_dir: Path) -> None:
    """Print file sizes present."""
    print("Verification:")
    expected = [Path(f).name for f in _ONNX_FILES] + _TOKENIZER_FILES
    for name in expected:
        p = out_dir / name
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {name}: {size_mb:.0f} MB  ✓")
        else:
            print(f"  {name}: MISSING  ✗")

    print()
    print("Next steps:")
    print("  1. Start Triton (the model is auto-discovered from the model repository)")
    print("  2. Verify model readiness:")
    print("     curl http://triton.nanai.khoofia.com:8001/v2/models/embeddinggemma-300m/ready")
    print("  3. Verify the embedding dimension matches settings.yaml:")
    print("     grep 'dim:' cognitive-companion/config/settings.yaml")
    print("     Expected: dim: 768")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/embeddinggemma-300m/1"),
        help="Output directory for downloaded files",
    )
    args = parser.parse_args()
    download(args.out)


if __name__ == "__main__":
    main()
