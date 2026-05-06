"""Download Florence-2-large ONNX files from onnx-community for Triton.

Downloads INT8 ONNX model files and the tokenizer from
``onnx-community/Florence-2-large`` on HuggingFace Hub.

Usage:
    pip install huggingface_hub
    python triton-models/scripts/download_florence.py

Output (written to triton-models/florence-2/1/):
    vision_encoder_int8.onnx
    encoder_model_int8.onnx
    decoder_model_merged_int8.onnx
    embed_tokens_int8.onnx
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    generation_config.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

_REPO = "onnx-community/Florence-2-large"

# ONNX model files to download (INT8 variants for performance).
_ONNX_FILES = [
    "onnx/vision_encoder_int8.onnx",
    "onnx/encoder_model_int8.onnx",
    "onnx/decoder_model_merged_int8.onnx",
    "onnx/embed_tokens_int8.onnx",
]

# Tokenizer / config files needed for client-side preprocessing.
_TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
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
        import shutil

        shutil.copy(local_path, dest)
        print(f"  {filename} → {dest}")

    print(f"Downloading tokenizer files from {_REPO}...")
    for filename in _TOKENIZER_FILES:
        try:
            local_path = hf_hub_download(
                repo_id=_REPO,
                filename=filename,
            )
            dest = out_dir / filename
            import shutil

            shutil.copy(local_path, dest)
            print(f"  {filename} → {dest}")
        except Exception:
            print(f"  {filename} — skipped (not found in repo)")

    print()
    print("Done. Model directory ready for Triton:")
    print(f"  {out_dir}")
    print()
    print("Next steps:")
    print("  1. Verify model files:")
    for f in _ONNX_FILES:
        name = Path(f).name
        path = out_dir / name
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"     {name}: {size_mb:.0f} MB")
        else:
            print(f"     {name}: MISSING")
    print("  2. Run: python triton-models/scripts/configure_gpu.py --vendor nvidia|intel")
    print("  3. Start Triton: docker compose up triton")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("triton-models/florence-2/1"),
        help="Output directory for downloaded files",
    )
    args = parser.parse_args()
    download(args.out)


if __name__ == "__main__":
    main()
