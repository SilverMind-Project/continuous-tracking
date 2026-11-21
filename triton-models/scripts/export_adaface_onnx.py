#!/usr/bin/env python3
"""Export an AdaFace IR-50 PyTorch checkpoint to Triton FP32 ONNX and verify numerical parity.

Usage:
    python triton-models/scripts/export_adaface_onnx.py \\
        --checkpoint /path/to/adaface_ir50_webface4m.ckpt \\
        --adaface-repo /path/to/AdaFace \\
        --output triton-models/face-recognition-adaface-ir50/1/model.onnx

Input contract (matches ``face-recognition-adaface-ir50/config.pbtxt`` and person-id profile
``adaface_ir50_webface4m``):
- Input tensor ``input``: FP32 ``[1, 3, 112, 112]``, BGR channel order, ``(bgr - 127.5) / 127.5``.
- Output tensor ``embedding``: FP32 ``[1, 512]`` (person-id L2-normalizes client-side).

Licence note: AdaFace repository code is MIT, while official pretrained weights trained on
WebFace4M / MS1MV2 inherit the non-commercial research terms of their training datasets. Model
weights are never committed to Git.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_and_verify(
    *,
    checkpoint: Path,
    adaface_repo: Path,
    output: Path,
    architecture: str = "ir_50",
    opset: int = 17,
) -> None:
    import onnxruntime as ort
    import torch

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not adaface_repo.is_dir():
        raise FileNotFoundError(f"AdaFace source checkout not found: {adaface_repo}")

    sys.path.insert(0, str(adaface_repo))
    import net  # type: ignore[import-not-found]

    model = net.build_model(architecture)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    sd = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )
    cleaned = {
        k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")
    }
    model.load_state_dict(cleaned or sd)
    model.eval()

    class _EmbeddingOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.inner(x)
            return out[0] if isinstance(out, tuple) else out

    wrapper = _EmbeddingOnly(model).eval()
    rng = np.random.default_rng(42)
    sample = rng.uniform(-1.0, 1.0, size=(1, 3, 112, 112)).astype(np.float32)
    sample_t = torch.from_numpy(sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ref = wrapper(sample_t).cpu().numpy().astype(np.float32)
        torch.onnx.export(
            wrapper,
            sample_t,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["embedding"],
        )

    sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    (got,) = sess.run(["embedding"], {"input": sample})
    got = np.asarray(got, dtype=np.float32)
    if got.shape != (1, 512):
        raise RuntimeError(
            f"Unexpected ONNX output shape {got.shape}; expected (1, 512)"
        )

    ref_u = ref[0] / np.linalg.norm(ref[0])
    got_u = got[0] / np.linalg.norm(got[0])
    cosine = float(np.dot(ref_u, got_u))
    max_abs = float(np.max(np.abs(ref - got)))
    if cosine < 0.9999:
        raise RuntimeError(
            f"ONNX parity check failed: cosine={cosine:.6f}, max_abs={max_abs:.6f}"
        )

    digest = _sha256(output)
    print(f"Exported: {output}")
    print(f"SHA-256:  {digest}")
    print(f"Parity:   cosine={cosine:.6f} max_abs={max_abs:.6e} shape={got.shape}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--adaface-repo", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("triton-models/face-recognition-adaface-ir50/1/model.onnx"),
    )
    parser.add_argument("--architecture", default="ir_50", choices=("ir_50", "ir_101"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    export_and_verify(
        checkpoint=args.checkpoint,
        adaface_repo=args.adaface_repo,
        output=args.output,
        architecture=args.architecture,
        opset=args.opset,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
