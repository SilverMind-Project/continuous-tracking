"""Export SOLIDER-REID Swin-Tiny to ONNX.

Loads a Swin-Tiny backbone with SOLIDER semantic controller from the
SOLIDER-REID repository, applies MSMT17 fine-tuned weights, adds the BN
bottleneck and L2 normalisation, and exports to ONNX for Triton.

Input:  [batch, 3, 384, 128]  FP32  (H×W = 384×128)
Output: [batch, 768]          FP32  L2-normalised

The input size matches the SOLIDER-REID training configuration
(configs/MSMT17/swin_tiny.yml: INPUT.SIZE_TRAIN = [384, 128]).

Requirements (run from the repo root):
    uv run --with torch --with onnx --with yacs \\
        python triton-models/scripts/export_reid.py

The SOLIDER-REID repository must be available at ../../SOLIDER-REID
relative to this script (i.e. alongside the continuous-tracking repo).
If it is elsewhere, set the SOLIDER_REID_PATH environment variable.

Download the MSMT17-fine-tuned Swin-Tiny weights from:
    https://drive.google.com/file/d/10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr/view

Place the checkpoint at:
    triton-models/reid-solider/solider_swin_tiny_msmt17.pth

Output:
    triton-models/reid-solider/1/model.onnx  (FP32, ~110 MB)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


def _find_solider_reid() -> Path:
    """Locate the SOLIDER-REID repository."""
    env_path = os.environ.get("SOLIDER_REID_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"SOLIDER_REID_PATH={env_path} is not a directory")

    # Default: sibling of continuous-tracking repo
    script_dir = Path(__file__).resolve().parent  # triton-models/scripts/
    ct_root = script_dir.parent.parent  # continuous-tracking/
    default = ct_root.parent / "SOLIDER-REID"
    if default.is_dir():
        return default
    # Fallback: /tmp clone
    tmp = Path("/tmp/SOLIDER-REID")
    if tmp.is_dir():
        return tmp
    raise FileNotFoundError(
        "SOLIDER-REID repository not found. "
        "Clone it: git clone https://github.com/tinyvision/SOLIDER-REID.git "
        "or set SOLIDER_REID_PATH."
    )


def _load_checkpoint(model, checkpoint_path: str) -> None:
    """Load a SOLIDER-REID checkpoint without mmcv dependency.

    The checkpoint dict has key 'state_dict' containing model weights.
    Keys may be prefixed with 'module.' from DDP training.
    """
    ckpt: dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_state = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, tensor in state_dict.items():
        clean = key.replace("module.", "")
        if clean in model_state:
            if model_state[clean].shape == tensor.shape:
                loaded[clean] = tensor
            else:
                skipped.append(f"{key} (shape mismatch: {tensor.shape} vs {model_state[clean].shape})")
        elif "classifier" in clean:
            skipped.append(f"{key} (classifier — excluded)")
        else:
            skipped.append(f"{key} (not in model)")

    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)

    print(f"Loaded {len(loaded)} parameters, skipped {len(skipped)}")
    if skipped:
        for s in skipped[:10]:
            print(f"  skip: {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")


class _ReIDInferenceWrapper(torch.nn.Module):
    """Wraps the SOLIDER-REID model for inference-only ONNX export.

    Forward path: backbone → BN bottleneck → L2-normalise.
    Matches the eval-mode path in ``build_transformer.forward()`` with
    ``neck_feat='after'`` and ``TEST.FEAT_NORM='yes'``.
    """

    def __init__(self, model):
        super().__init__()
        self.base = model.base
        self.bottleneck = model.bottleneck

    def forward(self, x):
        # Build semantic_weight on the input device to avoid the hardcoded
        # .cuda() call inside SwinTransformer.forward().
        sw = torch.full((x.shape[0], 2), self.base.semantic_weight, device=x.device)
        sw[:, 1] = 1.0 - self.base.semantic_weight
        # self.base(x) returns (global_feat, featmaps) in eval mode
        global_feat, _featmaps = self.base(x, semantic_weight=sw)
        feat = self.bottleneck(global_feat)
        # L2-normalise (equivalent to TEST.FEAT_NORM='yes'), then reshape to
        # pin the 768-dim output so the ONNX graph has a concrete last dimension
        # rather than an unresolved symbolic one.
        normed = torch.nn.functional.normalize(feat, p=2, dim=1)
        return normed.view(normed.size(0), 768)


def export(checkpoint_path: str, out: Path) -> None:
    from yacs.config import CfgNode as CN

    solider_reid_path = _find_solider_reid()
    print(f"SOLIDER-REID at: {solider_reid_path}")
    sys.path.insert(0, str(solider_reid_path))

    from model.make_model import build_transformer
    from model.backbones.swin_transformer import (
        swin_tiny_patch4_window7_224,
        swin_small_patch4_window7_224,
        swin_base_patch4_window7_224,
    )

    # Build a minimal config matching configs/MSMT17/swin_tiny.yml
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.LAST_STRIDE = 1
    cfg.MODEL.NAME = "transformer"
    cfg.MODEL.TRANSFORMER_TYPE = "swin_tiny_patch4_window7_224"
    cfg.MODEL.PRETRAIN_PATH = ""  # we load weights separately
    cfg.MODEL.PRETRAIN_CHOICE = "self"
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.NECK = "bnneck"
    cfg.MODEL.NECK_FEAT = "after"
    cfg.MODEL.REDUCE_FEAT_DIM = False
    cfg.MODEL.FEAT_DIM = 512  # unused when REDUCE_FEAT_DIM=False
    cfg.MODEL.DROPOUT_RATE = 0.0
    cfg.MODEL.DROP_PATH = 0.1
    cfg.MODEL.DROP_OUT = 0.0
    cfg.MODEL.ATT_DROP_RATE = 0.0
    cfg.MODEL.JPM = False
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    cfg.MODEL.SEMANTIC_WEIGHT = 1.0
    cfg.MODEL.ID_LOSS_TYPE = "softmax"

    cfg.INPUT = CN()
    cfg.INPUT.SIZE_TRAIN = [384, 128]
    cfg.INPUT.SIZE_TEST = [384, 128]

    cfg.TEST = CN()
    cfg.TEST.NECK_FEAT = "after"
    cfg.TEST.FEAT_NORM = "yes"

    cfg.SOLVER = CN()
    cfg.SOLVER.COSINE_SCALE = 30
    cfg.SOLVER.COSINE_MARGIN = 0.5

    factory = {
        "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
        "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
        "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    }

    print("Building SOLIDER-REID Swin-Tiny model (384×128 input)...")
    model = build_transformer(
        num_classes=1041,  # MSMT17 has 1041 identities
        camera_num=0,
        view_num=0,
        cfg=cfg,
        factory=factory,
        semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT,
    )
    model.eval()

    # Load MSMT17 fine-tuned weights.
    print(f"Loading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Download from:\n"
            "  https://drive.google.com/file/d/10YLhMbwvmxZl3gTVo2BN_828SKZHdCjr/view\n"
            "Place it at triton-models/reid-solider/solider_swin_tiny_msmt17.pth"
        )
    _load_checkpoint(model, checkpoint_path)

    # Wrap for inference-only export (backbone + BN neck + L2 norm).
    wrapper = _ReIDInferenceWrapper(model)
    wrapper.eval()

    # Verify output shape and normalisation.
    dummy = torch.zeros(1, 3, 384, 128)
    with torch.no_grad():
        output = wrapper(dummy)
    assert output.shape == (1, 768), (
        f"Unexpected output shape: {output.shape}, expected (1, 768)"
    )
    norm = torch.norm(output, p=2, dim=1).item()
    assert abs(norm - 1.0) < 0.01, (
        f"Output not L2-normalised: norm={norm:.6f}, expected ~1.0"
    )
    print(f"  Verified: shape={tuple(output.shape)}, L2 norm={norm:.6f}")

    # Export to ONNX.
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX → {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("Input:  input   [batch, 3, 384, 128]")
    print("Output: output  [batch, 768]  L2-normalised")

    # Verify ONNX and fix output shape if needed.
    import onnx
    from onnx import helper, TensorProto

    m = onnx.load(str(out))
    for i in m.graph.input:
        dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
        print(f"  Verify input:  {i.name} {dims}")
    for o in m.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  Verify output: {o.name} {dims}")

    # Triton requires explicit output dimensions when max_batch_size > 0.
    # Patch the output shape to [-1, 768] if the 768 dim is symbolic.
    output = m.graph.output[0]
    output_dim0 = output.type.tensor_type.shape.dim[1]
    if output_dim0.dim_value == 0 and not output_dim0.HasField("dim_param"):
        output_dim0.dim_value = 768
        onnx.save(m, str(out))
        print(f"  Patched output dim to: [batch, 768]")

    # Final sanity check with onnxruntime.
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    test_input = np.zeros((1, 3, 384, 128), dtype=np.float32)
    test_out = session.run(None, {"input": test_input})
    assert test_out[0].shape == (1, 768), f"Runtime shape mismatch: {test_out[0].shape}"
    assert abs(float(np.linalg.norm(test_out[0])) - 1.0) < 0.01, "Runtime norm mismatch"
    print(f"  onnxruntime OK: shape={test_out[0].shape}, norm={np.linalg.norm(test_out[0]):.6f}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent  # triton-models/
    checkpoint = repo_root / "reid-solider" / "solider_swin_tiny_msmt17.pth"
    out = repo_root / "reid-solider" / "1" / "model.onnx"
    export(str(checkpoint), out)


if __name__ == "__main__":
    main()
