"""SOLIDER-REID appearance embedder backed by Triton Inference Server.

Input: person crop as RGB uint8 numpy array (H, W, 3).
Preprocessing: resize to 256x128 (HxW), ImageNet normalise, CHW layout.

Triton model "reid-solider" output tensor "output" shape [batch, 768]:
  768-dim L2-normalised appearance embedding.
  The ONNX model includes L2 normalisation — do NOT re-normalise.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.inference.schemas import EMBEDDING_DIM, Embedding
from app.inference.triton_client import TritonClientProtocol

_MODEL_NAME = "reid-solider"
_CROP_H = 256
_CROP_W = 128

# ImageNet statistics for ReID normalisation
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _preprocess(crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Resize crop to 256x128, ImageNet-normalise, return CHW float32."""
    h, w = crop.shape[:2]
    scale_h = h / _CROP_H
    scale_w = w / _CROP_W
    row_idx = np.floor(np.arange(_CROP_H) * scale_h).astype(np.intp)
    col_idx = np.floor(np.arange(_CROP_W) * scale_w).astype(np.intp)
    resized = crop[row_idx[:, None], col_idx[None, :]]  # (_CROP_H, _CROP_W, 3)

    chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))  # CHW
    return (chw - _MEAN) / _STD  # normalised CHW float32


class ReidEmbedder:
    """Async appearance embedder using SOLIDER-REID on Triton."""

    def __init__(
        self,
        client: TritonClientProtocol,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._client = client
        self._model_name = model_name

    async def embed_batch(
        self,
        crops: list[npt.NDArray[np.uint8]],
    ) -> list[Embedding]:
        """Embed a batch of RGB person crops.  Returns L2-normalised embeddings."""
        if not crops:
            return []

        batch = np.stack([_preprocess(c) for c in crops])  # (N, 3, 256, 128)
        outputs = await self._client.infer(
            model_name=self._model_name,
            inputs=[("input", batch)],
            output_names=["output"],
        )
        embeddings: npt.NDArray[np.float32] = outputs["output"]  # (N, 768)

        if embeddings.shape[-1] != EMBEDDING_DIM:
            raise ValueError(f"Expected {EMBEDDING_DIM}-dim embedding, got {embeddings.shape[-1]}")

        return [embeddings[i] for i in range(len(crops))]

    async def embed(self, crop: npt.NDArray[np.uint8]) -> Embedding:
        """Embed a single RGB person crop."""
        return (await self.embed_batch([crop]))[0]
