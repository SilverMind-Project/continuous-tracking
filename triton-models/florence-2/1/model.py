"""Florence-2-large autoregressive scene description — Triton Python backend.

Uses Business Logic Scripting (BLS) to call native ONNX Runtime backend models:

  1. florence-2-vision   — pixel_values → image_features
  2. florence-2-embed    — input_ids → inputs_embeds
  3. florence-2-encoder  — attention_mask + inputs_embeds → last_hidden_state
  4. florence-2-decoder  — autoregressive decode with KV cache

All ONNX models are INT8 QDQ-quantized (from onnx-community/Florence-2-large)
and run on GPU via Triton's native onnxruntime backend (CUDA EP or OpenVINO EP).
The Python model handles only the generation loop orchestration.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]

# ---------------------------------------------------------------------------
# Model architecture constants (Florence-2-large)
# ---------------------------------------------------------------------------
_NUM_LAYERS = 12
_NUM_HEADS = 16
_HEAD_DIM = 64

_KV_KINDS = ("decoder.key", "decoder.value", "encoder.key", "encoder.value")


def _past_kv_name(layer: int, kind: str) -> str:
    return f"past_key_values.{layer}.{kind}"


def _present_kv_name(layer: int, kind: str) -> str:
    return f"present.{layer}.{kind}"


_PAST_KV_INPUTS = [_past_kv_name(i, k) for i in range(_NUM_LAYERS) for k in _KV_KINDS]
_PRESENT_KV_OUTPUTS = [_present_kv_name(i, k) for i in range(_NUM_LAYERS) for k in _KV_KINDS]
_DECODER_REQUESTED_OUTPUTS = ["logits"] + _PRESENT_KV_OUTPUTS


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert a pb_utils Tensor to a CPU numpy array.

    BLS child-model responses may store tensors on GPU.  ``.as_numpy()``
    alone can fail in that case — we fall back to the PyTorch DLPack bridge.
    """
    try:
        return tensor.as_numpy()
    except Exception:
        pass

    import torch

    return torch.from_dlpack(tensor).cpu().numpy()


class TritonPythonModel:
    """Florence-2-large Python orchestrator backed by native ONNX Runtime models."""

    def initialize(self, args: dict[str, Any]) -> None:
        model_path = os.path.join(args["model_repository"], str(args["model_version"]), "")

        gen_config_path = os.path.join(model_path, "generation_config.json")
        self._eos_token_id = 2
        self._max_new_tokens = 1024
        if os.path.exists(gen_config_path):
            with open(gen_config_path) as f:
                gen_cfg = json.load(f)
            self._eos_token_id = gen_cfg.get("eos_token_id", 2)
            self._max_new_tokens = gen_cfg.get("max_new_tokens", 1024)

        print(f"Florence-2 BLS orchestrator initialized")
        print(f"  EOS token:     {self._eos_token_id}")
        print(f"  Max new tokens: {self._max_new_tokens}")
        print(f"  Decoder layers: {_NUM_LAYERS}")
        print(f"  Sub-models: florence-2-vision, florence-2-embed, "
              f"florence-2-encoder, florence-2-decoder")

    def execute(self, requests: list[Any]) -> list[Any]:
        responses: list[Any] = []
        for request in requests:
            responses.append(self._process_request(request))
        return responses

    def finalize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    def _process_request(self, request: Any) -> Any:
        pixel_values = pb_utils.get_input_tensor_by_name(request, "pixel_values")
        input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")

        if pixel_values is None or input_ids is None:
            return pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError("Missing pixel_values or input_ids"),
            )

        pv = pixel_values.as_numpy().astype(np.float32)
        ids = input_ids.as_numpy().astype(np.int64)

        try:
            output_ids = self._generate(pv, ids)
        except Exception as exc:
            return pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError(f"Generation failed: {exc}"),
            )

        return pb_utils.InferenceResponse(
            output_tensors=[pb_utils.Tensor("output_ids", output_ids)]
        )

    # ------------------------------------------------------------------
    # Generation pipeline
    # ------------------------------------------------------------------

    def _generate(
        self,
        pixel_values: np.ndarray,  # (1, 3, H, W)  FP32
        input_ids: np.ndarray,     # (1, T)        INT64
    ) -> np.ndarray:
        """Run the autoregressive generation loop via BLS.

        Returns:
            (1, output_len) int64 array of generated token IDs (includes input_ids).
        """
        # 1. Vision encode.
        image_features = self._call_vision(pixel_values)       # (1, N_vis, 1024)

        # 2. Embed prompt tokens.
        prompt_embeds = self._call_embed(input_ids)            # (1, T, 1024)

        # 3. Concatenate visual + text embeddings.
        combined_embeds = np.concatenate(
            [image_features, prompt_embeds], axis=1
        )                                                      # (1, N_vis+T, 1024)
        enc_seq_len = combined_embeds.shape[1]
        attention_mask = np.ones((1, enc_seq_len), dtype=np.int64)

        # 4. Encode (text attends to visual features via the merged embeddings).
        encoder_hidden_states = self._call_encoder(
            attention_mask, combined_embeds
        )                                                      # (1, N_vis+T, 1024)

        # 5. Autoregressive decode loop.
        generated_ids: list[int] = list(input_ids[0])
        current_token = np.array([[generated_ids[-1]]], dtype=np.int64)
        past_kv = self._empty_kv_cache()

        for _ in range(self._max_new_tokens):
            token_embed = self._call_embed(current_token)     # (1, 1, 1024)

            logits, past_kv = self._call_decoder(
                encoder_attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                inputs_embeds=token_embed,
                past_kv=past_kv,
            )

            next_token = int(np.argmax(logits[0, -1]))
            generated_ids.append(next_token)
            current_token = np.array([[next_token]], dtype=np.int64)

            if next_token == self._eos_token_id:
                break

        return np.array([generated_ids], dtype=np.int64)

    # ------------------------------------------------------------------
    # BLS calls to native ONNX Runtime models
    # ------------------------------------------------------------------

    def _call_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        """florence-2-vision: pixel_values → image_features."""
        request = pb_utils.InferenceRequest(
            model_name="florence-2-vision",
            requested_output_names=["image_features"],
            inputs=[pb_utils.Tensor("pixel_values", pixel_values)],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Vision encode failed: {response.error().message()}")
        tensor = pb_utils.get_output_tensor_by_name(response, "image_features")
        return _tensor_to_numpy(tensor).astype(np.float32)

    def _call_embed(self, input_ids: np.ndarray) -> np.ndarray:
        """florence-2-embed: input_ids → inputs_embeds."""
        request = pb_utils.InferenceRequest(
            model_name="florence-2-embed",
            requested_output_names=["inputs_embeds"],
            inputs=[pb_utils.Tensor("input_ids", input_ids.astype(np.int64))],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Token embedding failed: {response.error().message()}")
        tensor = pb_utils.get_output_tensor_by_name(response, "inputs_embeds")
        return _tensor_to_numpy(tensor).astype(np.float32)

    def _call_encoder(
        self,
        attention_mask: np.ndarray,
        inputs_embeds: np.ndarray,
    ) -> np.ndarray:
        """florence-2-encoder: attention_mask + inputs_embeds → last_hidden_state."""
        request = pb_utils.InferenceRequest(
            model_name="florence-2-encoder",
            requested_output_names=["last_hidden_state"],
            inputs=[
                pb_utils.Tensor("attention_mask", attention_mask.astype(np.int64)),
                pb_utils.Tensor("inputs_embeds", inputs_embeds.astype(np.float32)),
            ],
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Encoder failed: {response.error().message()}")
        tensor = pb_utils.get_output_tensor_by_name(response, "last_hidden_state")
        return _tensor_to_numpy(tensor).astype(np.float32)

    def _call_decoder(
        self,
        encoder_attention_mask: np.ndarray,
        encoder_hidden_states: np.ndarray,
        inputs_embeds: np.ndarray,
        past_kv: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """florence-2-decoder: full decode step with KV cache.

        Returns (logits, new_past_kv).
        """
        inputs = [
            pb_utils.Tensor("encoder_attention_mask", encoder_attention_mask.astype(np.int64)),
            pb_utils.Tensor("encoder_hidden_states", encoder_hidden_states.astype(np.float32)),
            pb_utils.Tensor("inputs_embeds", inputs_embeds.astype(np.float32)),
        ]
        for name in _PAST_KV_INPUTS:
            inputs.append(pb_utils.Tensor(name, past_kv[name].astype(np.float32)))
        inputs.append(
            pb_utils.Tensor("use_cache_branch", np.array([True], dtype=np.bool_))
        )

        request = pb_utils.InferenceRequest(
            model_name="florence-2-decoder",
            requested_output_names=_DECODER_REQUESTED_OUTPUTS,
            inputs=inputs,
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(f"Decoder failed: {response.error().message()}")

        logits_tensor = pb_utils.get_output_tensor_by_name(response, "logits")
        logits = _tensor_to_numpy(logits_tensor).astype(np.float32)

        new_past_kv: dict[str, np.ndarray] = {}
        for present_name in _PRESENT_KV_OUTPUTS:
            tensor = pb_utils.get_output_tensor_by_name(response, present_name)
            if tensor is None:
                raise RuntimeError(f"Missing decoder output: {present_name}")
            # present.0.decoder.key → past_key_values.0.decoder.key
            _, layer_kind = present_name.split(".", 1)
            input_name = f"past_key_values.{layer_kind}"
            new_past_kv[input_name] = _tensor_to_numpy(tensor).astype(np.float32)

        return logits, new_past_kv

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _empty_kv_cache(self) -> dict[str, np.ndarray]:
        """Create empty (seq_len=0) KV cache tensors for the first decoder step."""
        empty_self = np.zeros((1, _NUM_HEADS, 0, _HEAD_DIM), dtype=np.float32)
        empty_cross = np.zeros((1, _NUM_HEADS, 0, _HEAD_DIM), dtype=np.float32)
        kv: dict[str, np.ndarray] = {}
        for name in _PAST_KV_INPUTS:
            kv[name] = empty_cross.copy() if ".encoder." in name else empty_self.copy()
        return kv
