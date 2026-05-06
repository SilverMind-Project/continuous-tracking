"""Florence-2-large autoregressive scene description — Triton Python backend.

Orchestrates the ONNX models from onnx-community/Florence-2-large:

  1. vision_encoder_int8.onnx     image → visual features
  2. embed_tokens_int8.onnx        token IDs → embeddings
  3. encoder_model_int8.onnx       text + visual → encoder hidden states
  4. decoder_model_merged_int8.onnx  autoregressive decode loop

GPU support: NVIDIA (CUDAExecutionProvider) or Intel Arc (OpenVINOExecutionProvider).

The backend auto-detects ONNX I/O names at load time so it works with
different export formats without code changes.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# Triton Python backend API — available inside the Triton container.
# pyright: reportMissingImports=false
import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]


class TritonPythonModel:
    """Florence-2-large Python backend model for Triton Inference Server."""

    def initialize(self, args: dict[str, Any]) -> None:
        """Load ONNX models and tokenizer at startup."""
        model_dir = os.path.dirname(__file__)
        model_dir = args.get("model_repository", model_dir)
        model_path = os.path.join(
            args["model_repository"], str(args["model_version"]), ""
        )

        self._max_new_tokens = 1024

        # Detect execution provider from instance group kind.
        # Intel configs set the OpenVINO accelerator; NVIDIA uses CUDA.
        self._providers = self._resolve_providers(args)

        # Load ONNX sessions.
        import onnxruntime as ort

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._vision = ort.InferenceSession(
            os.path.join(model_path, "vision_encoder_int8.onnx"),
            sess_options=sess_opts,
            providers=self._providers,
        )
        self._embed = ort.InferenceSession(
            os.path.join(model_path, "embed_tokens_int8.onnx"),
            sess_options=sess_opts,
            providers=self._providers,
        )
        self._encoder = ort.InferenceSession(
            os.path.join(model_path, "encoder_model_int8.onnx"),
            sess_options=sess_opts,
            providers=self._providers,
        )
        self._decoder = ort.InferenceSession(
            os.path.join(model_path, "decoder_model_merged_int8.onnx"),
            sess_options=sess_opts,
            providers=self._providers,
        )

        # Discover ONNX I/O names.
        self._vision_input = self._vision.get_inputs()[0].name
        self._vision_output = self._vision.get_outputs()[0].name

        self._embed_input = self._embed.get_inputs()[0].name
        self._embed_output = self._embed.get_outputs()[0].name

        # Encoder: map inputs by name.
        enc_inputs = {i.name: i for i in self._encoder.get_inputs()}
        self._enc_embeds_input = self._find_input(enc_inputs, ["inputs_embeds", "input_ids"])
        self._enc_visual_input = self._find_input(
            enc_inputs, ["visual_features", "encoder_hidden_states", "pixel_values"]
        )
        self._enc_output = self._encoder.get_outputs()[0].name

        # Decoder: map inputs by name.
        dec_inputs = {i.name: i for i in self._decoder.get_inputs()}
        self._dec_input_ids = self._find_input(dec_inputs, ["input_ids", "decoder_input_ids"])
        self._dec_enc_hidden = self._find_input(
            dec_inputs, ["encoder_hidden_states", "encoder_attention_mask"]
        )
        self._dec_past = self._find_input(
            dec_inputs,
            [
                "past_key_values",
                "past_key_values.0.key",
                "past_key_values.0.decoder.key",
            ],
        )
        dec_outputs = {o.name: o for o in self._decoder.get_outputs()}
        self._dec_logits = self._find_output(dec_outputs, ["logits"])
        self._dec_present = self._find_output(
            dec_outputs,
            [
                "present_key_values",
                "present_key_values.0.key",
                "present_key_values.0.decoder.key",
            ],
        )

        # Load generation config for EOS token ID.
        gen_config_path = os.path.join(model_path, "generation_config.json")
        self._eos_token_id = 2  # Florence-2 default
        if os.path.exists(gen_config_path):
            with open(gen_config_path) as f:
                gen_cfg = json.load(f)
            self._eos_token_id = gen_cfg.get("eos_token_id", 2)
            self._max_new_tokens = gen_cfg.get("max_new_tokens", 1024)

        print(f"Florence-2 loaded with providers: {self._providers}")
        print(f"  Vision input:  {self._vision_input} -> {self._vision_output}")
        print(f"  Embed input:   {self._embed_input} -> {self._embed_output}")
        print(f"  Encoder:       embeds={self._enc_embeds_input}, "
              f"visual={self._enc_visual_input} -> {self._enc_output}")
        print(f"  Decoder:       ids={self._dec_input_ids}, "
              f"enc_hidden={self._dec_enc_hidden}, "
              f"past={self._dec_past} -> {self._dec_logits}, {self._dec_present}")
        print(f"  EOS token:     {self._eos_token_id}")
        print(f"  Max new tokens: {self._max_new_tokens}")

    def execute(self, requests: list[Any]) -> list[Any]:
        """Run the Florence-2 generation pipeline for each request."""
        responses: list[Any] = []
        for request in requests:
            responses.append(self._process_request(request))
        return responses

    def finalize(self) -> None:
        pass  # ONNX sessions auto-cleanup

    # ------------------------------------------------------------------
    # Internal: generation pipeline
    # ------------------------------------------------------------------

    def _process_request(self, request: Any) -> Any:
        """Run full Florence-2 generation for a single request."""
        pixel_values = pb_utils.get_input_tensor_by_name(request, "pixel_values")
        input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")

        if pixel_values is None or input_ids is None:
            return pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError("Missing pixel_values or input_ids"),
            )

        pv = pb_utils.to_numpy(pixel_values).astype(np.float32)  # (1, 3, H, W)
        ids = pb_utils.to_numpy(input_ids).astype(np.int64)  # (1, seq_len)

        try:
            output_ids = self._generate(pv, ids)
        except Exception as exc:
            return pb_utils.InferenceResponse(
                output_tensors=[],
                error=pb_utils.TritonError(f"Generation failed: {exc}"),
            )

        out_tensor = pb_utils.Tensor("output_ids", output_ids)
        return pb_utils.InferenceResponse(output_tensors=[out_tensor])

    def _generate(
        self,
        pixel_values: np.ndarray,
        input_ids: np.ndarray,
    ) -> np.ndarray:
        """Run the autoregressive generation loop.

        Returns:
            (1, output_len) int64 array of generated token IDs (includes input_ids).
        """
        # 1. Vision encode.
        vis_out = self._vision.run(
            [self._vision_output], {self._vision_input: pixel_values}
        )
        visual_features = vis_out[0]  # (1, num_visual_tokens, hidden)

        # 2. Embed tokens.
        emb_out = self._embed.run(
            [self._embed_output], {self._embed_input: input_ids}
        )
        inputs_embeds = emb_out[0]  # (1, seq_len, hidden)

        # 3. Encode (cross-attention: text + visual features).
        enc_inputs = {
            self._enc_embeds_input: inputs_embeds,
            self._enc_visual_input: visual_features,
        }
        enc_out = self._encoder.run([self._enc_output], enc_inputs)
        encoder_hidden_states = enc_out[0]  # (1, src_len, hidden)

        # 4. Autoregressive decode loop.
        generated_ids = list(input_ids[0])
        past_key_values = None
        current_token = np.array([[generated_ids[-1]]], dtype=np.int64)

        for _ in range(self._max_new_tokens):
            dec_inputs: dict[str, np.ndarray] = {
                self._dec_input_ids: current_token,
                self._dec_enc_hidden: encoder_hidden_states,
            }
            if self._dec_past and past_key_values is not None:
                dec_inputs[self._dec_past] = past_key_values

            dec_outs = self._decoder.run(
                [self._dec_logits, self._dec_present], dec_inputs
            )
            logits = dec_outs[0]  # (1, 1, vocab_size)
            past_key_values = dec_outs[1] if len(dec_outs) > 1 else None

            next_token = int(np.argmax(logits[0, -1]))
            generated_ids.append(next_token)
            current_token = np.array([[next_token]], dtype=np.int64)

            if next_token == self._eos_token_id:
                break

        return np.array([generated_ids], dtype=np.int64)

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_input(inputs: dict[str, Any], candidates: list[str]) -> str | None:
        """Return the first candidate name that exists in the ONNX inputs."""
        for name in candidates:
            if name in inputs:
                return name
        return list(inputs.keys())[0] if inputs else None

    @staticmethod
    def _find_output(outputs: dict[str, Any], candidates: list[str]) -> str | None:
        """Return the first candidate name that exists in the ONNX outputs."""
        for name in candidates:
            if name in outputs:
                return name
        return list(outputs.keys())[0] if outputs else None

    @staticmethod
    def _resolve_providers(args: dict[str, Any]) -> list[str]:
        """Resolve ONNX Runtime execution providers from model config."""
        # Check if the instance group has an OpenVINO accelerator (Intel Arc).
        try:
            instance_groups = args.get("model_config", {}).get("instance_group", [])
            for group in instance_groups:
                accelerators = group.get("optimization", {}).get(
                    "execution_accelerators", {}
                ).get("gpu_execution_accelerator", [])
                for acc in accelerators:
                    if acc.get("name") == "openvino":
                        return [
                            ("OpenVINOExecutionProvider", {"device_type": "GPU"}),
                            "CPUExecutionProvider",
                        ]
        except (KeyError, TypeError, AttributeError):
            pass

        # Default: CUDA, then CPU fallback.
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
