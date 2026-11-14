"""Triton Inference Server gRPC client with Prometheus latency instrumentation.

The underlying transport lives in ``triton_shared.client``; this wrapper
observes ``cts_triton_inference_latency_ms{model}`` on every ``infer()`` call.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from triton_shared.client import TritonClientProtocol
from triton_shared.client import TritonGrpcClient as _BaseTritonGrpcClient

from ..observability import metrics as _metrics


class TritonGrpcClient(_BaseTritonGrpcClient):
    """Triton gRPC client that records per-model inference latency."""

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, npt.NDArray[np.float32]]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        start = time.monotonic()
        try:
            return await super().infer(model_name, inputs, output_names)
        finally:
            latency_ms = (time.monotonic() - start) * 1000.0
            _metrics.metrics.triton_inference_latency_ms.labels(model=model_name).observe(
                latency_ms
            )


__all__ = [
    "TritonClientProtocol",
    "TritonGrpcClient",
]
