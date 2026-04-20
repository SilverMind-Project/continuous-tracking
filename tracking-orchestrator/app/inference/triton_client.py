"""Async Triton Inference Server gRPC client.

Design:
  - TritonClientProtocol: structural subtyping interface used by inference
    wrappers.  Tests inject a mock that satisfies this Protocol without
    importing tritonclient.
  - TritonGrpcClient: concrete async gRPC implementation backed by
    tritonclient.grpc.aio (install with `uv sync --extra triton`).

tritonclient has no type stubs; the library import is deferred to the
first call to keep the module importable without the extra installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class TritonClientProtocol(Protocol):
    """Minimal Triton client surface used by all inference wrappers."""

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, npt.NDArray[np.float32]]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run a batched inference request and return named output tensors."""
        ...

    async def is_model_ready(self, model_name: str) -> bool:
        """Return True if Triton reports the model as ready."""
        ...


class TritonGrpcClient:
    """Async gRPC Triton client.  Use as an async context manager.

    Example::

        async with TritonGrpcClient("localhost:8001") as client:
            outputs = await client.infer("person-detector", inputs, ["output0"])
    """

    def __init__(self, url: str, timeout_ms: int = 150) -> None:
        self._url = url
        self._timeout_ms = timeout_ms
        self._client: Any = None  # tritonclient.grpc.aio.InferenceServerClient

    async def __aenter__(self) -> TritonGrpcClient:
        self._client = self._make_client()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Public interface (satisfies TritonClientProtocol)
    # ------------------------------------------------------------------

    async def infer(
        self,
        model_name: str,
        inputs: list[tuple[str, npt.NDArray[np.float32]]],
        output_names: list[str],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Run a batched inference request on Triton."""
        client = self._require_client()
        tc = self._triton_common()

        triton_inputs: list[Any] = []
        for name, array in inputs:
            arr = np.ascontiguousarray(array.astype(np.float32))
            inp: Any = tc.InferInput(name, list(arr.shape), "FP32")
            inp.set_data_from_numpy(arr)
            triton_inputs.append(inp)

        triton_outputs: list[Any] = [tc.InferRequestedOutput(n) for n in output_names]

        result: Any = await client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            client_timeout=self._timeout_ms / 1000.0,
        )

        return {name: np.asarray(result.as_numpy(name), dtype=np.float32) for name in output_names}

    async def is_model_ready(self, model_name: str) -> bool:
        client = self._require_client()
        return bool(await client.is_model_ready(model_name))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> Any:
        try:
            import tritonclient.grpc.aio as _aio
        except ImportError as exc:
            raise ImportError("tritonclient is not installed. Run: uv sync --extra triton") from exc
        return _aio.InferenceServerClient(url=self._url, verbose=False)

    @staticmethod
    def _triton_common() -> Any:
        try:
            import tritonclient.grpc as _tc
        except ImportError as exc:
            raise ImportError("tritonclient is not installed. Run: uv sync --extra triton") from exc
        return _tc

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "TritonGrpcClient is not open. Use 'async with TritonGrpcClient(...)'."
            )
        return self._client
