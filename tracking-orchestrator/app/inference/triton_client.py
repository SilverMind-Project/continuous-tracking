"""Triton Inference Server gRPC client (re-exported from triton_shared).

The canonical implementations live in ``triton_shared.client``.
This module re-exports for backwards compatibility within CTS.
"""

from __future__ import annotations

from triton_shared.client import TritonClientProtocol, TritonGrpcClient

__all__ = [
    "TritonClientProtocol",
    "TritonGrpcClient",
]
