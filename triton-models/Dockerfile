# Custom Triton Inference Server image for CTS.
# Extends the NVIDIA Triton image.
# florence-2 uses BLS to call native ONNX Runtime sub-models (INT8 QDQ).
# torch is kept for the DLPack bridge fallback in _tensor_to_numpy.

FROM nvcr.io/nvidia/tritonserver:26.04-py3

RUN pip install --no-cache-dir \
    torch
