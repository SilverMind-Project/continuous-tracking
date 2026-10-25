# Custom Triton Inference Server image for CTS.
# Extends the NVIDIA Triton image.
# florence-2 uses BLS to call native ONNX Runtime sub-models (INT8 QDQ).
# torch is kept for the DLPack bridge fallback in _tensor_to_numpy.
#
# torch is used for the DLPack bridge fallback in _tensor_to_numpy.
# The base image does not ship torch in the standard Python path; install it
# pinned to the version that matches the container's bundled CUDA runtime.
#
# tritonserver:26.04-py3 ships CUDA 13.0; use the cu130 PyTorch wheel index.
# To find the right version when bumping FROM:
#   pip index versions torch --index-url https://download.pytorch.org/whl/cu<MAJOR><MINOR>
# Update both the index URL suffix and the version pin together.

FROM nvcr.io/nvidia/tritonserver:26.07-py3

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.12.0
