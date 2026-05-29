"""Frame processing pipeline.

Wires together:
1. Transport (Redis Streams consumer)
2. Inference (Triton person detector)
3. Tracking (BoT-SORT per-camera tracker)
4. Tracklet management (lifecycle, gallery append)
5. Persistence (repository layer)
6. Event emission (Redis Streams producer)

The pipeline runs as a background task in the FastAPI lifespan.
"""

from __future__ import annotations

from .frame_pipeline import FrameProcessingPipeline

__all__ = ["FrameProcessingPipeline"]
