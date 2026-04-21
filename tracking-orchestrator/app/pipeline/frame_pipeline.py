"""Frame processing pipeline for M4.

This is the core orchestrator that wires together:
1. Transport (Redis Streams consumer for FrameReady)
2. Inference (Triton person detector)
3. Tracking (BoT-SORT per-camera tracker)
4. Tracklet management (lifecycle, gallery append)
5. Persistence (repository layer)
6. Event emission (Redis Streams producer)

The pipeline runs as a background task in the FastAPI lifespan.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from structlog import get_logger

from ..domain import CameraConfig, Detection, FrameRef, TrackingEvent
from ..inference.detector import PersonDetector
from ..inference.schemas import DetectionBox
from ..storage.base import InMemoryTrackingRepository, TrackingRepository
from ..tracking.tracker import PerCameraTrackers, TrackerConfig
from ..tracking.tracklet_manager import TrackletConfig, TrackletManager
from ..transport.redis_streams import (
    FrameReady,
    RedisStreamsTransport,
    TransportConfig,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the frame processing pipeline."""

    transport: TransportConfig = field(default_factory=TransportConfig)
    tracklet: TrackletConfig = field(default_factory=TrackletConfig)
    detector_confidence: float = 0.25
    max_concurrent_frames: int = 4
    shutdown_timeout: float = 5.0


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class FrameProcessingPipeline:
    """End-to-end frame processing pipeline for M4.

    Usage::

        config = PipelineConfig()
        pipeline = FrameProcessingPipeline(config)
        await pipeline.initialize(detector)
        await pipeline.start()
        # ... run until shutdown ...
        await pipeline.stop()
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._transport: RedisStreamsTransport | None = None
        self._detector: PersonDetector | None = None
        self._tracklet_manager: TrackletManager | None = None
        self._tracker: PerCameraTrackers | None = None
        self._repo: TrackingRepository | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    @property
    def is_running(self) -> bool:
        return self._running

    async def initialize(
        self,
        detector: PersonDetector | None = None,
        repo: TrackingRepository | None = None,
    ) -> None:
        """Initialize all pipeline components.

        Args:
            detector: Triton-backed person detector. If None, skeleton mode.
            repo: Tracking repository. If None, uses InMemoryTrackingRepository.
        """
        # Transport
        self._transport = RedisStreamsTransport(self._config.transport)
        await self._transport.connect()

        # Repository
        self._repo = repo or InMemoryTrackingRepository()

        # Detector
        self._detector = detector

        # Tracklet manager
        tracker = PerCameraTrackers(TrackerConfig())
        from ..storage.base import InMemoryGalleryRepository

        gallery = InMemoryGalleryRepository()

        self._tracklet_manager = TrackletManager(
            repo=self._repo,
            gallery=gallery,
            config=self._config.tracklet,
        )

        # Store tracker reference for pipeline step
        self._tracker = tracker

        logger.info("Pipeline initialized", detector=bool(detector))

    async def start(self) -> None:
        """Start the pipeline background tasks."""
        if self._running:
            return

        self._running = True
        self._tasks = [
            asyncio.create_task(self._consume_loop()),
        ]
        logger.info("Pipeline started")

    async def stop(self) -> None:
        """Stop the pipeline and wait for tasks to complete."""
        self._running = False

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._transport:
            await self._transport.disconnect()

        self._tasks.clear()
        logger.info("Pipeline stopped")

    async def _consume_loop(self) -> None:
        """Main consumption loop: read FrameReady, process, publish."""
        if self._transport is None:
            logger.error("Cannot start consume loop: transport not initialized")
            return

        logger.info("Consume loop started")

        while self._running:
            try:
                # Read frames from Redis Streams
                async for frame in self._transport.consume_frames(count=1):
                    if not self._running:
                        break

                    start = time.monotonic()
                    try:
                        await self._process_frame(frame)
                        await self._transport.ack_frame(frame)
                        latency_us = int((time.monotonic() - start) * 1e6)
                        logger.debug(
                            "Frame processed",
                            camera_id=frame.camera_id,
                            frame_index=frame.frame_index,
                            latency_us=latency_us,
                        )
                    except Exception:
                        logger.exception(
                            "Frame processing failed",
                            camera_id=frame.camera_id,
                            frame_index=frame.frame_index,
                        )
                        await self._transport.publish_response(
                            frame,
                            success=False,
                            error_code="processing_error",
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Consume loop error, retrying in 1s")
                await asyncio.sleep(1)

        logger.info("Consume loop stopped")

    async def _process_frame(self, frame: FrameReady) -> None:
        """Process a single FrameReady message through the full pipeline.

        Steps:
        1. (Skeleton mode) Skip MinIO fetch — in production, fetch JPEG.
        2. Run person detection via Triton (if detector available).
        3. Run per-camera tracking (BoT-SORT).
        4. Run TrackletManager step.
        5. Persist results.
        6. Publish tracking event.
        """
        if self._detector is None or self._tracklet_manager is None or self._tracker is None:
            # Skeleton mode: produce a zero-detection event
            await self._skeleton_frame(frame)
            return

        # Step 1: Fetch frame from MinIO (skeleton: skip)
        # image = await minio.fetch_jpeg(frame.minio_key)

        # Step 2: Run detection
        # We need a placeholder image for detection
        # In production: detections = await self._detector.detect(image)
        detections: list[DetectionBox] = []

        if detections:
            # Step 3: Per-camera tracking
            from ..inference.schemas import Embedding

            # Create Detection domain objects from inference results
            domain_detections: list[Detection] = []
            embeddings: list[Embedding] = []

            for det in detections:
                from ..domain import BoundingBox

                bbox = BoundingBox(
                    x_min=int(det.x1 * frame.width),
                    y_min=int(det.y1 * frame.height),
                    x_max=int(det.x2 * frame.width),
                    y_max=int(det.y2 * frame.height),
                )
                # Embedding comes from ReID model separately in production.
                # Placeholder: zero embedding (float32 to match Embedding schema).
                emb = np.zeros(768, dtype=np.float32)

                domain_det = Detection(
                    detection_id=str(uuid.uuid4()),
                    camera_id=frame.camera_id,
                    bbox=bbox,
                    embedding=emb.tolist(),
                    capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
                    event_time=datetime.now(UTC),
                    confidence=det.confidence,
                )
                domain_detections.append(domain_det)
                embeddings.append(emb)

            # Step 3: Per-camera tracking
            local_tracks = self._tracker.update(
                camera_id=frame.camera_id,
                detections=domain_detections,
                embeddings=embeddings,
                frame_index=frame.frame_index,
            )

            # Step 4: Tracklet management
            camera_config = CameraConfig(camera_id=frame.camera_id)
            await self._tracklet_manager.step(
                camera=camera_config,
                local_tracks=local_tracks,
                detections=domain_detections,
                embeddings=embeddings,
                event_time=datetime.now(UTC),
                frame_index=frame.frame_index,
            )

            detection_count = len(local_tracks)
        else:
            detection_count = 0

        # Step 5 & 6: Publish tracking event
        assert self._transport is not None
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=datetime.now(UTC),
            frame_index=frame.frame_index,
            detection_count=detection_count,
        )

    async def _skeleton_frame(self, frame: FrameReady) -> None:
        """Process a frame in skeleton mode (no detector, no tracking).

        Produces a zero-detection tracking event and publishes it.
        This allows the pipeline to run end-to-end without Triton.
        """
        event_time = datetime.now(UTC)

        # Build a minimal tracking event
        event = TrackingEvent(
            event_id=str(uuid.uuid4()),
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            frame_ref=FrameRef(
                minio_key=frame.minio_key,
                width=frame.width,
                height=frame.height,
                frame_index=frame.frame_index,
                capture_time=datetime.fromtimestamp(frame.capture_time_unix_ns / 1e9, tz=UTC),
            ),
            detections=[],
        )

        if self._repo:
            await self._repo.save_tracking_event(event)

        assert self._transport is not None
        await self._transport.publish_event(
            camera_id=frame.camera_id,
            event_time=event_time,
            frame_index=frame.frame_index,
            detection_count=0,
        )

        logger.debug(
            "Skeleton frame processed",
            camera_id=frame.camera_id,
            frame_index=frame.frame_index,
        )
