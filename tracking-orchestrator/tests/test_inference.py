"""Unit tests for the app.inference module.

All Triton calls are replaced with an AsyncMockClient so no GPU or
tritonclient installation is required — these tests run under `make check`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import numpy.typing as npt
import pytest
from triton_shared.inference.detection import decode_output, letterbox_preprocess

from app.inference.detector import PersonDetector
from app.inference.pose import PoseEstimator, _decode_simcc, _preprocess
from app.inference.reid_embedder import ReidEmbedder
from app.inference.schemas import (
    COCO_KEYPOINTS,
    EMBEDDING_DIM,
    NUM_KEYPOINTS,
    DetectionBox,
    Keypoint,
    PoseResult,
)
from app.inference.triton_client import TritonClientProtocol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTritonClient:
    """Minimal TritonClientProtocol implementation backed by AsyncMock."""

    def __init__(self, return_values: dict[str, npt.NDArray[np.float32]]) -> None:
        self._returns = return_values
        self.infer = AsyncMock(return_value=return_values)
        self.is_model_ready = AsyncMock(return_value=True)


def _make_image(h: int = 120, w: int = 160) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _make_crop(h: int = 80, w: int = 60) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# TritonClientProtocol structural check
# ---------------------------------------------------------------------------


def test_mock_client_satisfies_protocol() -> None:
    client = _MockTritonClient({})
    assert isinstance(client, TritonClientProtocol)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


def test_detection_box_area() -> None:
    b = DetectionBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8, confidence=0.9)
    assert abs(b.area - 0.24) < 1e-6


def test_pose_result_requires_17_keypoints() -> None:
    kpts = tuple(Keypoint(0.5, 0.5, 0.9) for _ in range(17))
    pr = PoseResult(keypoints=kpts)
    assert len(pr.keypoints) == NUM_KEYPOINTS


def test_pose_result_wrong_count_raises() -> None:
    kpts = tuple(Keypoint(0.5, 0.5, 0.9) for _ in range(5))
    with pytest.raises(ValueError, match="17 keypoints"):
        PoseResult(keypoints=kpts)


def test_pose_result_get_by_name() -> None:
    kpts = tuple(Keypoint(float(i) * 0.05, float(i) * 0.03, 0.8) for i in range(17))
    pr = PoseResult(keypoints=kpts)
    nose = pr.get("nose")
    assert nose == kpts[0]


def test_coco_keypoints_count() -> None:
    assert len(COCO_KEYPOINTS) == NUM_KEYPOINTS


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def testletterbox_preprocess_square_output() -> None:
    img = _make_image(480, 640)
    tensor, _px, _py, _scale = letterbox_preprocess(img, 640)
    assert tensor.shape == (3, 640, 640)
    assert tensor.dtype == np.float32


def testletterbox_preprocess_values_in_range() -> None:
    img = _make_image(100, 200)
    tensor, *_ = letterbox_preprocess(img, 640)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_reid_preprocess_shape() -> None:
    from app.inference.reid_embedder import _preprocess as reid_preprocess

    crop = _make_crop(300, 150)
    out = reid_preprocess(crop)
    assert out.shape == (3, 256, 128)
    assert out.dtype == np.float32


def test_reid_preprocess_samples_full_crop() -> None:
    """Regression: the resize must sample from the entire source crop, not
    collapse to a single row/column of the top-left pixel (the M3 bug)."""
    from app.inference.reid_embedder import _preprocess as reid_preprocess

    # Create a crop with a clear gradient — bottom-right corner is bright.
    crop = np.zeros((300, 150, 3), dtype=np.uint8)
    crop[:, :, 0] = np.arange(300, dtype=np.uint8)[:, None]  # R: top-to-bottom gradient
    crop[:, :, 1] = np.arange(150, dtype=np.uint8)[None, :]  # G: left-to-right gradient

    out = reid_preprocess(crop)  # (3, 256, 128)

    # If the resize samples the full crop, the output must have non-zero
    # variance in every channel.  A collapsed resize (bug) would produce
    # near-zero variance because all rows/cols map to index 0.
    assert out[0, :, :].std() > 0.5, "Red channel has near-zero std — resize collapsed"
    assert out[1, :, :].std() > 0.5, "Green channel has near-zero std — resize collapsed"


def test_pose_preprocess_shape() -> None:
    crop = _make_crop(300, 150)
    tensor, _px, _py, _scale = _preprocess(crop)
    assert tensor.shape == (3, 256, 192)
    assert tensor.dtype == np.float32


# ---------------------------------------------------------------------------
# YOLO26L decode
# ---------------------------------------------------------------------------


def _make_yolo_output(
    batch: int,
    x1: float = 270.0,
    y1: float = 165.0,
    x2: float = 370.0,
    y2: float = 315.0,
    conf: float = 0.9,
    class_id: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Construct a fake YOLO26L NMS-free output0 tensor [batch, 300, 6].

    Columns: x1, y1, x2, y2 (letterbox pixel space), confidence, class_id.
    One person detection placed at row 0; remaining 299 rows are zero (filtered
    out by conf threshold).
    """
    out = np.zeros((batch, 300, 6), dtype=np.float32)
    out[:, 0, 0] = x1
    out[:, 0, 1] = y1
    out[:, 0, 2] = x2
    out[:, 0, 3] = y2
    out[:, 0, 4] = conf
    out[:, 0, 5] = class_id
    return out


def testdecode_output_finds_person() -> None:
    raw = _make_yolo_output(1)[0]  # single sample (300, 6)
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=80, scale=1.0)
    assert len(boxes) == 1
    b = boxes[0]
    assert 0.0 <= b.x1 <= b.x2 <= 1.0
    assert 0.0 <= b.y1 <= b.y2 <= 1.0
    assert b.confidence >= 0.25


def testdecode_output_empty_on_low_confidence() -> None:
    raw = np.zeros((300, 6), dtype=np.float32)
    raw[0, 4] = 0.1  # below threshold
    raw[0, 5] = 0.0  # class 0
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=0, scale=1.0)
    assert boxes == []


def testdecode_output_filters_non_person_class() -> None:
    raw = _make_yolo_output(1, conf=0.9, class_id=1.0)[0]  # class 1 (not person)
    boxes = decode_output(raw, orig_h=480, orig_w=640, pad_x=0, pad_y=0, scale=1.0)
    assert boxes == []


def testdecode_output_multiple_detections() -> None:
    raw = np.zeros((300, 6), dtype=np.float32)
    raw[0] = [100.0, 50.0, 200.0, 150.0, 0.9, 0.0]
    raw[1] = [300.0, 200.0, 400.0, 350.0, 0.8, 0.0]
    boxes = decode_output(raw, orig_h=640, orig_w=640, pad_x=0, pad_y=0, scale=1.0)
    assert len(boxes) == 2


# ---------------------------------------------------------------------------
# PersonDetector (mocked Triton)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_detector_returns_boxes() -> None:
    raw = _make_yolo_output(1)
    client = _MockTritonClient({"output0": raw})
    detector = PersonDetector(client)
    img = _make_image()
    boxes = await detector.detect(img)
    assert isinstance(boxes, list)
    client.infer.assert_awaited_once()


@pytest.mark.asyncio
async def test_detector_batch() -> None:
    raw = _make_yolo_output(3)
    client = _MockTritonClient({"output0": raw})
    detector = PersonDetector(client)
    imgs = [_make_image() for _ in range(3)]
    results = await detector.detect_batch(imgs)
    assert len(results) == 3
    client.infer.assert_awaited_once()


@pytest.mark.asyncio
async def test_detector_empty_batch() -> None:
    client = _MockTritonClient({})
    detector = PersonDetector(client)
    results = await detector.detect_batch([])
    assert results == []
    client.infer.assert_not_awaited()


# ---------------------------------------------------------------------------
# ReidEmbedder (mocked Triton)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedder_returns_correct_shape() -> None:
    fake_emb = np.random.rand(2, EMBEDDING_DIM).astype(np.float32)
    client = _MockTritonClient({"output": fake_emb})
    embedder = ReidEmbedder(client)
    crops = [_make_crop(), _make_crop()]
    result = await embedder.embed_batch(crops)
    assert len(result) == 2
    assert result[0].shape == (EMBEDDING_DIM,)


@pytest.mark.asyncio
async def test_embedder_wrong_dim_raises() -> None:
    bad_emb = np.random.rand(1, 512).astype(np.float32)  # wrong dim
    client = _MockTritonClient({"output": bad_emb})
    embedder = ReidEmbedder(client)
    with pytest.raises(ValueError, match="768"):
        await embedder.embed(_make_crop())


@pytest.mark.asyncio
async def test_embedder_empty_batch() -> None:
    client = _MockTritonClient({})
    embedder = ReidEmbedder(client)
    assert await embedder.embed_batch([]) == []


# ---------------------------------------------------------------------------
# PoseEstimator (mocked Triton)
# ---------------------------------------------------------------------------


def _make_simcc_output(batch: int) -> dict[str, npt.NDArray[np.float32]]:
    rng = np.random.default_rng(2)
    sx = rng.random((batch, 17, 384)).astype(np.float32)
    sy = rng.random((batch, 17, 512)).astype(np.float32)
    return {"simcc_x": sx, "simcc_y": sy}


@pytest.mark.asyncio
async def test_pose_returns_17_keypoints() -> None:
    client = _MockTritonClient(_make_simcc_output(1))
    estimator = PoseEstimator(client)
    result = await estimator.infer(_make_crop())
    assert len(result.keypoints) == 17
    for kp in result.keypoints:
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0
        assert 0.0 <= kp.score <= 1.0


@pytest.mark.asyncio
async def test_pose_batch() -> None:
    client = _MockTritonClient(_make_simcc_output(3))
    estimator = PoseEstimator(client)
    crops = [_make_crop() for _ in range(3)]
    results = await estimator.infer_batch(crops)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_pose_empty_batch() -> None:
    client = _MockTritonClient({})
    estimator = PoseEstimator(client)
    assert await estimator.infer_batch([]) == []


# ---------------------------------------------------------------------------
# SimCC decode
# ---------------------------------------------------------------------------


def test_decode_simcc_coords_in_range() -> None:
    rng = np.random.default_rng(7)
    sx = rng.random((17, 384)).astype(np.float32)
    sy = rng.random((17, 512)).astype(np.float32)
    result = _decode_simcc(sx, sy, orig_h=200, orig_w=100, pad_x=16, pad_y=0, scale=0.96)
    for kp in result.keypoints:
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0
        assert 0.0 <= kp.score <= 1.0


def test_decode_simcc_score_is_bounded() -> None:
    sx = np.ones((17, 384), dtype=np.float32)
    sy = np.ones((17, 512), dtype=np.float32)
    result = _decode_simcc(sx, sy, orig_h=100, orig_w=80, pad_x=16, pad_y=0, scale=1.0)
    for kp in result.keypoints:
        assert kp.score <= 1.0
