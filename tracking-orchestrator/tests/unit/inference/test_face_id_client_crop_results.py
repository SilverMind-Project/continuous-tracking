"""FaceIdentificationClient reports exactly one CropQueryResult per crop, in input order."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from app.inference.face_id_client import CropQueryStatus, FaceIdentificationClient

_FACE = {
    "person_id": "resident-alpha",
    "name": "Resident Alpha",
    "confidence": 0.52,
    "bbox": [10.0, 20.0, 40.0, 60.0],
    "recognition_state": "recognized",
    "best_candidate_id": "resident-alpha",
    "similarity": 0.52,
    "raw_similarity": 0.52,
    "det_score": 0.88,
    "face_width_px": 30.0,
    "face_height_px": 40.0,
    "interocular_px": 14.5,
}


def _crop(width_px: int, height_px: int) -> np.ndarray:
    return np.zeros((height_px, width_px, 3), dtype=np.uint8)


async def _client(handler: httpx.MockTransport) -> FaceIdentificationClient:
    client = FaceIdentificationClient("http://person-id.test", transport=handler)
    await client.connect()
    return client


@pytest.mark.asyncio
async def test_each_crop_gets_its_own_status_in_input_order() -> None:
    responses = iter(
        [
            httpx.Response(200, json={"faces": [_FACE]}),
            httpx.Response(200, json={"faces": []}),
            httpx.Response(503, json={"detail": "unavailable"}),
        ]
    )
    client = await _client(httpx.MockTransport(lambda _request: next(responses)))
    crops = [_crop(64, 128), _crop(30, 128), _crop(64, 64), _crop(80, 80)]
    boxes = [(0.0, 0.0, 0.5, 0.5)] * 4

    results = await client.identify_crops(crops, boxes)

    assert [r.crop_index for r in results] == [0, 1, 2, 3]
    assert [r.status for r in results] == [
        CropQueryStatus.FACE_FOUND,
        CropQueryStatus.TOO_SMALL,
        CropQueryStatus.NO_FACE,
        CropQueryStatus.ERROR,
    ]
    assert [(r.crop_width_px, r.crop_height_px) for r in results] == [
        (64, 128),
        (30, 128),
        (64, 64),
        (80, 80),
    ]
    assert [len(r.faces) for r in results] == [1, 0, 0, 0]
    await client.disconnect()


@pytest.mark.asyncio
async def test_too_small_crop_is_never_sent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"faces": []})

    client = await _client(httpx.MockTransport(handler))
    [result] = await client.identify_crops([_crop(39, 200)], [(0.0, 0.0, 1.0, 1.0)])

    assert result.status is CropQueryStatus.TOO_SMALL
    assert requests == []
    await client.disconnect()


@pytest.mark.asyncio
async def test_geometry_fields_are_parsed_and_bbox_mapped_to_frame() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["save_guest_images"] is False
        return httpx.Response(200, json={"faces": [_FACE]})

    client = await _client(httpx.MockTransport(handler))
    [result] = await client.identify_crops([_crop(100, 200)], [(0.2, 0.1, 0.4, 0.5)])

    [face] = result.faces
    assert (face.face_width_px, face.face_height_px, face.interocular_px) == (30.0, 40.0, 14.5)
    # x: 0.2 + (10/100)*0.2 = 0.22 ... 0.2 + (40/100)*0.2 = 0.28; y: 0.1 + (20/200)*0.4 = 0.14
    assert face.bbox_normalized == pytest.approx([0.22, 0.14, 0.28, 0.22])
    await client.disconnect()


@pytest.mark.asyncio
async def test_missing_interocular_from_older_service_is_none() -> None:
    legacy_face = {k: v for k, v in _FACE.items() if not k.endswith("_px")}
    client = await _client(
        httpx.MockTransport(lambda _r: httpx.Response(200, json={"faces": [legacy_face]}))
    )
    [result] = await client.identify_crops([_crop(64, 64)], [(0.0, 0.0, 1.0, 1.0)])

    [face] = result.faces
    assert face.interocular_px is None
    assert face.face_width_px == 0.0
    await client.disconnect()


@pytest.mark.asyncio
async def test_disconnected_client_reports_error_for_every_crop() -> None:
    client = FaceIdentificationClient("http://person-id.test")
    results = await client.identify_crops([_crop(64, 64), _crop(64, 64)], [(0, 0, 1, 1)] * 2)
    assert [r.status for r in results] == [CropQueryStatus.ERROR, CropQueryStatus.ERROR]


@pytest.mark.asyncio
async def test_crops_are_encoded_at_the_configured_jpeg_quality() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["image"]))
        return httpx.Response(200, json={"faces": []})

    rng = np.random.default_rng(7)
    textured = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    for quality in (60, 95):
        client = FaceIdentificationClient(
            "http://person-id.test", transport=httpx.MockTransport(handler), jpeg_quality=quality
        )
        await client.connect()
        await client.identify_crops([textured], [(0.0, 0.0, 1.0, 1.0)])
        await client.disconnect()

    low, high = sizes
    assert high > low


@pytest.mark.parametrize("quality", [0, 101])
def test_out_of_range_jpeg_quality_is_rejected(quality: int) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        FaceIdentificationClient("http://person-id.test", jpeg_quality=quality)
