"""FaceIdentificationClient: per-crop detection ids and the attestation call."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import numpy as np
import pytest

from app.inference.face_id_client import (
    CaptureContext,
    FaceAttestationResult,
    FaceIdentificationClient,
)

_CONTEXT = CaptureContext(
    camera_id="cam-7",
    captured_at=datetime(2026, 11, 20, 9, 0, tzinfo=UTC),
    crop_pad_top_frac=0.15,
    crop_pad_side_frac=0.1,
)


def _crop() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


async def _client(handler: object) -> FaceIdentificationClient:
    client = FaceIdentificationClient(
        "http://person-id.test",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )
    await client.connect()
    return client


async def test_each_crop_carries_its_own_detection_id() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"faces": []})

    client = await _client(handler)
    context = replace(_CONTEXT, detection_ids=("det-a", "det-b"))
    await client.identify_crops([_crop(), _crop()], [(0, 0, 1, 1), (0, 0, 1, 1)], context)
    await client.disconnect()
    assert [b["capture_context"]["detection_id"] for b in bodies] == ["det-a", "det-b"]  # type: ignore[index]


async def test_no_detection_ids_leaves_the_context_unchanged() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"faces": []})

    client = await _client(handler)
    await client.identify_crops([_crop()], [(0, 0, 1, 1)], _CONTEXT)
    await client.disconnect()
    assert "detection_id" not in bodies[0]["capture_context"]  # type: ignore[operator]


async def test_detection_ids_must_be_parallel_to_the_crops() -> None:
    client = await _client(lambda _r: httpx.Response(200, json={"faces": []}))
    with pytest.raises(ValueError, match="parallel"):
        await client.identify_crops(
            [_crop(), _crop()], [(0, 0, 1, 1)] * 2, replace(_CONTEXT, detection_ids=("only-one",))
        )
    await client.disconnect()


async def test_attest_faces_posts_the_batch_and_parses_counts() -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"moved": 2, "missed": 1})

    client = await _client(handler)
    result = await client.attest_faces(
        association_id="seg-1",
        identity_id="alice",
        attestation_grade="trajectory",
        detection_ids=["d1", "d2", "d3"],
    )
    await client.disconnect()
    assert result == FaceAttestationResult(moved=2, missed=1)
    path, body = seen[0]
    assert path == "/api/v1/calibration/attestations"
    assert body == {
        "association_id": "seg-1",
        "identity_id": "alice",
        "attestation_grade": "trajectory",
        "detection_ids": ["d1", "d2", "d3"],
        "attested_by": "wearable",
    }


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(422, json={"detail": {"code": "unknown_identity"}}),
        httpx.Response(200, json={"moved": "many"}),
    ],
)
async def test_attest_faces_errors_return_none(response: httpx.Response) -> None:
    client = await _client(lambda _r: response)
    result = await client.attest_faces(
        association_id="seg-1", identity_id="alice", attestation_grade="room", detection_ids=["d1"]
    )
    await client.disconnect()
    assert result is None


async def test_attest_faces_without_ids_sends_nothing() -> None:
    calls: list[object] = []
    client = await _client(lambda r: calls.append(r) or httpx.Response(200, json={}))
    assert (
        await client.attest_faces(
            association_id="s", identity_id="a", attestation_grade="room", detection_ids=[]
        )
        is None
    )
    await client.disconnect()
    assert calls == []
