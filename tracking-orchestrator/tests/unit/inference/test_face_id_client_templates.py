"""FaceIdentificationClient: embeddings on /identify and the /match-template call."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
import pytest

from app.inference.face_id_client import (
    CaptureContext,
    FaceIdentificationClient,
    FaceQualityReading,
    TemplateMember,
)

_CONTEXT = CaptureContext(
    camera_id="cam-7",
    captured_at=datetime(2026, 11, 12, 9, 0, tzinfo=UTC),
    crop_pad_top_frac=0.15,
    crop_pad_side_frac=0.1,
)
_QUALITY = FaceQualityReading(
    score=0.5, resolution=0.4, sharpness=0.6, detection=0.9, pitch=1.0, version="heuristic-v1"
)


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _f16(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype("<f2").tobytes()).decode("ascii")


def _face(**changes: Any) -> dict[str, Any]:
    return {
        "person_id": "alice",
        "name": "Alice",
        "bbox": [10.0, 20.0, 40.0, 60.0],
        "recognition_state": "recognized",
        "best_candidate_id": "alice",
        "raw_similarity": 0.52,
        "det_score": 0.88,
    } | changes


async def _client(handler: Any) -> FaceIdentificationClient:
    client = FaceIdentificationClient(
        "http://person-id.test", transport=httpx.MockTransport(handler)
    )
    await client.connect()
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("asked", [False, True])
async def test_identify_sends_return_embedding_and_parses_the_embedding(asked: bool) -> None:
    sent: list[dict[str, Any]] = []
    embedding = _unit(3)

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        face = _face(embedding_f16_b64=_f16(embedding), embedding_model_version="buffalo_l")
        return httpx.Response(200, json={"faces": [face if asked else _face()]})

    client = await _client(handler)
    [result] = await client.identify_crops(
        [np.zeros((80, 60, 3), dtype=np.uint8)],
        [(0.1, 0.1, 0.3, 0.5)],
        _CONTEXT,
        return_embedding=asked,
    )

    assert sent[0]["return_embedding"] is asked
    face = result.faces[0]
    if asked:
        assert face.embedding is not None
        np.testing.assert_allclose(face.embedding, embedding, atol=1e-3)
        assert face.embedding_model_version == "buffalo_l"
    else:
        assert face.embedding is None
        assert face.embedding_model_version == ""
    # Biometric data never reaches a log line through repr.
    assert "embedding=" not in repr(face)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not base64!", base64.b64encode(b"\x00" * 6).decode()])
async def test_a_malformed_embedding_is_dropped_not_fatal(bad: str) -> None:
    client = await _client(
        lambda request: httpx.Response(200, json={"faces": [_face(embedding_f16_b64=bad)]})
    )
    [result] = await client.identify_crops(
        [np.zeros((80, 60, 3), dtype=np.uint8)],
        [(0.1, 0.1, 0.3, 0.5)],
        _CONTEXT,
        return_embedding=True,
    )
    assert result.faces[0].embedding is None
    assert result.faces[0].person_id == "alice"


_MATCH = {
    "person_id": "alice",
    "name": "Alice",
    "recognition_state": "recognized",
    "best_candidate_id": "alice",
    "raw_similarity": 0.71,
    "second_best_similarity": 0.12,
    "member_count": 3,
    "aggregator_version": "qwmean-v1",
    "template_quality": {
        "score": 0.5,
        "resolution": 0.4,
        "sharpness": 0.6,
        "detection": 0.9,
        "pitch": 1.0,
    },
    "quality_version": "heuristic-v1",
    "calibrated_confidence": 0.88,
    "calibration_profile": "template",
    "calibration_status": "ready",
    "calibration_artifact_version": "t-v1",
}


@pytest.mark.asyncio
async def test_match_template_sends_members_and_parses_the_verdict() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/match-template"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_MATCH)

    client = await _client(handler)
    members = [TemplateMember(embedding=_unit(i), quality=_QUALITY) for i in range(3)]

    result = await client.match_template(members, "buffalo_l")

    assert sent[0]["embedding_model_version"] == "buffalo_l"
    assert len(sent[0]["members"]) == 3
    decoded = np.frombuffer(
        base64.b64decode(sent[0]["members"][0]["embedding_f16_b64"]), dtype="<f2"
    )
    np.testing.assert_allclose(decoded, _unit(0), atol=1e-3)
    assert sent[0]["members"][0]["quality"]["version"] == "heuristic-v1"
    assert result is not None
    assert (result.person_id, result.recognition_state, result.member_count) == (
        "alice",
        "recognized",
        3,
    )
    assert result.calibrated_confidence == pytest.approx(0.88)
    assert (result.calibration_profile, result.quality.score) == ("template", pytest.approx(0.5))
    assert "embedding=" not in repr(members[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(422, json={"detail": {"code": "template.quality_version_mismatch"}}),
        httpx.Response(503, text="down"),
        httpx.Response(200, json={"person_id": "alice"}),  # malformed: fields missing
    ],
)
async def test_match_template_failures_return_none(response: httpx.Response) -> None:
    client = await _client(lambda request: response)
    members = [TemplateMember(embedding=_unit(i), quality=_QUALITY) for i in range(2)]
    assert await client.match_template(members, "buffalo_l") is None


@pytest.mark.asyncio
async def test_match_template_when_disconnected_returns_none() -> None:
    client = FaceIdentificationClient("http://person-id.test")
    members = [TemplateMember(embedding=_unit(i), quality=_QUALITY) for i in range(2)]
    assert await client.match_template(members, "buffalo_l") is None
