"""Per-PH best faces for face templates: a bounded, in-memory state container with no I/O.

A walking person yields many small faces that are weak one at a time. The buffer keeps the best
``top_k`` faces (by service-measured quality) seen on each Person Hypothesis so several of them
can be matched together as one template by person-identification-service.

Embeddings are biometric data. They live here in memory only, are excluded from every repr, are
never logged or persisted, and leave the process only in the template request to person-id.

Epoch: one PH partition. A PH's buffer resets when its committed identity changes to a different
identity, on an operator split, merge, or correction, and when an identity is purged; faces older
than ``max_age_s`` leave on their own, which also keeps a revived PH's faces across a brief
occlusion. Every reset advances a generation counter, so a template match requested before the
reset is recognized as stale when its result arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from app.inference.face_id_client import FaceQualityReading

FaceTemplateMode = Literal["off", "shadow", "enabled"]


@dataclass(frozen=True)
class FaceTemplatePolicy:
    """Settings ``face_templates.*``. Thresholds on ``quality`` are person-id quality scores."""

    # "off": no embeddings requested, no buffer, no requests. "shadow": templates are matched and
    # evaluated alongside live decisions without changing them. "enabled": template anchors are
    # live evidence.
    mode: FaceTemplateMode = "off"
    # Faces kept per PH, best quality first.
    top_k: int = 8
    # Faces a PH needs before a template is matched.
    min_faces: int = 3
    # Faces below this quality score never enter (a face with no score never enters either).
    min_face_quality: float = 0.15
    # Faces leave the buffer this long after capture.
    max_age_s: float = 15.0
    # A face whose cosine to the buffer's mean embedding is below this is rejected as another
    # person's (an association swap), not averaged in.
    outlier_min_similarity: float = 0.25
    # Minimum time between two template requests for one PH.
    trigger_cooldown_s: float = 2.0
    # Concurrent template requests, and requests allowed in flight at all (more are dropped).
    max_concurrency: int = 2
    max_in_flight: int = 16

    def __post_init__(self) -> None:
        if self.mode not in get_args(FaceTemplateMode):
            raise ValueError(f"face_templates.mode must be one of {get_args(FaceTemplateMode)}")
        if not 2 <= self.min_faces <= self.top_k <= 16:
            raise ValueError("face_templates needs 2 <= min_faces <= top_k <= 16")
        if not 0.0 <= self.min_face_quality <= 1.0:
            raise ValueError("face_templates.min_face_quality must be in [0, 1]")
        if not -1.0 <= self.outlier_min_similarity <= 1.0:
            raise ValueError("face_templates.outlier_min_similarity must be in [-1, 1]")
        if self.max_age_s <= 0.0 or self.trigger_cooldown_s < 0.0:
            raise ValueError("face_templates.max_age_s must be positive, the cooldown not negative")
        if self.max_concurrency < 1 or self.max_in_flight < self.max_concurrency:
            raise ValueError("face_templates needs 1 <= max_concurrency <= max_in_flight")


@dataclass(frozen=True)
class BufferedFace:
    """One face attributed to a person, kept for a template. Never logged, never persisted."""

    detection_id: str
    camera_id: str
    # Frame capture time (timezone-aware).
    captured_at: datetime
    # Unit-length ArcFace embedding from person-id; excluded from repr and equality.
    embedding: npt.NDArray[np.float32] = field(repr=False, compare=False)
    quality: FaceQualityReading
    det_score: float
    yaw_deg: float
    # The identity person-id recognized for this single face, if any (below the tracker's floor
    # included); the template's authority consistency gate reads it.
    single_frame_person_id: str | None
    embedding_model_version: str


class BufferAdd(StrEnum):
    """What ``FaceTemplateBuffer.add`` did with a face."""

    KEPT = "kept"  # the face is now among the PH's best faces
    NOT_KEPT = "not_kept"  # the buffer is full of better faces
    LOW_QUALITY = "low_quality"
    OUTLIER = "outlier"


class FaceTemplateBuffer:
    """Per-PH best faces. Pure state: callers pass the time, and nothing here does I/O."""

    def __init__(self, policy: FaceTemplatePolicy) -> None:
        self._policy = policy
        self._faces: dict[str, list[BufferedFace]] = {}
        self._identity: dict[str, str] = {}
        self._last_trigger: dict[str, datetime] = {}
        # Generation counter: a ticket handed out for a request is valid only if no reset of its
        # PH happened after it was issued.
        self._generation = 0
        self._reset_generation: dict[str, tuple[int, datetime]] = {}

    def add(self, ph_id: str, face: BufferedFace) -> BufferAdd:
        """Offer one face; ``KEPT`` means the PH's best faces changed."""
        if face.quality.score < self._policy.min_face_quality:
            return BufferAdd.LOW_QUALITY
        current = self._faces.get(ph_id, [])
        if (
            current
            and _cosine_to_mean(face.embedding, current) < self._policy.outlier_min_similarity
        ):
            return BufferAdd.OUTLIER
        ranked = sorted(
            [*current, face],
            key=lambda f: (f.quality.score, f.captured_at),
            reverse=True,
        )[: self._policy.top_k]
        self._faces[ph_id] = ranked
        return BufferAdd.KEPT if any(f is face for f in ranked) else BufferAdd.NOT_KEPT

    def snapshot(self, ph_id: str) -> tuple[BufferedFace, ...]:
        """The PH's best faces, best quality first."""
        return tuple(self._faces.get(ph_id, ()))

    def size(self, ph_id: str) -> int:
        return len(self._faces.get(ph_id, ()))

    def ph_ids(self) -> tuple[str, ...]:
        return tuple(self._faces)

    def reset(self, ph_id: str, at: datetime) -> None:
        """Drop the PH's faces and invalidate every ticket issued for it so far."""
        self._faces.pop(ph_id, None)
        self._identity.pop(ph_id, None)
        self._last_trigger.pop(ph_id, None)
        self._generation += 1
        self._reset_generation[ph_id] = (self._generation, at)

    def reset_identity(self, identity_id: str, at: datetime) -> list[str]:
        """Reset every PH committed to ``identity_id`` or holding a face recognized as it."""
        affected = [
            ph_id
            for ph_id in self._faces.keys() | self._identity.keys()
            if self._identity.get(ph_id) == identity_id
            or any(f.single_frame_person_id == identity_id for f in self._faces.get(ph_id, ()))
        ]
        for ph_id in affected:
            self.reset(ph_id, at)
        return sorted(affected)

    def observe_identity(self, ph_id: str, identity_id: str | None, at: datetime) -> bool:
        """Record the PH's committed identity; True when it changed identity and was reset.

        Unknown in between does not reset: a PH that drops to Unknown and comes back as the same
        person is still the same body.
        """
        if identity_id is None:
            return False
        previous = self._identity.get(ph_id)
        changed = previous is not None and previous != identity_id
        if changed:
            self.reset(ph_id, at)
        self._identity[ph_id] = identity_id
        return changed

    def evict_older_than(self, now: datetime) -> None:
        """Faces older than ``max_age_s`` leave; a PH with no faces left is forgotten."""
        horizon = now - timedelta(seconds=self._policy.max_age_s)
        for ph_id in list(self._faces):
            kept = [f for f in self._faces[ph_id] if f.captured_at >= horizon]
            if kept:
                self._faces[ph_id] = kept
            else:
                del self._faces[ph_id]
                self._last_trigger.pop(ph_id, None)
        # A reset marker only has to outlive any request issued before it; requests are bounded
        # by the client timeout and results by max_age_s, so older markers can go.
        for ph_id, (_, at) in list(self._reset_generation.items()):
            if at < horizon:
                del self._reset_generation[ph_id]
        for ph_id in list(self._identity):
            if ph_id not in self._faces and ph_id not in self._reset_generation:
                del self._identity[ph_id]

    def should_trigger(self, ph_id: str, now: datetime) -> bool:
        """True (and the cooldown starts) when the PH has enough faces and is off cooldown."""
        if self.size(ph_id) < self._policy.min_faces:
            return False
        last = self._last_trigger.get(ph_id)
        if last is not None and (now - last).total_seconds() < self._policy.trigger_cooldown_s:
            return False
        self._last_trigger[ph_id] = now
        return True

    def ticket(self) -> int:
        """A ticket for a request about to be issued; see ``is_current``."""
        self._generation += 1
        return self._generation

    def is_current(self, ph_id: str, ticket: int) -> bool:
        """False when the PH was reset after the ticket was issued."""
        reset = self._reset_generation.get(ph_id)
        return reset is None or ticket > reset[0]


def _cosine_to_mean(embedding: npt.NDArray[np.float32], faces: list[BufferedFace]) -> float:
    mean = np.mean([f.embedding for f in faces], axis=0)
    norm = float(np.linalg.norm(mean)) * float(np.linalg.norm(embedding))
    if norm <= 0.0:
        return -1.0
    return float(np.dot(mean, embedding)) / norm
