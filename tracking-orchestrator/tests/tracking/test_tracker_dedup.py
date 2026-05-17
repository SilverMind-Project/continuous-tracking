"""Unit tests for tracker-level tracklet dedup (Phase 1 §3.3.2)."""

from __future__ import annotations

from app.domain import BoundingBox, Detection
from app.tracking.tracker import PerCameraTracker, PerCameraTrackers, TrackerConfig


def _det(
    did: str,
    x_min: int = 10,
    y_min: int = 10,
    x_max: int = 100,
    y_max: int = 200,
    conf: float = 0.9,
) -> Detection:
    from datetime import UTC, datetime

    bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    return Detection(
        detection_id=did,
        camera_id="cam-1",
        bbox=bbox,
        embedding=[],
        capture_time=datetime.now(UTC),
        event_time=datetime.now(UTC),
        confidence=conf,
    )


class TestTrackerDedup:
    def _config(self, dedup_iou: float = 0.7, dedup_min_age: int = 5) -> TrackerConfig:
        return TrackerConfig(
            new_track_thresh=0.5,
            min_hits=1,
            dedup_iou_threshold=dedup_iou,
            dedup_min_age=dedup_min_age,
        )

    def test_no_dedup_when_stable_track_below_min_age(self) -> None:
        """A newly-spawned track is not deduped when the existing track is too young."""
        tracker = PerCameraTracker(self._config(dedup_min_age=10))
        # Create an existing track with age < 10
        d1 = _det("d1", x_min=10, y_min=10, x_max=110, y_max=210)
        for _ in range(3):
            tracker.update([d1], frame_index=0)

        # Now submit an overlapping detection — existing track has age=3, min_age=10
        d2 = _det("d2", x_min=15, y_min=15, x_max=115, y_max=215)
        tracks = tracker.update([d2], frame_index=1)
        # Both the existing track (age=4) and the new tracklet should appear
        assert len(tracks) >= 1
        assert tracker._dedup_dropped == 0

    def test_dedup_drops_new_tracklet_overlapping_stable_track(self) -> None:
        """A newly-spawned tracklet heavily overlapping a stable (age>=5) track is dropped."""
        tracker = PerCameraTracker(self._config(dedup_iou=0.7, dedup_min_age=5))
        d1 = _det("d1", x_min=10, y_min=10, x_max=110, y_max=210)
        # Age the existing track past dedup_min_age
        for _ in range(5):
            tracker.update([d1], frame_index=0)

        # Submit a heavily-overlapping new detection that won't match existing track
        # (use a completely different detection_id so it goes through step 4)
        # For the new detection to be unmatched, existing track must miss it — we
        # submit ONLY the new detection (not d1) so d1's track enters "lost" state
        # and won't consume d2 in association. But we still want d1's age >= 5.
        # Simplest: submit d2 alongside d1 but make d2 high-conf so it spawns new track.
        # To ensure d2 is unmatched: give it a bbox that overlaps d1 heavily but with
        # a different position so association doesn't pick it.
        d2 = _det("d2", x_min=12, y_min=12, x_max=112, y_max=212)
        # Association will match d1 to existing track. d2 is unmatched → new track.
        result = tracker.update([d1, d2], frame_index=5)
        assert tracker._dedup_dropped == 1
        # Only the stable d1 track should be in result, not d2
        assert len(result) == 1

    def test_dedup_counter_exposed_by_per_camera_trackers(self) -> None:
        """PerCameraTrackers.get_dedup_dropped() returns the per-camera dedup count."""
        trackers = PerCameraTrackers(
            TrackerConfig(
                new_track_thresh=0.5,
                min_hits=1,
                dedup_iou_threshold=0.7,
                dedup_min_age=5,
            )
        )
        d1 = _det("d1", x_min=10, y_min=10, x_max=110, y_max=210)
        for _ in range(5):
            trackers.update("cam-1", [d1])
        d2 = _det("d2", x_min=12, y_min=12, x_max=112, y_max=212)
        trackers.update("cam-1", [d1, d2])
        assert trackers.get_dedup_dropped("cam-1") == 1
        assert trackers.get_dedup_dropped("cam-99") == 0  # unknown camera

    def test_no_dedup_when_disabled(self) -> None:
        """Setting dedup_iou_threshold=1.0 effectively disables dedup."""
        tracker = PerCameraTracker(
            TrackerConfig(
                new_track_thresh=0.5,
                min_hits=1,
                dedup_iou_threshold=1.0,
                dedup_min_age=1,
            )
        )
        d1 = _det("d1", x_min=10, y_min=10, x_max=110, y_max=210)
        tracker.update([d1], frame_index=0)
        d2 = _det("d2", x_min=12, y_min=12, x_max=112, y_max=212)
        tracker.update([d1, d2], frame_index=1)
        assert tracker._dedup_dropped == 0
