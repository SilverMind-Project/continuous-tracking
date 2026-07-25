"""Tests for daily_centroid/compare pure math and AppearanceEvaluator (DL-M07)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from app.domain import Identity, PersonHypothesis
from app.storage.appearance import InMemoryDailyAppearanceRepo
from app.storage.base import InMemoryPHRepository
from app.trajectory.appearance_profile import (
    AppearanceEntry,
    AppearanceEvaluator,
    AppearanceSettings,
    compare,
    daily_centroid,
)

_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Pure math: daily_centroid
# ---------------------------------------------------------------------------


class TestDailyCentroid:
    def test_weighted_centroid_hand_computed(self) -> None:
        # Two entries, weights 0.5 and 1.0: weighted mean = (v1*0.5 + v2*1.0) / 1.5
        entries = [
            AppearanceEntry(embedding=(1.0, 0.0), quality=0.5, observation_count=5),
            AppearanceEntry(embedding=(0.0, 1.0), quality=1.0, observation_count=5),
        ]
        result = daily_centroid(entries, quality_floor=0.3, min_observations=3)
        assert result is not None
        centroid, sample_count, mean_quality = result
        assert sample_count == 2
        assert mean_quality == pytest.approx(0.75)
        # Weighted sum = (0.5, 1.0), normalised to unit length.
        expected_norm = (0.5**2 + 1.0**2) ** 0.5
        assert centroid[0] == pytest.approx(0.5 / expected_norm)
        assert centroid[1] == pytest.approx(1.0 / expected_norm)

    def test_quality_floor_excludes_low_quality_entries(self) -> None:
        entries = [
            AppearanceEntry(embedding=(1.0, 0.0), quality=0.1, observation_count=5),
            AppearanceEntry(embedding=(0.0, 1.0), quality=0.9, observation_count=5),
        ]
        result = daily_centroid(entries, quality_floor=0.3, min_observations=3)
        assert result is not None
        _centroid, sample_count, mean_quality = result
        assert sample_count == 1
        assert mean_quality == pytest.approx(0.9)

    def test_min_observations_excludes_transient_ph(self) -> None:
        entries = [
            AppearanceEntry(embedding=(1.0, 0.0), quality=0.9, observation_count=1),
            AppearanceEntry(embedding=(0.0, 1.0), quality=0.9, observation_count=5),
        ]
        result = daily_centroid(entries, quality_floor=0.3, min_observations=3)
        assert result is not None
        _centroid, sample_count, _mean_quality = result
        assert sample_count == 1

    def test_result_is_unit_normalised(self) -> None:
        entries = [AppearanceEntry(embedding=(3.0, 4.0), quality=1.0, observation_count=5)]
        result = daily_centroid(entries)
        assert result is not None
        centroid, _sample_count, _mean_quality = result
        norm = sum(v * v for v in centroid) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_empty_input_returns_none(self) -> None:
        assert daily_centroid([]) is None

    def test_single_sample_below_min_observations_returns_none(self) -> None:
        entries = [AppearanceEntry(embedding=(1.0, 0.0), quality=0.9, observation_count=1)]
        assert daily_centroid(entries, min_observations=3) is None

    def test_all_below_quality_floor_returns_none(self) -> None:
        entries = [AppearanceEntry(embedding=(1.0, 0.0), quality=0.1, observation_count=5)]
        assert daily_centroid(entries, quality_floor=0.3) is None


# ---------------------------------------------------------------------------
# Pure math: compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_identical_vectors_similarity_one(self) -> None:
        assert compare((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_zero(self) -> None:
        assert compare((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)

    def test_zero_vector_degrades_to_zero(self) -> None:
        assert compare((0.0, 0.0), (1.0, 0.0)) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            compare((1.0, 0.0), (1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# AppearanceEvaluator
# ---------------------------------------------------------------------------


class _FakeGalleryRepo:
    """Minimal stand-in for GalleryRepository.list_identities."""

    def __init__(self, identity_ids: list[str]) -> None:
        self._identities = [
            Identity(identity_id=iid, display_name=iid, enrolled_at=datetime.now(UTC))
            for iid in identity_ids
        ]

    async def list_identities(self, active_only: bool = True) -> list[Identity]:
        return self._identities


def _utc(y: int, m: int, d: int, h: int = 0, minute: int = 0) -> datetime:
    return datetime(y, m, d, h, minute, tzinfo=UTC)


def _make_ph(
    ph_id: str,
    identity_id: str,
    born_at: datetime,
    embedding: tuple[float, ...],
    quality: float = 0.9,
    observation_count: int = 5,
    closed_at: datetime | None = None,
) -> PersonHypothesis:
    return PersonHypothesis(
        ph_id=ph_id,
        state_mean=(0.0, 0.0, 0.0, 0.0),
        state_cov=(0.1,) * 16,
        born_at=born_at,
        last_seen_at=born_at,
        last_seen_camera="cam-1",
        observation_count=observation_count,
        current_identity_id=identity_id,
        gallery_mean=list(embedding),
        mean_quality=quality,
        active_cameras=frozenset(["cam-1"]),
        closed_at=closed_at,
    )


async def _seed_day(
    ph_repo: InMemoryPHRepository,
    identity_id: str,
    day: date,
    embedding: tuple[float, ...],
    tz_name: str = _TZ,
    count: int = 5,
) -> None:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name)
    day_start_local = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    for i in range(count):
        born = (day_start_local + timedelta(hours=10, minutes=i)).astimezone(UTC)
        await ph_repo.save(
            _make_ph(
                f"{identity_id}-{day}-{i}",
                identity_id,
                born,
                embedding,
                closed_at=born + timedelta(minutes=5),
            )
        )


@pytest.fixture
def ph_repo() -> InMemoryPHRepository:
    return InMemoryPHRepository()


@pytest.fixture
def profile_repo() -> InMemoryDailyAppearanceRepo:
    return InMemoryDailyAppearanceRepo()


def _evaluator(
    ph_repo: InMemoryPHRepository,
    profile_repo: InMemoryDailyAppearanceRepo,
    identity_ids: list[str],
    *,
    enabled: bool = False,
    similarity_threshold: float = 0.90,
    min_samples_per_day: int = 5,
    evaluate_local_hour: int = 11,
) -> AppearanceEvaluator:
    cfg = AppearanceSettings(
        enabled=enabled,
        similarity_threshold=similarity_threshold,
        min_samples_per_day=min_samples_per_day,
        evaluate_local_hour=evaluate_local_hour,
        tz_name=_TZ,
    )
    return AppearanceEvaluator(
        ph_repo=ph_repo,
        profile_repo=profile_repo,
        gallery_repo=_FakeGalleryRepo(identity_ids),
        keyframe_repo=None,
        cfg=cfg,
    )


class TestEvaluatorThresholdAndEmission:
    async def test_emits_signal_when_enabled_and_above_threshold(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        # Same embedding both days -> similarity 1.0.
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True)
        now = _utc(2026, 2, 10, 16, 0)  # 11:00 NY

        signals = await evaluator.run_once(now)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.signal_kind == "same_clothes_suspected"
        assert sig.identity_id == alice
        assert sig.value == pytest.approx(1.0)
        assert sig.severity == "info"

    async def test_no_signal_when_below_threshold(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (0.0, 1.0))  # orthogonal -> similarity 0

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True)
        now = _utc(2026, 2, 10, 16, 0)

        signals = await evaluator.run_once(now)
        assert signals == []

    async def test_shadow_mode_never_returns_signals(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=False)
        now = _utc(2026, 2, 10, 16, 0)

        signals = await evaluator.run_once(now)
        assert signals == []

    async def test_shadow_mode_still_persists_yesterday_profile(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=False)
        now = _utc(2026, 2, 10, 16, 0)
        await evaluator.run_once(now)

        profile = await profile_repo.get_profile(alice, yesterday)
        assert profile is not None
        assert profile.sample_count == 5


class TestFailSilentInsufficientEvidence:
    async def test_fail_silent_when_yesterday_below_min_samples(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0), count=2)  # below min 5
        await _seed_day(ph_repo, alice, today, (1.0, 0.0), count=5)

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True, min_samples_per_day=5)
        now = _utc(2026, 2, 10, 16, 0)

        signals = await evaluator.run_once(now)
        assert signals == []

    async def test_fail_silent_when_today_below_min_samples(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0), count=5)
        await _seed_day(ph_repo, alice, today, (1.0, 0.0), count=2)  # below min 5

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True, min_samples_per_day=5)
        now = _utc(2026, 2, 10, 16, 0)

        signals = await evaluator.run_once(now)
        assert signals == []


class TestOncePerDay:
    async def test_second_run_same_day_does_not_reemit(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True)
        now = _utc(2026, 2, 10, 16, 0)

        first = await evaluator.run_once(now)
        second = await evaluator.run_once(now + timedelta(minutes=1))

        assert len(first) == 1
        assert second == []

    async def test_before_evaluate_hour_returns_empty_and_does_not_mark_evaluated(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True, evaluate_local_hour=11)
        before_hour = _utc(2026, 2, 10, 14, 0)  # 09:00 NY, before hour 11
        after_hour = _utc(2026, 2, 10, 16, 0)  # 11:00 NY

        before_result = await evaluator.run_once(before_hour)
        after_result = await evaluator.run_once(after_hour)

        assert before_result == []
        assert len(after_result) == 1


class TestProtoRoundTrip:
    """Enabled-mode signals must serialize to a valid, non-UNSPECIFIED proto kind."""

    async def test_emitted_signal_round_trips_through_proto_bindings(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        from app.proto.continuoustracking.v1 import signals_pb2
        from app.transport.signal_publisher import _to_proto

        alice = "alice"
        yesterday = date(2026, 2, 9)
        today = date(2026, 2, 10)
        await _seed_day(ph_repo, alice, yesterday, (1.0, 0.0))
        await _seed_day(ph_repo, alice, today, (1.0, 0.0))

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=True)
        now = _utc(2026, 2, 10, 16, 0)
        signals = await evaluator.run_once(now)
        assert len(signals) == 1

        proto = _to_proto(signals[0])
        assert proto.kind == signals_pb2.DEMENTIA_SIGNAL_KIND_SAME_CLOTHES_SUSPECTED
        assert proto.kind != signals_pb2.DEMENTIA_SIGNAL_KIND_UNSPECIFIED
        assert proto.severity == signals_pb2.DEMENTIA_SIGNAL_SEVERITY_INFO


class TestLocalDayBoundary:
    async def test_ph_at_2350_local_lands_in_correct_day(
        self, ph_repo: InMemoryPHRepository, profile_repo: InMemoryDailyAppearanceRepo
    ) -> None:
        from zoneinfo import ZoneInfo

        alice = "alice"
        yesterday = date(2026, 2, 9)
        tz = ZoneInfo(_TZ)
        late_local = datetime(2026, 2, 9, 23, 50, tzinfo=tz).astimezone(UTC)

        for i in range(5):
            await ph_repo.save(
                _make_ph(
                    f"late-{i}",
                    alice,
                    late_local + timedelta(seconds=i),
                    (1.0, 0.0),
                    closed_at=late_local + timedelta(minutes=1),
                )
            )

        evaluator = _evaluator(ph_repo, profile_repo, [alice], enabled=False)
        # Force a backfill of "yesterday" (Feb 9) directly.
        result = await evaluator._get_or_backfill_day(alice, yesterday)
        assert result is not None
        assert result.sample_count == 5

        # A PH at 00:10 local (Feb 10) must NOT land in Feb 9's profile.
        next_day_repo = InMemoryPHRepository()
        early_next_day = datetime(2026, 2, 10, 0, 10, tzinfo=tz).astimezone(UTC)
        for i in range(5):
            await next_day_repo.save(
                _make_ph(
                    f"early-{i}",
                    alice,
                    early_next_day + timedelta(seconds=i),
                    (1.0, 0.0),
                    closed_at=early_next_day + timedelta(minutes=1),
                )
            )
        evaluator2 = _evaluator(
            next_day_repo, InMemoryDailyAppearanceRepo(), [alice], enabled=False
        )
        result2 = await evaluator2._get_or_backfill_day(alice, yesterday)
        assert result2 is None  # nothing overlapping Feb 9 in this repo
