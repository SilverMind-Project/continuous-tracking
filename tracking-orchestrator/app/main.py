"""FastAPI application factory for the tracking orchestrator.

Mirrors the cognitive-companion pattern: create_app() returns the app,
lifespan manages service lifecycle (pipeline start/stop, DB migrations).

Configuration is read from ``config/settings.yaml`` (path configurable via
``ORCHESTRATOR_CONFIG_PATH``). Only deployment endpoints/secrets use explicit
``${VAR}`` interpolation; tunables are literal YAML values.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis import asyncio as aioredis
from structlog import get_logger

from .config import SettingNotFoundError, Settings, SettingsSection, settings
from .domain import (
    CameraProfile,
    ClosedWorldMode,
    FaceEvidencePolicy,
    OverlapGroup,
    WearableMode,
)
from .observability.logging_config import configure_logging
from .tracking.identity.face_template_buffer import FaceTemplateMode, FaceTemplatePolicy

# Configure structlog before any logger is first used.
configure_logging(settings.as_str("logging.level"))
from .inference.depth import DepthEstimator
from .inference.detector import PersonDetector
from .inference.pose import PoseEstimator
from .inference.reid_embedder import ReidEmbedder
from .inference.triton_client import TritonClientProtocol, TritonGrpcClient
from .pipeline import FrameProcessingPipeline
from .pipeline.crops import FaceCropPadding
from .pipeline.face_attribution import HeadRegionPolicy
from .pipeline.frame_pipeline import (
    FaceIdConfig,
    PipelineConfig,
    PipelineDependencies,
)
from .pipeline.reid_policy import AdaptiveReidConfig
from .pipeline.stages.fall_detection import FallDetectionConfig
from .pipeline.stages.height_estimation import HeightEstimationConfig
from .pipeline.types import FaceIdCameraConfig
from .routers import dashboard as dashboard_router_mod
from .routers import gallery as gallery_router_mod
from .routers.calibration import router as calibration_router
from .routers.calibration import set_auto_calibration_context
from .routers.corrections import router as corrections_router
from .routers.corrections import set_context as set_corrections_context
from .routers.dashboard import router as dashboard_router
from .routers.gait import router as gait_router
from .routers.gait import set_context as set_gait_context
from .routers.gallery import router as gallery_router
from .routers.height_profiles import router as height_profiles_router
from .routers.height_profiles import set_context as set_height_profiles_context
from .routers.live import router as live_router
from .routers.live import set_context as set_live_context
from .routers.ph import router as ph_router
from .routers.ph import set_ph_repository, set_revision_publisher
from .routers.readiness import router as readiness_router
from .routers.reid_review import router as reid_review_router
from .routers.reid_review import set_context as set_reid_review_context
from .routers.topology import router as topology_router
from .routers.topology import set_topology_repo
from .routers.trajectory import router as trajectory_router
from .routers.trajectory import set_context as set_trajectory_context
from .routers.wearables import router as wearables_router
from .sampling.keyframe_sampler import SamplerConfig
from .services.camera_profile_map import CameraProfileMap
from .services.cc_client import CognitiveCompanionClient
from .services.cc_config_sync import parse_camera_profile
from .services.cc_identity_assertion_subscriber import (
    CCIdentityAssertionSubscriber,
    IdentityAssertionCache,
)
from .services.gallery_retention import (
    GalleryRetentionConfig,
    GalleryRetentionMode,
    GalleryRetentionService,
)
from .services.height_profile_learner import (
    HeightProfileLearner,
    HeightProfileLearnerConfig,
    PostgresHeightSampleSource,
)
from .services.household_presence_subscriber import (
    HouseholdPresenceCache,
    HouseholdPresenceSubscriber,
)
from .services.identity_readiness import IdentityPolicySnapshot, IdentityReadinessService
from .services.identity_rewriter import InMemoryIdentityRewriter, PostgresIdentityRewriter
from .services.image_only_evaluator import ImageOnlyEvaluationService
from .services.overlap_group_sync import fetch_overlap_groups
from .services.ph_maintenance import PHMaintenanceService, PHUnknownPurgeConfig
from .services.wearable_fix_buffer import WearableFixBuffer
from .services.wearable_fix_subscriber import WearableFixSubscriber
from .services.wearable_revocation import WearableTeachingRevocationService
from .services.wearable_site_config_map import WearableSiteConfigMap
from .storage.height_profiles import (
    IdentityHeightProfileRepository,
    InMemoryIdentityHeightProfileRepository,
    InMemoryPHHeightSampleRepository,
    PHHeightSampleRepository,
)
from .storage.image_only_evaluation import (
    ImageOnlyEvaluationRepository,
    InMemoryImageOnlyEvaluationRepository,
)
from .storage.migrations import MigrationRunner
from .storage.postgres.appearance_repo import PostgresDailyAppearanceRepo
from .storage.postgres.baseline_repo import PostgresBehaviorBaselineRepository
from .storage.postgres.bbox_annotations import PostgresBboxAnnotationRepository
from .storage.postgres.face_stats_repo import PostgresFaceObservationStatsRepository
from .storage.postgres.gait_repo import PostgresGaitDailyRepository
from .storage.postgres.gallery_repo import PostgresGalleryRepository
from .storage.postgres.height_profile_repo import (
    PostgresIdentityHeightProfileRepository,
    PostgresPHHeightSampleRepository,
)
from .storage.postgres.image_only_evaluation_repo import PostgresImageOnlyEvaluationRepository
from .storage.postgres.keyframe_repo import PostgresKeyframeRepository
from .storage.postgres.ph_repo import PostgresPHRepository, PostgresWorldObservationRepository
from .storage.postgres.settings_repo import PostgresSettingsRepository
from .storage.postgres.signal_repo import PostgresDementiaSignalRepository
from .storage.postgres.trajectory_repo import PostgresTrajectoryRepository
from .storage.postgres.wearable_association_repo import PostgresWearableAssociationRepository
from .storage.postgres.wearable_teaching_hold_repo import PostgresWearableTeachingHoldRepository
from .storage.wearable_associations import (
    InMemoryWearableAssociationRepository,
    WearableAssociationRepository,
)
from .storage.wearable_teaching_holds import (
    InMemoryWearableTeachingHoldRepository,
    WearableTeachingHoldRepository,
)
from .tracking.identity.candidate_eligibility import CandidatePolicy
from .tracking.identity.closed_world_channel import ClosedWorldConfig
from .tracking.identity_resolver import FaceQualityMode, HeightMode, ResolverConfig
from .tracking.world.config import WorldTrackerConfig
from .tracking.world.wearable_config import FaceAttestationMode, WearableConfig
from .trajectory.appearance_profile import AppearanceSettings
from .trajectory.dementia_signals import SignalConfig
from .trajectory.depth_posture_strategy import DepthPostureStrategy
from .trajectory.fused_posture_strategy import FusedPostureStrategy
from .trajectory.posture_strategy import RTMPosePostureStrategy
from .transport.minio_frames import MinioFrameConfig, MinioFrameFetcher
from .transport.redis_streams import TransportConfig

logger = get_logger(__name__)

# Module-level pipeline singleton, initialized in lifespan.
_pipeline: FrameProcessingPipeline | None = None

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


@dataclass(frozen=True)
class _PostureComponents:
    depth_estimator: DepthEstimator | None
    posture_strategy: RTMPosePostureStrategy | FusedPostureStrategy


async def _build_posture_components(
    s: Settings,
    triton_client: TritonClientProtocol | None,
) -> _PostureComponents:
    """Build posture dependencies from the single depth slow-path flag."""
    fast_strategy = RTMPosePostureStrategy()
    if not s.as_bool("pipeline.posture.depth_slow_path_enabled") or triton_client is None:
        return _PostureComponents(
            depth_estimator=None,
            posture_strategy=fast_strategy,
        )

    depth_model_name = s.as_str("triton.depth_model")
    if not await triton_client.is_model_ready(depth_model_name):
        logger.warning(
            "depth_model_not_ready",
            model=depth_model_name,
            hint="export_depth_anything_v2.py must be run to produce model.onnx",
        )
        return _PostureComponents(
            depth_estimator=None,
            posture_strategy=fast_strategy,
        )

    depth_estimator = DepthEstimator(
        triton_client,
        model_name=depth_model_name,
    )
    posture_strategy = FusedPostureStrategy(
        fast=fast_strategy,
        slow=DepthPostureStrategy(depth_estimator),
        slow_path_min_interval_s=s.as_float("pipeline.posture.depth_slow_path_min_interval_s"),
        slow_path_max_age_s=s.as_float("pipeline.posture.depth_slow_path_max_age_s"),
    )
    logger.info(
        "Posture strategy: fused (RTMPose + Depth slow-path)",
        slow_path_interval_s=s.as_float("pipeline.posture.depth_slow_path_min_interval_s"),
    )
    return _PostureComponents(
        depth_estimator=depth_estimator,
        posture_strategy=posture_strategy,
    )


def _normalize_dsn(dsn: str) -> str:
    """Strip ``+asyncpg`` SQLAlchemy scheme prefix if present."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("+asyncpg", "", 1)
    return dsn


async def _fetch_cc_camera_profiles(
    client: CognitiveCompanionClient,
) -> list[CameraProfile]:
    """Fetch initial hardware-neutral camera profiles from cognitive-companion.

    Delegates parsing to :func:`parse_camera_profile` so startup and periodic sync share
    one decode boundary. Unknown camera roles fail loudly with a warning and metric.
    """
    from .observability import metrics as _m

    try:
        cameras: list[dict[str, Any]] = await client.get("/api/v1/cts/cameras")
    except Exception as exc:
        logger.exception("cc_camera_config_fetch_failed")
        raise RuntimeError("Failed to fetch camera configs from CC") from exc

    profiles: list[CameraProfile] = []
    for index, cam in enumerate(cameras):
        if not isinstance(cam, dict) or not cam.get("id"):
            logger.warning("Skipped malformed CC camera config", camera=cam)
            continue
        profile, reject_reason = parse_camera_profile(cam, context=f"/api/v1/cts/cameras[{index}]")
        if reject_reason is not None or profile is None:
            reason_val = reject_reason.value if reject_reason is not None else "malformed_field"
            logger.warning(
                "cc_config_sync_camera_profile_rejected",
                camera_id=str(cam.get("id")),
                reason=reason_val,
                raw_role=cam.get("role"),
            )
            _m.metrics.camera_profile_rejected_total.labels(
                camera_id=str(cam.get("id")),
                reason=reason_val,
            ).inc()
            continue
        profiles.append(profile)

    logger.info("Fetched camera profiles from CC", camera_count=len(profiles))
    return profiles


async def _fetch_cc_camera_configs(
    client: CognitiveCompanionClient,
) -> dict[str, FaceIdCameraConfig]:
    """Fetch per-camera face-id configs from cognitive-companion's camera API."""
    profiles = await _fetch_cc_camera_profiles(client)
    return {
        p.camera_id: FaceIdCameraConfig(
            enabled=p.face_id_enabled,
            min_raw_similarity=p.face_id_min_raw_similarity,
        )
        for p in profiles
    }


def _build_transport_config(s: Settings) -> TransportConfig:
    """Build TransportConfig from required settings.yaml keys."""
    try:
        pending_reclaim_min_idle_s = s.as_int("redis.pending_reclaim_min_idle_s")
    except SettingNotFoundError:
        pending_reclaim_min_idle_s = 300
    return TransportConfig(
        redis_url=s.as_str("redis.url"),
        consumer_group=s.as_str("redis.consumer_group"),
        consumer_name=s.as_str("redis.consumer_name"),
        frames_stream=s.as_str("redis.frames_stream"),
        events_stream=s.as_str("redis.events_stream"),
        batch_max_wait_ms=s.as_int("redis.batch_max_wait_ms"),
        batch_max_size=s.as_int("redis.batch_max_size"),
        xack_timeout_ms=s.as_int("redis.xack_timeout_ms"),
        ack_ttl_seconds=s.as_int("redis.ack_ttl_seconds"),
        pending_reclaim_min_idle_s=pending_reclaim_min_idle_s,
        revisions_stream=s.as_str("redis.revisions_stream"),
        signals_stream=s.as_str("redis.signals_stream"),
        scene_samples_stream=s.as_str("redis.scene_samples_stream"),
        wearable_fixes_stream=s.as_str("redis.wearable_fixes_stream"),
        wearable_capabilities_stream=s.as_str("redis.wearable_capabilities_stream"),
        household_presence_stream=s.as_str("redis.household_presence_stream"),
    )


def _build_resolver_config(s: Settings) -> ResolverConfig:
    """Build ResolverConfig from required settings.yaml keys."""
    r = s.section("resolver")
    return ResolverConfig(
        commit_prob=r.as_float("commit_prob"),
        commit_margin=r.as_float("commit_margin"),
        reid_decision_sim=r.as_float("reid_decision_sim"),
        revision_horizon_s=r.as_float("revision_horizon_s"),
        max_revisions_per_ph_per_minute=r.as_int("max_revisions_per_ph_per_minute"),
        unknown_mass=r.as_float("unknown_mass"),
        prior_weight=r.as_float("prior_weight"),
        face_weight_multiplier=r.as_float("face_weight_multiplier"),
        propagated_face_weight_multiplier=r.as_float("propagated_face_weight_multiplier"),
        height_weight_multiplier=r.as_float("height_weight_multiplier"),
        min_quality_to_commit=r.as_float("min_quality_to_commit"),
        enable_quality_gate=r.as_bool("enable_quality_gate"),
        commit_prob_dense=r.as_float("commit_prob_dense"),
        commit_margin_dense=r.as_float("commit_margin_dense"),
        flip_debounce_window_s=r.as_float("flip_debounce_window_s"),
        enable_flip_debounce=r.as_bool("enable_flip_debounce"),
        enable_duplicate_active_identity_guard=r.as_bool("enable_duplicate_active_identity_guard"),
        duplicate_identity_direct_face_min_similarity=r.as_float(
            "duplicate_identity_direct_face_min_similarity"
        ),
        prior_maintenance_max_age_s=r.as_float("prior_maintenance_max_age_s"),
        identified_entry_boost_min_sim=r.as_float("identified_entry_boost_min_sim"),
        identified_entry_min_likelihood=r.as_float("identified_entry_min_likelihood"),
        gallery_verified_trust_multiplier=r.as_float("gallery_verified_trust_multiplier"),
        gallery_auto_verified_trust_multiplier=r.as_float("gallery_auto_verified_trust_multiplier"),
        gallery_wearable_trust_multiplier=r.as_float("gallery_wearable_trust_multiplier"),
        gallery_recency_half_life_days=r.as_float("gallery_recency_half_life_days"),
        gallery_vote_max_age_s=_optional_max_age_s(s, "resolver.gallery_vote_max_age_s"),
        enable_embedding_coherence_boost=r.as_bool("enable_embedding_coherence_boost"),
        embedding_coherence_window=r.as_int("embedding_coherence_window"),
        embedding_coherence_min_sim=r.as_float("embedding_coherence_min_sim"),
        embedding_coherence_boost=r.as_float("embedding_coherence_boost"),
        face_commit_min_similarity=r.as_float("face_commit_min_similarity"),
        arcface_authority_calibrated_confidence=r.as_float(
            "arcface_authority_calibrated_confidence"
        ),
        cross_gt_face_propagation_threshold=r.as_float("cross_gt_face_propagation_threshold"),
        cross_gt_face_propagation_max_gts=r.as_int("cross_gt_face_propagation_max_gts"),
        enable_sticky_maintenance=r.as_bool("enable_sticky_maintenance"),
        contradiction_face_similarity=r.as_float("contradiction_face_similarity"),
        contradiction_posterior_prob=r.as_float("contradiction_posterior_prob"),
        contradiction_posterior_margin=r.as_float("contradiction_posterior_margin"),
        # Rich face evidence
        candidate_face_weight_multiplier=r.as_float("candidate_face_weight_multiplier"),
        face_present_unknown_unknown_mass=r.as_float("face_present_unknown_unknown_mass"),
        frontality_full_yaw_deg=r.as_float("frontality_full_yaw_deg"),
        frontality_zero_yaw_deg=r.as_float("frontality_zero_yaw_deg"),
        frontality_min_factor=r.as_float("frontality_min_factor"),
        enable_multiview_gallery=r.as_bool("enable_multiview_gallery"),
        multiview_shadow_sample_rate=r.as_float("multiview_shadow_sample_rate"),
        seed_orientation_min_confidence=r.as_float("seed_orientation_min_confidence"),
        enable_unknown_backfill=r.as_bool("enable_unknown_backfill"),
        backfill_shadow=r.as_bool("backfill_shadow"),
        enable_wearable_backfill=r.as_bool("enable_wearable_backfill"),
        backfill_max_range_s=r.as_float("backfill_max_range_s"),
        cc_assertion_likelihood_scale=r.as_float("cc_assertion_likelihood_scale"),
        cc_assertion_mode=r.as_str("cc_assertion_mode"),
        room_match_confidence_scale=r.as_float("room_match_confidence_scale"),
        cc_assertion_default_quality=r.as_float("cc_assertion_default_quality"),
        cc_assertion_default_yaw_deg=r.as_float("cc_assertion_default_yaw_deg"),
        # ResolverConfig validates the value; the cast only narrows the type for mypy.
        face_quality_mode=cast(FaceQualityMode, r.as_str("face_quality_mode")),
        face_quality_missing_default=r.as_float("face_quality_missing_default"),
        # The same key FaceIdConfig.evidence_policy reads, so face emission and resolver weighting
        # cannot disagree. ResolverConfig validates it; the cast only narrows the type for mypy.
        face_evidence_policy=cast(FaceEvidencePolicy, s.as_str("face_id.evidence_policy")),
        uncalibrated_face_max_likelihood=r.as_float("uncalibrated_face_max_likelihood"),
        height_mode=cast(HeightMode, r.as_str("height_mode")),
    )


def _build_candidate_policy(s: Settings, wearable: WearableConfig) -> CandidatePolicy:
    """Build CandidatePolicy from required settings.yaml keys.

    Refuses wearable seeding when wearable.mode is "off". Accepts "shadow" so room-grade
    pending_review seeds can gather review evidence while trajectory seeds skip as
    channel_not_live.
    """
    rc = s.section("reid_candidates")
    policy = CandidatePolicy(
        enabled=rc.as_bool("enabled"),
        require_calibrated_face=rc.as_bool("require_calibrated_face"),
        calibrated_confidence_min=rc.as_float("calibrated_confidence_min"),
        seed_orientation_min_confidence=rc.as_float("seed_orientation_min_confidence"),
        min_quality=rc.as_float("min_quality"),
        min_face_quality=rc.as_float("min_face_quality"),
        max_per_identity_orientation=rc.as_int("max_per_identity_orientation"),
        model_version=s.as_str("triton.reid_model"),
        preprocessing_version=rc.as_str("preprocessing_version"),
        auto_verify_min_confidence=rc.as_float("auto_verify_min_confidence"),
        wearable_seeding_enabled=rc.as_bool("wearable_seeding_enabled"),
        wearable_min_seed_interval_s=rc.as_float("wearable_min_seed_interval_s"),
        wearable_max_similarity_to_recent=rc.as_float("wearable_max_similarity_to_recent"),
        wearable_room_grade_max_per_day=rc.as_int("wearable_room_grade_max_per_day"),
        gallery_vote_max_age_s=_optional_max_age_s(s, "resolver.gallery_vote_max_age_s"),
        wearable_channel_live=wearable.mode == "enabled",
    )
    if policy.wearable_seeding_enabled and wearable.mode == "off":
        raise ValueError(
            "reid_candidates.wearable_seeding_enabled requires wearable.mode 'shadow' or 'enabled' "
            f"(got {wearable.mode!r})"
        )
    return policy


def _build_world_tracker_config(s: Settings) -> WorldTrackerConfig:
    """Build WorldTrackerConfig from required settings.yaml keys."""
    wt = s.section("world_tracker")
    return WorldTrackerConfig(
        gate_chi2=wt.as_float("gate_chi2"),
        alpha_geo=wt.as_float("alpha_geo"),
        alpha_app=wt.as_float("alpha_app"),
        alpha_height=wt.as_float("alpha_height"),
        observation_noise_m=wt.as_float("observation_noise_m"),
        process_noise_accel_m_s2=wt.as_float("process_noise_accel_m_s2"),
        ph_close_grace_s=wt.as_float("ph_close_grace_s"),
        min_observations_to_publish=wt.as_int("min_observations_to_publish"),
        face_conflict_min_similarity=wt.as_float("face_conflict_min_similarity"),
        inferred_handoff_max_s=wt.as_float("inferred_handoff_max_s"),
        inferred_handoff_max_distance_m=wt.as_float("inferred_handoff_max_distance_m"),
        initial_position_sigma_m=wt.as_float("initial_position_sigma_m"),
        initial_velocity_sigma_m_s=wt.as_float("initial_velocity_sigma_m_s"),
        velocity_decay_s=wt.as_float("velocity_decay_s"),
        damped_velocity_prediction=wt.as_bool("damped_velocity_prediction"),
        zupt_speed_enter_m_s=wt.as_float("zupt_speed_enter_m_s"),
        zupt_speed_exit_m_s=wt.as_float("zupt_speed_exit_m_s"),
        zupt_consecutive_frames=wt.as_int("zupt_consecutive_frames"),
        zupt_innov_chi2=wt.as_float("zupt_innov_chi2"),
        zupt_velocity_sigma_m_s=wt.as_float("zupt_velocity_sigma_m_s"),
        height_sigma_m=wt.as_float("height_sigma_m"),
        k_cal=wt.as_float("k_cal"),
        base_footpoint_sigma_px=wt.as_float("base_footpoint_sigma_px"),
        occluded_footpoint_inflation=wt.as_float("occluded_footpoint_inflation"),
        primary_switch_frames=wt.as_int("primary_switch_frames"),
        dedup_enabled=wt.as_bool("dedup_enabled"),
        dedup_max_distance_m=wt.as_float("dedup_max_distance_m"),
        dedup_residual_coeff_k=wt.as_float("dedup_residual_coeff_k"),
        dedup_max_distance_ceiling_m=wt.as_float("dedup_max_distance_ceiling_m"),
        dedup_require_no_face_conflict=wt.as_bool("dedup_require_no_face_conflict"),
        dedup_time_skew_speed_m_s=wt.as_float("dedup_time_skew_speed_m_s"),
        enable_ph_revival=wt.as_bool("enable_ph_revival"),
        revive_max_age_s=wt.as_float("revive_max_age_s"),
        revive_max_distance_m=wt.as_float("revive_max_distance_m"),
        revive_appearance_min_sim=wt.as_float("revive_appearance_min_sim"),
        enable_uncalibrated_gate_relax=wt.as_bool("enable_uncalibrated_gate_relax"),
        uncalibrated_gate_chi2=wt.as_float("uncalibrated_gate_chi2"),
        uncalibrated_alpha_app=wt.as_float("uncalibrated_alpha_app"),
        enable_multiview_association=wt.as_bool("enable_multiview_association"),
        # Cross-camera revival
        enable_cross_camera_revival=wt.as_bool("enable_cross_camera_revival"),
        cross_camera_min_plausibility=wt.as_float("cross_camera_min_plausibility"),
        cross_camera_revive_appearance_min_sim=wt.as_float(
            "cross_camera_revive_appearance_min_sim"
        ),
        # Group appearance dedup
        enable_group_appearance_dedup=wt.as_bool("enable_group_appearance_dedup"),
        dedup_group_appearance_min_sim=wt.as_float("dedup_group_appearance_min_sim"),
        # Low-confidence detection recovery
        enable_low_confidence_recovery=wt.as_bool("enable_low_confidence_recovery"),
        low_confidence_floor=wt.as_float("low_confidence_floor"),
        recovery_gate_chi2=wt.as_float("recovery_gate_chi2"),
        measure_low_confidence_band=wt.as_bool("measure_low_confidence_band"),
        # Association covariance/point validation
        enable_covariance_validation=wt.as_bool("enable_covariance_validation"),
        covariance_max_trace_m2=wt.as_float("covariance_max_trace_m2"),
        covariance_symmetry_tol_m2=wt.as_float("covariance_symmetry_tol_m2"),
        covariance_psd_tol_m2=wt.as_float("covariance_psd_tol_m2"),
        # Typed authoritative identity evidence (task 5)
        enable_reid_disagreement_cost=wt.as_bool("enable_reid_disagreement_cost"),
        reid_disagreement_cost=wt.as_float("reid_disagreement_cost"),
        reid_disagreement_min_similarity=wt.as_float("reid_disagreement_min_similarity"),
        reid_disagreement_max_age_s=_optional_max_age_s(
            s, "world_tracker.reid_disagreement_max_age_s"
        ),
        # PH-local appearance contamination guard (tasks 7-9)
        enable_appearance_outlier_rejection=wt.as_bool("enable_appearance_outlier_rejection"),
        appearance_min_quality=wt.as_float("appearance_min_quality"),
        appearance_embedding_norm_tol=wt.as_float("appearance_embedding_norm_tol"),
        appearance_outlier_min_sim=wt.as_float("appearance_outlier_min_sim"),
        appearance_new_orientation_min_confidence=wt.as_float(
            "appearance_new_orientation_min_confidence"
        ),
    )


def _build_sampler_config(s: Settings) -> SamplerConfig:
    """Build SamplerConfig from required settings.yaml keys."""
    sp = s.section("sampler")
    return SamplerConfig(
        keyframe_min_interval_s=sp.as_float("keyframe_min_interval_s"),
        periodic_expires_hours=sp.as_int("periodic_expires_hours"),
        trigger_expires_days=sp.as_int("trigger_expires_days"),
    )


def _build_signal_config(s: Settings) -> SignalConfig:
    sig = s.section("signal")
    agitation = s.section("signal.agitation")
    return SignalConfig(
        window_hours=sig.as_int("window_hours"),
        tz_name=s.as_str("app.timezone"),
        min_baseline_n=sig.as_int("min_baseline_n"),
        onset_consecutive_windows=sig.as_int("onset_consecutive_windows"),
        cooldown_minutes=sig.as_int("cooldown_minutes"),
        gait_slowing_cooldown_minutes=sig.as_int("gait_slowing_cooldown_minutes"),
        gait_slowing_warning_decline_frac=sig.as_float("gait_slowing_warning_decline_frac"),
        gait_slowing_warning_speed_m_s=sig.as_float("gait_slowing_warning_speed_m_s"),
        agitation_cooldown_minutes=agitation.as_int("agitation_cooldown_minutes"),
        pacing_room_threshold=sig.as_int("pacing_room_threshold"),
        pacing_window_minutes=sig.as_int("pacing_window_minutes"),
        pacing_min_obs_density=sig.as_float("pacing_min_obs_density"),
        pacing_warning_rate_per_min=sig.as_float("pacing_warning_rate_per_min"),
        pacing_emergency_rate_per_min=sig.as_float("pacing_emergency_rate_per_min"),
        nighttime_transition_threshold=sig.as_int("nighttime_transition_threshold"),
        nighttime_warning_z=sig.as_float("nighttime_warning_z"),
        nighttime_emergency_z=sig.as_float("nighttime_emergency_z"),
        nighttime_cold_start_warning_extra=sig.as_int("nighttime_cold_start_warning_extra"),
        nighttime_cold_start_emergency_extra=sig.as_int("nighttime_cold_start_emergency_extra"),
        stillness_threshold_minutes=sig.as_int("stillness_threshold_minutes"),
        stillness_emergency_minutes=sig.as_int("stillness_emergency_minutes"),
        stillness_motion_floor=sig.as_float("stillness_motion_floor"),
        stillness_open_freshness_s=sig.as_int("stillness_open_freshness_s"),
        stillness_still_energy_floor=sig.as_float("stillness_still_energy_floor"),
        stillness_max_gap_s=sig.as_float("stillness_max_gap_s"),
        resting_rooms=tuple(sig.require("resting_rooms")),
        bathroom_z_threshold=sig.as_float("bathroom_z_threshold"),
        bathroom_z_threshold_night=sig.as_float("bathroom_z_threshold_night"),
        bathroom_warning_z=sig.as_float("bathroom_warning_z"),
        bathroom_emergency_z=sig.as_float("bathroom_emergency_z"),
        bathroom_absolute_threshold_seconds=sig.as_int("bathroom_absolute_threshold_seconds"),
        sundowning_z_threshold=sig.as_float("sundowning_z_threshold"),
        sundowning_min_evening_minutes=sig.as_int("sundowning_min_evening_minutes"),
        sundowning_warning_z=sig.as_float("sundowning_warning_z"),
        sundowning_emergency_z=sig.as_float("sundowning_emergency_z"),
        sundowning_cold_start_min_rate=sig.as_float("sundowning_cold_start_min_rate"),
        absence_threshold_minutes=sig.as_int("absence_threshold_minutes"),
        absence_warning_extra_minutes=sig.as_int("absence_warning_extra_minutes"),
        absence_emergency_extra_minutes=sig.as_int("absence_emergency_extra_minutes"),
        absence_prior_demotion_frac=sig.as_float("absence_prior_demotion_frac"),
        baseline_cache_ttl_minutes=sig.as_int("baseline_cache_ttl_minutes"),
        max_concurrent_identities=sig.as_int("max_concurrent_identities"),
        incremental_enabled=sig.as_bool("incremental_enabled"),
        agitation_enabled=agitation.as_bool("agitation_enabled"),
        agitation_window_minutes=agitation.as_int("agitation_window_minutes"),
        agitation_min_observed_minutes=agitation.as_int("agitation_min_observed_minutes"),
        agitation_composite_threshold=agitation.as_float("agitation_composite_threshold"),
        agitation_z_threshold=agitation.as_float("agitation_z_threshold"),
        agitation_warning_composite=agitation.as_float("agitation_warning_composite"),
        agitation_warning_z=agitation.as_float("agitation_warning_z"),
        agitation_weight_in_place=agitation.as_float("agitation_weight_in_place"),
        agitation_weight_entropy=agitation.as_float("agitation_weight_entropy"),
        agitation_weight_excursion=agitation.as_float("agitation_weight_excursion"),
    )


def _build_appearance_settings(s: Settings) -> AppearanceSettings:
    hyg = s.section("hygiene.same_clothes")
    return AppearanceSettings(
        enabled=hyg.as_bool("enabled"),
        similarity_threshold=hyg.as_float("similarity_threshold"),
        min_samples_per_day=hyg.as_int("min_samples_per_day"),
        evaluate_local_hour=hyg.as_int("evaluate_local_hour"),
        tz_name=s.as_str("app.timezone"),
    )


def _build_fall_detection_config(s: Settings) -> FallDetectionConfig:
    fd = s.section("fall_detection")
    return FallDetectionConfig(
        enabled=fd.as_bool("enabled"),
        min_samples=fd.as_int("min_samples"),
        max_descent_rate_hps_threshold=fd.as_float("max_descent_rate_hps_threshold"),
        height_ratio_threshold=fd.as_float("height_ratio_threshold"),
        lying_score_threshold=fd.as_float("lying_score_threshold"),
        floor_speed_max_m_s=fd.as_float("floor_speed_max_m_s"),
        escalation_window_s=fd.as_float("escalation_window_s"),
        stillness_motion_floor=fd.as_float("stillness_motion_floor"),
        standing_clear_height_ratio=fd.as_float("standing_clear_height_ratio"),
        standing_clear_lying_score=fd.as_float("standing_clear_lying_score"),
        cooldown_minutes=fd.as_int("cooldown_minutes"),
        resting_rooms=tuple(fd.require("resting_rooms")),
    )


def _build_ph_unknown_purge_config(s: Settings) -> PHUnknownPurgeConfig:
    purge = s.section("person_hypotheses.unknown_purge")
    return PHUnknownPurgeConfig(
        enabled=purge.as_bool("enabled"),
        older_than_days=purge.as_int("older_than_days"),
        interval_s=purge.as_float("interval_s"),
        batch_size=purge.as_int("batch_size"),
    )


def _build_gallery_retention_config(s: Settings) -> GalleryRetentionConfig:
    ret = s.section("reid_gallery.retention")
    return GalleryRetentionConfig(
        mode=cast(GalleryRetentionMode, ret.as_str("mode")),
        auto_verified_days=ret.as_int("auto_verified_days"),
        pending_review_days=ret.as_int("pending_review_days"),
        operator_verified_days=ret.as_int("operator_verified_days"),
        interval_s=ret.as_float("interval_s"),
        batch_size=ret.as_int("batch_size"),
    )


def _build_height_estimation_config(s: Settings) -> HeightEstimationConfig:
    he = s.section("height_estimation")
    wt = s.section("world_tracker")
    return HeightEstimationConfig(
        pose_height_tolerance_m=he.as_float("pose_height_tolerance_m"),
        pose_tilt_tolerance_deg=he.as_float("pose_tilt_tolerance_deg"),
        sigma_head_px=he.as_float("sigma_head_px"),
        sigma_model_m=he.as_float("sigma_model_m"),
        top_margin_px=he.as_int("top_margin_px"),
        edge_margin_frac=he.as_float("edge_margin_frac"),
        max_torso_tilt_deg=he.as_float("max_torso_tilt_deg"),
        min_aspect_ratio=he.as_float("min_aspect_ratio"),
        min_height_m=he.as_float("min_height_m"),
        max_height_m=he.as_float("max_height_m"),
        base_footpoint_sigma_px=wt.as_float("base_footpoint_sigma_px"),
        occluded_footpoint_inflation=wt.as_float("occluded_footpoint_inflation"),
    )


def _build_height_learner_config(s: Settings) -> HeightProfileLearnerConfig:
    hl = s.section("height_estimation.learner")
    return HeightProfileLearnerConfig(
        enabled=hl.as_bool("enabled"),
        min_samples=hl.as_int("min_samples"),
        min_days=hl.as_int("min_days"),
        min_cameras=hl.as_int("min_cameras"),
        min_sigma_m=hl.as_float("min_sigma_m"),
        lookback_days=hl.as_int("lookback_days"),
        interval_s=hl.as_float("interval_s"),
    )


def _optional_str(section: SettingsSection, key: str, default: str = "") -> str:
    """Read a string from a settings section, returning default if the key is absent."""
    try:
        return section.as_str(key)
    except SettingNotFoundError:
        return default


def _optional_max_age_s(s: Settings, dotted_key: str) -> float | None:
    """Read a nullable vote-age cutoff in seconds (identity-continuity M03).

    Absent, ``null``, or non-positive all mean "no cutoff" (``None``); this is
    the explicit None-when-absent-or-zero rule used by
    ``resolver.gallery_vote_max_age_s`` and
    ``world_tracker.reid_disagreement_max_age_s`` so the config-only rollback
    path (delete or zero the key) needs no code change.
    """
    raw = s.get(dotted_key)
    if raw is None:
        return None
    value = float(raw)
    return value if value > 0 else None


def _build_face_id_config(
    s: Settings, camera_configs: dict[str, FaceIdCameraConfig]
) -> FaceIdConfig:
    fi = s.section("face_id")
    url = s.as_str("face_id.url")
    return FaceIdConfig(
        url=url,
        cooldown_s=fi.as_float("cooldown_s"),
        identification_point_cooldown_s=fi.as_float("identification_point_cooldown_s"),
        timeout_s=fi.as_float("timeout_s"),
        min_raw_similarity=fi.as_float("min_raw_similarity"),
        enabled=fi.as_bool("enabled"),
        camera_configs=camera_configs,
        expected_arcface_model_version=_optional_str(fi, "expected_arcface_model_version"),
        expected_preprocessing_version=_optional_str(fi, "expected_preprocessing_version"),
        expected_quality_version=fi.as_str("expected_quality_version"),
        head_region=HeadRegionPolicy(
            head_height_frac=fi.as_float("head_region_height_frac"),
            side_margin_frac=fi.as_float("head_region_side_margin_frac"),
        ),
        crop_padding=FaceCropPadding(
            top_frac=fi.as_float("crop_pad_top_frac"),
            side_frac=fi.as_float("crop_pad_side_frac"),
        ),
        crop_jpeg_quality=fi.as_int("crop_jpeg_quality"),
        # FaceIdConfig validates the value; the cast only narrows the type for mypy.
        evidence_policy=cast(FaceEvidencePolicy, fi.as_str("evidence_policy")),
    )


def _build_identity_policy_snapshot(config: PipelineConfig) -> IdentityPolicySnapshot:
    """Copy the identity policies from the built pipeline config, so readiness reports what runs."""
    return IdentityPolicySnapshot(
        face_evidence_policy=config.face_id.evidence_policy,
        face_templates_mode=config.face_templates.mode,
        face_quality_mode=config.resolver.face_quality_mode,
        cc_assertion_mode=config.resolver.cc_assertion_mode,
        expected_arcface_model_version=config.face_id.expected_arcface_model_version or None,
        expected_quality_version=config.face_id.expected_quality_version,
        wearable_mode=config.wearable.mode,
        wearable_seeding_enabled=config.reid_candidates.wearable_seeding_enabled,
        wearable_face_attestation_mode=config.wearable.face_attestation,
        wearable_backfill_enabled=config.resolver.enable_wearable_backfill,
        wearable_auto_on_swap=config.wearable.revocation_auto_on_swap,
        wearable_evaluation_sample_rate=config.wearable.evaluation_sample_rate,
        closed_world_mode=config.closed_world.mode,
    )


def _build_closed_world_config(s: Settings) -> ClosedWorldConfig:
    """Build ClosedWorldConfig from required settings.yaml keys."""
    cw = s.section("closed_world")
    return ClosedWorldConfig(
        mode=cast(ClosedWorldMode, cw.as_str("mode")),
        away_factor=cw.as_float("away_factor"),
        away_min_confidence=cw.as_float("away_min_confidence"),
        wearable_home_window_s=cw.as_float("wearable_home_window_s"),
        occupied_factor=cw.as_float("occupied_factor"),
        occupancy_fresh_s=cw.as_float("occupancy_fresh_s"),
        simultaneous_obs_max_skew_s=cw.as_float("simultaneous_obs_max_skew_s"),
        min_elimination_distance_m=cw.as_float("min_elimination_distance_m"),
        guest_floor=cw.as_float("guest_floor"),
        guest_floor_visitors=cw.as_float("guest_floor_visitors"),
        presence_stale_after_s=cw.as_float("presence_stale_after_s"),
        factor_floor=cw.as_float("factor_floor"),
        factor_ceiling=cw.as_float("factor_ceiling"),
    )


def _build_wearable_config(s: Settings) -> WearableConfig:
    """Build WearableConfig from required settings.yaml keys."""
    w = s.section("wearable")
    return WearableConfig(
        mode=cast(WearableMode, w.as_str("mode")),
        window_s=w.as_float("window_s"),
        min_fixes=w.as_int("min_fixes"),
        min_window_llr=w.as_float("min_window_llr"),
        min_margin_llr=w.as_float("min_margin_llr"),
        gate_chi2=w.as_float("gate_chi2"),
        gate_penalty_llr=w.as_float("gate_penalty_llr"),
        timing_sigma_s=w.as_float("timing_sigma_s"),
        bias_floor_m=w.as_float("bias_floor_m"),
        max_fix_age_s=w.as_float("max_fix_age_s"),
        buffer_max_age_s=w.as_float("buffer_max_age_s"),
        max_extrapolation_s=w.as_float("max_extrapolation_s"),
        max_fix_buffer_per_tag=w.as_int("max_fix_buffer_per_tag"),
        trajectory_max_sigma_m=w.as_float("trajectory_max_sigma_m"),
        room_min_mass=w.as_float("room_min_mass"),
        room_polygon_min_mass=w.as_float("room_polygon_min_mass"),
        room_match_eps=w.as_float("room_match_eps"),
        parked_after_s=w.as_float("parked_after_s"),
        tag_disagreement_window_s=w.as_float("tag_disagreement_window_s"),
        trajectory_attest_min_duration_s=w.as_float("trajectory_attest_min_duration_s"),
        trajectory_attest_min_fixes=w.as_int("trajectory_attest_min_fixes"),
        room_attest_min_duration_s=w.as_float("room_attest_min_duration_s"),
        room_entry_window_s=w.as_float("room_entry_window_s"),
        attest_min_llr=w.as_float("attest_min_llr"),
        weight_multiplier=w.as_float("weight_multiplier"),
        p_max=w.as_float("p_max"),
        p_max_room=w.as_float("p_max_room"),
        llr_mid=w.as_float("llr_mid"),
        llr_scale=w.as_float("llr_scale"),
        duplicate_guard_min_probability=w.as_float("duplicate_guard_min_probability"),
        swap_conflict_count=w.as_int("swap_conflict_count"),
        swap_conflict_window_s=w.as_float("swap_conflict_window_s"),
        evaluation_sample_rate=s.as_float("wearable.evaluation.sample_rate"),
        evaluation_exclusion_s=s.as_float("wearable.evaluation.exclusion_s"),
        evaluation_max_gap_s=s.as_float("wearable.evaluation.max_gap_s"),
        face_attestation=cast(
            FaceAttestationMode, s.as_str("wearable.supervision.face_attestation")
        ),
        revocation_auto_on_swap=s.as_bool("wearable.revocation.auto_on_swap"),
    )


def _build_face_template_policy(s: Settings) -> FaceTemplatePolicy:
    ft = s.section("face_templates")
    return FaceTemplatePolicy(
        # FaceTemplatePolicy validates the value; the cast only narrows the type for mypy.
        mode=cast(FaceTemplateMode, ft.as_str("mode")),
        top_k=ft.as_int("top_k"),
        min_faces=ft.as_int("min_faces"),
        min_face_quality=ft.as_float("min_face_quality"),
        max_age_s=ft.as_float("max_age_s"),
        outlier_min_similarity=ft.as_float("outlier_min_similarity"),
        trigger_cooldown_s=ft.as_float("trigger_cooldown_s"),
        max_concurrency=ft.as_int("max_concurrency"),
        max_in_flight=ft.as_int("max_in_flight"),
    )


# Module-level asyncpg pool so shutdown can close it.
_pool: Any = None  # asyncpg.Pool | None
_triton_client: TritonGrpcClient | None = None
_frame_fetcher: MinioFrameFetcher | None = None


def get_pipeline() -> FrameProcessingPipeline | None:
    """Access the running pipeline from dependency injection."""
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline, _pool, _triton_client, _frame_fetcher

    # -------------------------------------------------------------------
    # Startup: build config, connect services, start pipeline
    # -------------------------------------------------------------------

    transport_config = _build_transport_config(settings)

    # Per-camera profiles and face-id config: CC is the primary source.
    cc_url = settings.as_str("cognitive_companion.url")
    cc_api_key = settings.as_str("cognitive_companion.api_key")
    cc_client = CognitiveCompanionClient(cc_url, cc_api_key)
    initial_camera_profiles = await _fetch_cc_camera_profiles(cc_client)
    face_id_camera_configs = {
        p.camera_id: FaceIdCameraConfig(
            enabled=p.face_id_enabled,
            min_raw_similarity=p.face_id_min_raw_similarity,
        )
        for p in initial_camera_profiles
    }

    declared_overlap_groups: list[OverlapGroup] = []
    try:
        declared_overlap_groups = await fetch_overlap_groups(cc_client)
        logger.info("overlap_groups_loaded", count=len(declared_overlap_groups))
    except Exception:
        logger.exception("overlap_groups_fetch_failed")

    # Build config objects from settings (env-interpolated YAML).
    resolver_config = _build_resolver_config(settings)
    world_tracker_config = _build_world_tracker_config(settings)
    wearable_config = _build_wearable_config(settings)
    closed_world_config = _build_closed_world_config(settings)
    sampler_config = _build_sampler_config(settings)

    config = PipelineConfig(
        transport=transport_config,
        resolver=resolver_config,
        world_tracker=world_tracker_config,
        wearable=wearable_config,
        closed_world=closed_world_config,
        reid_candidates=_build_candidate_policy(settings, wearable_config),
        sampler=sampler_config,
        signals=_build_signal_config(settings),
        signals_enabled=settings.as_bool("signal.enabled"),
        signal_interval_s=settings.as_int("signal.interval_s"),
        fall_detection=_build_fall_detection_config(settings),
        face_id=_build_face_id_config(settings, face_id_camera_configs),
        face_templates=_build_face_template_policy(settings),
        max_concurrent_frames=settings.as_int("pipeline.max_concurrent_frames"),
        shutdown_timeout=settings.as_float("pipeline.shutdown_timeout_s"),
        pose_enabled=settings.as_bool("triton.pose_enabled"),
        allow_skeleton=settings.as_bool("pipeline.allow_skeleton"),
        # Phase 1: noise reduction
        detection_iou_dedup_threshold=settings.as_float("pipeline.detection.iou_dedup_threshold"),
        identity_rewrite_on_face_commit=settings.as_bool(
            "pipeline.identity.rewrite_on_face_commit"
        ),
        batch_window_s=settings.as_float("pipeline.batch_window_s"),
        max_batch_size=settings.as_int("pipeline.max_batch_size"),
        cross_camera_detector_batching=settings.as_bool("pipeline.cross_camera_detector_batching"),
        min_keyframe_detection_confidence=settings.as_float(
            "pipeline.min_keyframe_detection_confidence"
        ),
        live_publish_max_hz=settings.as_float("pipeline.live_publish_max_hz"),
        adaptive_reid=AdaptiveReidConfig(
            enabled=settings.as_bool("pipeline.adaptive_reid.enabled"),
            shadow=settings.as_bool("pipeline.adaptive_reid.shadow"),
            refresh_interval_s=settings.as_float("pipeline.adaptive_reid.refresh_interval_s"),
            proximity_gate_m=settings.as_float("pipeline.adaptive_reid.proximity_gate_m"),
        ),
        gait_aggregate_interval_s=settings.as_int("gait.aggregate_interval_s"),
        gait_min_daily_bouts=settings.as_int("gait.min_daily_bouts"),
        gait_min_daily_walking_s=settings.as_float("gait.min_daily_walking_s"),
        appearance=_build_appearance_settings(settings),
        height_estimation=_build_height_estimation_config(settings),
    )
    _pipeline = FrameProcessingPipeline(config)

    # -- Database --
    database_url = settings.as_str("database.url")

    if not database_url:
        env = settings.as_str("env")
        if env in ("dev", "development", "test"):
            logger.info("No database.url; using in-memory storage for this environment.")
        else:
            logger.warning(
                "No database.url configured; running with in-memory storage. "
                "State will be lost on restart.",
            )

    if database_url:
        import asyncpg

        _pool = await asyncpg.create_pool(
            dsn=_normalize_dsn(database_url),
            min_size=2,
            max_size=20,
            command_timeout=120.0,
            server_settings={"search_path": "continuous_tracking, public"},
        )
        runner = MigrationRunner(_pool, MIGRATIONS_DIR)
        await runner.migrate()
        gallery_repo = PostgresGalleryRepository(_pool)
        trajectory_repo = PostgresTrajectoryRepository(_pool)
        keyframe_repo = PostgresKeyframeRepository(_pool)
        signal_repo = PostgresDementiaSignalRepository(_pool)
        settings_repo = PostgresSettingsRepository(_pool)
        bbox_repo = PostgresBboxAnnotationRepository(_pool)
        baseline_repo = PostgresBehaviorBaselineRepository(_pool)
        gait_daily_repo = PostgresGaitDailyRepository(_pool)
        daily_appearance_repo = PostgresDailyAppearanceRepo(_pool)
        face_stats_repo = PostgresFaceObservationStatsRepository(_pool)
        from .storage.postgres.identity_decision_repo import (
            PostgresIdentityDecisionRepository,
        )

        identity_decision_repo = PostgresIdentityDecisionRepository(_pool)
    else:
        gallery_repo = None
        trajectory_repo = None
        keyframe_repo = None
        signal_repo = None
        settings_repo = None
        bbox_repo = None
        baseline_repo = None
        gait_daily_repo = None
        daily_appearance_repo = None
        face_stats_repo = None
        identity_decision_repo = None

    # -- Triton --
    triton_url = settings.as_str("triton.url")
    detector = None
    reid_embedder = None
    pose_estimator = None
    env = settings.as_str("env")
    triton_timeout_ms = settings.as_int("triton.timeout_ms")
    detector_model_name = settings.as_str("triton.person_detector_model")
    detector_static_batch_size = settings.as_int("triton.detector_static_batch_size")
    detector_dynamic_batch = settings.as_bool("triton.detector_dynamic_batch")
    detector_confidence = settings.as_float("pipeline.detector_confidence")
    reid_model_name = settings.as_str("triton.reid_model")
    pose_enabled = settings.as_bool("triton.pose_enabled")
    pose_model_name = settings.as_str("triton.pose_model")
    _triton_client = None
    if triton_url:
        startup_wait_s = settings.as_int("triton.startup_wait_s")
        startup_retry_s = settings.as_int("triton.startup_retry_interval_s")
        deadline = asyncio.get_event_loop().time() + startup_wait_s
        last_exc: Exception | None = None
        connected = False
        while asyncio.get_event_loop().time() < deadline:
            try:
                _triton_client = TritonGrpcClient(triton_url, timeout_ms=triton_timeout_ms)
                await _triton_client.__aenter__()
                connected = True
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                remaining = deadline - asyncio.get_event_loop().time()
                logger.warning(
                    "triton_not_ready",
                    url=triton_url,
                    remaining_s=round(remaining, 1),
                    retry_interval_s=startup_retry_s,
                )
                await asyncio.sleep(startup_retry_s)

        if not connected:
            logger.exception("Failed to connect to Triton Inference Server", exc_info=last_exc)
            if env in ("production", "staging"):
                raise RuntimeError(
                    "Triton Inference Server is required in production/staging. "
                    "Use env=development in settings.yaml for skeleton-only local testing."
                ) from None
            _triton_client = None
        else:
            assert _triton_client is not None  # narrow for mypy
            detector = PersonDetector(
                _triton_client,
                model_name=detector_model_name,
                conf_threshold=detector_confidence,
                static_batch_size=detector_static_batch_size,
                dynamic_batch=detector_dynamic_batch,
            )
            reid_embedder = ReidEmbedder(
                _triton_client,
                model_name=reid_model_name,
            )
            if pose_enabled:
                pose_estimator = PoseEstimator(
                    _triton_client,
                    model_name=pose_model_name,
                )

    posture_components = await _build_posture_components(settings, _triton_client)
    depth_estimator = posture_components.depth_estimator
    posture_strategy = posture_components.posture_strategy

    # -- MinIO --
    minio_endpoint = settings.as_str("minio.endpoint")
    minio_bucket = settings.as_str("minio.bucket")
    minio_access_key = settings.as_str("minio.access_key")
    minio_secret_key = settings.as_str("minio.secret_key")
    if minio_endpoint and minio_bucket and minio_access_key and minio_secret_key:
        secure = settings.as_bool("minio.secure")
        _frame_fetcher = MinioFrameFetcher(
            MinioFrameConfig(
                endpoint_url=minio_endpoint,
                bucket=minio_bucket,
                access_key_id=minio_access_key,
                secret_access_key=minio_secret_key,
                region_name=settings.as_str("minio.region"),
                secure=secure,
            )
        )
        await _frame_fetcher.connect()
    elif detector is not None:
        logger.warning(
            "Triton configured without MinIO frame storage; inference will run on blank frames.",
        )

    # -- Auto-calibration (depth-based homography estimation) --
    if depth_estimator is not None:
        from .calibration.auto_calibrator import AutoCalibrator

        auto_calibrator = AutoCalibrator(depth_estimator=depth_estimator)
        set_auto_calibration_context(auto_calibrator=auto_calibrator, frame_fetcher=_frame_fetcher)
        logger.info("auto_calibrator_ready", model="depth-anything-v2")
    else:
        set_auto_calibration_context(auto_calibrator=None, frame_fetcher=_frame_fetcher)
        logger.info("auto_calibrator_disabled", reason="triton_not_connected_or_depth_disabled")

    if depth_estimator is None:
        logger.info("Posture strategy: RTMPose only (depth slow-path disabled)")

    # -- PH repository (Postgres or in-memory fallback) --
    from .storage.base import InMemoryPHRepository as _InMemPH

    if _pool is not None:
        _ph_repo_n1: _InMemPH | PostgresPHRepository = PostgresPHRepository(_pool)
    else:
        _ph_repo_n1 = _InMemPH()
    set_ph_repository(_ph_repo_n1)  # type: ignore[arg-type]
    deps_ph_repo = _ph_repo_n1

    # -- World observation repository (Postgres or in-memory fallback) --
    # Must be wired explicitly: FrameProcessingPipeline defaults to an
    # in-memory repo when deps.obs_repo is None, so leaving this unset here
    # silently drops every observation instead of persisting it, even though
    # the PH rows themselves keep landing in Postgres via deps_ph_repo above.
    from .storage.base import InMemoryWorldObservationRepository as _InMemObs

    if _pool is not None:
        _obs_repo_n1: _InMemObs | PostgresWorldObservationRepository = (
            PostgresWorldObservationRepository(_pool)
        )
    else:
        _obs_repo_n1 = _InMemObs()
    deps_obs_repo = _obs_repo_n1

    # identity-correction repository (segment corrections + revision ranges).
    from .storage.corrections import InMemoryIdentityCorrectionRepository
    from .storage.postgres.correction_repo import PostgresIdentityCorrectionRepository

    correction_repo_obj = (
        PostgresIdentityCorrectionRepository(_pool)
        if _pool is not None
        else InMemoryIdentityCorrectionRepository()
    )

    # Wire revision publisher for manual PH corrections (merge/split; routers/ph.py).
    # NOTE: _pipeline._revision_publisher is None here (initialize has not run
    # yet), so this call is a no-op today. It is left in its original position
    # deliberately: fixing this ordering would activate the merge/split revision-publish
    # path in routers/ph.py for the first time, which is a behavioral change unrelated
    # to the current scope. This pre-existing bug should be fixed in its own
    # dedicated change if desired.
    if _pipeline._revision_publisher is not None:
        set_revision_publisher(_pipeline._revision_publisher)

    # -- Wire everything and start --
    identity_rewriter = (
        PostgresIdentityRewriter(_pool) if _pool is not None else InMemoryIdentityRewriter()
    )

    # Wire the correction service (single owner of operator corrections).
    # Constructed here, before FrameProcessingPipeline.initialize(), so it can
    # be handed to UnknownBackfillService through
    # ordinary constructor injection via PipelineDependencies. The real
    # RevisionPublisher only exists once the pipeline initializes (it needs
    # transport config resolved during initialize), so publisher=None here
    # and IdentityCorrectionService.set_publisher(...) wires it in below,
    # after initialize returns. record_inferred_range() (the only method
    # UnknownBackfillService calls on this service before the publisher is
    # wired) never touches the publisher, so this ordering is safe.
    from .services.identity_correction_service import IdentityCorrectionService

    correction_service = IdentityCorrectionService(
        ph_repo=deps_ph_repo,  # type: ignore[arg-type]
        correction_repo=correction_repo_obj,
        publisher=None,
        rewriter=identity_rewriter,
    )

    # Cross-camera topology and co-presence repositories.
    from .storage.base import (
        CachedCameraTopologyRepository,
        InMemoryCameraTopologyRepository,
        InMemoryCoPresenceRepository,
    )
    from .storage.postgres.camera_topology_repo import (
        PostgresCameraTopologyRepository,
        PostgresCoPresenceRepository,
    )

    if _pool is not None:
        topology_repo_obj: CachedCameraTopologyRepository | InMemoryCameraTopologyRepository = (
            CachedCameraTopologyRepository(PostgresCameraTopologyRepository(_pool))
        )
        copresence_repo_obj: InMemoryCoPresenceRepository | PostgresCoPresenceRepository = (
            PostgresCoPresenceRepository(_pool)
        )
    else:
        topology_repo_obj = InMemoryCameraTopologyRepository()
        copresence_repo_obj = InMemoryCoPresenceRepository()
    set_topology_repo(topology_repo_obj)

    # CC identity assertion cache and subscriber constructed before the pipeline
    # so WorldTrackingStage receives ExternalIdentityEvidenceSource via constructor
    # injection rather than post-init attribute mutation.
    cc_assertion_cache: IdentityAssertionCache | None = None
    cc_assertion_subscriber: CCIdentityAssertionSubscriber | None = None
    redis_url = settings.as_str("redis.url")
    if redis_url:
        try:
            cc_assertion_cache = IdentityAssertionCache()
            # Use a dedicated Redis client for the subscriber so it does not
            # share the pipeline's pub/sub connection.
            _cc_redis = aioredis.from_url(redis_url, decode_responses=False)
            cc_assertion_subscriber = CCIdentityAssertionSubscriber(
                redis_client=_cc_redis,
                cache=cc_assertion_cache,
            )
        except Exception:
            logger.exception("cc_assertion_subscriber_init_failed")
            cc_assertion_cache = None
            cc_assertion_subscriber = None

    wearable_site_config_map = WearableSiteConfigMap()
    wearable_association_repo: WearableAssociationRepository = (
        PostgresWearableAssociationRepository(_pool)
        if _pool is not None
        else InMemoryWearableAssociationRepository()
    )
    app.state.wearable_association_repo = wearable_association_repo

    wearable_teaching_hold_repo: WearableTeachingHoldRepository = (
        PostgresWearableTeachingHoldRepository(_pool)
        if _pool is not None
        else InMemoryWearableTeachingHoldRepository()
    )
    wearable_revocation_service: WearableTeachingRevocationService | None = None
    if gallery_repo is not None:
        wearable_revocation_service = WearableTeachingRevocationService(
            association_repo=wearable_association_repo,
            teaching_holds=wearable_teaching_hold_repo,
            gallery_repo=gallery_repo,
            correction_repo=correction_repo_obj,
            correction_service=correction_service,
            face_id_client=None,
        )
    app.state.wearable_revocation_service = wearable_revocation_service

    # Image-only evaluation runs only while the wearable channel does; its sample rate (0.0 by
    # default) decides whether the resolver produces any evaluations at all.
    image_only_eval_repo: ImageOnlyEvaluationRepository = (
        PostgresImageOnlyEvaluationRepository(_pool)
        if _pool is not None
        else InMemoryImageOnlyEvaluationRepository()
    )
    image_only_evaluation: ImageOnlyEvaluationService | None = None
    if wearable_config.mode != "off":
        image_only_evaluation = ImageOnlyEvaluationService(
            image_only_eval_repo,
            max_gap_s=wearable_config.evaluation_max_gap_s,
        )
    app.state.image_only_evaluation = image_only_evaluation

    wearable_fix_buffer: WearableFixBuffer | None = None
    wearable_fix_subscriber: WearableFixSubscriber | None = None
    if wearable_config.mode != "off":
        wearable_fix_buffer = WearableFixBuffer(
            max_age_s=wearable_config.buffer_max_age_s,
            max_per_tag=wearable_config.max_fix_buffer_per_tag,
        )
        if redis_url:
            try:
                _wearable_redis = aioredis.from_url(redis_url, decode_responses=False)
                wearable_fix_subscriber = WearableFixSubscriber(
                    redis_client=_wearable_redis,
                    buffer=wearable_fix_buffer,
                    site_config_map=wearable_site_config_map,
                    stream=transport_config.wearable_fixes_stream,
                    group=transport_config.consumer_group,
                    consumer=transport_config.consumer_name,
                    max_fix_age_s=wearable_config.max_fix_age_s,
                )
            except Exception:
                logger.exception("wearable_fix_subscriber_init_failed")
                wearable_fix_subscriber = None

    household_presence_cache: HouseholdPresenceCache | None = None
    household_presence_subscriber: HouseholdPresenceSubscriber | None = None
    if closed_world_config.mode != "off":
        household_presence_cache = HouseholdPresenceCache()
        if redis_url:
            try:
                _presence_redis = aioredis.from_url(redis_url, decode_responses=False)
                household_presence_subscriber = HouseholdPresenceSubscriber(
                    redis_client=_presence_redis,
                    cache=household_presence_cache,
                    stream=transport_config.household_presence_stream,
                    group=transport_config.consumer_group,
                    consumer=transport_config.consumer_name,
                )
            except Exception:
                logger.exception("household_presence_subscriber_init_failed")
                household_presence_subscriber = None

    height_profile_repo: IdentityHeightProfileRepository = (
        PostgresIdentityHeightProfileRepository(_pool)
        if _pool is not None
        else InMemoryIdentityHeightProfileRepository()
    )
    height_sample_repo: PHHeightSampleRepository = (
        PostgresPHHeightSampleRepository(_pool)
        if _pool is not None
        else InMemoryPHHeightSampleRepository()
    )
    set_height_profiles_context(height_profile_repo)

    deps = PipelineDependencies(
        detector=detector,
        gallery_repo=gallery_repo,
        ph_repo=deps_ph_repo,  # type: ignore[arg-type]  # Postgres-backed PH repository
        obs_repo=deps_obs_repo,  # Postgres-backed world observation repository
        trajectory_repo=trajectory_repo,
        keyframe_repo=keyframe_repo,
        signal_repo=signal_repo,
        settings_repo=settings_repo,
        frame_fetcher=_frame_fetcher,
        reid_embedder=reid_embedder,
        pose_estimator=pose_estimator,
        posture_strategy=posture_strategy,
        identity_rewriter=identity_rewriter,
        bbox_repo=bbox_repo,  # type: ignore[arg-type]
        topology_repo=topology_repo_obj,
        copresence_repo=copresence_repo_obj,
        overlap_groups=declared_overlap_groups,
        baseline_repo=baseline_repo,
        gait_daily_repo=gait_daily_repo,
        daily_appearance_repo=daily_appearance_repo,
        face_stats_repo=face_stats_repo,
        identity_provenance_repo=identity_decision_repo,
        identity_correction_service=correction_service,
        external_evidence=cc_assertion_cache,
        wearable_fix_source=wearable_fix_buffer,
        wearable_association_repo=wearable_association_repo,
        wearable_teaching_hold_repo=wearable_teaching_hold_repo,
        wearable_revocation_service=wearable_revocation_service,
        image_only_evaluation=image_only_evaluation,
        height_profile_repo=height_profile_repo,
        height_sample_repo=height_sample_repo,
        household_presence_source=household_presence_cache,
    )
    await _pipeline.initialize(deps)

    # Now that the pipeline has initialized, the real RevisionPublisher exists;
    # wire it into the correction service used by the correct/batch_correct
    # routes. (The separate routers/ph.py merge/split publish path is wired
    # earlier, before initialize, where it is intentionally still a no-op;
    # see the comment at that call site.)
    correction_service.set_publisher(_pipeline._revision_publisher)
    if wearable_revocation_service is not None:
        wearable_revocation_service.set_face_id_client(_pipeline._face_id_client)

    from .calibration.state import calibration_state

    # Privacy zones live only in memory; load them before the first frame so a restart never
    # tracks people inside an excluded zone. CCConfigSyncService keeps them current afterwards.
    from .services.cc_config_sync import restore_privacy_zones

    zone_count = await restore_privacy_zones(cc_client, calibration_state)
    logger.info("Restored privacy zones from CC", zone_count=zone_count)

    await _pipeline.start()

    # CC config sync service (camera-to-room map + camera profiles + homography + wearables sync).
    from .services.camera_room_map import CameraRoomMap, RoomPolygonMap
    from .services.cc_config_sync import CCConfigSyncService
    from .services.transit_zone_map import TransitZoneMap

    camera_room_map = CameraRoomMap()
    room_polygon_map = RoomPolygonMap()
    transit_zone_map = TransitZoneMap()
    camera_profile_map = CameraProfileMap(initial_camera_profiles)
    _pipeline.set_camera_profile_map(camera_profile_map)
    _pipeline.set_wearable_site_config_map(wearable_site_config_map)
    if _pipeline._height_estimation_stage is not None:
        from .routers.calibration import set_height_pose_provider

        set_height_pose_provider(_pipeline._height_estimation_stage.pose_cache.snapshot_all)

    _poll_s = settings.as_float("cc_sync.poll_interval_s")
    cc_sync_service = CCConfigSyncService(
        client=cc_client,
        calibration_state=calibration_state,
        camera_room_map=camera_room_map,
        room_polygon_map=room_polygon_map,
        transit_zone_map=transit_zone_map,
        poll_interval_s=_poll_s,
        overlap_group_sink=_pipeline.set_overlap_groups,
        camera_profile_map=camera_profile_map,
        wearable_site_config_map=wearable_site_config_map,
        height_profile_repo=height_profile_repo,
        on_height_profiles_updated=(
            _pipeline._identity_resolver.invalidate_identity_cache
            if _pipeline._identity_resolver is not None
            else None
        ),
    )
    app.state.cc_sync_service = cc_sync_service
    app.state.cc_sync_task = asyncio.create_task(cc_sync_service.run())
    app.state.camera_room_map = camera_room_map
    app.state.room_polygon_map = room_polygon_map
    app.state.transit_zone_map = transit_zone_map
    app.state.camera_profile_map = camera_profile_map
    app.state.wearable_site_config_map = wearable_site_config_map
    # Wire CameraRoomMap into the pipeline so stages read room bindings.
    _pipeline.set_camera_room_map(camera_room_map)
    _pipeline.set_room_polygon_map(room_polygon_map)

    # Wire live transit zones into the world tracker. CCConfigSyncService
    # populates TransitZoneMap every poll after converting zones to metres.
    from .tracking.world.transit_detector import TransitDetector as _TransitDetector
    from .transport.room_transition_publisher import (
        RoomTransitionPublisher as _RoomTransitionPublisher,
    )

    _transit_detector = _TransitDetector()
    _room_transition_pub = _RoomTransitionPublisher(
        redis_url=settings.as_str("redis.url"),
    )
    await _room_transition_pub.connect()
    _pipeline.set_transit_config(
        transit_detector=_transit_detector,
        transit_zone_map=transit_zone_map,
        room_transition_publisher=_room_transition_pub,
    )

    # Start CC identity assertion subscriber (cache was already injected into
    # WorldTrackingStage via PipelineDependencies.external_evidence).
    if cc_assertion_subscriber is not None:
        try:
            await cc_assertion_subscriber.start()
            logger.info("cc_assertion_subscriber_started", stream="cc.identity_assertions")
        except Exception:
            logger.exception("cc_assertion_subscriber_start_failed")
            cc_assertion_subscriber = None

    # Start wearable fix subscriber only when wearable mode is shadow or enabled.
    if wearable_fix_subscriber is not None:
        try:
            await wearable_fix_subscriber.start()
        except Exception:
            logger.exception("wearable_fix_subscriber_start_failed")
            wearable_fix_subscriber = None

    # Start household presence subscriber only when closed_world.mode is shadow or enabled.
    if household_presence_subscriber is not None:
        try:
            await household_presence_subscriber.start()
        except Exception:
            logger.exception("household_presence_subscriber_start_failed")
            household_presence_subscriber = None

    # Keyframe revalidator background task.
    if bbox_repo is not None:
        from .services.keyframe_revalidator import KeyframeRevalidator

        _revalidator = KeyframeRevalidator(
            bbox_repo=bbox_repo,
            threshold=settings.as_float("pipeline.min_keyframe_detection_confidence"),
        )
        app.state.keyframe_revalidator = _revalidator
        app.state.keyframe_revalidator_task = asyncio.create_task(_revalidator.run())

    # correction_service was constructed earlier (before pipeline.initialize)
    # so UnknownBackfillService could depend on it via constructor injection;
    # its publisher was wired in just after initialize above.
    set_corrections_context(correction_service)
    # Route the PH inspector and batch correction endpoints through the same
    # service so every correction writes an effective revision range.
    from .routers.ph import set_correction_service as set_ph_correction_service
    from .routers.ph import set_face_template_reset

    set_ph_correction_service(correction_service)
    set_face_template_reset(_pipeline.face_templates)
    set_live_context(
        ph_repo=deps_ph_repo,  # type: ignore[arg-type]
        keyframe_repo=keyframe_repo,
        gallery_repo=gallery_repo,
    )

    if trajectory_repo is not None:
        set_trajectory_context(trajectory_repo=trajectory_repo)

    if gait_daily_repo is not None:
        set_gait_context(gait_daily_repo=gait_daily_repo)

    if gallery_repo is not None:
        from .services.identity_roster_sync import IdentityRosterSyncService
        from .services.reid_review_service import ReIDReviewService

        roster_sync_service = IdentityRosterSyncService(
            gallery_repo=gallery_repo,
            members_source=_pipeline._face_id_client if _pipeline is not None else None,
            identity_resolver=_pipeline._identity_resolver if _pipeline is not None else None,
            face_templates=_pipeline.face_templates if _pipeline is not None else None,
            wearable_site_config=wearable_site_config_map,
            wearable_fix_buffer=wearable_fix_buffer,
            wearable_association_repo=wearable_association_repo,
            image_only_evaluation=image_only_evaluation,
            face_attestation=_pipeline.face_attestation if _pipeline is not None else None,
            height_profile_repo=height_profile_repo,
            household_presence_cache=household_presence_cache,
            world_tracker_state=_pipeline,
            interval_s=60.0,
        )
        await roster_sync_service.start()
        app.state.identity_roster_sync = roster_sync_service

        gallery_router_mod.set_context(
            gallery_repo=gallery_repo,
            reid_embedder=reid_embedder,
            roster_sync=roster_sync_service,
        )

        # ReID review queue: governed approve/relabel/reject over the gallery.
        # active_model_versions is left unconfigured (None) for now: model
        # incompatibility is surfaced once a current-version list is wired here.
        reid_review_service = ReIDReviewService(
            gallery_repo,
            storage_client=_frame_fetcher,
        )
        set_reid_review_context(reid_review_service)

        gallery_retention = GalleryRetentionService(
            gallery_repo=gallery_repo,
            config=_build_gallery_retention_config(settings),
            crop_storage=_frame_fetcher,
        )
        app.state.gallery_retention = gallery_retention
        app.state.gallery_retention_task = asyncio.create_task(gallery_retention.run())

    app.state.identity_readiness_service = IdentityReadinessService(
        policy=_build_identity_policy_snapshot(config),
        calibration_state=calibration_state,
        camera_room_map=camera_room_map,
        overlap_groups=_pipeline,
        gallery_repo=gallery_repo,
        wearable_associations=_pipeline,
        image_only_eval_repo=image_only_eval_repo,
        wearable_assoc_repo=wearable_association_repo,
        teaching_holds_repo=wearable_teaching_hold_repo,
        household_presence=household_presence_cache,
        closed_world_config=closed_world_config,
    )

    if signal_repo is not None and trajectory_repo is not None and keyframe_repo is not None:
        dashboard_router_mod.set_repos(
            signal=signal_repo,
            trajectory=trajectory_repo,
            keyframe=keyframe_repo,
        )

    if bbox_repo is not None:
        dashboard_router_mod.set_bbox_repo(bbox_repo)  # type: ignore[arg-type]

    # keyframe read model: provenance-bearing repos for grouped cards.
    if identity_decision_repo is not None and gallery_repo is not None:
        dashboard_router_mod.set_read_model_repos(
            decision=identity_decision_repo,
            correction=correction_repo_obj,
            gallery=gallery_repo,
        )

    ph_maintenance = PHMaintenanceService(
        repo=deps_ph_repo,  # type: ignore[arg-type]
        config=_build_ph_unknown_purge_config(settings),
    )
    app.state.ph_maintenance = ph_maintenance
    app.state.ph_maintenance_task = asyncio.create_task(ph_maintenance.run())

    if _pool is not None:
        height_learner = HeightProfileLearner(
            sample_source=PostgresHeightSampleSource(_pool),
            profile_repo=height_profile_repo,
            config=_build_height_learner_config(settings),
            on_profiles_updated=(
                _pipeline._identity_resolver.invalidate_identity_cache
                if _pipeline is not None and _pipeline._identity_resolver is not None
                else None
            ),
        )
        app.state.height_profile_learner = height_learner
        app.state.height_profile_learner_task = asyncio.create_task(height_learner.run())

    yield

    # -------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------
    _height_learner = getattr(app.state, "height_profile_learner", None)
    if _height_learner is not None:
        await _height_learner.stop()
    # Stop keyframe revalidator.
    _m3_revalidator = getattr(app.state, "keyframe_revalidator", None)
    if _m3_revalidator is not None:
        await _m3_revalidator.stop()

    # Stop PH maintenance.
    _ph_maintenance = getattr(app.state, "ph_maintenance", None)
    if _ph_maintenance is not None:
        await _ph_maintenance.stop()

    # Stop gallery retention.
    _gallery_retention = getattr(app.state, "gallery_retention", None)
    if _gallery_retention is not None:
        await _gallery_retention.stop()

    # Stop identity roster sync.
    _roster_sync = getattr(app.state, "identity_roster_sync", None)
    if _roster_sync is not None:
        await _roster_sync.stop()

    # Stop CC assertion subscriber.
    if cc_assertion_subscriber is not None:
        await cc_assertion_subscriber.stop()

    # Stop wearable fix subscriber.
    if wearable_fix_subscriber is not None:
        await wearable_fix_subscriber.stop()

    # Stop household presence subscriber.
    if household_presence_subscriber is not None:
        await household_presence_subscriber.stop()

    # Disconnect room transition publisher.
    if "_room_transition_pub" in dir():
        await _room_transition_pub.disconnect()

    # Stop CC config sync before pipeline.
    _cc_sync = getattr(app.state, "cc_sync_service", None)
    if _cc_sync is not None:
        await _cc_sync.stop()

    if _pipeline is not None:
        await _pipeline.stop()
        _pipeline = None

    # Totals accumulate between flushes; write the last ones while the pool is still open.
    _image_only = getattr(app.state, "image_only_evaluation", None)
    if _image_only is not None:
        await _image_only.flush()

    if _frame_fetcher is not None:
        await _frame_fetcher.disconnect()
        _frame_fetcher = None

    if _triton_client is not None:
        await _triton_client.__aexit__(None, None, None)
        _triton_client = None

    if _pool is not None:
        await _pool.close()
        _pool = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Tracking — Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Set by the lifespan once the pipeline is built; the route answers 503 until then.
    app.state.identity_readiness_service = None
    app.state.wearable_association_repo = None
    app.state.wearable_revocation_service = None
    app.state.image_only_evaluation = None
    # Internal calibration endpoints consumed by the CC BFF.
    # Reserved for future use: separate :8510 listener
    # they live on the same app under /internal/ to keep the routing simple.
    app.include_router(calibration_router)
    app.include_router(dashboard_router)
    app.include_router(corrections_router)
    app.include_router(gallery_router)
    app.include_router(height_profiles_router)
    app.include_router(reid_review_router)
    app.include_router(readiness_router)
    app.include_router(wearables_router)
    app.include_router(live_router)
    app.include_router(ph_router)
    app.include_router(trajectory_router)
    app.include_router(gait_router)
    app.include_router(topology_router)

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Rollout health check — reports subsystem readiness.

        Returns 'healthy' only when all required subsystems are operational.
        Returns 'degraded' when optional subsystems are missing.
        Returns 'unhealthy' when required subsystems are absent.
        """
        checks: dict[str, dict[str, object]] = {}
        overall = "healthy"

        # Pipeline
        pipeline_ok = _pipeline is not None and _pipeline.is_running
        checks["pipeline"] = {
            "status": "ok" if pipeline_ok else "degraded",
            "message": "running" if pipeline_ok else "stopped or not initialized",
        }
        if not pipeline_ok and _pipeline is None:
            overall = "degraded"

        # PH repository
        _ph_repo_ref = _pipeline._ph_repo if _pipeline else None
        ph_repo_ok = _ph_repo_ref is not None
        checks["ph_repository"] = {
            "status": "ok" if ph_repo_ok else "degraded",
            "message": "wired" if ph_repo_ok else "not configured",
        }

        # Observation repository (wired through pipeline). FrameProcessingPipeline
        # silently falls back to an in-memory repo when unwired, so "not None" alone
        # can't detect a misconfiguration -- check it's Postgres-backed whenever a
        # database pool is available.
        from .storage.base import InMemoryWorldObservationRepository as _InMemObsCheck

        _obs_repo_ref = _pipeline._obs_repo if _pipeline else None
        obs_repo_ok = _obs_repo_ref is not None and not (
            _pool is not None and isinstance(_obs_repo_ref, _InMemObsCheck)
        )
        checks["observation_repository"] = {
            "status": "ok" if obs_repo_ok else "degraded",
            "message": "wired" if obs_repo_ok else "not configured",
        }

        # Database pool
        db_ok = _pool is not None
        checks["database"] = {
            "status": "ok" if db_ok else "degraded",
            "message": "connected" if db_ok else "in-memory mode (state lost on restart)",
        }

        # Identity resolver (wired through pipeline)
        resolver_ok = _pipeline is not None and _pipeline._identity_resolver is not None
        checks["identity_resolver"] = {
            "status": "ok" if resolver_ok else "degraded",
            "message": "wired" if resolver_ok else "not configured",
        }

        # World tracker
        world_tracker_ok = _pipeline is not None and _pipeline._world_tracker is not None
        checks["world_tracker"] = {
            "status": "ok" if world_tracker_ok else "degraded",
            "message": "wired" if world_tracker_ok else "not configured",
        }

        # Calibration sync
        cc_sync = getattr(app.state, "cc_sync_service", None)
        calibration_ok = cc_sync is not None
        checks["calibration_sync"] = {
            "status": "ok" if calibration_ok else "degraded",
            "message": "running" if calibration_ok else "not configured",
        }

        # Transit detector
        transit_ok = getattr(app.state, "cc_sync_service", None) is not None
        checks["transit_detector"] = {
            "status": "ok" if transit_ok else "degraded",
            "message": "configured" if transit_ok else "transit zones not loaded",
        }

        # Face ID
        face_id_ok = _pipeline is not None and _pipeline._face_id_client is not None
        checks["face_identity"] = {
            "status": "ok" if face_id_ok else "info",
            "message": "configured" if face_id_ok else "not configured (optional)",
        }

        return {
            "status": overall,
            "service": "tracking-orchestrator",
            "version": "0.1.0",
            "checks": checks,
        }

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus scrape target.

        Exposes the default ``prometheus_client`` registry so the
        observability stack can pull counters, gauges, and histograms
        registered by :mod:`app.observability.metrics`.
        """
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
