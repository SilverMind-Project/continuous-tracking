"""Prometheus metrics for the tracking orchestrator.

The metric names and labels follow phase-1 §1.9 (system-wide observability
table) and phase-3 §3.19 (per-pipeline counters).  Every metric is
registered against the default ``prometheus_client`` registry so the
``/metrics`` FastAPI route in :mod:`app.main` can expose them without
extra plumbing.

The :class:`Metrics` object is a lightweight namespace; tests construct
their own registry via :func:`build_metrics` to avoid leaking state
between cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

# Histogram buckets tuned for the latency budgets
# (frame-end-to-end p99 target = 450 ms).
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 200, 350, 500, 750, 1000, 2000, 5000)
# Posterior entropy is bits-per-decision; ranges 0..log2(N+1) for N
# residents. Values above 3 bits indicate an overly diffuse posterior.
ENTROPY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


class RevisionReasonKind(StrEnum):
    """Bounded label set for ``cts_tracking_revisions_published_total{kind}``."""

    INITIAL_ASSIGNMENT = "initial_assignment"
    DEMOTED_TO_UNKNOWN = "demoted_to_unknown"
    IDENTITY_CHANGED = "identity_changed"
    ARCFACE_AUTHORITY = "arcface_authority"
    ARCFACE_AUTHORITY_CONFLICT = "arcface_authority_conflict"
    UNKNOWN_BACKFILL = "unknown_backfill"
    OPERATOR_CORRECTION = "operator_correction"
    OTHER = "other"


class CameraProfileRejectReason(StrEnum):
    """Bounded label set for ``cts_camera_profile_rejected_total{reason}``."""

    UNKNOWN_ROLE = "unknown_role"
    MALFORMED_FIELD = "malformed_field"


class CCSyncResource(StrEnum):
    """Bounded label set for ``cts_cc_sync_failures_total{resource}``."""

    WEARABLES = "wearables"
    PERSON_HEIGHTS = "person_heights"


class WearableFixKindLabel(StrEnum):
    """Bounded label set for ``cts_wearable_fixes_total{kind}``."""

    POSITION = "position"
    ROOM = "room"
    PROXIMITY = "proximity"
    UNSPECIFIED = "unspecified"


class WearableFixIngestResult(StrEnum):
    """Bounded label set for ``cts_wearable_fixes_total{result}``."""

    ACCEPTED = "accepted"
    DROPPED_OFF_BODY = "dropped_off_body"
    DROPPED_STALE = "dropped_stale"
    DROPPED_UNBOUND = "dropped_unbound"
    DROPPED_MALFORMED = "dropped_malformed"


class WearableAssociationOutcome(StrEnum):
    """Bounded label set for ``cts_wearable_associations_total{result}``."""

    ACCEPTED = "accepted"
    AMBIGUOUS_TWO_PHS = "ambiguous_two_phs"
    AMBIGUOUS_TWO_TAGS = "ambiguous_two_tags"
    AMBIGUOUS_TIE = "ambiguous_tie"
    UNCALIBRATED_SHARED_ROOM = "uncalibrated_shared_room"
    PARKED = "parked"
    TAG_DISAGREEMENT = "tag_disagreement"
    REJECTED_GATE = "rejected_gate"
    INSUFFICIENT_FIXES = "insufficient_fixes"


class WearableSpatialFusionOutcome(StrEnum):
    """Bounded label set for ``cts_wearable_spatial_fusion_total{outcome}``."""

    APPLIED = "applied"
    BRIDGED = "bridged"
    GATED = "gated"
    NO_FIX = "no_fix"


class PositionTrackIngestResult(StrEnum):
    """Bounded label set for ``cts_position_tracks_total{result}``."""

    ACCEPTED = "accepted"
    DROPPED_STALE = "dropped_stale"
    DROPPED_UNREGISTERED = "dropped_unregistered"
    DROPPED_FLOOR_PLAN = "dropped_floor_plan"
    DROPPED_MALFORMED = "dropped_malformed"


class DeviceFreeObservationOutcome(StrEnum):
    """Bounded label set for ``cts_device_free_observations_total{outcome}``."""

    UPDATED = "updated"
    SPAWNED = "spawned"
    REVIVED = "revived"
    SPAWN_SUPPRESSED = "spawn_suppressed"


class DeviceFreeRoundOutcome(StrEnum):
    """Bounded label set for ``cts_device_free_rounds_total{outcome}``."""

    COMPLETED = "completed"
    FAILED = "failed"


class DeviceFreeShadowOutcome(StrEnum):
    """Bounded label set for ``cts_device_free_shadow_total{outcome}``."""

    WOULD_UPDATE = "would_update"
    WOULD_SPAWN = "would_spawn"
    WOULD_DROP = "would_drop"


class WearableShadowOutcome(StrEnum):
    """Bounded label set for ``cts_wearable_shadow_outcomes_total{outcome}``."""

    AGREES = "agrees"
    WOULD_NAME_UNKNOWN = "would_name_unknown"
    WOULD_CHANGE_IDENTITY = "would_change_identity"
    WOULD_CONFLICT = "would_conflict"


class WearableSeedOutcome(StrEnum):
    """Bounded label set for ``cts_wearable_seed_outcomes_total{outcome}``."""

    MINTED_AUTO_VERIFIED = "minted_auto_verified"
    MINTED_PENDING_REVIEW = "minted_pending_review"
    WEARABLE_NOT_ATTESTED = "wearable_not_attested"
    IDENTITY_MISMATCH = "identity_mismatch"
    FACE_CONFLICT = "face_conflict"
    UNKNOWN_ORIENTATION = "unknown_orientation"
    LOW_ORIENTATION_CONFIDENCE = "low_orientation_confidence"
    NO_EMBEDDING = "no_embedding"
    NON_FINITE_EMBEDDING = "non_finite_embedding"
    LOW_QUALITY = "low_quality"
    SEED_INTERVAL = "seed_interval"
    TOO_SIMILAR_TO_RECENT = "too_similar_to_recent"
    ROOM_DAILY_CAP = "room_daily_cap"
    CAP_REACHED = "cap_reached"
    CREATE_FAILED = "create_failed"
    TEACHING_HELD = "teaching_held"
    CHANNEL_NOT_LIVE = "channel_not_live"
    NO_IDENTITY = "no_identity"


class GalleryRetentionState(StrEnum):
    """Bounded lifecycle-state label set for ``cts_gallery_retention_rows_total{state, mode}``."""

    AUTO_VERIFIED = "auto_verified"
    PENDING_REVIEW = "pending_review"
    OPERATOR_VERIFIED = "operator_verified"


class GalleryRetentionModeLabel(StrEnum):
    """Bounded mode label set for ``cts_gallery_retention_rows_total{state, mode}``."""

    SHADOW = "shadow"
    ENABLED = "enabled"


class ImageOnlyArm(StrEnum):
    """Bounded label set for the image-only evaluation arm."""

    CONTROL = "control"
    TREATMENT = "treatment"
    CLASSIFIER = "classifier"


class ImageOnlyOutcome(StrEnum):
    """Bounded label set for an image-only evaluation outcome against wearable truth."""

    CORRECT = "correct"
    WRONG = "wrong"
    UNKNOWN = "unknown"


class WearableFaceAttestationResult(StrEnum):
    """Bounded label set for ``cts_wearable_face_attestations_total{result}``."""

    QUEUED = "queued"
    EXCLUDED_CONFLICT = "excluded_conflict"
    SENT = "sent"
    MOVED = "moved"
    BUFFER_MISS = "buffer_miss"
    FAILED = "failed"
    TEACHING_HELD = "teaching_held"


class DeadLetterStream(StrEnum):
    """Bounded label set for ``cts_messages_dead_lettered_total{stream}``."""

    WEARABLE_FIXES = "wearable_fixes"
    POSITION_TRACKS = "position_tracks"
    HOUSEHOLD_PRESENCE = "household_presence"


class ClosedWorldInactiveReason(StrEnum):
    """Bounded label set for ``cts_closed_world_inactive_total{reason}``."""

    ROSTER_INCOMPLETE = "roster_incomplete"
    NO_PRESENCE = "no_presence"
    NO_IDENTITIES = "no_identities"


class ClosedWorldShadowOutcome(StrEnum):
    """Bounded label set for ``cts_closed_world_shadow_outcomes_total{outcome}``."""

    WOULD_NAME_UNKNOWN = "would_name_unknown"
    WOULD_CHANGE = "would_change"
    AGREES = "agrees"


class HouseholdClassifierHealthLabel(StrEnum):
    """Bounded label set for ``cts_household_classifier_health{state}``."""

    OFF = "off"
    READY = "ready"
    DEGRADED_MISSING = "degraded_missing"
    DEGRADED_INCOMPATIBLE = "degraded_incompatible"
    DEGRADED_INVALID = "degraded_invalid"


class HouseholdClassifierShadowOutcome(StrEnum):
    """Bounded label set for ``cts_household_classifier_shadow_outcomes_total{outcome}``."""

    WOULD_NAME_UNKNOWN = "would_name_unknown"
    WOULD_CHANGE = "would_change"
    AGREES = "agrees"


class AppearanceSampleDropReason(StrEnum):
    """Bounded label set for ``cts_appearance_training_samples_dropped_total{reason}``."""

    NOT_ELIGIBLE = "not_eligible"
    LOW_QUALITY = "low_quality"
    NO_EMBEDDING = "no_embedding"
    NON_FINITE_EMBEDDING = "non_finite_embedding"
    DEDUP_WINDOW = "dedup_window"
    NEGATIVE_DAILY_CAP = "negative_daily_cap"
    TEACHING_HELD = "teaching_held"
    FACE_CONFLICT = "face_conflict"
    ROOM_GRADE_EXCLUDED = "room_grade_excluded"


_REVISION_REASON_PREFIX_MAP: dict[str, RevisionReasonKind] = {
    "initial_assignment": RevisionReasonKind.INITIAL_ASSIGNMENT,
    "demoted_to_unknown": RevisionReasonKind.DEMOTED_TO_UNKNOWN,
    "identity_change": RevisionReasonKind.IDENTITY_CHANGED,
    "identity_changed": RevisionReasonKind.IDENTITY_CHANGED,
    "arcface_authority": RevisionReasonKind.ARCFACE_AUTHORITY,
    "arcface_authority_conflict": RevisionReasonKind.ARCFACE_AUTHORITY_CONFLICT,
    "unknown_backfill": RevisionReasonKind.UNKNOWN_BACKFILL,
    "operator_correction": RevisionReasonKind.OPERATOR_CORRECTION,
    "operator_unknown": RevisionReasonKind.OPERATOR_CORRECTION,
    "operator_frame_correction": RevisionReasonKind.OPERATOR_CORRECTION,
}


def classify_revision_reason(reason: str | None) -> RevisionReasonKind:
    """Extract the prefix before the first ``:`` and map to a bounded ``RevisionReasonKind``."""
    if not reason:
        return RevisionReasonKind.OTHER
    prefix = reason.split(":", 1)[0].strip().lower()
    return _REVISION_REASON_PREFIX_MAP.get(prefix, RevisionReasonKind.OTHER)


@dataclass
class Metrics:
    """Container for the orchestrator's Prometheus metrics."""

    # ---- Stream consumption / publishing -----------------------------
    frames_consumed_total: Counter
    frames_failed_total: Counter
    frames_abandoned_total: Counter
    frames_stream_pending: Gauge
    frames_stream_lag: Gauge
    tracking_events_published_total: Counter
    tracking_revisions_published_total: Counter
    dementia_signals_published_total: Counter
    scene_samples_published_total: Counter

    # ---- Codec / wire-format -----------------------------------------
    proto_messages_emitted_total: Counter
    proto_messages_decoded_total: Counter
    proto_decode_errors_total: Counter

    # ---- Pipeline state ----------------------------------------------
    tracklets_active: Gauge
    global_tracks_active: Gauge
    gallery_size: Gauge

    # ---- Identity resolution -----------------------------------------
    identity_commits_total: Counter
    identity_revisions_total: Counter
    identity_demotions_total: Counter
    face_propagations_total: Counter
    face_id_cooldown_skips_total: Counter
    gallery_backfills_skipped_total: Counter
    reid_cross_camera_assist_total: Counter
    height_evidence_frames_total: Counter
    unknown_gts_merged_total: Counter
    identity_decays_total: Counter
    identity_quality_gate_blocks_total: Counter
    identity_flips_total: Counter
    identity_shadow_mismatch_total: Counter
    posterior_entropy: Histogram
    homography_rejected_total: Counter
    homography_warning_total: Counter
    camera_profile_rejected_total: Counter
    camera_height_pose_unreliable_total: Counter
    cc_sync_failures_total: Counter

    # ---- Calibration and transit metrics -----------------------------------
    worldtracker_observations_by_calibration_total: Counter  # labels: camera_id, calibrated
    overlap_groups_loaded: Gauge
    transit_event_published_total: Counter
    transit_event_unknown_identity_total: Counter
    transit_zones_loaded: Gauge
    transit_zone_rejected_total: Counter

    # ---- Staleness / backlog -----------------------------------------
    frames_dropped_stale_total: Counter

    # ---- Privacy enforcement ------------------------------------------
    privacy_detections_dropped_total: Counter

    # ---- Phase 1: noise reduction ------------------------------------
    detections_suppressed_total: Counter
    tracklets_dedup_dropped_total: Counter
    tracklets_held_below_stability_gate: Gauge
    revision_rows_rewritten_total: Counter

    # ---- Signal worker -------------------------------------------------
    signal_worker_run_seconds: Histogram
    signal_worker_identities: Gauge
    signal_worker_emitted_total: Counter
    signal_baseline_cache_hits_total: Counter
    appearance_signal_candidates_total: Counter

    # ---- World tracker -------------------------------------------------
    world_tracker_ph_open: Gauge
    world_tracker_ph_spawned_total: Counter
    world_tracker_ph_closed_total: Counter

    # ---- PH lifecycle ---------------------------------------------------
    ph_lifetime_seconds: Histogram
    ph_observations_at_close: Histogram
    identity_unknown_after_known_total: Counter

    world_tracker_assignment_cost: Histogram
    world_tracker_continuations_total: Counter
    world_tracker_spawn_rejected_out_of_room_total: Counter

    # ---- cross-camera dedup -----------------------------------------
    worldtracker_observations_deduped_total: Counter
    worldtracker_dedup_clusters_total: Counter
    worldtracker_observation_missing_floorpoint_total: Counter
    worldtracker_dedup_pair_distance_m: Histogram

    # ---- association integrity (primary pass only) -----------------
    worldtracker_association_rejections_total: Counter  # label: reason
    worldtracker_association_outcome_total: Counter  # label: outcome
    worldtracker_appearance_updates_rejected_total: Counter  # label: reason
    worldtracker_batch_skew_ms: Histogram  # label: camera_id

    # ---- Keyframe quality ----------------------------------------------
    keyframe_dropped_low_confidence_total: Counter

    # ---- Latency -----------------------------------------------------
    frame_end_to_end_latency_ms: Histogram
    triton_inference_latency_ms: Histogram

    # ---- Posture slow-path -------------------------------------------
    cts_posture_slow_path_runs_total: Counter
    cts_posture_slow_path_latency_seconds: Histogram

    # ---- Posture fusion / hysteresis ----------------------------------
    cts_posture_hysteresis_flips_total: Counter
    cts_posture_camera_contributions_total: Counter
    cts_posture_cameras_fused: Histogram
    cts_posture_view_weight: Histogram
    cts_posture_fused_class_total: Counter

    # ---- Stage latency -----------------------------------------------
    stage_latency_ms: Histogram

    # ---- Batching ----------------------------------------------------
    batch_size_metric: Histogram

    # ---- PH operations -------------------------------------------------
    cts_ph_corrections_total: Counter
    cts_ph_merges_total: Counter
    cts_ph_splits_total: Counter
    cts_ph_api_latency_seconds: Histogram

    # ---- PH continuity ----
    cts_ph_revived_total: Counter
    world_tracker_shadow_revival_total: Counter
    world_tracker_shadow_assoc_mismatch_total: Counter

    # ---- Rich face evidence ----
    cts_face_anchors_total: Counter
    cts_face_gate_outcomes_total: Counter
    cts_face_queries_total: Counter
    cts_face_stats_write_failures_total: Counter
    cts_face_attribution_total: Counter
    cts_face_anchor_duplicate_detection_total: Counter
    cts_face_anchor_grade_overlap_total: Counter
    cts_face_shadow_outcomes_total: Counter
    cts_face_template_requests_total: Counter
    cts_face_template_buffer_size: Histogram
    cts_face_template_outliers_total: Counter
    cts_face_template_latency_seconds: Histogram
    cts_face_template_authority_blocked_total: Counter
    cts_face_template_shadow_outcomes_total: Counter
    cts_face_template_results_discarded_total: Counter

    # ---- Cross-camera & co-presence ----
    world_tracker_shadow_cross_camera_revival_total: Counter
    worldtracker_group_appearance_dedup_total: Counter
    worldtracker_copresence_links_total: Counter

    # ---- CC load decoupling (Tier 2 event emission) ----
    cts_presence_events_published_total: Counter
    cts_dwell_events_published_total: Counter

    # ---- Fall detection fast path ----
    cts_fall_suspected_total: Counter
    cts_fall_descent_rate: Histogram
    cts_fall_suspected_unidentified_total: Counter

    # ---- Low-confidence detection recovery ----
    worldtracker_low_band_matches_total: Counter
    worldtracker_low_band_dropped_total: Counter
    worldtracker_out_of_order_observations_total: Counter

    # ---- Low-confidence band measurement (diagnostic, gated, read-only) ----
    # Quantifies how often the detector_confidence cut hides a present person:
    #   band="high"     — >=1 person box at/above the high threshold this frame
    #   band="low_only" — 0 high boxes but >=1 box in [low_floor, high) (a gap
    #                     caused purely by the threshold — the discriminator)
    #   band="empty"    — no person boxes at all (genuine no-detection)
    detector_band_frames_total: Counter
    detector_lowband_boxes_total: Counter

    # ---- Adaptive ReID cadence ----
    cts_reid_executed_total: Counter
    cts_reid_skipped_total: Counter

    # ---- identity-integrity observability ----
    # Bounded labels only; PH/decision/identity IDs go to structlog, never here.
    identity_duplicate_active_blocks_total: Counter  # enforced duplicate / tie-clear blocks
    reid_rejected_vector_vote_attempts_total: Counter  # invariant=0; alert if >0
    identity_prior_only_updates_total: Counter  # temporal-prior maintenance updates
    # The two breach counters below are page-now invariants: each is a runtime
    # assertion whose value must stay zero. Any increment is a defect.
    identity_duplicate_active_breach_total: Counter  # >1 active PH holds one identity post-commit
    identity_prior_only_evidence_advance_total: Counter  # prior-only advanced evidence time

    # ---- governed candidate creation ----
    reid_candidate_rejected_total: Counter  # label: reason (bounded, typed set)
    reid_candidate_created_total: Counter  # label: state (pending_review|auto_verified)

    # ---- temporal vote window ----
    # Vote volume per query path. Compare against candidate-creation volume to
    # see the vote corpus collapse if gallery_vote_max_age_s starves ReID.
    reid_gallery_hits_total: Counter  # label: path (multiview|fallback|shadow)

    # ---- unknown-segment backfill ----
    identity_backfill_total: Counter  # labels: mode (shadow|enabled), outcome
    identity_backfill_clip_total: Counter  # label: reason
    identity_backfill_range_seconds: Histogram

    # ---- CC assertion evidence integrity ----
    cc_assertions_matched_total: Counter  # label: gate (floor|room)
    cc_assertions_rejected_total: Counter  # label: reason
    cc_assertions_shadow_total: Counter  # label: outcome (would_name_unknown|agrees|disagrees)

    # ---- identity roster reconciliation ----
    # labels: mode (shadow|enabled), action (bounded RosterAction set)
    identity_roster_reconcile_total: Counter

    # ---- wearable association and evidence ----
    cts_wearable_fixes_total: Counter  # labels: kind, result
    cts_wearable_associations_total: Counter  # label: result
    cts_wearable_assoc_residual_m: Histogram  # label: room_id
    cts_wearable_face_conflict_total: Counter
    cts_wearable_shadow_outcomes_total: Counter  # label: outcome
    cts_wearable_spatial_fusion_total: Counter  # labels: mode, outcome
    cts_wearable_spatial_fusion_shift_m: Histogram  # label: mode
    cts_messages_dead_lettered_total: Counter  # label: stream

    # ---- device-free position sensors ----
    cts_position_tracks_total: Counter  # label: result (PositionTrackIngestResult)
    cts_device_free_observations_total: Counter  # label: outcome
    cts_device_free_rounds_total: Counter  # label: outcome (DeviceFreeRoundOutcome)
    cts_device_free_shadow_total: Counter  # label: outcome (DeviceFreeShadowOutcome)

    # ---- wearable supervision ----
    cts_wearable_seed_outcomes_total: Counter  # label: outcome (WearableSeedOutcome)
    # labels: arm (ImageOnlyArm), outcome (ImageOnlyOutcome), grade (room|trajectory)
    cts_image_only_eval_seconds_total: Counter
    cts_wearable_face_attestations_total: Counter  # label: result

    # ---- gallery retention ----
    # labels: state (GalleryRetentionState), mode (GalleryRetentionModeLabel)
    cts_gallery_retention_rows_total: Counter
    cts_gallery_retention_crop_delete_failures_total: Counter

    # ---- closed-world prior shaping ----
    cts_closed_world_inactive_total: Counter  # label: reason (ClosedWorldInactiveReason)
    cts_closed_world_shadow_outcomes_total: Counter  # label: outcome (ClosedWorldShadowOutcome)

    # ---- household appearance classifier ----
    cts_household_classifier_health: Gauge  # label: state (HouseholdClassifierHealthLabel)
    cts_household_classifier_shadow_outcomes_total: Counter  # label: outcome
    cts_appearance_training_samples_recorded_total: Counter  # label: source (AppearanceLabelSource)
    cts_appearance_training_samples_dropped_total: Counter  # label: reason


def build_metrics(registry: CollectorRegistry = REGISTRY) -> Metrics:
    """Create a :class:`Metrics` bound to *registry*.

    The default registry is the global ``prometheus_client.REGISTRY`` so
    the FastAPI ``/metrics`` route exposes them automatically. Tests
    pass a fresh ``CollectorRegistry()`` to keep state isolated.
    """

    def _counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
        return Counter(name, doc, labelnames=labels or [], registry=registry)

    def _gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
        return Gauge(name, doc, labelnames=labels or [], registry=registry)

    def _hist(
        name: str,
        doc: str,
        buckets: tuple[float, ...],
        labels: list[str] | None = None,
    ) -> Histogram:
        return Histogram(name, doc, labelnames=labels or [], buckets=buckets, registry=registry)

    return Metrics(
        frames_consumed_total=_counter(
            "cts_frames_consumed_total",
            "FrameReady messages successfully consumed from frames.ready.",
            ["camera_id"],
        ),
        frames_failed_total=_counter(
            "cts_frames_failed_total",
            "FrameReady messages whose pipeline processing raised.",
            ["camera_id", "reason"],
        ),
        frames_abandoned_total=_counter(
            "cts_frames_abandoned_total",
            "Stale pending FrameReady entries reclaimed and acknowledged at startup.",
        ),
        frames_stream_pending=_gauge(
            "cts_frames_stream_pending",
            "Unacknowledged entries in the frames.ready consumer group PEL.",
        ),
        frames_stream_lag=_gauge(
            "cts_frames_stream_lag",
            "Unread entries behind the frames.ready consumer group cursor.",
        ),
        tracking_events_published_total=_counter(
            "cts_tracking_events_published_total",
            "TrackingEvent messages emitted to tracking.events.",
            ["camera_id"],
        ),
        tracking_revisions_published_total=_counter(
            "cts_tracking_revisions_published_total",
            "IdentityRevision messages emitted to tracking.revisions.",
            ["kind"],
        ),
        dementia_signals_published_total=_counter(
            "cts_dementia_signals_published_total",
            "DementiaSignal messages emitted to tracking.signals.",
            ["signal_kind", "severity"],
        ),
        scene_samples_published_total=_counter(
            "cts_scene_samples_published_total",
            "SceneSample messages emitted to scene.samples.",
            ["reason"],
        ),
        proto_messages_emitted_total=_counter(
            "cts_proto_messages_emitted_total",
            "Messages serialised in proto wire format.",
            ["stream"],
        ),
        proto_messages_decoded_total=_counter(
            "cts_proto_messages_decoded_total",
            "Messages successfully decoded as proto on consumption.",
            ["stream"],
        ),
        proto_decode_errors_total=_counter(
            "cts_proto_decode_errors_total",
            "Messages that failed proto decode (codec sentinel present, parse failed).",
            ["stream"],
        ),
        tracklets_active=_gauge(
            "cts_tracklets_active",
            "Number of active per-camera tracklets.",
            ["camera_id"],
        ),
        global_tracks_active=_gauge(
            "cts_global_tracks_active",
            "Number of open GlobalTrack records.",
        ),
        gallery_size=_gauge(
            "cts_gallery_size",
            "Number of embeddings currently in the ReID gallery.",
        ),
        identity_commits_total=_counter(
            "cts_identity_commits_total",
            "Identity decisions that fired the commit rule.",
            ["source"],
        ),
        identity_revisions_total=_counter(
            "cts_identity_revisions_total",
            "Identity revisions emitted (post-commit changes).",
            ["reason"],
        ),
        identity_demotions_total=_counter(
            "cts_identity_demotions_total",
            "Identity demotions to UNKNOWN (all evidence in window said UNKNOWN).",
        ),
        face_propagations_total=_counter(
            "cts_face_propagations_total",
            "Face anchors propagated to adjacent GlobalTracks.",
        ),
        gallery_backfills_skipped_total=_counter(
            "cts_gallery_backfills_skipped_total",
            "Gallery backfills skipped because the committed identity"
            " has not yet survived the confirmation delay.",
        ),
        reid_cross_camera_assist_total=_counter(
            "cts_reid_cross_camera_assist_total",
            "Identity commits whose strongest ReID support came from another camera.",
        ),
        homography_rejected_total=_counter(
            "cts_homography_rejected_total",
            "Homography matrices rejected by the server-side validator during CC sync.",
            ["reason", "camera_id"],
        ),
        homography_warning_total=_counter(
            "cts_homography_warning_total",
            "Homography matrices accepted with warnings during CC sync.",
            ["reason", "camera_id"],
        ),
        camera_profile_rejected_total=_counter(
            "cts_camera_profile_rejected_total",
            "Camera profile entries rejected during CC config sync (e.g. unknown role).",
            ["camera_id", "reason"],
        ),
        camera_height_pose_unreliable_total=_counter(
            "cts_camera_height_pose_unreliable_total",
            "Cameras marked unreliable for single-view standing height estimation by reason.",
            ["reason"],
        ),
        cc_sync_failures_total=_counter(
            "cts_cc_sync_failures_total",
            "CC config sync fetch or decode failures that kept the previous live snapshot.",
            ["resource"],
        ),
        # ---- Calibration and transit metrics ------------------------------------
        worldtracker_observations_by_calibration_total=_counter(
            "cts_worldtracker_observations_by_calibration_total",
            (
                "World observations built per camera, split by whether the floor point came from "
                "a homography (calibrated=true) or the per-camera virtual room (calibrated=false)."
            ),
            ["camera_id", "calibrated"],
        ),
        overlap_groups_loaded=_gauge(
            "cts_overlap_groups_loaded",
            "Camera overlap groups currently loaded from CC.",
        ),
        transit_event_published_total=_counter(
            "cts_transit_event_published_total",
            "RoomTransitionEvent messages published to tracking.room_transitions.",
            ["direction"],
        ),
        transit_event_unknown_identity_total=_counter(
            "cts_transit_event_unknown_identity_total",
            "RoomTransitionEvent messages where identity_id was unknown.",
        ),
        transit_zones_loaded=_gauge(
            "cts_transit_zones_loaded",
            "Transit zones currently loaded from CC in floor-plan metre coordinates.",
        ),
        transit_zone_rejected_total=_counter(
            "cts_transit_zone_rejected_total",
            "Transit zones rejected during CC sync contract validation.",
            ["reason"],
        ),
        face_id_cooldown_skips_total=_counter(
            "cts_face_id_cooldown_skips_total",
            "Tracklets skipped for face ID due to per-tracklet cooldown.",
        ),
        height_evidence_frames_total=_counter(
            "cts_height_evidence_frames_total",
            "Frames where height evidence was available for at least one tracklet.",
        ),
        unknown_gts_merged_total=_counter(
            "cts_unknown_gts_merged_total",
            "UNKNOWN GlobalTracks merged via temporal+spatial heuristic.",
        ),
        identity_decays_total=_counter(
            "cts_identity_decays_total",
            "Identities cleared because the maintenance window expired"
            " without fresh confirming evidence.",
        ),
        identity_quality_gate_blocks_total=_counter(
            "cts_identity_quality_gate_blocks_total",
            "Identity commits suppressed by the PH quality gate.",
        ),
        identity_flips_total=_counter(
            "cts_identity_flips_total",
            "Committed identity changes from one non-UNKNOWN identity to another.",
        ),
        identity_shadow_mismatch_total=_counter(
            "cts_identity_shadow_mismatch_total",
            "Shadow-mode identity decisions that would differ from live behavior.",
            ["feature"],
        ),
        posterior_entropy=_hist(
            "cts_posterior_entropy_bits",
            "Per-decision Bayesian posterior entropy (bits).",
            ENTROPY_BUCKETS,
        ),
        frames_dropped_stale_total=_counter(
            "cts_frames_dropped_stale_total",
            "FrameReady messages silently dropped because capture_time_unix_ns"
            " was older than the max-frame-age threshold.",
            ["camera_id"],
        ),
        privacy_detections_dropped_total=_counter(
            "cts_privacy_detections_dropped_total",
            "Person detections dropped by privacy zone enforcement.",
            ["camera_id"],
        ),
        detections_suppressed_total=_counter(
            "cts_detections_suppressed_total",
            "Detections suppressed before tracker: score_threshold or post-decode IoU dedup.",
            ["stage"],
        ),
        tracklets_dedup_dropped_total=_counter(
            "cts_tracklets_dedup_dropped_total",
            "Newly-spawned tracklets dropped because they overlapped a stable existing tracklet.",
            ["camera_id"],
        ),
        tracklets_held_below_stability_gate=_gauge(
            "cts_tracklets_held_below_stability_gate",
            "Active tracklets currently below the frames_alive stability gate.",
            ["camera_id"],
        ),
        revision_rows_rewritten_total=_counter(
            "cts_revision_rows_rewritten_total",
            "DB rows retroactively relabelled by the identity rewriter.",
            ["table"],
        ),
        # ---- World tracker ---------------------------------------------------
        world_tracker_ph_open=_gauge(
            "cts_world_tracker_ph_open",
            "Currently open Person Hypotheses.",
        ),
        world_tracker_ph_spawned_total=_counter(
            "cts_world_tracker_ph_spawned_total",
            "PHs created since process start.",
            ["reason"],
        ),
        world_tracker_ph_closed_total=_counter(
            "cts_world_tracker_ph_closed_total",
            "PHs closed since process start.",
        ),
        # ---- PH lifecycle -----------------------------------------------------
        ph_lifetime_seconds=_hist(
            "cts_ph_lifetime_seconds",
            "Seconds from PH creation to close.",
            (0.5, 1, 2, 5, 10, 30, 60, 300),
        ),
        ph_observations_at_close=_hist(
            "cts_ph_observations_at_close",
            "Observation count at PH close time.",
            (1, 2, 3, 5, 10, 30, 100),
        ),
        identity_unknown_after_known_total=_counter(
            "cts_identity_unknown_after_known_total",
            "PHs that previously had a known identity now resolving to UNKNOWN.",
        ),
        world_tracker_assignment_cost=_hist(
            "cts_world_tracker_assignment_cost",
            "Final cost of matched (PH, obs) pairs.",
            (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
        ),
        world_tracker_continuations_total=_counter(
            "cts_world_tracker_continuations_total",
            "PH continuation candidates published.",
        ),
        world_tracker_spawn_rejected_out_of_room_total=_counter(
            "cts_world_tracker_spawn_rejected_out_of_room_total",
            "WorldTracker PH spawns rejected because a calibrated observation was outside rooms.",
        ),
        # ---- cross-camera dedup -----------------------------------------
        worldtracker_observations_deduped_total=_counter(
            "cts_worldtracker_observations_deduped_total",
            "Source observations merged into a dedup cluster representative (one per collapsed).",
        ),
        worldtracker_dedup_clusters_total=_counter(
            "cts_worldtracker_dedup_clusters_total",
            "Multi-observation dedup clusters formed (one per cluster with >1 member).",
        ),
        worldtracker_observation_missing_floorpoint_total=_counter(
            "cts_worldtracker_observation_missing_floorpoint_total",
            "Observations skipped from dedup weighting due to missing calibrated floor point.",
        ),
        worldtracker_dedup_pair_distance_m=_hist(
            "cts_worldtracker_dedup_pair_distance_m",
            (
                "Floor distance between two calibrated observations from different cameras that "
                "the floor-distance dedup pass merged: the cross-camera agreement measurement."
            ),
            buckets=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5),
        ),
        # ---- association integrity (emitted from the primary pass only) ----
        worldtracker_association_rejections_total=_counter(
            "cts_worldtracker_association_rejections_total",
            "Gated (PH, obs) pairs by typed reason in the primary association pass.",
            ["reason"],
        ),
        worldtracker_association_outcome_total=_counter(
            "cts_worldtracker_association_outcome_total",
            "Primary-pass association outcomes: matched / unmatched_obs / unmatched_ph.",
            ["outcome"],
        ),
        worldtracker_appearance_updates_rejected_total=_counter(
            "cts_worldtracker_appearance_updates_rejected_total",
            "Matched observations whose embedding was barred from PH appearance, by reason.",
            ["reason"],
        ),
        worldtracker_batch_skew_ms=_hist(
            "cts_worldtracker_batch_skew_ms",
            "Per-camera observation age (now - captured_at) within a tracker batch.",
            (1, 5, 10, 25, 50, 100, 200, 500, 1000),
            ["camera_id"],
        ),
        # ---- Keyframe quality ----------------------------------------------
        keyframe_dropped_low_confidence_total=_counter(
            "cts_keyframe_dropped_low_confidence_total",
            "Keyframe bbox annotations skipped due to low detection confidence.",
        ),
        signal_worker_run_seconds=_hist(
            "cts_signal_worker_run_seconds",
            "Wall-clock duration of a DementiaSignalWorker.run_once cycle.",
            LATENCY_BUCKETS_MS,
        ),
        signal_worker_identities=_gauge(
            "cts_signal_worker_identities",
            "Number of identities processed in the last run_once cycle.",
        ),
        signal_worker_emitted_total=_counter(
            "cts_signal_worker_emitted_total",
            "Dementia signals emitted by the signal worker.",
            ["kind", "severity"],
        ),
        signal_baseline_cache_hits_total=_counter(
            "cts_signal_baseline_cache_hits_total",
            "Baseline repository cache hits in the signal worker.",
        ),
        appearance_signal_candidates_total=_counter(
            "cts_appearance_signal_candidates_total",
            "same_clothes_suspected candidate evaluations (shadow-mode observability, DL-M07).",
        ),
        frame_end_to_end_latency_ms=_hist(
            "cts_frame_end_to_end_latency_ms",
            "Wall-clock latency from FrameReady receipt to TrackingEvent publish.",
            LATENCY_BUCKETS_MS,
            ["camera_id"],
        ),
        triton_inference_latency_ms=_hist(
            "cts_triton_inference_latency_ms",
            "Per-model Triton gRPC inference latency.",
            LATENCY_BUCKETS_MS,
            ["model"],
        ),
        cts_posture_slow_path_runs_total=_counter(
            "cts_posture_slow_path_runs_total",
            "Number of depth-based posture inference runs.",
            ["camera_id"],
        ),
        cts_posture_slow_path_latency_seconds=_hist(
            "cts_posture_slow_path_latency_seconds",
            "Latency of depth-based posture inference.",
            (0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
        ),
        cts_posture_hysteresis_flips_total=_counter(
            "cts_posture_hysteresis_flips_total",
            "Total number of posture hysteresis state flips (committed posture changed).",
            ["camera_id"],
        ),
        cts_posture_camera_contributions_total=_counter(
            "cts_posture_camera_contributions_total",
            "Total number of per-camera posture score updates submitted to GlobalPostureTracker.",
            ["camera_id"],
        ),
        cts_posture_cameras_fused=_hist(
            "cts_posture_cameras_fused",
            "Number of cameras contributing to each fusion cycle (non-stale).",
            (1.0, 2.0, 3.0, 4.0, 5.0),
        ),
        cts_posture_view_weight=_hist(
            "cts_posture_view_weight",
            "Geometry suitability multiplier applied to each per-camera posture contribution.",
            (0.0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0),
        ),
        cts_posture_fused_class_total=_counter(
            "cts_posture_fused_class_total",
            "Posture class assigned after fusion, before hysteresis.",
            ["posture"],
        ),
        stage_latency_ms=_hist(
            "cts_stage_latency_ms",
            "Per-stage wall-clock latency within the frame pipeline.",
            LATENCY_BUCKETS_MS,
            ["stage", "camera_id"],
        ),
        batch_size_metric=_hist(
            "cts_batch_size",
            "Number of frames accumulated in a batch flush (across cameras).",
            (1, 2, 3, 4, 5, 6, 8, 12, 16),
        ),
        # ---- PH operations -------------------------------------------------
        cts_ph_corrections_total=_counter(
            "cts_ph_corrections_total",
            "Total PH identity corrections applied",
            ["actor"],
        ),
        cts_ph_merges_total=_counter(
            "cts_ph_merges_total",
            "Total PH merge operations",
            ["actor"],
        ),
        cts_ph_splits_total=_counter(
            "cts_ph_splits_total",
            "Total PH split operations",
            ["actor"],
        ),
        cts_ph_api_latency_seconds=_hist(
            "cts_ph_api_latency_seconds",
            "PH API endpoint request latency in seconds",
            (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            ["endpoint"],
        ),
        # ---- PH continuity ----
        cts_ph_revived_total=_counter(
            "cts_ph_revived_total",
            "PHs revived from recently-closed state instead of spawning new.",
        ),
        world_tracker_shadow_revival_total=_counter(
            "cts_world_tracker_shadow_revival_total",
            "Shadow count: revivals that would have fired if enable_ph_revival were on.",
        ),
        world_tracker_shadow_assoc_mismatch_total=_counter(
            "cts_world_tracker_shadow_assoc_mismatch_total",
            "Shadow count: assoc decisions that would differ under relaxed uncalibrated gate.",
        ),
        # ---- Rich face evidence ----
        cts_face_anchors_total=_counter(
            "cts_face_anchors_total",
            "Face anchors produced by FaceIdentityStage.",
            ["recognition_state"],
        ),
        cts_face_gate_outcomes_total=_counter(
            "cts_face_gate_outcomes_total",
            "Faces returned by person-identification-service, by what FaceIdentityStage did "
            "with them (emitted as an anchor or dropped, and why).",
            ["camera_id", "outcome"],
        ),
        cts_face_queries_total=_counter(
            "cts_face_queries_total",
            "Person crops sent for face identification, by result "
            "(face_found, no_face, too_small, error).",
            ["camera_id", "result"],
        ),
        cts_face_stats_write_failures_total=_counter(
            "cts_face_stats_write_failures_total",
            "Failed writes of face observation statistics rows; face evidence is unaffected.",
        ),
        cts_face_attribution_total=_counter(
            "cts_face_attribution_total",
            "Person crops with faces, by how a face was attributed to the person "
            "(assigned, no_face_in_head_region, ambiguous).",
            ["camera_id", "result"],
        ),
        cts_face_anchor_duplicate_detection_total=_counter(
            "cts_face_anchor_duplicate_detection_total",
            "Same-grade face anchors that collided on one detection_id; weaker match dropped.",
        ),
        cts_face_anchor_grade_overlap_total=_counter(
            "cts_face_anchor_grade_overlap_total",
            "Cross-grade face anchors that overlapped on one detection_id; higher precedence kept.",
            ["kept", "dropped"],
        ),
        cts_face_template_requests_total=_counter(
            "cts_face_template_requests_total",
            "Face template match requests to person-identification-service, by result "
            "(recognized, candidate, unrecognized, error, dropped_in_flight_limit).",
            ["result"],
        ),
        cts_face_template_buffer_size=_hist(
            "cts_face_template_buffer_size",
            "Faces held for a PH's face template after a tracker round, per PH with faces.",
            buckets=(1, 2, 3, 4, 5, 6, 8, 12, 16),
        ),
        cts_face_template_outliers_total=_counter(
            "cts_face_template_outliers_total",
            "Faces kept out of a PH's template because they did not resemble its other faces "
            "(a likely association swap).",
        ),
        cts_face_template_latency_seconds=_hist(
            "cts_face_template_latency_seconds",
            "Wall time of one face template match request, including queueing for a slot.",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        ),
        cts_face_template_authority_blocked_total=_counter(
            "cts_face_template_authority_blocked_total",
            "Recognized face templates that may vote but cannot grant ArcFace authority, by reason "
            "(member_disagreement, uncalibrated).",
            ["reason"],
        ),
        cts_face_template_shadow_outcomes_total=_counter(
            "cts_face_template_shadow_outcomes_total",
            "Per-PH decisions with a face template under face_templates.mode shadow, by how the "
            "decision with the template compares with live (agrees, would_commit_new, "
            "would_change_identity, would_become_unknown).",
            ["outcome"],
        ),
        cts_face_template_results_discarded_total=_counter(
            "cts_face_template_results_discarded_total",
            "Face template results never applied, by reason (stale_after_reset, expired, "
            "not_identifying).",
            ["reason"],
        ),
        cts_face_shadow_outcomes_total=_counter(
            "cts_face_shadow_outcomes_total",
            "Per-PH decisions with a recognized face under the shadow face evidence policy, by "
            "how the calibrated policy's decision compares with live (agrees, would_commit_new, "
            "would_change_identity, would_become_unknown).",
            ["outcome"],
        ),
        # ---- Cross-camera and co-presence ----
        world_tracker_shadow_cross_camera_revival_total=_counter(
            "cts_world_tracker_shadow_cross_camera_revival_total",
            "Shadow cross-camera PH revivals that would fire if enabled.",
        ),
        worldtracker_group_appearance_dedup_total=_counter(
            "cts_worldtracker_group_appearance_dedup_total",
            "Uncalibrated observation pairs merged by appearance inside a declared overlap group.",
        ),
        worldtracker_copresence_links_total=_counter(
            "cts_worldtracker_copresence_links_total",
            "Co-presence links written between PHs sharing an identity.",
        ),
        # ---- CC load decoupling (Tier 2 event emission) ----
        cts_presence_events_published_total=_counter(
            "cts_presence_events_published_total",
            "Presence events published to tracking.presence.",
            ["event_type"],
        ),
        cts_dwell_events_published_total=_counter(
            "cts_dwell_events_published_total",
            "Dwell events published to tracking.dwell.",
            ["event_type"],
        ),
        # ---- Fall detection fast path ----
        cts_fall_suspected_total=_counter(
            "cts_fall_suspected_total",
            "fall_suspected signals emitted by FallDetectionStage.",
            ["severity"],
        ),
        cts_fall_descent_rate=_hist(
            "cts_fall_descent_rate",
            "Max descent rate (heights/s) at fall_suspected emission.",
            (0.1, 0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0),
        ),
        cts_fall_suspected_unidentified_total=_counter(
            "cts_fall_suspected_unidentified_total",
            "Fall detections suppressed because the PH had no committed identity.",
        ),
        # ---- Low-confidence detection recovery ----
        worldtracker_low_band_matches_total=_counter(
            "cts_worldtracker_low_band_matches_total",
            "PHs updated via low-confidence second association pass.",
        ),
        worldtracker_low_band_dropped_total=_counter(
            "cts_worldtracker_low_band_dropped_total",
            "Low-band observations dropped (no open PH within recovery gate).",
        ),
        worldtracker_out_of_order_observations_total=_counter(
            "cts_worldtracker_out_of_order_observations_total",
            "Observations whose captured_at preceded the matched PH's last_seen_at.",
            ["camera_id"],
        ),
        # ---- Low-confidence band measurement (diagnostic) ----
        detector_band_frames_total=_counter(
            "cts_detector_band_frames_total",
            "Per-frame detection-band classification (high / low_only / empty).",
            ["camera_id", "band"],
        ),
        detector_lowband_boxes_total=_counter(
            "cts_detector_lowband_boxes_total",
            "Person boxes in [low_floor, high_threshold) discarded by the cut.",
            ["camera_id"],
        ),
        # ---- Adaptive ReID cadence ----
        cts_reid_executed_total=_counter(
            "cts_reid_executed_total",
            "Frames where SOLIDER-ReID embed_batch was called.",
        ),
        cts_reid_skipped_total=_counter(
            "cts_reid_skipped_total",
            "Frames where adaptive policy skipped ReID (or would have skipped in shadow mode).",
            ["reason"],
        ),
        # ---- identity-integrity observability ----
        identity_duplicate_active_blocks_total=_counter(
            "cts_identity_duplicate_active_blocks_total",
            "New identity assignments demoted to UNKNOWN by the enforced "
            "duplicate-active-identity guard (occupied holder or tie-clear).",
        ),
        reid_rejected_vector_vote_attempts_total=_counter(
            "cts_reid_rejected_vector_vote_attempts_total",
            "Gallery vectors that reached the resolver vote without "
            "operator_verified state. Invariant is zero; any increment is a "
            "governance breach and pages.",
        ),
        identity_prior_only_updates_total=_counter(
            "cts_identity_prior_only_updates_total",
            "Temporal-prior maintenance updates applied (prior-only, never "
            "advancing independent identity evidence time).",
        ),
        identity_duplicate_active_breach_total=_counter(
            "cts_identity_duplicate_active_breach_total",
            "Post-commit invariant breach: a household identity is held by more "
            "than one active PH. Must stay zero; any increment pages.",
        ),
        identity_prior_only_evidence_advance_total=_counter(
            "cts_identity_prior_only_evidence_advance_total",
            "Invariant breach: a prior-only maintenance decision advanced "
            "independent identity evidence time to the current frame. Must stay "
            "zero; any increment pages.",
        ),
        identity_roster_reconcile_total=_counter(
            "cts_identity_roster_reconcile_total",
            "Identity roster reconciliation against person-identification-service enrollment, "
            "by mode and action (registered, retired, gallery_rejected, ph_cleared, skipped_*).",
            ["mode", "action"],
        ),
        reid_candidate_rejected_total=_counter(
            "cts_reid_candidate_rejected_total",
            "Detections that failed the governed ReID candidate eligibility gate.",
            ["reason"],
        ),
        reid_candidate_created_total=_counter(
            "cts_reid_candidate_created_total",
            "Governed ReID gallery candidates created by ReIDCandidateStage, by mint state.",
            ["state"],
        ),
        reid_gallery_hits_total=_counter(
            "cts_reid_gallery_hits_total",
            "Gallery search_similar hits returned per query path, after the M03 vote-age "
            "cutoff and voting-states filter.",
            ["path"],
        ),
        identity_backfill_total=_counter(
            "cts_identity_backfill_total",
            "Unknown-segment backfill attempts by mode (shadow|enabled) and outcome "
            "(applied, shadow, operator_conflict, skipped_<reason>).",
            ["mode", "outcome"],
        ),
        identity_backfill_clip_total=_counter(
            "cts_identity_backfill_clip_total",
            "Unknown-segment backfill range clips by reason (clipped_prior_decision, "
            "clipped_operator_range, clipped_cap).",
            ["reason"],
        ),
        identity_backfill_range_seconds=_hist(
            "cts_identity_backfill_range_seconds",
            "Computed Unknown-segment backfill range span, in seconds.",
            (60, 300, 900, 1800, 3600, 7200, 14400, 28800),
        ),
        cc_assertions_matched_total=_counter(
            "cts_cc_assertions_matched_total",
            "CC identity assertions matched to a WorldObservation, by spatial gate "
            "(floor|room). Always-on regardless of resolver.cc_assertion_mode.",
            ["gate"],
        ),
        cc_assertions_rejected_total=_counter(
            "cts_cc_assertions_rejected_total",
            "CC identity assertions rejected before matching, by reason "
            "(no_spatial_evidence, room_mismatch, stale, uncalibrated, low_confidence). "
            "Distinguishes 'flowing but not matching' from 'not flowing'.",
            ["reason"],
        ),
        cc_assertions_shadow_total=_counter(
            "cts_cc_assertions_shadow_total",
            "Shadow-mode outcome for a matched CC assertion, by outcome "
            "(would_name_unknown|agrees|disagrees), compared against the PH's "
            "post-resolve committed identity. Only recorded when "
            "resolver.cc_assertion_mode == 'shadow'.",
            ["outcome"],
        ),
        cts_wearable_fixes_total=_counter(
            "cts_wearable_fixes_total",
            "Wearable TagFix messages processed from wearable.fixes, by fix kind and result.",
            ["kind", "result"],
        ),
        cts_wearable_associations_total=_counter(
            "cts_wearable_associations_total",
            "Wearable-to-PH association evaluations, by bounded outcome.",
            ["result"],
        ),
        cts_wearable_assoc_residual_m=_hist(
            "cts_wearable_assoc_residual_m",
            "Trajectory-class wearable-to-PH mean floor residual in metres, by room.",
            (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
            ["room_id"],
        ),
        cts_wearable_face_conflict_total=_counter(
            "cts_wearable_face_conflict_total",
            "Rounds where a qualifying calibrated direct-face anchor disagreed with accepted "
            "wearable evidence on the same PH, forcing Unknown.",
        ),
        cts_wearable_shadow_outcomes_total=_counter(
            "cts_wearable_shadow_outcomes_total",
            "Shadow-mode wearable counterfactual outcomes against the live resolver decision.",
            ["outcome"],
        ),
        cts_wearable_spatial_fusion_total=_counter(
            "cts_wearable_spatial_fusion_total",
            "Trajectory-attested position fixes considered for PH spatial fusion, by fusion "
            "mode (shadow measures, enabled applies) and outcome (applied to a camera-tracked "
            "PH, bridged a camera-occluded PH, gated, or no fresh tight fix).",
            ["mode", "outcome"],
        ),
        cts_position_tracks_total=_counter(
            "cts_position_tracks_total",
            "Anonymous position tracks read from position.tracks, by ingest result.",
            ["result"],
        ),
        cts_device_free_observations_total=_counter(
            "cts_device_free_observations_total",
            "Position-sensor observations in device-free tracker rounds, by what they did: "
            "updated a PH, spawned or revived one, or were kept from spawning outside a "
            "camera-free room.",
            ["outcome"],
        ),
        cts_device_free_rounds_total=_counter(
            "cts_device_free_rounds_total",
            "Device-free tracker rounds driven by position sensors, by outcome.",
            ["outcome"],
        ),
        cts_device_free_shadow_total=_counter(
            "cts_device_free_shadow_total",
            "Shadow-mode position tracks by what a device-free round would do with them.",
            ["outcome"],
        ),
        cts_wearable_spatial_fusion_shift_m=_hist(
            "cts_wearable_spatial_fusion_shift_m",
            "Distance in metres a gated fused fix moved (enabled) or would move (shadow) the PH "
            "position estimate.",
            (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5),
            ["mode"],
        ),
        cts_messages_dead_lettered_total=_counter(
            "cts_messages_dead_lettered_total",
            "Malformed stream messages acknowledged and discarded without silent fallback.",
            ["stream"],
        ),
        cts_wearable_seed_outcomes_total=_counter(
            "cts_wearable_seed_outcomes_total",
            "Wearable-attested ReID gallery seed attempts, by minted state or rejection reason.",
            ["outcome"],
        ),
        cts_image_only_eval_seconds_total=_counter(
            "cts_image_only_eval_seconds_total",
            "Sample-weighted seconds of image-only identity scored against wearable truth, by "
            "arm (control without wearable-taught rows, treatment with them), outcome, and "
            "attestation grade.",
            ["arm", "outcome", "grade"],
        ),
        cts_wearable_face_attestations_total=_counter(
            "cts_wearable_face_attestations_total",
            "Faces of wearable-attested PHs offered to person-identification-service's "
            "attested-exemplar queue, by result.",
            ["result"],
        ),
        cts_gallery_retention_rows_total=_counter(
            "cts_gallery_retention_rows_total",
            "ReID gallery rows counted (shadow) or expired (enabled) by retention, by prior "
            "lifecycle state and retention mode.",
            ["state", "mode"],
        ),
        cts_gallery_retention_crop_delete_failures_total=_counter(
            "cts_gallery_retention_crop_delete_failures_total",
            "Crop deletion failures during ReID gallery retention sweeps (retried on next tick).",
        ),
        cts_closed_world_inactive_total=_counter(
            "cts_closed_world_inactive_total",
            "Closed-world channel evaluations skipped because a readiness gate was not satisfied.",
            ["reason"],
        ),
        cts_closed_world_shadow_outcomes_total=_counter(
            "cts_closed_world_shadow_outcomes_total",
            "Shadow-mode closed-world counterfactual outcomes against the live resolver decision.",
            ["outcome"],
        ),
        cts_household_classifier_health=_gauge(
            "cts_household_classifier_health",
            "Household appearance classifier readiness state (1 for active state, 0 otherwise).",
            ["state"],
        ),
        cts_household_classifier_shadow_outcomes_total=_counter(
            "cts_household_classifier_shadow_outcomes_total",
            "Shadow-mode household appearance classifier outcomes against live decisions.",
            ["outcome"],
        ),
        cts_appearance_training_samples_recorded_total=_counter(
            "cts_appearance_training_samples_recorded_total",
            "Body-appearance training samples recorded for offline classifier training, by source.",
            ["source"],
        ),
        cts_appearance_training_samples_dropped_total=_counter(
            "cts_appearance_training_samples_dropped_total",
            "Candidate body-appearance training samples skipped, by bounded reason.",
            ["reason"],
        ),
    )


# Module-level singleton against the global registry.  Reset in tests via
# ``observability.metrics_module.metrics = build_metrics(registry=...)``.
metrics = build_metrics()


__all__ = [
    "ENTROPY_BUCKETS",
    "LATENCY_BUCKETS_MS",
    "Metrics",
    "build_metrics",
    "metrics",
]
