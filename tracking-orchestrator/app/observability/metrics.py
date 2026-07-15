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

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

# Histogram buckets tuned for the latency budgets in phase-1 §1.9b.4
# (frame-end-to-end p99 target = 450 ms).
LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 200, 350, 500, 750, 1000, 2000, 5000)
# Posterior entropy is bits-per-decision; ranges 0..log2(N+1) for N
# residents. Values above 3 bits indicate an overly diffuse posterior.
ENTROPY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


@dataclass
class Metrics:
    """Container for the orchestrator's Prometheus metrics."""

    # ---- Stream consumption / publishing -----------------------------
    frames_consumed_total: Counter
    frames_failed_total: Counter
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

    # ---- Calibration and transit metrics -----------------------------------
    uncalibrated_detection_total: Counter
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

    # ---- World tracker -------------------------------------------------
    world_tracker_ph_open: Gauge
    world_tracker_ph_spawned_total: Counter
    world_tracker_ph_closed_total: Counter

    # ---- PH lifecycle ---------------------------------------------------
    ph_lifetime_seconds: Histogram
    ph_observations_at_close: Histogram
    identity_unknown_after_known_total: Counter

    world_tracker_observations_total: Counter
    world_tracker_assignment_cost: Histogram
    world_tracker_continuations_total: Counter
    world_tracker_clock_skew_ms: Histogram
    world_tracker_spawn_rejected_out_of_room_total: Counter

    # ---- U1 cross-camera dedup -----------------------------------------
    worldtracker_observations_deduped_total: Counter
    worldtracker_dedup_clusters_total: Counter
    worldtracker_observation_missing_floorpoint_total: Counter

    # ---- M03 association integrity (primary pass only) -----------------
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

    # ---- M12 identity-integrity observability ----
    # Bounded labels only; PH/decision/identity IDs go to structlog, never here.
    identity_duplicate_active_blocks_total: Counter  # enforced duplicate / tie-clear blocks
    reid_rejected_vector_vote_attempts_total: Counter  # invariant=0; alert if >0
    identity_prior_only_updates_total: Counter  # temporal-prior maintenance updates
    # The two breach counters below are page-now invariants: each is a runtime
    # assertion whose value must stay zero. Any increment is a defect.
    identity_duplicate_active_breach_total: Counter  # >1 active PH holds one identity post-commit
    identity_prior_only_evidence_advance_total: Counter  # prior-only advanced evidence time

    # ---- M04 governed candidate creation ----
    reid_candidate_rejected_total: Counter  # label: reason (bounded, typed set)
    reid_candidate_created_total: Counter


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
        tracking_events_published_total=_counter(
            "cts_tracking_events_published_total",
            "TrackingEvent messages emitted to tracking.events.",
            ["camera_id"],
        ),
        tracking_revisions_published_total=_counter(
            "cts_tracking_revisions_published_total",
            "IdentityRevision messages emitted to tracking.revisions.",
            ["reason"],
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
        # ---- Calibration and transit metrics ------------------------------------
        uncalibrated_detection_total=_counter(
            "cts_uncalibrated_detection_total",
            "Detections dropped because the source camera lacks a valid homography.",
            ["camera_id"],
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
            "Identity commits or face locks suppressed by the PH quality gate.",
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
        world_tracker_observations_total=_counter(
            "cts_world_tracker_observations_total",
            "Observations consumed by the world tracker.",
            ["camera_id", "result"],
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
        world_tracker_clock_skew_ms=_hist(
            "cts_world_tracker_clock_skew_ms",
            "Per-camera clock skew vs orchestrator wall-clock.",
            (1, 5, 10, 25, 50, 100, 200, 500),
            ["camera_id"],
        ),
        world_tracker_spawn_rejected_out_of_room_total=_counter(
            "cts_world_tracker_spawn_rejected_out_of_room_total",
            "WorldTracker PH spawns rejected because a calibrated observation was outside rooms.",
        ),
        # ---- U1 cross-camera dedup -----------------------------------------
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
        # ---- M03 association integrity (emitted from the primary pass only) ----
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
        # ---- Cross-camera and co-presence ----
        world_tracker_shadow_cross_camera_revival_total=_counter(
            "cts_world_tracker_shadow_cross_camera_revival_total",
            "Shadow cross-camera PH revivals that would fire if enabled.",
        ),
        worldtracker_group_appearance_dedup_total=_counter(
            "cts_worldtracker_group_appearance_dedup_total",
            "Group-appearance dedup clusters formed.",
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
        # ---- M12 identity-integrity observability ----
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
        reid_candidate_rejected_total=_counter(
            "cts_reid_candidate_rejected_total",
            "Detections that failed the governed ReID candidate eligibility gate.",
            ["reason"],
        ),
        reid_candidate_created_total=_counter(
            "cts_reid_candidate_created_total",
            "Governed pending_review ReID gallery candidates created by ReIDCandidateStage.",
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
