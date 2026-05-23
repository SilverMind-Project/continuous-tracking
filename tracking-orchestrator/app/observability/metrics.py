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
    tracking_responses_published_total: Counter
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
    height_evidence_frames_total: Counter
    unknown_gts_merged_total: Counter
    identity_decays_total: Counter
    posterior_entropy: Histogram

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

    # ---- Latency -----------------------------------------------------
    frame_end_to_end_latency_ms: Histogram
    triton_inference_latency_ms: Histogram

    # ---- Posture slow-path -------------------------------------------
    cts_posture_slow_path_runs_total: Counter
    cts_posture_slow_path_latency_seconds: Histogram

    # ---- Stage latency -----------------------------------------------
    stage_latency_ms: Histogram

    # ---- Batching ----------------------------------------------------
    batch_size_metric: Histogram


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
        tracking_responses_published_total=_counter(
            "cts_tracking_responses_published_total",
            "FrameResponse messages published to tracking.responses (dead-letter).",
            ["outcome"],
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
