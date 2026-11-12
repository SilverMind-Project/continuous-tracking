# Face template buffer: a per-round side effect

`WorldTrackingStage.run_many` is the only place the per-PH face template buffer is fed, and every
execution path reaches the tracker through it (`run` delegates to `run_many`):

1. Before `tracker.step`: `FaceTemplateService.pending_anchors(batch_time)` returns template
   results that arrived since the last round; they are appended to the `face_anchors` passed to
   `WorldTracker.step` (identity resolution only) and their evidence to `face_evidence`. They are
   never passed to `_attach_face_anchors`.
2. After `tracker.step`: `after_round(faces, det_to_ph, committed, now)` adds each
   `ctx.face_template_faces` entry to the PH its detection was associated with, resets PHs whose
   identity changed, consumes templates offered to PHs resolved this round, evicts old faces, and
   starts background requests.

`FaceIdentityStage` produces `ctx.face_template_faces` (only while `face_templates.mode` is not
`off`); `WorldTrackingStage` is its only reader. The service is built in
`FrameProcessingPipeline.initialize` next to the face client, stopped in `stop()`, and exposed as
`FrameProcessingPipeline.face_templates` for the PH router and roster-sync resets.

Tests: `tests/pipeline/test_template_execution_paths.py` drives the direct
(`_process_frame`), per-camera (`_handle_batch`), and cross-camera
(`_process_cross_camera_post_detect_batch`) entry points;
`tests/pipeline/test_template_matcher_nonblocking.py` proves a blocked person-id never delays a
round.
