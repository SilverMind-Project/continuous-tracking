# Identity Path Validation, June 2026

## Decision

Decision gate outcome: **(a)**. `PersonTrackingService` identity and location
conclusions are redundant for CTS-enabled deployments.

CTS is the authoritative identity path when `cognitive-companion` has
`cts.enabled=true`. It combines direct face anchors, ReID evidence, temporal
continuity, commit thresholds, and revision events. The CC path performs a
single person-identification-service face match per camera workflow and has no
equivalent posterior or revision model.

`PersonTrackingService` is not removable yet. It remains the identity path for
non-CTS reCamera workflows, writes raw `old_sightings_record` records, correlates Home
Assistant presence, and owns activity methods used by `ActivityService`.

## Concrete Path Map

### CC face path

1. A workflow `person_identification` step calls
   `PersonTrackingService.process_camera_event()`.
2. The service calls person-identification-service, keeps the
   highest-confidence face match per person, and returns
   `PersonDetection` values to pipeline data.
3. When `record_presence=true`, it writes `PersonLocationState` and
   `old_location_history`. When `record_sightings=true`, it writes
   `old_sightings_record`.
4. The HA polling job can infer a person from recent camera sightings and write
   another location conclusion.

### CTS identity path

1. CTS `FaceIdentityStage` converts person-identification-service results into
   face anchors.
2. `WorldTracker` and the Bayesian identity resolver combine face, body ReID,
   and temporal evidence. A commit requires probability, margin, and sensory
   evidence.
3. `tracking.events` carries identity snapshots to CC.
4. CC's `TrackingEventSubscriber` writes the legacy location tables through
   `LocationWriter` and feeds `PersonLocationService` through the CTS runtime's
   world-observation path.
5. `tracking.revisions` retroactively corrects CTS observations and presence
   segments.

## Surface Inventory

| Surface | CC `PersonTrackingService` source | CTS source | Both feed it? |
| --- | --- | --- | --- |
| Workflow `person_identification` outputs (`person_detections`, `room_transitions`) | Direct return from person-identification-service | None | No |
| Effective `GET /api/v1/persons/locations` | The legacy `routers/persons.py` handler reads `PersonLocationState` written by CC | `LocationWriter` also writes the table from `tracking.events` | Yes |
| Effective `GET /api/v1/persons/{id}/location` and legacy `/history` | The legacy handlers read `PersonLocationState` / `old_location_history` | `LocationWriter` and identity revision handling write the same tables | Yes |
| Persons admin location/history tabs | Calls the legacy person routes | Indirectly visible through shared legacy tables | Yes |
| Persons admin sightings tab and MCP `get_old_sightings_table` | Reads `old_sightings_record` | CTS does not write `old_sightings_record` | No |
| New `routers/persons_location.py` implementations for `/persons/{id}/location` and `/persons/locations` | None | Reads `PersonLocationService` presence segments | No, but these duplicate paths are registered after the legacy handlers and are shadowed at runtime |
| BFF `/persons/{id}/presence-history` and `/dwell` | None | Reads `PersonLocationService` | No |
| MCP `get_person_location(s)` | None | Reads `PersonLocationService` | No |
| Rule filters: room, room transition, person presence, presence dwell, presence status, home state, scene trend | None for location identity | Reads `PersonLocationService` | No |
| CTS live view, floor plan, PH list, corrections | None | WebSocket and PH APIs sourced from `tracking.events` and revisions | No |
| Legacy presence providers (`cts_location`, `night_anchor`, `stale_fallback`) | Read shared `PersonLocationState` | `LocationWriter` writes the table | Yes |
| Activity timeline and daily report location entries | Read `old_location_history`; sightings also read `old_sightings_record` | CTS writes location history, not sightings | Yes for location entries |
| Dashboard occupancy | None | CTS occupancy read model / canonical location service | No |

The overlapping surfaces are the risk. `SourceAuthority` gives recent CTS
writes priority, but it does not prove identity agreement and permits a stale
CTS row to be replaced by a lower-priority CC face conclusion.

There is also a routing boundary defect: `backend.main` includes
`persons.router` before `persons_location.router`. FastAPI resolves the first
matching route, so the duplicate location paths currently execute the legacy
handlers even though the newer module documents `PersonLocationService` as the
single source of truth.

## Divergence Check

No live divergence data was collected for this report. The available
development environment does not include two enrolled residents, physical
camera walks, or a running person-identification and Triton stack. Fabricated
results would not satisfy the end-to-end validation requirement.

Run one dev-stack session with residents A and B enrolled in both galleries.
Use synchronized UTC timestamps and perform these known patterns:

| Pattern | Expected observation |
| --- | --- |
| A walks camera 1 to camera 2 while B remains in camera 1 | Cross-camera continuity for A; no identity swap |
| A and B cross in one room | Stable PH assignments through overlap |
| A leaves and re-enters after PH closure | New PH may form, but identity should recommit consistently |
| Partial face or back-facing walk | CC may be absent; CTS may retain or recommit using ReID |
| Deliberate ambiguous face frame followed by clear evidence | CTS may revise; CC has no equivalent revision |

Capture these records for each camera frame or workflow batch:

- CC conclusion: person ID, face confidence, sensor, room, and write time from
  `old_sightings_record` plus `old_location_history` rows with `source='camera'`.
- CTS conclusion: PH ID, committed identity, top and second probabilities,
  direct-face flag, camera, room, event time, and revision ID from
  `cts_tracking_event_identity_decode`, `tracking.events`, and
  `cts_identity_revision_log`.
- Shared-table result: final `PersonLocationState.last_sensor_id`, confidence,
  and update time.

Classify every mismatch as:

1. **Identity mismatch:** both paths commit different resident IDs for the same
   person and time window.
2. **Commit timing mismatch:** IDs agree, but commit times differ. Record
   `cts_commit_time - cc_face_time`.
3. **Coverage mismatch:** only one path produces an identity.
4. **Revision mismatch:** CTS changes a prior identity and the CC conclusion
   remains unrevised.
5. **Authority mismatch:** the final legacy state reflects the lower-priority
   path despite a valid CTS conclusion.

Join records by camera, overlapping bounding box or PH-linked frame, and a
two-second capture-time window. Preserve the raw logs and query output as the
validation artifact. The deprecation decision above does not depend on a zero
mismatch rate: the two algorithms have different evidence and revision
semantics, so dual authority is itself the defect.

## Deprecation Plan

1. Add an explicit CC setting such as
   `person_tracking.identity_location_writes_enabled`. Its default derives from
   deployment mode: false when `cts.enabled=true`, true when CTS is disabled.
   Reject a production configuration that enables both identity/location
   writers unless a temporary comparison flag is also enabled.
2. In CTS-enabled mode, keep `PersonTrackingService.process_camera_event()` for
   workflow detection output and optional `old_sightings_record` capture, but force
   `record_presence=false`. Keep HA presence as a separate provider rather than
   allowing it to infer identity from the CC face path.
3. Remove the duplicate route registration by migrating the legacy person
   location handlers and Persons admin location/history tabs to
   `PersonLocationService`. Also migrate daily report location reads, activity
   timeline location reads, and legacy presence providers.
4. Keep MCP location tools and rule filters on `PersonLocationService`; they
   are already migrated.
5. After the legacy readers are gone, stop CTS `LocationWriter` writes to
   `PersonLocationState` / `old_location_history`, then remove those tables in
   a dedicated migration milestone.
6. Retain `PersonTrackingService` only for non-CTS face workflows, sightings,
   and activity APIs. Split those responsibilities before any final class
   removal.

Suggested removal milestone: the first post-M7 architecture milestone, after
the presence-provider and legacy person-route migrations have shipped and the
comparison session above has been archived.
