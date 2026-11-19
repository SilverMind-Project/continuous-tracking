# RF Wearable Hardware Decision Record

Date decided: 2026-11-13; amended 2026-11-14 (dual-mode Wear OS companion and standalone UWB probe); finalized 2026-11-19.
Status: Accepted decision record (supersedes the time-boxed bench spike originally planned in `identity-recognition-m11-rf-hardware-feasibility-spike.md`).

## 1. Context and household constraints

The Continuous Tracking System (CTS) tracks individuals across RTSP cameras in a single floor-plane Kalman filter (`PersonHypothesis`) and resolves identity through a Bayesian posterior (`IdentityResolver`). Wide-angle surveillance cameras yield small face crops at range (12 to 21 pixels interocular median on four of five cameras), so the wearable track adds an RF identity sensor as a strong corroborating evidence channel and a governed teacher for body appearance (`reid_gallery`), height profiles, and operator-reviewed face exemplars.

Four constraints govern the hardware choice for this household:

1. **Vulnerable primary wearer:** The primary resident is a senior with early dementia who cannot manage a daily charging routine and may remove unfamiliar wristbands.
   - Her default wearable must require **zero charging** (replaceable or sealed coin cell with 6 months or more of projected operating life at a 0.5 to 1.0 s interval, IP67 waterproof, comfortable silicone or clip/pendant form factor).
   - A person may have multiple active wearables bound simultaneously (wristband, pendant, shoe or garment clip, smartwatch), all resolving to the same `member_id`.
   - Any powered wearable must last at least a full day with margin: at least **36 hours** measured at the operating duty cycle before anyone wears it, at least **72 hours** before a standalone rechargeable tag is offered to the senior, and at least **24 hours with 20% or more remaining** (30 hours or more projected) for a family member's daily-charged smartwatch.
2. **Off-the-shelf procurement:** All parts must be purchasable from Amazon or Seeed Studio without enterprise RTLS contracts, proprietary cloud subscriptions, or closed server software.
3. **Abstractions first, hardware last:**
   - All RF hardware sits behind adapter modules in `continuous-tracking/rf-ingress` that emit one hardware-neutral protobuf contract (`TagFix` on `wearable.fixes` and `AdapterCapability` on `wearable.capabilities`).
   - Neither `tracking-orchestrator` nor `cognitive-companion` imports vendor SDKs or branches on an adapter or radio technology name.
   - Wearables provide identity evidence only: they never create, close, or move a `PersonHypothesis`, and a missing fix is never evidence that a person is absent (removed, parked, charging, or out of coverage).
   - `rf-ingress` emits per-epoch fixes with honest measurement covariance and performs no temporal filtering across epochs; `tracking-orchestrator` is the sole temporal filter.
   - Downstream association and supervision classify fixes strictly by their covariance (`trajectory` class vs. `room` class), never by technology name.
4. **Tiered floor-frame prerequisites:**
   - **Tier 1** (floor-plan image, tape-validated `floor_meters_per_pixel` scale from two orthogonal measurements, and room polygons in `cognitive-companion`) is required before placing RF nodes on the floor plan or running room-level wearable association.
   - **Tier 2** (validated per-camera homography, overlap groups, and a multi-camera acceptance walk) plus capture-time clock alignment are required before running trajectory-level wearable association in a room.

## 2. Options evaluated

| Option | Mechanism | Typical indoor accuracy | Wearable power | Verdict |
| --- | --- | --- | --- | --- |
| **Path A0:** Consumer item trackers (Google Pixel Tag / Find Hub Network) as fixed anchors | Watch ranges to fixed consumer tags over UWB or Bluetooth Channel Sounding | N/A (blocked by protocol auth) | CR2032 coin cell (1 year under occasional finding; days under continuous ranging) | **Rejected.** Cryptographically locked to the owner's Fast Pair Account Key inside Google Play services; unsuitable coin-cell power budget |
| **Path C:** Coin-cell BLE beacons with per-room ESP32 receivers (`ble_rssi`) | Fixed ESP32 nodes running ESPresense publish per-node RSSI over MQTT; `rf-ingress` solves wide-covariance position (3+ nodes) or room fix (1 to 2 nodes) | Room reliable; 1 to 3 m horizontal position | Coin cell (CR3032 / CR2477 / CR2032), 6 months to several years | **Selected as primary first hardware (`IR-M23`)** and **Tier 1 of the Wear OS app (`IR-M14`)** |
| **Path A:** Pixel Watch 5 (Wear OS 7) as UWB Controller against fixed USB FiRa anchors (`uwb`) | Watch ranges via `com.google.android.gms.nearby.uwb` or `androidx.core.uwb` (`CONFIG_MULTICAST_DS_TWR`) to 3+ fixed USB-powered FiRa MAC 1.3 Controlees | 0.10 to 0.30 m horizontal | Daily-charged watch LiPo; continuous 1 Hz UWB drains watch in 4 to 8 h, but **12 s locomotion bursts** keep 24+ h life | **Continuous mode rejected; motion-duty-cycled 12 s burst mode selected as Tier 2 of `IR-M14`**, gated on the standalone **`IR-M25` Wear OS UWB probe** |
| **Path B:** Standalone deep-sleep UWB tag with fixed USB FiRa anchors (`uwb`) | Rechargeable DW3000-class clip/pendant tag with accelerometer deep-sleep firmware ranging to fixed USB FiRa MAC 1.3 anchors | 0.10 to 0.30 m horizontal | 400 to 600 mAh LiPo; 4 h without sleep, 36 to 72+ h with deep-sleep duty cycling | **Selected as optional last precision tier (`IR-M24`)**, gated on a 30-day shared-room necessity gate and a 36/72 h battery bench gate |
| **Path A2:** Bluetooth 6.0 Channel Sounding (`ble_channel_sounding`) | Phase-based ranging (PBR) and round-trip time (RTT) between Bluetooth 6.0 radios (for example Seeed XIAO nRF54L15 or watch) | Nominal 0.5 m; no vendor guarantee yet (> 1 m observed in early reports) | Low power (coin cell or small LiPo) | **Watch list.** Contract supported via `technology = ble_channel_sounding`; watch capability audited in `IR-M25` Stage P1 |
| **Device-free mmWave radar** (24 GHz Hi-Link LD2450 / 60 GHz XIAO ESP32-C6) | Mains-powered radar reports anonymous `(x, y, speed)` of up to 3 moving targets | Decimetres within 6 m and 120 deg; no identity | Mains / USB powered (no wearable) | **Contract reserved only (`AnonymousPositionTrack` on `position.tracks`).** No adapter or consumer in this program |

### 2.1 Why consumer Pixel Tags are excluded (Path A0 analysis)

Desk verification of the Google Find Hub Network (FMDN) accessory specification, the Android UWB developer documentation, and Google Play services (`com.google.android.gms.nearby.uwb`) established three blockers that eliminate consumer Pixel Tags without bench purchasing:

1. **Out-of-Band (OOB) Fast Pair Account Key HMAC lock:** `com.google.android.gms.nearby.uwb.UwbClient` and `androidx.core.uwb.UwbManager` control only the local UWB radio after OOB BLE negotiation has woken the peer and agreed on session ID, channel, preamble index, and Static STS keys. On an FMDN tag, the Beacon Actions BLE GATT opcodes `0x0A` (Ranging Capability Request), `0x0B` (Set Ranging Configuration), and `0x0D` (Stop Ranging) require an 8-byte truncated `HMAC-SHA256` signature keyed by the owner's 16-byte Fast Pair **Account Key**. That key is held privately inside Google Play services for Find Hub and is not exposed to third-party applications.
2. **Address rotation and missing Channel Sounding bond:** FMDN tags rotate their BLE advertising address approximately every 1,024 seconds, and Bluetooth Channel Sounding requires a BLE bond that the FMDN specification provides no command to create for third-party initiators.
3. **Coin-cell anchor depletion:** A Pixel Tag is a unicast responder powered by a 220 mAh CR2032 coin cell designed for brief, occasional lost-item finding sessions. Fixed room anchors must listen and respond continuously; UWB active receive/transmit bursts draw 80 to 180 mA, which exceeds a CR2032's continuous pulse capability and would exhaust fixed coin-cell anchors in days. Fixed UWB anchors must be **USB- or mains-powered programmable FiRa MAC 1.3 devices**.

### 2.2 Why wrist azimuth (`RangingPosition.getAzimuth()`) is logged for observability only

`com.google.android.gms.nearby.uwb.RangingPosition.getAzimuth()` returns Angle of Arrival (AoA) in `[-180, 180]` degrees relative to the **watch's local antenna boresight (`device_local`)**, not the household floor plan. Because a wearer's wrist pronates, supinates, and swings continuously while walking, projecting a watch-local azimuth into room coordinates requires an accurate 3D world-frame yaw of the watch face, and indoor magnetometers suffer 15 to 45 degrees of heading distortion near appliances and structural steel.

By contrast, **distance-only multilateration** (`getDistance()`) from 3 or more fixed anchors at known `(x_m, y_m, z_m)` positions solves `(x, y, z)` directly in the shared floor frame and is invariant to wrist rotation. `RangeMeasurement` records optional `azimuth_deg`, `elevation_deg`, and `azimuth_reference_frame` for diagnostic analysis (`IR-M25` Stage P3), and `MultilaterationSolver` ignores `device_local` azimuth whenever 3 or more anchors report.

### 2.3 Why motion-duty-cycled 12-second UWB bursts solve the smartwatch battery constraint

Continuous 1 Hz UWB ranging plus active Wi-Fi MQTT publishing draws 60 to 120 mW on a smartwatch, draining a 350 to 400 mAh watch battery in 4 to 8 hours. However, `tracking-orchestrator` does not need continuous decimetre fixes while a person sits or stands still:

- While worn (`ON_BODY`), the Pixel Watch 5 continuously runs **Tier 1 BLE iBeacon advertising** (< 5% daily battery drain, zero Wi-Fi traffic), heard by the `IR-M23` ESP32 room receivers.
- **Tier 2 UWB burst mode** wakes the watch's UWB radio only when a two-stage locomotion filter (`LocomotionDutyCycleFsm`: 3 or more hardware step-detector events within 4.0 s plus linear acceleration magnitude variance in `[0.8, 15.0] m^2/s^4` over 2.0 s to reject > 98% of seated wrist fidgets) and a token-bucket battery guard (`BatteryBudgetLimiter`: 120 s/hour maximum UWB radio budget, <= 3.3% duty cycle, hard cutoff below 20% battery or at `THERMAL_STATUS_MODERATE`) confirm active walking.
- Upon entering a burst, `PreSharedStsNegotiator` broadcasts the watch's HAL-assigned `(sessionId, controllerAddress, channel, preambleIndex)` in a 300 ms non-connectable BLE advertisement (`ADVERTISE_MODE_LOW_LATENCY`). All USB-powered FiRa anchors in the room hear the broadcast within 100 ms and configure their Controlee sessions using a pre-shared 8-byte Static STS key with **zero BLE GATT connections**.
- One **12-second UWB burst** at 1 to 2 Hz (flushed in 3-second micro-batches via `EpochBatchPublisher` using schema `device_ranging.v1`, with top-level `sent_at_unix_ns` decoupled from per-epoch `measured_at_unix_ns` by `BurstClockOffsetTracker`) delivers 10 to 12 trajectory fixes (`sigma ~ 0.15 m`). That satisfies `WearableAssociator`'s `trajectory_attest_min_duration_s = 10.0 s` (`>= 8` trajectory fixes), latching the identity on the `PersonHypothesis` and minting governed gallery rows while the UWB radio sleeps for a 60-second cooldown and Tier 1 BLE room presence sustains the evidence clock.

## 3. Decision summary (`R17` to `R30`)

| Decision ID | Decision |
| --- | --- |
| **R17** | Consumer item trackers (Pixel Tag and other Find Hub Network tags) cannot be ranged by third-party apps and are not used as anchors or tags. |
| **R18** | First wearable hardware is a never-charged coin-cell BLE beacon wristband/pendant heard by per-room USB-powered ESP32 receivers (`IR-M23`). UWB is an optional precision tier built last (`IR-M24`). |
| **R19** | The senior and willing household members wear tags. A person may have multiple active devices bound simultaneously (wristband, pendant, clip, watch); all resolve to that member. |
| **R20** | A powered wearable must last at least a full day (36 h measured before anyone wears it; 72 h before a standalone tag is offered to the senior; >= 24 h with >= 20% margin / >= 30 h projected for a smartwatch). The senior's primary tag is never charged. |
| **R21** | Device-free mmWave radar: reserve the protobuf contract only (`AnonymousPositionTrack` on `position.tracks` in `IR-M12`); no adapter or consumer in this program. |
| **R22 / R27** | `wear-companion` on the Pixel Watch 5 (`IR-M14`) is a dual-mode app: Tier 1 continuous on-body BLE iBeacon (`BLE_ONLY` default) plus Tier 2 motion-duty-cycled 12 s UWB burst controller (`BLE_PLUS_UWB_BURST`) against USB-powered FiRa MAC 1.3 anchors (`IR-M24`), gated on the standalone `IR-M25` Wear OS UWB probe. |
| **R23** | Abstractions first, hardware last. `IR-M12`, `IR-M13` (core, simulator, adapter conformance kit), `IR-M15`, `IR-M16`, and `IR-M17` are built and accepted in CI on the simulator without physical radios. Hardware milestones (`IR-M23`, `IR-M25`, `IR-M14`, `IR-M24`) may add adapters, settings, and firmware, and may not alter contracts, solver interfaces, or `tracking-orchestrator/app/`. |
| **R24** | Accuracy is expressed solely by covariance on `TagFix`. Consumers classify fixes into `trajectory` or `room` evidence from covariance and never branch on `adapter_kind` or `technology`. |
| **R25** | A tag that is not moving (`MOTION_STATE_STILL`) is parked and yields no evidence; a tag already stationary in a room yields no evidence for a person who walks in until the tag moves. |
| **R26** | Attestation is graded (`trajectory` vs. `room`). `trajectory` attestation mints `auto_verified` body gallery rows (`seed_source = wearable_attested`, trust multiplier `1.25`) and queues operator-gated face exemplars; `room` attestation (sole room occupant, arrived with a moving tag) mints body gallery rows only as `pending_review` (`seed_source = wearable_room_attested`) and never backfills or supplies height samples. |
| **R28** | Wearables are differentiated by wearer profile: the senior always keeps a zero-charging coin-cell BLE wristband/pendant (`IR-M23`) as her primary tag (optionally paired with a secondary bound UWB clip/pendant or watch under `R19`, never a single point of charging failure). Family members use the dual-mode Pixel Watch 5 (`IR-M14`) or a BLE wristband. |
| **R29** | 12-second locomotion-gated UWB bursts (`IR-M14`) flushed in 3-second micro-batches (`BurstClockOffsetTracker` in `IR-M13`) suffice for `trajectory` attestation (`trajectory_attest_min_duration_s = 10.0 s`, `>= 8` fixes in `IR-M16`), after which minted gallery rows and Tier 1 BLE room presence sustain identity while the UWB radio sleeps. |
| **R30** | `RangeMeasurement` (`IR-M13`) carries optional `azimuth_deg`, `elevation_deg`, and `azimuth_reference_frame` for observability, but `MultilaterationSolver` ignores `device_local` wrist azimuth and solves `(x, y, z)` from distances alone whenever 3 or more anchors report. |

## 4. Disposition of original spike experiments (`E0` to `E5`)

Rather than blocking software abstractions on a 10-day upfront hardware spike, each experiment from the original `IR-M11` protocol was either closed analytically or assigned to a dedicated hardware milestone with explicit pass/fail gates:

| Original Experiment | Original Purpose | Disposition and Owning Milestone | Pass / Acceptance Gate |
| --- | --- | --- | --- |
| **E0** (Path A0) | Test third-party GATT/UWB ranging against a consumer Pixel Tag | **Closed analytically (`R17`).** Not run; zero Pixel Tags purchased | FMDN Beacon Actions `0x0A`/`0x0B` require private Fast Pair Account Key `HMAC-SHA256`; 1,024 s MAC rotation; CR2032 power limit |
| **E1** (Path A / A2) | Verify Wear OS 7 third-party UWB and `android.ranging.RangingManager` API availability on Pixel Watch 5 | **Moved to `IR-M25` Stage P1** (`continuous-tracking/wear-uwb-probe/`, watch-only, 30 min upon watch arrival) | At least one of `GmsNearbyUwbBackend` (`com.google.android.gms.nearby.uwb.UwbClient`) or `JetpackUwbBackend` (`androidx.core.uwb.UwbManager`) opens a controller session in a screen-off `connectedDevice` foreground service |
| **E2** (Path A) | Verify 1-to-4 multicast DS-TWR session stability and OOB setup latency from watch to FiRa anchors | **Moved to `IR-M25` Stage P2** (`PreSharedStsNegotiator` 300 ms BLE broadcast wake vs. `BleGattOobNegotiator`) | Time-to-first-3-anchor-epoch <= 2.0 s (target <= 0.5 s with broadcast wake); sustained >= 1.0 Hz across >= 3 anchors for >= 90% of epochs over 5 min |
| **E3** (Path A / B) | Measure wearable battery drain and thermal behavior under operating duty cycle | **Watch BLE:** `IR-M14` Section 6 day-one probe.<br>**Watch UWB bursts:** `IR-M25` Stage P4 (`BatteryProfiler` coulomb counter `BATTERY_PROPERTY_CHARGE_COUNTER`).<br>**Standalone BLE tag:** `IR-M23` Task 3 step 5.<br>**Standalone UWB tag:** `IR-M24` Part A (U2 stock vs. U3 deep-sleep profiler) | **Watch BLE:** <= 10%/24 h added drain.<br>**Watch UWB burst mode:** >= 30 h projected runtime (or <= 15%/24 h added drain over BLE), zero thermal throttling, <= 1 seated false trigger/h.<br>**Senior BLE beacon:** >= 6 months projected coin-cell life.<br>**Standalone UWB tag:** >= 36 h measured (general), >= 72 h (senior) |
| **E4** (All paths) | Measure floor-frame horizontal accuracy, body shadowing, 3D vs. 2D height-prior solve, and room classification accuracy | **BLE room & RSSI path-loss:** `IR-M23` Task 3 (bench) and Task 5 (8-room walk test).<br>**Watch UWB trilateration & wrist azimuth coupling:** `IR-M25` Stage P3 (12 points, 4 facing directions, 90 deg wrist rotation).<br>**Standalone UWB tag:** `IR-M24` Part A (U1) | **BLE:** >= 90% in-room fixes assigned to the true room across all 8 rooms.<br>**UWB (Watch or Tag):** distance-only median horizontal error <= 0.30 m, p95 <= 1.0 m, fix availability >= 80% across 3+ anchors |
| **E5** (All paths) | Verify on-body / off-body detection and parked-tag motion gating | **BLE beacons:** `IR-M23` Task 2 (`ble_motion.py`), Task 3 step 4, and Task 6 step 3 (24 h bedroom dresser parked-tag drill).<br>**Watch:** `IR-M14` Section 6 & `IR-M25` Stage P4 (`Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT` -> immediate `OFF_BODY` stop) | **BLE parked-tag drill:** zero accepted associations after parked timeout.<br>**Watch off-body:** advertising and UWB stop within 10 s of watch removal |

## 5. Architectural consequences for Track C milestones

1. **Unified RF node and device vocabulary (`IR-M12`):**
   - Fixed radios are stored in `cognitive-companion` table `rf_nodes` (`RfNodeService`) with `node_role` (`receiver` or `ranging_anchor`), `technology` (`ble_rssi`, `ble_channel_sounding`, `uwb`), floor-plan coordinates `(x_norm, y_norm, z_m)`, nullable `room_id`, and nullable FiRa UWB session fields (`uwb_address_hex` matching `^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$`, `uwb_channel` in `{5, 9}`, `uwb_preamble_index` in `9..32`).
   - Worn radios are stored in `wearable_devices` with `device_kind` (`ble_beacon`, `watch`, `uwb_tag`), `adapter_id`, and `hardware_key` (`UNIQUE (adapter_id, hardware_key)`), mapped at ingestion to the hardware-neutral `tag_id`.
   - A dual-mode Pixel Watch 5 (`IR-M14`) is registered as two logical device rows bound to the same `member_id`: one for `ble_scanner_mqtt` (`hardware_key = "<uuid>:<major>:<minor>"`) and one for `device_ranging_mqtt` (`hardware_key = "<watch-uwb-tag-id>"`).
   - `TagFix` (`proto/continuoustracking/v1/wearable.proto`) carries `motion_state` (`MotionState`: `UNSPECIFIED`, `MOVING`, `STILL`) and optional `battery_fraction` (`has_battery`, `battery_fraction` in `[0.0, 1.0]`).
2. **Hardware-free `rf-ingress` core and adapter build order (`IR-M13`, `IR-M23`, `IR-M14`, `IR-M24`):**
   - `IR-M13` builds **only** the hardware-free core (`EpochBuilder` with generic `< 3` node room fallback, `MultilaterationSolver` weighting strictly by `variance_m2` and ignoring `device_local` azimuth, `BurstClockOffsetTracker`, `SiteConfigClient`, `RedisTagFixPublisher`), the deterministic `simulator` adapter (modeling `uwb`, `ble_channel_sounding`, `ble_rssi`, and `two_people_uwb_burst.yaml`), and the parameterized adapter conformance test suite (`rf-ingress/tests/conformance/`).
   - `IR-M23` builds the first production hardware adapter: `ble_scanner_mqtt` (payload profile `espresense_node_v1`) and `ble_motion.py`, plus the LAN Eclipse Mosquitto broker in `docker-compose.yml`.
   - `IR-M14` Tier 2 / `IR-M24` Part B builds the UWB MQTT adapters (`device_ranging_mqtt` with profile `device_ranging_v1`, and `anchor_ranging_mqtt` if standalone anchor-reported tags are deployed), passing `rf-ingress/tests/conformance/` without any changes to `tracking-orchestrator`.
3. **Shared USB-powered FiRa UWB anchor infrastructure (`IR-M24`, `IR-M14` Tier 2):**
   - Fixed UWB room anchors in `IR-M24` must be FiRa MAC 1.3 DS-TWR compliant (such as Qorvo DWM3001CDK, NXP SR150 modules, or DW3000 + ESP32-S3 running FiRa Static STS responder firmware) with multi-controller arbitration (`AnchorSessionArbitrator`: concurrent UCI Controlee sessions or 1,000 ms round-robin time-slicing across active BLE wake requests).
   - This allows the exact same 4 USB-powered anchors in a gated room to serve both **Endpoint A** (Pixel Watch 5 in 12-second UWB burst mode) and **Endpoint B** (standalone deep-sleep UWB clip/pendant tags for the senior).

## 6. Household RF node placement and survey guidance

### 6.1 Floor-frame prerequisites

- **Before placing any RF node (`IR-S04` Tier 1):** Upload the household floor plan in `cognitive-companion`, validate `floor_meters_per_pixel` against two orthogonal tape measurements, and draw closed room polygons for all rooms. The Wearables admin page blocks node placement when the floor plan or scale is missing.
- **Before trajectory-class UWB evaluation or association (`IR-S04` Tier 2):** Calibrate a floor homography with residual checks and overlap groups for every camera in the UWB-covered rooms, and verify cross-camera floor continuity with the acceptance walk.

### 6.2 Tier 1: BLE receiver placement (`node_role = receiver`, `technology = ble_rssi`, `IR-M23`)

- **Count per room (8-room household starter layout: 9 active nodes + 1 spare = 10 ESP32 boards):**
  - Place **1 receiver node** in each enclosed room (bedrooms, study/office, hallway/corridor, entryway, bathroom exterior/threshold).
  - Place **2 receiver nodes** at opposite ends of the large open-plan living room and kitchen space so RSSI margins separate the two zones cleanly along the room polygon boundary.
- **Mounting height and environment:**
  - Power from continuous USB mains adapters (never batteries).
  - Mount on interior walls at **`z_m = 1.0 to 1.5 m`** above the floor (approximate wrist/torso height).
  - Keep at least **0.5 m clear** of large metal appliances (refrigerator, microwave), mirrors, television panels, and the floor or ceiling slab corner.
  - Bind each receiver node in `cognitive-companion` to the `room_id` it serves so 1- or 2-node epochs fall back to the strongest receiver's room accurately.

### 6.3 Tier 2: UWB FiRa ranging anchor placement (`node_role = ranging_anchor`, `technology = uwb`, `IR-M24` / `IR-M14` Tier 2)

- **Which rooms:** Install UWB anchors only in rooms that meet the `IR-M24` Section 3 necessity gate (`room_only_shared_room >= 25%` of worn in-coverage time for the senior, typically shared living/dining/kitchen spaces) and have `IR-S04` Tier 2 calibrated cameras.
- **Count and 3D geometry per room:**
  - Install **4 anchors per gated room** near the room corners with unobstructed line of sight to the walking floor area (minimum 3 anchors for a 2D solve with wrist-height prior `assumed_tag_height_m = 1.0 +/- 0.3 m`; 4 anchors provide redundancy for single-anchor NLOS outlier rejection).
  - **Stagger mounting heights (`z_m`) between `1.2 m` and `2.3 m`** (for example, two diagonal corners at `2.2 m` and the other two at `1.3 m`). Coplanar anchors at a single height yield poor vertical dilution of precision (VDOP); staggered heights allow `IR-M25` Stage P3 and `IR-M24` Part A to compare full 3D multilateration against 2D multilateration with the wrist-height prior.
- **Power and FiRa addressing:**
  - Power every anchor from USB mains power.
  - Assign deterministic 2-byte FiRa short addresses (`uwb_address_hex` = `00:01` through `00:08`), `uwb_channel = 9` (7987.2 MHz), and `uwb_preamble_index = 10` in `rf_nodes`.

### 6.4 Tape-distance node survey rule

Clicking a point on a floor-plan image by eye can easily introduce a 0.3 to 0.5 m registration offset, which would cause trajectory-class UWB fixes (`sigma ~ 0.15 m`) to disagree systematically with camera homography trajectories. For every installed RF node:

1. Measure its height above the floor (`z_m`) and its horizontal perpendicular distances to two reference walls (or to an adjacent surveyed node) with a tape measure.
2. Enter the tape distances in the `cognitive-companion` RF node placement form (`IR-M12` Task 4).
3. Verify that the floor-plan coordinates agree with the tape measurements within **`0.20 m`** (`placement_residual_m <= 0.20 m`). Re-check the floor-plan scale or node position if the discrepancy exceeds `0.20 m`.

## 7. Review triggers

Re-open or advance this hardware decision when any of the following triggers fires:

1. **Physical Pixel Watch 5 arrival (expected November 2026):** Run `IR-M25` Stage P1 (`continuous-tracking/wear-uwb-probe/`, 30-minute watch-only capability audit of `com.google.android.gms.nearby.uwb.UwbClient`, `androidx.core.uwb.UwbManager`, and Android 16 `android.ranging.RangingManager`). If Gate P1 passes, run Stages P2 to P4 against 4 bench FiRa anchor boards and record the go/no-go decision for `IR-M14` Tier 2.
2. **Each major Wear OS or Google Play services (`com.google.android.gms.nearby`) update:** Re-run the `IR-M14` Section 6 BLE background advertising probe and `IR-M25` Stage P1 UWB capability audit to verify that background BLE advertising, off-body detection, and third-party UWB controller sessions remain permitted.
3. **30 days after `IR-M23` BLE rollout is enabled:** Evaluate the `IR-M24` Section 3 necessity gate (`room_only_shared_room` share of the senior's worn in-coverage time). Start `IR-M24` Part A only if that share is 25% or more in `IR-S04` Tier 2 calibrated rooms (or on explicit owner override after `IR-M25`).
4. **Market or platform shift:** Re-evaluate if a waterproof turnkey UWB or Bluetooth 6.0 Channel Sounding wristband with 14+ days of verified battery life becomes available on Amazon or Seeed Studio, or if Google opens a documented, local, non-rate-limited third-party ranging API for Find Hub Network accessories paired with mains-powered anchor hardware.
