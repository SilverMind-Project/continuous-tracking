---
name: wear-companion
description: Use when building or modifying continuous-tracking/wear-companion/, changing BLE iBeacon advertising, on-body gating, locomotion duty-cycle UWB burst scheduling, Pre-Shared STS negotiation, device_ranging.v1 batch publishing, or Wear OS manifest permissions.
paths: ["**/wear-companion/**"]
---

# wear-companion

Standalone Wear OS companion app (`continuous-tracking/wear-companion/`, Kotlin + Gradle 8.12.1) that lets a resident or caregiver's smartwatch (such as a Pixel Watch 5) serve as a dual-mode identity wearable:

- **Tier 1 (default, always-on BLE beacon):** continuous static iBeacon advertising (`0x004C` Apple/iBeacon manufacturer payload, 2 Hz on-body, 1 Hz screen-off, stopped immediately off-body) picked up by fixed room BLE receivers.
- **Tier 2 (opt-in `dualMode` flavor, UWB burst controller):** motion-duty-cycled 12-second FiRa 2.0 UWB ranging bursts triggered on walking locomotion, capped by a token-bucket battery and thermal guard (120 s/h, 20% SoC cutoff), and published as 3-second JSON micro-batches (`device_ranging.v1`) over LAN MQTT to `rf-ingress`'s `device_ranging_mqtt` adapter.

## Non-negotiable rules

1. **Pure-Kotlin `:core` owns all logic.** All state machines, frame encoding, STS payload framing, battery budget limits, and batch serialization live in `core/src/main/kotlin/io/silvermind/wearcompanion/core/` (`kotlin("jvm")`, zero Android SDK imports) so `make wear-check` runs natively on Linux `aarch64` and `x86_64` CI without requiring an Android SDK.
2. **Tier 1 zero-network manifest invariant.** `app/src/main/AndroidManifest.xml` (`bleOnly` flavor) declares ONLY `BLUETOOTH_ADVERTISE`, `BODY_SENSORS`, `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_CONNECTED_DEVICE`, and `RECEIVE_BOOT_COMPLETED`, with `android:allowBackup="false"` and standalone Wear OS metadata. It must NEVER declare `INTERNET`, `ACCESS_NETWORK_STATE`, `CHANGE_WIFI_STATE`, `UWB_RANGING`, or `WAKE_LOCK`. `core/src/test/kotlin/io/silvermind/wearcompanion/core/ManifestPermissionsTest.kt` enforces this invariant on every `make wear-check` run.
3. **Tier 2 stays opt-in at build time AND runtime.** UWB and LAN MQTT permissions live exclusively in `app/src/dualMode/AndroidManifest.xml`, and `app/src/dualMode/kotlin/io/silvermind/wearcompanion/app/uwb/Tier2Provider.kt` defaults `RuntimeMode` to `BLE_ONLY` until anchors, LAN MQTT broker, and pre-shared STS key are explicitly configured (and gated on physical device probe sign-off).
4. **Never log PII or secrets.** The watch holds only its anonymous beacon UUID/major/minor and optional `tag_id` plus LAN broker config. Never store household member names, floor-plan coordinates, or cloud credentials on the watch, and never log STS keys or MQTT passwords.
5. **Monotonic timestamps and burst clock decoupling.** `EpochBatchPublisher` (`core/src/main/kotlin/io/silvermind/wearcompanion/core/EpochBatchPublisher.kt`) converts monotonic `elapsedRealtimeNanos` into strictly monotonic `measured_at_unix_ns` using a boot-time Unix nanosecond anchor, and stamps top-level `sent_at_unix_ns` on every MQTT flush so `rf-ingress`'s `BurstClockOffsetTracker` decouples 3-second micro-batching delay from clock skew.
6. **Hardware stops at the engine interface.** `UwbRangingEngine` (`core/src/main/kotlin/io/silvermind/wearcompanion/core/UwbRangingEngine.kt`) isolates `:core` from Android UWB APIs. `:app`'s `dualMode` flavor implements `GmsNearbyUwbEngine` (primary) and `JetpackUwbEngine` (fallback), while `:core` tests inject `FakeUwbRangingEngine`.
7. **Wire contract parity.** Every field name and enum value in `DeviceRangingV1Batch` (`core/src/main/kotlin/io/silvermind/wearcompanion/core/EpochBatchPublisher.kt`) and `core/src/main/kotlin/io/silvermind/wearcompanion/core/RuntimeMode.kt` must stay in lockstep with `DeviceRangingV1Payload` in `rf-ingress/rf_ingress/adapters/device_ranging_mqtt.py` and `cts_contracts.wearables` (`unspecified`, `on_body`, `off_body`; `unspecified`, `moving`, `still`; `uwb`, `ble_channel_sounding`, `ble_rssi`).

## Modules and files

| Path | Responsibility |
| --- | --- |
| `settings.gradle.kts` | Always includes `:core`; conditionally includes `:app` when `ANDROID_HOME`/`ANDROID_SDK_ROOT`/`local.properties` `sdk.dir` is present and `-PskipAndroid=true` is not passed |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/BeaconIdentifier.kt` | Static iBeacon UUID/major/minor model, `hardwareKey` formatting, and `FileIdentifierStore` (0600 file persistence, no Android backup) |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/IBeaconFrameEncoder.kt` | Exact 23-byte iBeacon payload encoder (`0x02 0x15`, 16-byte UUID, big-endian major/minor, signed txPower) |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/BeaconStateMachine.kt` | Tier 1 advertising FSM: unpaired, off-body, Bluetooth-off, advertising 2 Hz / 1 Hz |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/LocomotionDutyCycleFsm.kt` | Tier 2 5-state walking-trigger FSM (`OFF_BODY`, `ON_BODY_STILL`, `LOCOMOTION_CANDIDATE`, `UWB_RANGING_BURST` 12 s max, `BURST_COOLDOWN` 60 s same-walk / 10 s stop-walk reset) with fidget filter (`>= 3` steps in 4 s + accel variance `[0.8, 15.0]` m^2/s^4) |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/BatteryBudgetLimiter.kt` | Token-bucket UWB guard (120 s capacity, 120 s/h refill = 3.3% max duty cycle), 20% battery cutoff, and `MODERATE` thermal cutoff |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/PreSharedStsNegotiator.kt` | Deterministic FiRa 2.0 static/provisioned STS parameter derivation and exact 10-byte big-endian BLE GATT wake payload (`version=0x01`, `sessionId:u32`, `controllerAddress:u16`, `channel:u8`, `preambleIndex:u8`, `burstDurationS:u8`) |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/EpochBatchPublisher.kt` | 60-epoch in-memory ring buffer, monotonic `measured_at_unix_ns`, 3 s `device_ranging.v1` JSON micro-batch flush to `rf/devices/{tag_id}/ranges` |
| `core/src/main/kotlin/io/silvermind/wearcompanion/core/CompanionCoordinator.kt` | Composes Tier 1 BLE, Tier 2 locomotion FSM, budget limiter, `UwbRangingEngine`, and `EpochBatchPublisher` |
| `app/src/main/AndroidManifest.xml` | Base `bleOnly` manifest (strictly zero network and zero UWB permissions) |
| `app/src/dualMode/AndroidManifest.xml` | Opt-in `dualMode` permission overlay (`UWB_RANGING`, `WAKE_LOCK`, `INTERNET`, `ACCESS_NETWORK_STATE`, `CHANGE_WIFI_STATE`) |
| `app/src/main/kotlin/io/silvermind/wearcompanion/app/` | `BleBeaconTransmitter`, `OffBodySensorMonitor`, `CompanionForegroundService`, `PairingActivity`, `BeaconStatusTileService`, `BootCompletedReceiver` |
| `app/src/dualMode/kotlin/io/silvermind/wearcompanion/app/uwb/` | `GmsNearbyUwbEngine`, `JetpackUwbEngine`, `MqttBatchTransport`, `Tier2Provider` |

## Quality gates

- Format: `make wear-format` (`./gradlew :core:ktlintFormat --no-daemon`)
- Lint: `make wear-lint` (`./gradlew :core:ktlintCheck :core:detekt --no-daemon`)
- Tests: `make wear-test` (`./gradlew :core:test --no-daemon`, JUnit 5)
- Module gate: `make wear-check` (also wired into `make check-all`, `make ci`, pre-commit, and CI `wear-check` job)
- Building Wear OS APKs on a host with Android SDK 35 (`ANDROID_HOME` set): `./gradlew :app:assembleBleOnlyDebug :app:assembleDualModeDebug --no-daemon`
