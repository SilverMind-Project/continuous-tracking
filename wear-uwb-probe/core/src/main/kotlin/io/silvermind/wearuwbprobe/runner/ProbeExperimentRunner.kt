package io.silvermind.wearuwbprobe.runner

import io.silvermind.wearuwbprobe.fsm.BatteryBudgetLimiter
import io.silvermind.wearuwbprobe.fsm.DutyCycleAction
import io.silvermind.wearuwbprobe.fsm.LocomotionDutyCycleFsm
import io.silvermind.wearuwbprobe.oob.OobNegotiationRequest
import io.silvermind.wearuwbprobe.oob.OobNegotiationResult
import io.silvermind.wearuwbprobe.oob.OobSessionNegotiator
import io.silvermind.wearuwbprobe.oob.OobStrategy
import io.silvermind.wearuwbprobe.telemetry.BatteryDrainSummary
import io.silvermind.wearuwbprobe.telemetry.BatteryProfiler
import io.silvermind.wearuwbprobe.telemetry.BatteryRegime
import io.silvermind.wearuwbprobe.telemetry.BatterySnapshotReader
import io.silvermind.wearuwbprobe.telemetry.BatteryTelemetrySample
import io.silvermind.wearuwbprobe.telemetry.OrientationSynchronizer
import io.silvermind.wearuwbprobe.telemetry.P1CapabilityRecord
import io.silvermind.wearuwbprobe.telemetry.P2OobTrialRecord
import io.silvermind.wearuwbprobe.telemetry.P2RangingSampleRecord
import io.silvermind.wearuwbprobe.telemetry.P2SessionSummaryRecord
import io.silvermind.wearuwbprobe.telemetry.P3RangingOrientationRecord
import io.silvermind.wearuwbprobe.telemetry.P4BatterySampleRecord
import io.silvermind.wearuwbprobe.telemetry.P4FsmTransitionRecord
import io.silvermind.wearuwbprobe.telemetry.ProbeTelemetryRecorder
import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import io.silvermind.wearuwbprobe.uwb.MulticastSessionConfig
import io.silvermind.wearuwbprobe.uwb.ProbeCapabilities
import io.silvermind.wearuwbprobe.uwb.SessionSummary
import io.silvermind.wearuwbprobe.uwb.UwbProbeBackend

/** Bench probe stages executed on the watch via UI or ADB broadcast intent. */
enum class ProbeStage {
    P1,
    P2,
    P3,
    P4,
    STOP,
}

/** Validated command parsed from an ADB broadcast intent or UI button click. */
data class ProbeCommand(
    val stage: ProbeStage,
    val backendFilter: String = BACKEND_ALL,
    val oobStrategy: OobStrategy = OobStrategy.PRESHARED_STS_BROADCAST,
    val regime: BatteryRegime = BatteryRegime.C_MOTION_DUTY_CYCLED,
    val durationSeconds: Int = DEFAULT_DURATION_S,
    val trials: Int = DEFAULT_TRIALS,
    val pointId: String = DEFAULT_POINT_ID,
    val facingDir: String = DEFAULT_FACING_DIR,
    val wristRotationTrial: Boolean = false,
    val screenOffTest: Boolean = false,
    val peers: List<AnchorPeerSpec> = DEFAULT_PEERS,
) {
    companion object {
        const val ACTION_RUN_STAGE: String = "io.silvermind.wearuwbprobe.RUN_STAGE"
        const val BACKEND_ALL: String = "all"
        const val DEFAULT_DURATION_S: Int = 60
        const val MAX_DURATION_S: Int = 14_400
        const val DEFAULT_TRIALS: Int = 20
        const val MAX_TRIALS: Int = 100
        const val DEFAULT_POINT_ID: String = "P01"
        const val DEFAULT_FACING_DIR: String = "N"
        private val ALLOWED_BACKENDS = setOf("all", "gms_nearby_uwb", "jetpack_uwb", "platform_ranging")
        private val ALLOWED_FACING = setOf("N", "E", "S", "W")
        private val POINT_ID_REGEX = Regex("^[A-Za-z0-9_-]{1,32}$")
        val DEFAULT_PEERS: List<AnchorPeerSpec> =
            listOf(
                AnchorPeerSpec("anchor-1", "00:01"),
                AnchorPeerSpec("anchor-2", "00:02"),
                AnchorPeerSpec("anchor-3", "00:03"),
                AnchorPeerSpec("anchor-4", "00:04"),
            )

        /**
         * Strictly parses and allow-list validates untrusted string extras from ADB intents.
         * Returns `null` when any field violates its allowed vocabulary or numeric range.
         */
        fun parse(extras: Map<String, String?>): ProbeCommand? {
            val stage = parseStage(extras["stage"])
            val backend = parseBackend(extras["backend"])
            val oob = parseOob(extras["oob"])
            val regime = parseRegime(extras["regime"])
            val durationS = parseBoundedInt(extras["duration_s"], DEFAULT_DURATION_S, 1..MAX_DURATION_S)
            val trials = parseBoundedInt(extras["trials"], DEFAULT_TRIALS, 1..MAX_TRIALS)
            val pointId = parsePointId(extras["point_id"])
            val facing = parseFacing(extras["facing_dir"])
            val peers = parsePeers(extras["anchors"])
            val invalid =
                stage == null || backend == null || oob == null || regime == null ||
                    durationS == null || trials == null || pointId == null || facing == null || peers == null
            if (invalid) {
                return null
            }
            return ProbeCommand(
                stage = stage!!,
                backendFilter = backend!!,
                oobStrategy = oob!!,
                regime = regime!!,
                durationSeconds = durationS!!,
                trials = trials!!,
                pointId = pointId!!,
                facingDir = facing!!,
                wristRotationTrial = extras["wrist_rotation"]?.trim()?.lowercase() == "true",
                screenOffTest = extras["screen_off"]?.trim()?.lowercase() == "true",
                peers = peers!!,
            )
        }

        private fun parseStage(raw: String?): ProbeStage? =
            raw?.trim()?.uppercase()?.let { runCatching { ProbeStage.valueOf(it) }.getOrNull() }

        private fun parseBackend(raw: String?): String? {
            val candidate = raw?.trim()?.lowercase()?.ifEmpty { BACKEND_ALL } ?: BACKEND_ALL
            return candidate.takeIf { it in ALLOWED_BACKENDS }
        }

        private fun parseOob(raw: String?): OobStrategy? =
            when (raw?.trim()?.lowercase()) {
                null, "", "preshared_sts", "sts" -> OobStrategy.PRESHARED_STS_BROADCAST
                "ble_gatt", "gatt" -> OobStrategy.BLE_GATT_SEQUENTIAL
                else -> null
            }

        private fun parseRegime(raw: String?): BatteryRegime? =
            when (raw?.trim()?.uppercase()) {
                null, "", "C" -> BatteryRegime.C_MOTION_DUTY_CYCLED
                "A" -> BatteryRegime.A_BLE_CONTROL
                "B" -> BatteryRegime.B_CONTINUOUS_UWB
                else -> null
            }

        private fun parseBoundedInt(
            raw: String?,
            defaultValue: Int,
            range: IntRange,
        ): Int? {
            val value = if (raw.isNullOrBlank()) defaultValue else raw.trim().toIntOrNull()
            return value?.takeIf { it in range }
        }

        private fun parsePointId(raw: String?): String? {
            val candidate = raw?.trim()?.ifEmpty { DEFAULT_POINT_ID } ?: DEFAULT_POINT_ID
            return candidate.takeIf { POINT_ID_REGEX.matches(it) }
        }

        private fun parseFacing(raw: String?): String? {
            val candidate = raw?.trim()?.uppercase()?.ifEmpty { DEFAULT_FACING_DIR } ?: DEFAULT_FACING_DIR
            return candidate.takeIf { it in ALLOWED_FACING }
        }

        private fun parsePeers(rawAnchors: String?): List<AnchorPeerSpec>? {
            if (rawAnchors.isNullOrBlank()) {
                return DEFAULT_PEERS
            }
            val tokens = rawAnchors.split(",").map { it.trim() }.filter { it.isNotEmpty() }
            if (tokens.isEmpty() || tokens.size > 8) {
                return null
            }
            return runCatching {
                tokens.mapIndexed { idx, addr ->
                    AnchorPeerSpec(anchorId = "anchor-${idx + 1}", uwbAddressHex = addr)
                }
            }.getOrNull()
        }
    }
}

/** Monotonic and wall-clock time source for deterministic JVM testing and watch execution. */
interface ProbeClock {
    fun elapsedRealtimeNanos(): Long

    fun wallClockUnixMs(): Long
}

/** Result of one Stage P2 trial combining OOB negotiation and multicast session summary. */
data class StageP2TrialOutcome(
    val trialIndex: Int,
    val oobResult: OobNegotiationResult,
    val sessionSummary: SessionSummary?,
)

/**
 * Coordinates Stages P1 to P4 across pluggable [UwbProbeBackend]s, [OobSessionNegotiator]s,
 * [OrientationSynchronizer], [LocomotionDutyCycleFsm], [BatteryProfiler], and
 * [ProbeTelemetryRecorder].
 */
class ProbeExperimentRunner(
    private val backends: List<UwbProbeBackend>,
    private val negotiators: Map<OobStrategy, OobSessionNegotiator>,
    private val recorder: ProbeTelemetryRecorder,
    private val orientationSynchronizer: OrientationSynchronizer,
    private val batteryProfiler: BatteryProfiler,
    private val clock: ProbeClock,
    private val sessionKeyInfo: ByteArray = DEFAULT_STATIC_STS_KEY_INFO,
) {
    /** Stage P1: Audit capabilities and controller-session readiness across all backends. */
    suspend fun runStageP1(
        backendFilter: String = ProbeCommand.BACKEND_ALL,
        screenOffForegroundService: Boolean = false,
    ): List<ProbeCapabilities> {
        val selected = selectBackends(backendFilter)
        val results = mutableListOf<ProbeCapabilities>()
        for (backend in selected) {
            val caps = backend.queryCapabilities()
            results.add(caps)
            recorder.append(
                ProbeTelemetryRecorder.P1_FILENAME,
                P1CapabilityRecord(
                    wallClockUnixMs = clock.wallClockUnixMs(),
                    elapsedRealtimeNanos = clock.elapsedRealtimeNanos(),
                    capabilities = caps,
                    screenOffForegroundService = screenOffForegroundService,
                ),
            )
        }
        return results
    }

    /** Stage P2: Run repeated OOB handshakes and multicast DS-TWR sessions against fixed anchors. */
    suspend fun runStageP2(
        backend: UwbProbeBackend,
        strategy: OobStrategy,
        peers: List<AnchorPeerSpec>,
        trials: Int,
        sessionDurationMs: Long,
        baseSessionId: Int = 0x534D0001,
    ): List<StageP2TrialOutcome> {
        val negotiator = requireNotNull(negotiators[strategy]) { "No negotiator registered for $strategy" }
        return (1..trials).map { trialIndex ->
            executeSingleP2Trial(
                backend = backend,
                negotiator = negotiator,
                peers = peers,
                trialIndex = trialIndex,
                sessionId = baseSessionId + trialIndex,
                sessionDurationMs = sessionDurationMs,
            )
        }
    }

    private suspend fun executeSingleP2Trial(
        backend: UwbProbeBackend,
        negotiator: OobSessionNegotiator,
        peers: List<AnchorPeerSpec>,
        trialIndex: Int,
        sessionId: Int,
        sessionDurationMs: Long,
    ): StageP2TrialOutcome {
        val prepared = backend.openControllerSession(sessionId = sessionId)
        val oobReq =
            OobNegotiationRequest(
                sessionId = sessionId,
                controllerAddress = prepared?.localControllerAddress ?: 0,
                channel = prepared?.channel ?: AnchorPeerSpec.DEFAULT_CHANNEL,
                preambleIndex = prepared?.preambleIndex ?: AnchorPeerSpec.DEFAULT_PREAMBLE_INDEX,
                burstDurationS = (sessionDurationMs / 1000L).toInt().coerceIn(1, 60),
                peers = peers,
                sessionKeyInfo = sessionKeyInfo,
            )
        val oobResult = negotiator.negotiate(oobReq)
        recorder.append(
            ProbeTelemetryRecorder.P2_FILENAME,
            P2OobTrialRecord(
                wallClockUnixMs = clock.wallClockUnixMs(),
                elapsedRealtimeNanos = clock.elapsedRealtimeNanos(),
                trialIndex = trialIndex,
                result = oobResult,
            ),
        )
        if (!oobResult.succeeded) {
            return StageP2TrialOutcome(trialIndex, oobResult, null)
        }
        val config =
            MulticastSessionConfig(
                sessionId = sessionId,
                channel = oobReq.channel,
                preambleIndex = oobReq.preambleIndex,
                sessionKeyInfo = sessionKeyInfo,
                peers = peers,
            )
        val summary =
            backend.runMulticastSession(config, sessionDurationMs) { sample ->
                recorder.append(
                    ProbeTelemetryRecorder.P2_FILENAME,
                    P2RangingSampleRecord(
                        wallClockUnixMs = sample.wallClockUnixMs,
                        elapsedRealtimeNanos = sample.elapsedRealtimeNanos,
                        trialIndex = trialIndex,
                        sample = sample,
                    ),
                )
            }
        recorder.append(
            ProbeTelemetryRecorder.P2_FILENAME,
            P2SessionSummaryRecord(
                wallClockUnixMs = clock.wallClockUnixMs(),
                elapsedRealtimeNanos = clock.elapsedRealtimeNanos(),
                trialIndex = trialIndex,
                summary = summary,
            ),
        )
        return StageP2TrialOutcome(trialIndex, oobResult, summary)
    }

    /** Stage P3: Record synchronized UWB ranges + SLERP-interpolated wrist quaternions at a floor point. */
    suspend fun runStageP3(
        backend: UwbProbeBackend,
        strategy: OobStrategy,
        peers: List<AnchorPeerSpec>,
        pointId: String,
        facingDir: String,
        durationMs: Long,
        wristRotationTrial: Boolean = false,
        sessionId: Int = 0x534D0301,
    ): SessionSummary? {
        val negotiator = requireNotNull(negotiators[strategy]) { "No negotiator registered for $strategy" }
        val prepared = backend.openControllerSession(sessionId = sessionId)
        val channel = prepared?.channel ?: AnchorPeerSpec.DEFAULT_CHANNEL
        val preamble = prepared?.preambleIndex ?: AnchorPeerSpec.DEFAULT_PREAMBLE_INDEX
        val oobResult =
            negotiator.negotiate(
                OobNegotiationRequest(
                    sessionId = sessionId,
                    controllerAddress = prepared?.localControllerAddress ?: 0,
                    channel = channel,
                    preambleIndex = preamble,
                    burstDurationS = (durationMs / 1000L).toInt().coerceIn(1, 60),
                    peers = peers,
                    sessionKeyInfo = sessionKeyInfo,
                ),
            )
        if (!oobResult.succeeded) {
            return null
        }

        val config =
            MulticastSessionConfig(
                sessionId = sessionId,
                channel = channel,
                preambleIndex = preamble,
                sessionKeyInfo = sessionKeyInfo,
                peers = peers,
            )
        return backend.runMulticastSession(config, durationMs) { sample ->
            val interpolated = orientationSynchronizer.interpolateAt(sample.elapsedRealtimeNanos)
            recorder.append(
                ProbeTelemetryRecorder.P3_FILENAME,
                P3RangingOrientationRecord(
                    wallClockUnixMs = sample.wallClockUnixMs,
                    elapsedRealtimeNanos = sample.elapsedRealtimeNanos,
                    pointId = pointId,
                    facingDir = facingDir,
                    wristRotationTrial = wristRotationTrial,
                    sample = sample,
                    orientation = interpolated,
                ),
            )
        }
    }

    /** Stage P4 helper: capture and log one battery/thermal snapshot tied to FSM state. */
    fun recordBatteryCheckpoint(
        regime: BatteryRegime,
        eventTag: String,
        reader: BatterySnapshotReader,
        fsm: LocomotionDutyCycleFsm,
        activityLabel: String = "unspecified",
    ): BatteryTelemetrySample {
        val raw = reader.readSnapshot()
        val sample =
            BatteryTelemetrySample(
                elapsedRealtimeNanos = raw.elapsedRealtimeNanos,
                wallClockUnixMs = raw.wallClockUnixMs,
                regime = regime,
                eventTag = eventTag,
                capacityPercent = raw.capacityPercent,
                chargeCounterUah = raw.chargeCounterUah,
                currentNowUa = raw.currentNowUa,
                voltageMv = raw.voltageMv,
                temperatureDeciC = raw.temperatureDeciC,
                thermalStatus = raw.thermalStatus,
                fsmState = fsm.state,
                burstsTriggeredTotal = fsm.counters.burstsStarted,
                fidgetsRejectedTotal = fsm.counters.fidgetsRejected,
                activityLabel = activityLabel,
            )
        batteryProfiler.record(sample)
        recorder.append(
            ProbeTelemetryRecorder.P4_FILENAME,
            P4BatterySampleRecord(
                wallClockUnixMs = raw.wallClockUnixMs,
                elapsedRealtimeNanos = raw.elapsedRealtimeNanos,
                battery = sample,
            ),
        )
        return sample
    }

    /** Stage P4 helper: log one FSM state transition for false-trigger auditing. */
    fun recordFsmTransition(
        regime: BatteryRegime,
        fromState: io.silvermind.wearuwbprobe.fsm.DutyCycleState,
        transition: io.silvermind.wearuwbprobe.fsm.DutyCycleTransition,
        activityLabel: String = "unspecified",
    ) {
        if (fromState == transition.state && transition.action == DutyCycleAction.NONE) {
            return
        }
        recorder.append(
            ProbeTelemetryRecorder.P4_FILENAME,
            P4FsmTransitionRecord(
                wallClockUnixMs = clock.wallClockUnixMs(),
                elapsedRealtimeNanos = clock.elapsedRealtimeNanos(),
                regime = regime,
                fromState = fromState,
                toState = transition.state,
                action = transition.action,
                denialReason = transition.denialReason,
                activityLabel = activityLabel,
            ),
        )
    }

    fun summarizeBatteryRegime(regime: BatteryRegime): BatteryDrainSummary = batteryProfiler.summarize(regime)

    fun createDutyCycleFsm(): LocomotionDutyCycleFsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())

    private fun selectBackends(filter: String): List<UwbProbeBackend> =
        if (filter == ProbeCommand.BACKEND_ALL) {
            backends
        } else {
            backends.filter { it.name.equals(filter, ignoreCase = true) }
        }

    companion object {
        /** Default bench Static STS key (`vendorId="SM"` + 6-byte bench IV; non-production bench fixture). */
        val DEFAULT_STATIC_STS_KEY_INFO: ByteArray =
            byteArrayOf(0x53, 0x4D, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06)
    }
}
