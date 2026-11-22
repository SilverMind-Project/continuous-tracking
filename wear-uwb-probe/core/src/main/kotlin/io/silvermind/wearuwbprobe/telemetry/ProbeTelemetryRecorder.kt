package io.silvermind.wearuwbprobe.telemetry

import io.silvermind.wearuwbprobe.fsm.BudgetDenialReason
import io.silvermind.wearuwbprobe.fsm.DutyCycleAction
import io.silvermind.wearuwbprobe.fsm.DutyCycleState
import io.silvermind.wearuwbprobe.oob.OobNegotiationResult
import io.silvermind.wearuwbprobe.uwb.ProbeCapabilities
import io.silvermind.wearuwbprobe.uwb.ProbeRangingSample
import io.silvermind.wearuwbprobe.uwb.SessionSummary
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/** Tagged union of all structured JSONL records emitted by the probe APK. */
@Serializable
sealed interface ProbeTelemetryRecord {
    val wallClockUnixMs: Long
    val elapsedRealtimeNanos: Long
}

@Serializable
@SerialName("p1_capability")
data class P1CapabilityRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val capabilities: ProbeCapabilities,
    val screenOffForegroundService: Boolean = false,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p2_oob_trial")
data class P2OobTrialRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val trialIndex: Int,
    val result: OobNegotiationResult,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p2_ranging_sample")
data class P2RangingSampleRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val trialIndex: Int,
    val sample: ProbeRangingSample,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p2_session_summary")
data class P2SessionSummaryRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val trialIndex: Int,
    val summary: SessionSummary,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p3_ranging_orientation")
data class P3RangingOrientationRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val pointId: String,
    val facingDir: String,
    val wristRotationTrial: Boolean = false,
    val sample: ProbeRangingSample,
    val orientation: InterpolatedOrientation,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p4_battery_sample")
data class P4BatterySampleRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val battery: BatteryTelemetrySample,
) : ProbeTelemetryRecord

@Serializable
@SerialName("p4_fsm_transition")
data class P4FsmTransitionRecord(
    override val wallClockUnixMs: Long,
    override val elapsedRealtimeNanos: Long,
    val regime: BatteryRegime,
    val fromState: DutyCycleState,
    val toState: DutyCycleState,
    val action: DutyCycleAction,
    val denialReason: BudgetDenialReason = BudgetDenialReason.NONE,
    val activityLabel: String = "unspecified",
) : ProbeTelemetryRecord

/**
 * Append-only JSONL telemetry writer scoped strictly inside a validated output directory.
 *
 * Output filenames are validated against a strict allow-list regex and verified to remain
 * inside [outputDir] after path normalization to prevent directory traversal.
 */
class ProbeTelemetryRecorder(
    private val outputDir: Path,
) {
    private val normalizedBaseDir: Path = outputDir.toAbsolutePath().normalize()
    private val json: Json =
        Json {
            encodeDefaults = true
            explicitNulls = true
            classDiscriminator = "record_type"
        }

    init {
        Files.createDirectories(normalizedBaseDir)
    }

    /** Resolves and validates [fileName] strictly inside [outputDir]. */
    fun resolveSafeFile(fileName: String): Path {
        require(SAFE_FILENAME_REGEX.matches(fileName)) {
            "Telemetry filename '$fileName' must match ${SAFE_FILENAME_REGEX.pattern}"
        }
        val resolved = normalizedBaseDir.resolve(fileName).normalize()
        require(resolved.parent == normalizedBaseDir && resolved.startsWith(normalizedBaseDir)) {
            "Resolved telemetry path escapes output directory: $resolved"
        }
        return resolved
    }

    fun append(
        fileName: String,
        record: ProbeTelemetryRecord,
    ): Path {
        val target = resolveSafeFile(fileName)
        val line = json.encodeToString<ProbeTelemetryRecord>(record) + "\n"
        synchronized(this) {
            Files.writeString(
                target,
                line,
                StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.APPEND,
            )
        }
        return target
    }

    fun readAll(fileName: String): List<ProbeTelemetryRecord> {
        val target = resolveSafeFile(fileName)
        if (!Files.exists(target)) {
            return emptyList()
        }
        return Files.readAllLines(target, StandardCharsets.UTF_8)
            .filter { it.isNotBlank() }
            .map { json.decodeFromString<ProbeTelemetryRecord>(it) }
    }

    companion object {
        const val P1_FILENAME: String = "p1_capabilities.jsonl"
        const val P2_FILENAME: String = "p2_multicast.jsonl"
        const val P3_FILENAME: String = "p3_accuracy.jsonl"
        const val P4_FILENAME: String = "p4_battery.jsonl"
        private val SAFE_FILENAME_REGEX = Regex("^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}\\.jsonl$")
    }
}
