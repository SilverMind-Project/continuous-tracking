package io.silvermind.wearuwbprobe.telemetry

import io.silvermind.wearuwbprobe.fsm.DutyCycleState
import io.silvermind.wearuwbprobe.fsm.ThermalSeverity
import kotlinx.serialization.Serializable
import kotlin.math.abs

/** Battery benchmark regime measured in Stage P4. */
@Serializable
enum class BatteryRegime {
    A_BLE_CONTROL,
    B_CONTINUOUS_UWB,
    C_MOTION_DUTY_CYCLED,
}

/** Raw hardware battery and thermal snapshot read via Android BatteryManager/PowerManager. */
data class RawBatterySnapshot(
    val elapsedRealtimeNanos: Long,
    val wallClockUnixMs: Long,
    val capacityPercent: Int,
    val chargeCounterUah: Long?,
    val currentNowUa: Long?,
    val voltageMv: Int?,
    val temperatureDeciC: Int?,
    val thermalStatus: ThermalSeverity,
)

/** Hardware boundary for reading instant battery and thermal telemetry on the watch. */
interface BatterySnapshotReader {
    fun readSnapshot(): RawBatterySnapshot
}

/** One persisted Stage P4 battery/thermal JSONL sample. */
@Serializable
data class BatteryTelemetrySample(
    val elapsedRealtimeNanos: Long,
    val wallClockUnixMs: Long,
    val regime: BatteryRegime,
    val eventTag: String,
    val capacityPercent: Int,
    val chargeCounterUah: Long? = null,
    val currentNowUa: Long? = null,
    val voltageMv: Int? = null,
    val temperatureDeciC: Int? = null,
    val thermalStatus: ThermalSeverity = ThermalSeverity.NONE,
    val fsmState: DutyCycleState = DutyCycleState.OFF_BODY,
    val burstsTriggeredTotal: Int = 0,
    val fidgetsRejectedTotal: Int = 0,
    val activityLabel: String = "unspecified",
)

/** Computed coulomb-counter drain and thermal summary over a Stage P4 regime window. */
@Serializable
data class BatteryDrainSummary(
    val regime: BatteryRegime,
    val sampleCount: Int,
    val durationHours: Double,
    val deltaUah: Long? = null,
    val drainUahPerHour: Double? = null,
    val drainPercentPerHour: Double? = null,
    val peakCurrentMa: Double? = null,
    val maxThermalStatus: ThermalSeverity = ThermalSeverity.NONE,
    val thermalThrottleEvents: Int = 0,
    val projectedRuntimeHoursFromUah: Double? = null,
    val projectedRuntimeHoursFromPercent: Double? = null,
)

/**
 * Records periodic (10 s) and burst-edge coulomb-counter battery/thermal telemetry and
 * computes high-resolution drain rates from `BATTERY_PROPERTY_CHARGE_COUNTER` (uAh).
 */
class BatteryProfiler(
    private val nominalCellCapacityMah: Double = DEFAULT_PIXEL_WATCH_CAPACITY_MAH,
) {
    private val samples: MutableList<BatteryTelemetrySample> = mutableListOf()

    init {
        require(nominalCellCapacityMah > 0.0) { "nominalCellCapacityMah must be > 0" }
    }

    fun record(sample: BatteryTelemetrySample) {
        samples.add(sample)
    }

    fun allSamples(): List<BatteryTelemetrySample> = samples.toList()

    fun clear() {
        samples.clear()
    }

    fun summarize(regime: BatteryRegime): BatteryDrainSummary {
        val filtered = samples.filter { it.regime == regime }.sortedBy { it.elapsedRealtimeNanos }
        if (filtered.size < 2) {
            return BatteryDrainSummary(
                regime = regime,
                sampleCount = filtered.size,
                durationHours = 0.0,
                maxThermalStatus = filtered.firstOrNull()?.thermalStatus ?: ThermalSeverity.NONE,
                thermalThrottleEvents = countThermalThrottles(filtered),
            )
        }

        val first = filtered.first()
        val last = filtered.last()
        val durationHours = (last.elapsedRealtimeNanos - first.elapsedRealtimeNanos).toDouble() / NANOS_PER_HOUR
        val deltaUah = computeDeltaUah(filtered)
        val drainUahPerHour = deltaUah?.takeIf { durationHours > 0.0 }?.let { it.toDouble() / durationHours }
        val deltaPct = (first.capacityPercent - last.capacityPercent).coerceAtLeast(0)
        val drainPercentPerHour = if (durationHours > 0.0) deltaPct.toDouble() / durationHours else null

        return BatteryDrainSummary(
            regime = regime,
            sampleCount = filtered.size,
            durationHours = durationHours,
            deltaUah = deltaUah,
            drainUahPerHour = drainUahPerHour,
            drainPercentPerHour = drainPercentPerHour,
            peakCurrentMa = filtered.mapNotNull { it.currentNowUa }.maxOfOrNull { abs(it).toDouble() / UA_PER_MA },
            maxThermalStatus = filtered.maxBy { it.thermalStatus.level }.thermalStatus,
            thermalThrottleEvents = countThermalThrottles(filtered),
            projectedRuntimeHoursFromUah = projectHours(nominalCellCapacityMah * UA_PER_MA, drainUahPerHour),
            projectedRuntimeHoursFromPercent = projectHours(100.0, drainPercentPerHour),
        )
    }

    private fun computeDeltaUah(filtered: List<BatteryTelemetrySample>): Long? {
        val uahSamples = filtered.mapNotNull { it.chargeCounterUah }
        return if (uahSamples.size >= 2) (uahSamples.first() - uahSamples.last()).coerceAtLeast(0L) else null
    }

    private fun projectHours(
        totalCapacity: Double,
        drainPerHour: Double?,
    ): Double? = if (drainPerHour != null && drainPerHour > 0.0) totalCapacity / drainPerHour else null

    private fun countThermalThrottles(items: List<BatteryTelemetrySample>): Int =
        items.count { it.thermalStatus.level >= ThermalSeverity.MODERATE.level }

    companion object {
        const val SAMPLE_INTERVAL_MS: Long = 10_000L
        const val DEFAULT_PIXEL_WATCH_CAPACITY_MAH: Double = 390.0
        private const val NANOS_PER_HOUR: Double = 3_600_000_000_000.0
        private const val UA_PER_MA: Double = 1_000.0
    }
}
