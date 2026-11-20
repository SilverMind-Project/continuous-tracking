package io.silvermind.wearcompanion.core

import kotlin.math.min

/** Platform-neutral thermal severity mirroring Android PowerManager.THERMAL_STATUS_*. */
enum class ThermalSeverity(val level: Int) {
    NONE(0),
    LIGHT(1),
    MODERATE(2),
    SEVERE(3),
    CRITICAL(4),
}

/** Configuration for [BatteryBudgetLimiter]. */
data class BatteryBudgetConfig(
    val maxBudgetSeconds: Double = 120.0,
    val refillSecondsPerHour: Double = 120.0,
    val minBurstCostSeconds: Double = 12.0,
    val minBatteryPercent: Int = 20,
    val maxAllowedThermal: ThermalSeverity = ThermalSeverity.LIGHT,
) {
    init {
        require(maxBudgetSeconds > 0.0) { "maxBudgetSeconds must be > 0" }
        require(refillSecondsPerHour > 0.0) { "refillSecondsPerHour must be > 0" }
        require(minBurstCostSeconds > 0.0 && minBurstCostSeconds <= maxBudgetSeconds) {
            "minBurstCostSeconds must be in (0, maxBudgetSeconds]"
        }
        require(minBatteryPercent in 0..100) { "minBatteryPercent must be in 0..100" }
    }
}

/**
 * Token-bucket battery and thermal guard for Tier 2 UWB bursts.
 *
 * Defaults enforce a 3.3% maximum UWB duty cycle (120 s per hour, at most ten
 * 12-second bursts per hour), a hard 20% state-of-charge floor, and thermal
 * shedding at [ThermalSeverity.MODERATE] or higher. When a burst is denied,
 * the watch stays strictly in Tier 1 BLE beacon mode.
 */
class BatteryBudgetLimiter(
    private val config: BatteryBudgetConfig = BatteryBudgetConfig(),
) {
    private var availableSeconds: Double = config.maxBudgetSeconds
    private var lastRefillNanos: Long? = null

    fun availableBudgetSeconds(nowNanos: Long): Double {
        refill(nowNanos)
        return availableSeconds
    }

    fun allowBurst(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): Boolean {
        refill(nowNanos)
        if (batteryPercent < config.minBatteryPercent) {
            return false
        }
        if (thermalSeverity.level > config.maxAllowedThermal.level) {
            return false
        }
        return availableSeconds >= config.minBurstCostSeconds - EPSILON
    }

    fun consumeBurstSeconds(
        nowNanos: Long,
        durationSeconds: Double,
    ) {
        require(durationSeconds >= 0.0) { "durationSeconds must be >= 0" }
        refill(nowNanos)
        availableSeconds = (availableSeconds - durationSeconds).coerceAtLeast(0.0)
    }

    private fun refill(nowNanos: Long) {
        val previous = lastRefillNanos
        if (previous == null) {
            lastRefillNanos = nowNanos
            return
        }
        val elapsedNanos = (nowNanos - previous).coerceAtLeast(0L)
        if (elapsedNanos == 0L) {
            return
        }
        val elapsedHours = elapsedNanos.toDouble() / NANOS_PER_HOUR
        val added = elapsedHours * config.refillSecondsPerHour
        availableSeconds = min(config.maxBudgetSeconds, availableSeconds + added)
        lastRefillNanos = nowNanos
    }

    companion object {
        private const val NANOS_PER_HOUR: Double = 3_600_000_000_000.0
        private const val EPSILON: Double = 1e-6
    }
}
