package io.silvermind.wearuwbprobe.fsm

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BatteryBudgetLimiterTest {
    private fun minNanos(minutes: Double): Long = (minutes * 60.0 * 1_000_000_000.0).toLong()

    @Test
    fun `token bucket enforces 120s per hour cap and refills smoothly over time`() {
        val limiter = BatteryBudgetLimiter()
        val t0 = 0L

        // Consume all 10 bursts (120 s).
        for (i in 0 until 10) {
            assertTrue(limiter.allowBurst(t0 + i * 1_000L, batteryPercent = 85))
            limiter.consumeBurstSeconds(t0 + i * 1_000L, 12.0)
        }

        // 11th burst immediately after is denied with BUDGET_EXHAUSTED.
        assertFalse(limiter.allowBurst(t0 + 20_000L, batteryPercent = 85))
        assertEquals(
            BudgetDenialReason.BUDGET_EXHAUSTED,
            limiter.evaluateBurst(t0 + 20_000L, batteryPercent = 85),
        )

        // After 6 minutes, 12.0 s of tokens have refilled (120 s / 60 min * 6 min = 12 s).
        val t6Min = minNanos(6.0)
        assertEquals(12.0, limiter.availableBudgetSeconds(t6Min), 0.1)
        assertTrue(limiter.allowBurst(t6Min, batteryPercent = 85))
    }

    @Test
    fun `enforces 20 percent battery floor and moderate thermal shedding`() {
        val limiter = BatteryBudgetLimiter()

        assertEquals(
            BudgetDenialReason.LOW_BATTERY,
            limiter.evaluateBurst(0L, batteryPercent = 19, thermalSeverity = ThermalSeverity.NONE),
        )
        assertEquals(
            BudgetDenialReason.NONE,
            limiter.evaluateBurst(0L, batteryPercent = 20, thermalSeverity = ThermalSeverity.LIGHT),
        )
        assertEquals(
            BudgetDenialReason.THERMAL_THROTTLED,
            limiter.evaluateBurst(0L, batteryPercent = 90, thermalSeverity = ThermalSeverity.MODERATE),
        )
        assertEquals(
            BudgetDenialReason.THERMAL_THROTTLED,
            limiter.evaluateBurst(0L, batteryPercent = 90, thermalSeverity = ThermalSeverity.CRITICAL),
        )
    }
}
