package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BatteryBudgetLimiterTest {
    private fun sec(s: Long): Long = s * 1_000_000_000L

    @Test
    fun `token bucket allows at most ten 12-second bursts per hour and refills linearly`() {
        val limiter = BatteryBudgetLimiter()
        var now = sec(0)
        repeat(10) {
            assertTrue(limiter.allowBurst(now, batteryPercent = 85))
            limiter.consumeBurstSeconds(now, 12.0)
            now += sec(60)
        }
        // Bucket depleted (10 * 12 s consumed in 10 min; < 12 s refilled so far? Actually 10 min refills 20 s!).
        // Test immediate burst depletion at t = 0:
        val tight = BatteryBudgetLimiter()
        repeat(10) {
            assertTrue(tight.allowBurst(sec(0), batteryPercent = 85))
            tight.consumeBurstSeconds(sec(0), 12.0)
        }
        assertFalse(tight.allowBurst(sec(0), batteryPercent = 85))

        // After 6 minutes (0.1 hr), 120 * 0.1 = 12.0 s refilled -> one burst allowed again.
        assertTrue(tight.allowBurst(sec(360), batteryPercent = 85))
        assertEquals(12.0, tight.availableBudgetSeconds(sec(360)), 1e-6)
    }

    @Test
    fun `hard battery cutoff below 20 percent denies bursts`() {
        val limiter = BatteryBudgetLimiter()
        assertFalse(limiter.allowBurst(sec(0), batteryPercent = 19))
        assertTrue(limiter.allowBurst(sec(0), batteryPercent = 20))
    }

    @Test
    fun `thermal cutoff at moderate or higher denies bursts`() {
        val limiter = BatteryBudgetLimiter()
        assertTrue(limiter.allowBurst(sec(0), batteryPercent = 90, thermalSeverity = ThermalSeverity.NONE))
        assertTrue(limiter.allowBurst(sec(0), batteryPercent = 90, thermalSeverity = ThermalSeverity.LIGHT))
        assertFalse(limiter.allowBurst(sec(0), batteryPercent = 90, thermalSeverity = ThermalSeverity.MODERATE))
        assertFalse(limiter.allowBurst(sec(0), batteryPercent = 90, thermalSeverity = ThermalSeverity.SEVERE))
    }
}
