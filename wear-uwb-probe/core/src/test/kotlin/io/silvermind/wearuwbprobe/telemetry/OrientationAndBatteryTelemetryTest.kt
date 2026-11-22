package io.silvermind.wearuwbprobe.telemetry

import io.silvermind.wearuwbprobe.fsm.DutyCycleState
import io.silvermind.wearuwbprobe.fsm.ThermalSeverity
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class OrientationAndBatteryTelemetryTest {
    private fun yawQuaternion(deg: Double): Quaternion {
        val half = Math.toRadians(deg) / 2.0
        return Quaternion(w = cos(half), x = 0.0, y = 0.0, z = sin(half))
    }

    @Test
    fun `OrientationSynchronizer SLERP-interpolates across antipodal quaternions accurately`() {
        val sync = OrientationSynchronizer()
        val q0 = yawQuaternion(0.0)
        // Negate q90 to verify antipodal shortest-path sign flip.
        val q90Pos = yawQuaternion(90.0)
        val q90Neg = Quaternion(-q90Pos.w, -q90Pos.x, -q90Pos.y, -q90Pos.z)

        sync.recordWorldRotation(1_000_000_000L, q0)
        sync.recordWorldRotation(2_000_000_000L, q90Neg)
        sync.recordGravity(1_000_000_000L, GravityVectorMps2(0.0, 0.0, 9.81))
        sync.recordGravity(2_000_000_000L, GravityVectorMps2(0.0, 9.81, 0.0))

        val mid = sync.interpolateAt(1_500_000_000L)
        assertFalse(mid.extrapolated)
        val qMid = requireNotNull(mid.worldQuaternion)
        // Rotating +X (1, 0, 0) by 45 deg around +Z produces (cos 45, sin 45, 0).
        val (rx, ry, rz) = qMid.rotateVector(1.0, 0.0, 0.0)
        val expected = cos(PI / 4.0)
        assertEquals(expected, rx, 1e-6)
        assertEquals(expected, ry, 1e-6)
        assertEquals(0.0, rz, 1e-6)

        val grav = requireNotNull(mid.gravityMps2)
        assertEquals(4.905, grav.yMps2, 1e-6)
        assertEquals(4.905, grav.zMps2, 1e-6)
    }

    @Test
    fun `BatteryProfiler computes coulomb-counter drain rate and projected runtime`() {
        val profiler = BatteryProfiler(nominalCellCapacityMah = 390.0)
        val hourNanos = 3_600_000_000_000L

        profiler.record(
            BatteryTelemetrySample(
                elapsedRealtimeNanos = 0L,
                wallClockUnixMs = 1_000L,
                regime = BatteryRegime.C_MOTION_DUTY_CYCLED,
                eventTag = "regime_start",
                capacityPercent = 100,
                chargeCounterUah = 390_000L,
                currentNowUa = -12_000L,
                voltageMv = 4200,
                temperatureDeciC = 295,
                thermalStatus = ThermalSeverity.NONE,
                fsmState = DutyCycleState.ON_BODY_STILL,
            ),
        )
        profiler.record(
            BatteryTelemetrySample(
                elapsedRealtimeNanos = 2L * hourNanos,
                wallClockUnixMs = 7_201_000L,
                regime = BatteryRegime.C_MOTION_DUTY_CYCLED,
                eventTag = "regime_end",
                capacityPercent = 94,
                chargeCounterUah = 366_000L,
                currentNowUa = -85_000L,
                voltageMv = 4120,
                temperatureDeciC = 310,
                thermalStatus = ThermalSeverity.LIGHT,
                fsmState = DutyCycleState.ON_BODY_STILL,
                burstsTriggeredTotal = 6,
                fidgetsRejectedTotal = 14,
            ),
        )

        val summary = profiler.summarize(BatteryRegime.C_MOTION_DUTY_CYCLED)
        assertEquals(2.0, summary.durationHours, 1e-6)
        assertEquals(24_000L, summary.deltaUah)
        assertEquals(12_000.0, summary.drainUahPerHour!!, 1e-3)
        assertEquals(3.0, summary.drainPercentPerHour!!, 1e-3)
        assertEquals(85.0, summary.peakCurrentMa!!, 1e-3)
        assertEquals(0, summary.thermalThrottleEvents)
        // 390,000 uAh / 12,000 uAh/h = 32.5 h (>= 30 h Gate P4).
        assertEquals(32.5, summary.projectedRuntimeHoursFromUah!!, 1e-3)
        assertTrue(summary.projectedRuntimeHoursFromUah!! >= 30.0)
    }
}
