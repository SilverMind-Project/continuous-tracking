package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LocomotionDutyCycleFsmTest {
    private fun sec(s: Double): Long = (s * 1_000_000_000.0).toLong()

    private fun feedWalkingAccel(
        fsm: LocomotionDutyCycleFsm,
        startS: Double,
    ) {
        // Alternating linear accel magnitudes [0.5, 3.5] -> variance = 2.25 m^2/s^4 (inside [0.8, 15.0]).
        val samples = listOf(0.5, 3.5, 0.5, 3.5, 0.5, 3.5)
        samples.forEachIndexed { idx, mag ->
            fsm.onLinearAcceleration(sec(startS + idx * 0.2), mag)
        }
    }

    @Test
    fun `rejects seated wrist fidgets with low accel variance or fewer than 3 steps`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        assertEquals(DutyCycleState.ON_BODY_STILL, fsm.state)

        // Seated typing: 2 false step events and low accel variance (0.04 < 0.8).
        fsm.onLinearAcceleration(sec(1.0), 0.1)
        fsm.onLinearAcceleration(sec(1.2), 0.5)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, fsm.state)
        fsm.onStepDetected(sec(2.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, fsm.state)

        // After 4.0 s window expires without confirmation -> back to ON_BODY_STILL, never wakes UWB.
        val expired = fsm.tick(sec(5.1), batteryPercent = 80)
        assertEquals(DutyCycleState.ON_BODY_STILL, expired.state)
        assertEquals(DutyCycleAction.NONE, expired.action)

        // Violent single wrist shake (variance > 15.0 m^2/s^4) even with 3 steps is also rejected.
        fsm.onLinearAcceleration(sec(10.0), 0.0)
        fsm.onLinearAcceleration(sec(10.2), 12.0) // variance = 36.0 > 15.0
        fsm.onStepDetected(sec(10.0), batteryPercent = 80)
        fsm.onStepDetected(sec(10.5), batteryPercent = 80)
        val shake = fsm.onStepDetected(sec(11.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, shake.state)
        assertEquals(DutyCycleAction.NONE, shake.action)
    }

    @Test
    fun `walking triggers burst, stops early on 10 epochs over 10s, and honors cooldown reset`() {
        val limiter = BatteryBudgetLimiter()
        val fsm = LocomotionDutyCycleFsm(limiter)
        fsm.onOnBody(sec(0.0))

        feedWalkingAccel(fsm, 1.0)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        fsm.onStepDetected(sec(1.6), batteryPercent = 80)
        val started = fsm.onStepDetected(sec(2.2), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)
        assertEquals(DutyCycleAction.START_UWB_BURST, started.action)

        // 10 consecutive epochs with >= 3 anchors reaching 10.0 s triggers early stop.
        var last = started
        for (i in 1..10) {
            last = fsm.onRangingEpochObserved(sec(2.2 + i * 1.0), anchorCount = 4)
        }
        assertEquals(DutyCycleState.BURST_COOLDOWN, last.state)
        assertEquals(DutyCycleAction.STOP_UWB_BURST, last.action)

        // Continuous walking during cooldown keeps UWB off until 60 s elapse.
        fsm.onStepDetected(sec(20.0), batteryPercent = 80)
        assertEquals(DutyCycleState.BURST_COOLDOWN, fsm.tick(sec(25.0), batteryPercent = 80).state)

        // Stopping for 10 s resets cooldown back to ON_BODY_STILL so a new walk bursts immediately.
        val reset = fsm.tick(sec(30.5), batteryPercent = 80)
        assertEquals(DutyCycleState.ON_BODY_STILL, reset.state)
    }

    @Test
    fun `max 12s duration caps burst and 60s continuous-walk cooldown resets at 60s`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        feedWalkingAccel(fsm, 0.0)
        fsm.onStepDetected(sec(0.5), batteryPercent = 80)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        val started = fsm.onStepDetected(sec(1.5), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)

        // Even with < 3 anchors, burst hard-stops at 12.0 s.
        val capped = fsm.tick(sec(13.6), batteryPercent = 80)
        assertEquals(DutyCycleState.BURST_COOLDOWN, capped.state)
        assertEquals(DutyCycleAction.STOP_UWB_BURST, capped.action)

        // Walking continuously every 5 s stays in cooldown until 60 s from 13.6 s.
        for (t in listOf(18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 63.0, 68.0)) {
            assertEquals(DutyCycleState.BURST_COOLDOWN, fsm.onStepDetected(sec(t), batteryPercent = 80).state)
        }
        assertEquals(DutyCycleState.ON_BODY_STILL, fsm.tick(sec(73.7), batteryPercent = 80).state)
    }

    @Test
    fun `low battery or off-body blocks or tears down burst immediately`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        feedWalkingAccel(fsm, 0.0)
        fsm.onStepDetected(sec(0.5), batteryPercent = 15)
        fsm.onStepDetected(sec(1.0), batteryPercent = 15)
        val denied = fsm.onStepDetected(sec(1.5), batteryPercent = 15)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, denied.state)
        assertEquals(DutyCycleAction.NONE, denied.action)

        // Start a real burst at 80%, then remove watch -> STOP_ALL_RADIOS.
        val started = fsm.onStepDetected(sec(2.0), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)
        val removed = fsm.onOffBody(sec(5.0))
        assertEquals(DutyCycleState.OFF_BODY, removed.state)
        assertEquals(DutyCycleAction.STOP_ALL_RADIOS, removed.action)
    }
}
