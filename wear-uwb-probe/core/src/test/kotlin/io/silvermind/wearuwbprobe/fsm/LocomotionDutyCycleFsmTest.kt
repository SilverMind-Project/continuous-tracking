package io.silvermind.wearuwbprobe.fsm

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class LocomotionDutyCycleFsmTest {
    private fun sec(s: Double): Long = (s * 1_000_000_000.0).toLong()

    private fun feedWalkingAccel(
        fsm: LocomotionDutyCycleFsm,
        startS: Double,
    ) {
        val samples = listOf(0.5, 3.5, 0.5, 3.5, 0.5, 3.5)
        samples.forEachIndexed { idx, mag ->
            fsm.onLinearAcceleration(sec(startS + idx * 0.2), mag)
        }
    }

    @Test
    fun `rejects seated wrist fidgets and tracks rejected fidget counter`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        assertEquals(DutyCycleState.ON_BODY_STILL, fsm.state)

        // Seated typing: 2 false step events and low accel variance (0.04 < 0.8 m^2/s^4).
        fsm.onLinearAcceleration(sec(1.0), 0.1)
        fsm.onLinearAcceleration(sec(1.2), 0.5)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, fsm.state)
        fsm.onStepDetected(sec(2.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, fsm.state)

        // Window expires at 4.0 s -> back to ON_BODY_STILL and fidget counter increments.
        val expired = fsm.tick(sec(5.1), batteryPercent = 80)
        assertEquals(DutyCycleState.ON_BODY_STILL, expired.state)
        assertEquals(DutyCycleAction.NONE, expired.action)
        assertEquals(1, fsm.counters.fidgetsRejected)
        assertEquals(0, fsm.counters.burstsStarted)

        // Violent single wrist shake (variance = 36.0 > 15.0 m^2/s^4) even with 3 steps is rejected.
        fsm.onLinearAcceleration(sec(10.0), 0.0)
        fsm.onLinearAcceleration(sec(10.2), 12.0)
        fsm.onStepDetected(sec(10.0), batteryPercent = 80)
        fsm.onStepDetected(sec(10.5), batteryPercent = 80)
        val shake = fsm.onStepDetected(sec(11.0), batteryPercent = 80)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, shake.state)
        assertEquals(DutyCycleAction.NONE, shake.action)
        assertEquals(0, fsm.counters.burstsStarted)
    }

    @Test
    fun `three steps within 4s transitions to burst and stops early on 10 solvable epochs`() {
        val limiter = BatteryBudgetLimiter()
        val fsm = LocomotionDutyCycleFsm(limiter)
        fsm.onOnBody(sec(0.0))

        feedWalkingAccel(fsm, 1.0)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        fsm.onStepDetected(sec(1.6), batteryPercent = 80)
        val started = fsm.onStepDetected(sec(2.2), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)
        assertEquals(DutyCycleAction.START_UWB_BURST, started.action)
        assertEquals(1, fsm.counters.burstsStarted)

        var last = started
        for (i in 1..10) {
            last = fsm.onRangingEpochObserved(sec(2.2 + i * 1.0), anchorCount = 4)
        }
        assertEquals(DutyCycleState.BURST_COOLDOWN, last.state)
        assertEquals(DutyCycleAction.STOP_UWB_BURST, last.action)
        assertEquals(1, fsm.counters.burstsEndedByEarlyStop)

        // Continuous walking stays in cooldown; stopping 10 s resets to ON_BODY_STILL.
        fsm.onStepDetected(sec(20.0), batteryPercent = 80)
        assertEquals(DutyCycleState.BURST_COOLDOWN, fsm.tick(sec(25.0), batteryPercent = 80).state)
        assertEquals(DutyCycleState.ON_BODY_STILL, fsm.tick(sec(30.5), batteryPercent = 80).state)
    }

    @Test
    fun `hard 12s timeout caps burst and 60s continuous-walk cooldown resets at 60s`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        feedWalkingAccel(fsm, 0.0)
        fsm.onStepDetected(sec(0.5), batteryPercent = 80)
        fsm.onStepDetected(sec(1.0), batteryPercent = 80)
        val started = fsm.onStepDetected(sec(1.5), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)

        val capped = fsm.tick(sec(13.6), batteryPercent = 80)
        assertEquals(DutyCycleState.BURST_COOLDOWN, capped.state)
        assertEquals(DutyCycleAction.STOP_UWB_BURST, capped.action)
        assertEquals(1, fsm.counters.burstsEndedByTimeout)

        for (t in listOf(18.0, 23.0, 28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 63.0, 68.0)) {
            assertEquals(DutyCycleState.BURST_COOLDOWN, fsm.onStepDetected(sec(t), batteryPercent = 80).state)
        }
        assertEquals(DutyCycleState.ON_BODY_STILL, fsm.tick(sec(73.7), batteryPercent = 80).state)
    }

    @Test
    fun `off-body immediately aborts active burst and low battery records denial`() {
        val fsm = LocomotionDutyCycleFsm(BatteryBudgetLimiter())
        fsm.onOnBody(sec(0.0))
        feedWalkingAccel(fsm, 0.0)
        fsm.onStepDetected(sec(0.5), batteryPercent = 15)
        fsm.onStepDetected(sec(1.0), batteryPercent = 15)
        val denied = fsm.onStepDetected(sec(1.5), batteryPercent = 15)
        assertEquals(DutyCycleState.LOCOMOTION_CANDIDATE, denied.state)
        assertEquals(BudgetDenialReason.LOW_BATTERY, denied.denialReason)
        assertEquals(1, fsm.counters.budgetDenials)

        val started = fsm.onStepDetected(sec(2.0), batteryPercent = 80)
        assertEquals(DutyCycleState.UWB_RANGING_BURST, started.state)

        val aborted = fsm.onOffBody(sec(5.0))
        assertEquals(DutyCycleState.OFF_BODY, aborted.state)
        assertEquals(DutyCycleAction.STOP_ALL_RADIOS, aborted.action)
        assertEquals(1, fsm.counters.offBodyAborts)
    }
}
