package io.silvermind.wearuwbprobe.fsm

import kotlinx.serialization.Serializable

/** Five states of the motion-duty-cycled UWB burst finite state machine. */
@Serializable
enum class DutyCycleState {
    OFF_BODY,
    ON_BODY_STILL,
    LOCOMOTION_CANDIDATE,
    UWB_RANGING_BURST,
    BURST_COOLDOWN,
}

/** Motion classification reported alongside FSM transitions for bench logging. */
@Serializable
enum class MotionStateLabel {
    STILL,
    MOVING,
}

/** Configuration matching the production locomotion gate and wrist-fidget filter. */
data class LocomotionFsmConfig(
    val locomotionConfirmWindowS: Double = 4.0,
    val minConfirmSteps: Int = 3,
    val accelVarianceWindowS: Double = 2.0,
    val minAccelVarianceM2S4: Double = 0.8,
    val maxAccelVarianceM2S4: Double = 15.0,
    val maxBurstDurationS: Double = 12.0,
    val earlyStopMinDurationS: Double = 10.0,
    val earlyStopSolvableEpochs: Int = 10,
    val minSolvableAnchors: Int = 3,
    val sameWalkCooldownS: Double = 60.0,
    val stopWalkResetS: Double = 10.0,
)

/** Side-effect action emitted by [LocomotionDutyCycleFsm] on state transitions. */
@Serializable
enum class DutyCycleAction {
    NONE,
    START_UWB_BURST,
    STOP_UWB_BURST,
    STOP_ALL_RADIOS,
}

/** Diagnostic counters exposed for battery and false-trigger benchmarking. */
@Serializable
data class FsmCounters(
    val burstsStarted: Int = 0,
    val burstsEndedByTimeout: Int = 0,
    val burstsEndedByEarlyStop: Int = 0,
    val fidgetsRejected: Int = 0,
    val budgetDenials: Int = 0,
    val offBodyAborts: Int = 0,
)

/** Result of advancing [LocomotionDutyCycleFsm]. */
data class DutyCycleTransition(
    val state: DutyCycleState,
    val action: DutyCycleAction,
    val motionState: MotionStateLabel,
    val denialReason: BudgetDenialReason = BudgetDenialReason.NONE,
)

/**
 * Pure deterministic 5-state locomotion gate and wrist-fidget filter driven
 * strictly by monotonic timestamps (`nowNanos`).
 *
 * Rejects seated wrist fidgets by requiring both sustained step cadence
 * (`>= 3` steps within 4.0 s) and linear acceleration magnitude variance in
 * `[0.8, 15.0] m^2/s^4` over a 2.0 s window before waking the UWB radio for
 * at most 12.0 s.
 */
class LocomotionDutyCycleFsm(
    private val budgetLimiter: BatteryBudgetLimiter,
    private val config: LocomotionFsmConfig = LocomotionFsmConfig(),
) {
    private data class AccelSample(
        val timestampNanos: Long,
        val magnitudeMps2: Double,
    )

    var state: DutyCycleState = DutyCycleState.OFF_BODY
        private set

    var counters: FsmCounters = FsmCounters()
        private set

    private var candidateEnteredNanos: Long = 0L
    private var burstStartedNanos: Long = 0L
    private var cooldownStartedNanos: Long = 0L
    private var lastStepNanos: Long? = null
    private var consecutiveSolvableEpochs: Int = 0
    private val stepTimestampsNanos: ArrayDeque<Long> = ArrayDeque()
    private val accelSamples: ArrayDeque<AccelSample> = ArrayDeque()

    fun onOffBody(nowNanos: Long): DutyCycleTransition {
        val wasBurst = state == DutyCycleState.UWB_RANGING_BURST
        if (wasBurst) {
            val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos).coerceAtLeast(0.0)
            budgetLimiter.consumeBurstSeconds(nowNanos, elapsedS)
            counters = counters.copy(offBodyAborts = counters.offBodyAborts + 1)
        }
        state = DutyCycleState.OFF_BODY
        clearBuffers()
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.STOP_ALL_RADIOS,
            motionState = MotionStateLabel.STILL,
        )
    }

    fun onOnBody(nowNanos: Long): DutyCycleTransition {
        if (state == DutyCycleState.OFF_BODY) {
            state = DutyCycleState.ON_BODY_STILL
            lastStepNanos = null
            pruneSteps(nowNanos)
            pruneAccelSamples(nowNanos)
        }
        return currentTransition(DutyCycleAction.NONE)
    }

    fun onLinearAcceleration(
        nowNanos: Long,
        magnitudeMps2: Double,
    ) {
        if (state == DutyCycleState.OFF_BODY) {
            return
        }
        accelSamples.addLast(AccelSample(nowNanos, magnitudeMps2))
        pruneAccelSamples(nowNanos)
    }

    fun onStepDetected(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): DutyCycleTransition {
        if (state == DutyCycleState.OFF_BODY) {
            return currentTransition(DutyCycleAction.NONE)
        }
        lastStepNanos = nowNanos
        stepTimestampsNanos.addLast(nowNanos)
        pruneSteps(nowNanos)

        return when (state) {
            DutyCycleState.OFF_BODY -> currentTransition(DutyCycleAction.NONE)
            DutyCycleState.ON_BODY_STILL -> {
                state = DutyCycleState.LOCOMOTION_CANDIDATE
                candidateEnteredNanos = nowNanos
                maybePromoteToBurst(nowNanos, batteryPercent, thermalSeverity)
            }
            DutyCycleState.LOCOMOTION_CANDIDATE ->
                maybePromoteToBurst(nowNanos, batteryPercent, thermalSeverity)
            DutyCycleState.UWB_RANGING_BURST -> tick(nowNanos, batteryPercent, thermalSeverity)
            DutyCycleState.BURST_COOLDOWN -> tick(nowNanos, batteryPercent, thermalSeverity)
        }
    }

    fun onRangingEpochObserved(
        nowNanos: Long,
        anchorCount: Int,
    ): DutyCycleTransition {
        if (state != DutyCycleState.UWB_RANGING_BURST) {
            return currentTransition(DutyCycleAction.NONE)
        }
        if (anchorCount >= config.minSolvableAnchors) {
            consecutiveSolvableEpochs += 1
        } else {
            consecutiveSolvableEpochs = 0
        }
        val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos)
        val earlyStopReached =
            elapsedS >= config.earlyStopMinDurationS &&
                consecutiveSolvableEpochs >= config.earlyStopSolvableEpochs
        if (earlyStopReached) {
            return endBurst(nowNanos, earlyStop = true)
        }
        if (elapsedS >= config.maxBurstDurationS) {
            return endBurst(nowNanos, earlyStop = false)
        }
        return currentTransition(DutyCycleAction.NONE)
    }

    fun tick(
        nowNanos: Long,
        batteryPercent: Int = 100,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): DutyCycleTransition {
        pruneSteps(nowNanos)
        pruneAccelSamples(nowNanos)

        return when (state) {
            DutyCycleState.OFF_BODY -> currentTransition(DutyCycleAction.NONE)
            DutyCycleState.ON_BODY_STILL -> currentTransition(DutyCycleAction.NONE)
            DutyCycleState.LOCOMOTION_CANDIDATE -> {
                val promoted = maybePromoteToBurst(nowNanos, batteryPercent, thermalSeverity)
                if (promoted.state == DutyCycleState.UWB_RANGING_BURST) {
                    promoted
                } else if (nanosToSeconds(nowNanos - candidateEnteredNanos) >= config.locomotionConfirmWindowS) {
                    state = DutyCycleState.ON_BODY_STILL
                    stepTimestampsNanos.clear()
                    counters = counters.copy(fidgetsRejected = counters.fidgetsRejected + 1)
                    currentTransition(DutyCycleAction.NONE)
                } else {
                    promoted
                }
            }
            DutyCycleState.UWB_RANGING_BURST -> {
                val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos)
                if (elapsedS >= config.maxBurstDurationS) {
                    endBurst(nowNanos, earlyStop = false)
                } else {
                    currentTransition(DutyCycleAction.NONE)
                }
            }
            DutyCycleState.BURST_COOLDOWN -> {
                val cooldownElapsedS = nanosToSeconds(nowNanos - cooldownStartedNanos)
                val referenceStep = lastStepNanos ?: cooldownStartedNanos
                val idleSinceStepS = nanosToSeconds(nowNanos - referenceStep)
                if (cooldownElapsedS >= config.sameWalkCooldownS || idleSinceStepS >= config.stopWalkResetS) {
                    state = DutyCycleState.ON_BODY_STILL
                    stepTimestampsNanos.clear()
                }
                currentTransition(DutyCycleAction.NONE)
            }
        }
    }

    private fun maybePromoteToBurst(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity,
    ): DutyCycleTransition {
        pruneSteps(nowNanos)
        pruneAccelSamples(nowNanos)
        val hasEnoughSteps = stepTimestampsNanos.size >= config.minConfirmSteps
        val variance = computeAccelVariance()
        val accelInBand =
            variance != null &&
                variance >= config.minAccelVarianceM2S4 &&
                variance <= config.maxAccelVarianceM2S4
        if (!hasEnoughSteps || !accelInBand) {
            return currentTransition(DutyCycleAction.NONE)
        }
        val denial = budgetLimiter.evaluateBurst(nowNanos, batteryPercent, thermalSeverity)
        if (denial != BudgetDenialReason.NONE) {
            counters = counters.copy(budgetDenials = counters.budgetDenials + 1)
            return currentTransition(DutyCycleAction.NONE, denial)
        }
        state = DutyCycleState.UWB_RANGING_BURST
        burstStartedNanos = nowNanos
        consecutiveSolvableEpochs = 0
        counters = counters.copy(burstsStarted = counters.burstsStarted + 1)
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.START_UWB_BURST,
            motionState = MotionStateLabel.MOVING,
        )
    }

    private fun endBurst(
        nowNanos: Long,
        earlyStop: Boolean,
    ): DutyCycleTransition {
        val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos).coerceIn(0.0, config.maxBurstDurationS)
        budgetLimiter.consumeBurstSeconds(nowNanos, elapsedS)
        state = DutyCycleState.BURST_COOLDOWN
        cooldownStartedNanos = nowNanos
        consecutiveSolvableEpochs = 0
        counters =
            if (earlyStop) {
                counters.copy(burstsEndedByEarlyStop = counters.burstsEndedByEarlyStop + 1)
            } else {
                counters.copy(burstsEndedByTimeout = counters.burstsEndedByTimeout + 1)
            }
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.STOP_UWB_BURST,
            motionState = MotionStateLabel.MOVING,
        )
    }

    private fun computeAccelVariance(): Double? {
        if (accelSamples.size < 2) {
            return null
        }
        val n = accelSamples.size.toDouble()
        var sum = 0.0
        for (sample in accelSamples) {
            sum += sample.magnitudeMps2
        }
        val mean = sum / n
        var sqDiff = 0.0
        for (sample in accelSamples) {
            val d = sample.magnitudeMps2 - mean
            sqDiff += d * d
        }
        return sqDiff / n
    }

    private fun pruneSteps(nowNanos: Long) {
        val cutoff = nowNanos - secondsToNanos(config.locomotionConfirmWindowS)
        while (stepTimestampsNanos.isNotEmpty() && stepTimestampsNanos.first() < cutoff) {
            stepTimestampsNanos.removeFirst()
        }
    }

    private fun pruneAccelSamples(nowNanos: Long) {
        val cutoff = nowNanos - secondsToNanos(config.accelVarianceWindowS)
        while (accelSamples.isNotEmpty() && accelSamples.first().timestampNanos < cutoff) {
            accelSamples.removeFirst()
        }
    }

    private fun clearBuffers() {
        stepTimestampsNanos.clear()
        accelSamples.clear()
        lastStepNanos = null
        consecutiveSolvableEpochs = 0
    }

    private fun currentTransition(
        action: DutyCycleAction,
        denialReason: BudgetDenialReason = BudgetDenialReason.NONE,
    ): DutyCycleTransition {
        val motion =
            when (state) {
                DutyCycleState.OFF_BODY, DutyCycleState.ON_BODY_STILL -> MotionStateLabel.STILL
                DutyCycleState.LOCOMOTION_CANDIDATE,
                DutyCycleState.UWB_RANGING_BURST,
                DutyCycleState.BURST_COOLDOWN,
                -> MotionStateLabel.MOVING
            }
        return DutyCycleTransition(
            state = state,
            action = action,
            motionState = motion,
            denialReason = denialReason,
        )
    }

    companion object {
        private const val NANOS_PER_SECOND: Double = 1_000_000_000.0

        private fun nanosToSeconds(nanos: Long): Double = nanos.toDouble() / NANOS_PER_SECOND

        private fun secondsToNanos(seconds: Double): Long = (seconds * NANOS_PER_SECOND).toLong()
    }
}
