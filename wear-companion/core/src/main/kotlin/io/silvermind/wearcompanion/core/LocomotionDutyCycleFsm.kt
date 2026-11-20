package io.silvermind.wearcompanion.core

/** Five states of the Tier 2 motion-duty-cycled UWB burst state machine. */
enum class DutyCycleState {
    OFF_BODY,
    ON_BODY_STILL,
    LOCOMOTION_CANDIDATE,
    UWB_RANGING_BURST,
    BURST_COOLDOWN,
}

/** Configuration for [LocomotionDutyCycleFsm]. */
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
enum class DutyCycleAction {
    NONE,
    START_UWB_BURST,
    STOP_UWB_BURST,
    STOP_ALL_RADIOS,
}

/** Result of advancing [LocomotionDutyCycleFsm]. */
data class DutyCycleTransition(
    val state: DutyCycleState,
    val action: DutyCycleAction,
    val motionState: MotionStateWire,
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
        }
        state = DutyCycleState.OFF_BODY
        clearBuffers()
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.STOP_ALL_RADIOS,
            motionState = MotionStateWire.STILL,
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
        if (earlyStopReached || elapsedS >= config.maxBurstDurationS) {
            return endBurst(nowNanos)
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
                    currentTransition(DutyCycleAction.NONE)
                } else {
                    currentTransition(DutyCycleAction.NONE)
                }
            }
            DutyCycleState.UWB_RANGING_BURST -> {
                val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos)
                if (elapsedS >= config.maxBurstDurationS) {
                    endBurst(nowNanos)
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
        if (!budgetLimiter.allowBurst(nowNanos, batteryPercent, thermalSeverity)) {
            return currentTransition(DutyCycleAction.NONE)
        }
        state = DutyCycleState.UWB_RANGING_BURST
        burstStartedNanos = nowNanos
        consecutiveSolvableEpochs = 0
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.START_UWB_BURST,
            motionState = MotionStateWire.MOVING,
        )
    }

    private fun endBurst(nowNanos: Long): DutyCycleTransition {
        val elapsedS = nanosToSeconds(nowNanos - burstStartedNanos).coerceIn(0.0, config.maxBurstDurationS)
        budgetLimiter.consumeBurstSeconds(nowNanos, elapsedS)
        state = DutyCycleState.BURST_COOLDOWN
        cooldownStartedNanos = nowNanos
        consecutiveSolvableEpochs = 0
        return DutyCycleTransition(
            state = state,
            action = DutyCycleAction.STOP_UWB_BURST,
            motionState = MotionStateWire.MOVING,
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

    private fun currentTransition(action: DutyCycleAction): DutyCycleTransition {
        val motion =
            when (state) {
                DutyCycleState.OFF_BODY, DutyCycleState.ON_BODY_STILL -> MotionStateWire.STILL
                DutyCycleState.LOCOMOTION_CANDIDATE,
                DutyCycleState.UWB_RANGING_BURST,
                DutyCycleState.BURST_COOLDOWN,
                -> MotionStateWire.MOVING
            }
        return DutyCycleTransition(state = state, action = action, motionState = motion)
    }

    companion object {
        private const val NANOS_PER_SECOND: Double = 1_000_000_000.0

        private fun nanosToSeconds(nanos: Long): Double = nanos.toDouble() / NANOS_PER_SECOND

        private fun secondsToNanos(seconds: Double): Long = (seconds * NANOS_PER_SECOND).toLong()
    }
}
