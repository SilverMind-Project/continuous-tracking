package io.silvermind.wearcompanion.core

/** Hardware-neutral BLE advertising sink driven by [CompanionCoordinator]. */
interface BleAdvertiserSink {
    fun startIBeacon(payload: ByteArray)

    fun stopAdvertising()

    /** Optional 300 ms low-latency wake broadcast used only in Tier 2 UWB burst mode. */
    fun broadcastWakeFrame(
        payload: ByteArray,
        durationMs: Long,
    ) {}
}

/** Snapshot of coordinator state for the Wear OS UI and status tile. */
data class CoordinatorSnapshot(
    val mode: RuntimeMode,
    val beaconState: BeaconState,
    val tileStatus: TileStatus,
    val dutyCycleState: DutyCycleState,
    val hardwareKey: String?,
)

/**
 * Pure dual-mode coordinator uniting Tier 1 BLE beacon advertising (default)
 * and Tier 2 motion-duty-cycled UWB burst ranging (opt-in when anchors exist).
 */
class CompanionCoordinator(
    private val identifierStore: IdentifierStore,
    private val bleSink: BleAdvertiserSink,
    private val mode: RuntimeMode = RuntimeMode.BLE_ONLY,
    private val budgetLimiter: BatteryBudgetLimiter = BatteryBudgetLimiter(),
    private val fsmConfig: LocomotionFsmConfig = LocomotionFsmConfig(),
    private val uwbAnchors: List<UwbAnchorEntry> = emptyList(),
    private val uwbEngine: UwbRangingEngine? = null,
    private val batchPublisher: EpochBatchPublisher? = null,
) {
    private val beaconFsm = BeaconStateMachine()
    private val dutyFsm = LocomotionDutyCycleFsm(budgetLimiter = budgetLimiter, config = fsmConfig)

    private var onBody: Boolean = false
    private var bluetoothEnabled: Boolean = true
    private var screenOn: Boolean = true
    private var activeSessionCounter: Int = 1

    fun pair(): BeaconIdentifier {
        val id = identifierStore.loadOrCreate()
        reconcileBle()
        return id
    }

    fun resetIdentifier(): BeaconIdentifier {
        val id = identifierStore.reset()
        reconcileBle()
        return id
    }

    fun updateBluetooth(enabled: Boolean): CoordinatorSnapshot {
        bluetoothEnabled = enabled
        reconcileBle()
        return snapshot()
    }

    fun updateScreen(on: Boolean): CoordinatorSnapshot {
        screenOn = on
        reconcileBle()
        return snapshot()
    }

    suspend fun updateBodyState(
        nowNanos: Long,
        isOnBody: Boolean,
        batteryFraction: Double? = null,
    ): CoordinatorSnapshot {
        onBody = isOnBody
        reconcileBle()
        if (tier2Active()) {
            val transition =
                if (isOnBody) {
                    dutyFsm.onOnBody(nowNanos)
                } else {
                    dutyFsm.onOffBody(nowNanos)
                }
            handleDutyAction(transition, nowNanos, batteryFraction)
        }
        return snapshot()
    }

    fun onLinearAcceleration(
        nowNanos: Long,
        magnitudeMps2: Double,
    ) {
        if (tier2Active()) {
            dutyFsm.onLinearAcceleration(nowNanos, magnitudeMps2)
        }
    }

    suspend fun onStepDetected(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): CoordinatorSnapshot {
        if (tier2Active()) {
            val transition = dutyFsm.onStepDetected(nowNanos, batteryPercent, thermalSeverity)
            handleDutyAction(transition, nowNanos, batteryPercent / 100.0)
        }
        return snapshot()
    }

    suspend fun onUwbEpoch(
        sample: RangingEpochSample,
        batteryPercent: Int,
    ): CoordinatorSnapshot {
        if (!tier2Active()) {
            return snapshot()
        }
        val publisher = batchPublisher
        if (publisher != null) {
            publisher.enqueue(sample)
            if (publisher.shouldFlush(sample.measuredAtElapsedNanos)) {
                publisher.flush(
                    nowElapsedNanos = sample.measuredAtElapsedNanos,
                    wearState = if (onBody) WearStateWire.ON_BODY else WearStateWire.OFF_BODY,
                    motionState = MotionStateWire.MOVING,
                    batteryFraction = batteryPercent / 100.0,
                )
            }
        }
        val transition = dutyFsm.onRangingEpochObserved(sample.measuredAtElapsedNanos, sample.ranges.size)
        handleDutyAction(transition, sample.measuredAtElapsedNanos, batteryPercent / 100.0)
        return snapshot()
    }

    suspend fun onTick(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): CoordinatorSnapshot {
        if (tier2Active()) {
            val transition = dutyFsm.tick(nowNanos, batteryPercent, thermalSeverity)
            handleDutyAction(transition, nowNanos, batteryPercent / 100.0)
        }
        return snapshot()
    }

    fun snapshot(): CoordinatorSnapshot {
        val currentId = identifierStore.current()
        val transition =
            beaconFsm.evaluate(
                BeaconInputs(
                    paired = currentId != null,
                    onBody = onBody,
                    bluetoothEnabled = bluetoothEnabled,
                    screenOn = screenOn,
                ),
            )
        return CoordinatorSnapshot(
            mode = mode,
            beaconState = transition.state,
            tileStatus = transition.tileStatus,
            dutyCycleState = dutyFsm.state,
            hardwareKey = currentId?.hardwareKey,
        )
    }

    private fun tier2Active(): Boolean {
        return mode == RuntimeMode.BLE_PLUS_UWB_BURST &&
            uwbAnchors.isNotEmpty() &&
            uwbEngine != null
    }

    private fun reconcileBle() {
        val currentId = identifierStore.current()
        val transition =
            beaconFsm.evaluate(
                BeaconInputs(
                    paired = currentId != null,
                    onBody = onBody,
                    bluetoothEnabled = bluetoothEnabled,
                    screenOn = screenOn,
                ),
            )
        if (transition.shouldAdvertise && currentId != null) {
            bleSink.startIBeacon(IBeaconFrameEncoder.encode(currentId))
        } else {
            bleSink.stopAdvertising()
        }
    }

    private suspend fun handleDutyAction(
        transition: DutyCycleTransition,
        nowNanos: Long,
        batteryFraction: Double?,
    ) {
        val engine = uwbEngine ?: return
        when (transition.action) {
            DutyCycleAction.NONE -> Unit
            DutyCycleAction.START_UWB_BURST -> {
                if (!engine.isAvailable()) {
                    return
                }
                val sessionId = activeSessionCounter++
                val request =
                    UwbBurstRequest(
                        sessionId = sessionId,
                        anchors = uwbAnchors,
                        burstDurationS = fsmConfig.maxBurstDurationS.toInt(),
                    )
                val prepared = engine.prepareBurst(request)
                val wakeFrame =
                    PreSharedStsNegotiator.encodeWakePayload(
                        UwbWakeFrame(
                            sessionId = prepared.sessionId,
                            controllerAddress = prepared.localControllerAddress,
                            channel = prepared.channel,
                            preambleIndex = prepared.preambleIndex,
                            burstDurationS = request.burstDurationS,
                        ),
                    )
                bleSink.broadcastWakeFrame(wakeFrame, PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS)
                batchPublisher?.markBurstStarted(nowNanos)
                engine.startRanging(request)
            }
            DutyCycleAction.STOP_UWB_BURST -> {
                engine.stopRanging()
                batchPublisher?.flush(
                    nowElapsedNanos = nowNanos,
                    wearState = if (onBody) WearStateWire.ON_BODY else WearStateWire.OFF_BODY,
                    motionState = transition.motionState,
                    batteryFraction = batteryFraction,
                )
            }
            DutyCycleAction.STOP_ALL_RADIOS -> {
                engine.stopRanging()
                bleSink.stopAdvertising()
                batchPublisher?.flush(
                    nowElapsedNanos = nowNanos,
                    wearState = WearStateWire.OFF_BODY,
                    motionState = MotionStateWire.STILL,
                    batteryFraction = batteryFraction,
                )
            }
        }
    }
}
