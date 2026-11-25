package io.silvermind.wearcompanion.core

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.security.SecureRandom

/** Hardware-neutral BLE advertising sink driven by [CompanionCoordinator]. */
interface BleAdvertiserSink {
    fun startIBeacon(payload: ByteArray)

    fun stopAdvertising()

    /**
     * Optional low-latency wake broadcast used only in Tier 2 UWB burst mode.
     * Implementations advertise under [PreSharedStsNegotiator.WAKE_MANUFACTURER_ID]
     * for [durationMs] and then restore the iBeacon.
     */
    fun broadcastWakeFrame(
        payload: ByteArray,
        durationMs: Long,
    ) {}
}

/**
 * Keeps the application processor awake for one UWB burst so the burst and
 * flush timers fire while the screen is off. [acquire] must carry a timeout so
 * a missed [release] cannot drain the battery.
 */
interface BurstWakeHold {
    fun acquire(timeoutMs: Long)

    fun release()
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
 *
 * Every public method is serialized by one [Mutex], so sensor callbacks, the
 * 1 Hz tick, and the ranging collector can call in from any coroutine. Callers
 * that need ordering (accelerometer samples) must still deliver in order.
 *
 * A Tier 2 burst runs as a job in [scope]: prepare the engine, broadcast the
 * wake frame, wait [PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS], then
 * collect the engine's cold ranging flow into [onUwbEpoch]. Stopping a burst
 * stops the engine, flushes the publisher, and cancels the job last, because a
 * burst usually ends from inside its own collector.
 */
@Suppress("TooManyFunctions")
class CompanionCoordinator(
    private val identifierStore: IdentifierStore,
    private val bleSink: BleAdvertiserSink,
    private val mode: RuntimeMode = RuntimeMode.BLE_ONLY,
    private val budgetLimiter: BatteryBudgetLimiter = BatteryBudgetLimiter(),
    private val fsmConfig: LocomotionFsmConfig = LocomotionFsmConfig(),
    private val uwbAnchors: List<UwbAnchorEntry> = emptyList(),
    private val uwbEngine: UwbRangingEngine? = null,
    private val batchPublisher: EpochBatchPublisher? = null,
    private val scope: CoroutineScope? = null,
    private val wakeHold: BurstWakeHold? = null,
    private val onBurstError: (Throwable) -> Unit = {},
) {
    init {
        if (tier2Configured()) {
            requireNotNull(scope) { "Tier 2 needs a CoroutineScope to collect the ranging flow" }
            require(uwbAnchors.map { it.channel to it.preambleIndex }.toSet().size == 1) {
                "All UWB anchors must share one channel and preamble index"
            }
        }
    }

    private val mutex = Mutex()
    private val beaconFsm = BeaconStateMachine()
    private val dutyFsm = LocomotionDutyCycleFsm(budgetLimiter = budgetLimiter, config = fsmConfig)

    private var onBody: Boolean = false
    private var bluetoothEnabled: Boolean = true
    private var screenOn: Boolean = true
    private var lastBatteryPercent: Int = FULL_BATTERY_PERCENT
    private var burstGeneration: Long = 0L
    private var burstJob: Job? = null

    suspend fun pair(): BeaconIdentifier =
        mutex.withLock {
            val id = identifierStore.loadOrCreate()
            reconcileBle()
            id
        }

    suspend fun resetIdentifier(): BeaconIdentifier =
        mutex.withLock {
            val id = identifierStore.reset()
            reconcileBle()
            id
        }

    suspend fun updateBluetooth(enabled: Boolean): CoordinatorSnapshot =
        mutex.withLock {
            bluetoothEnabled = enabled
            reconcileBle()
            snapshotLocked()
        }

    suspend fun updateScreen(on: Boolean): CoordinatorSnapshot =
        mutex.withLock {
            screenOn = on
            reconcileBle()
            snapshotLocked()
        }

    suspend fun updateBodyState(
        nowNanos: Long,
        isOnBody: Boolean,
        batteryFraction: Double? = null,
    ): CoordinatorSnapshot =
        mutex.withLock {
            onBody = isOnBody
            batteryFraction?.let { lastBatteryPercent = (it * FULL_BATTERY_PERCENT).toInt() }
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
            snapshotLocked()
        }

    suspend fun onLinearAcceleration(
        nowNanos: Long,
        magnitudeMps2: Double,
    ) {
        if (!tier2Active()) {
            return
        }
        mutex.withLock {
            dutyFsm.onLinearAcceleration(nowNanos, magnitudeMps2)
        }
    }

    suspend fun onStepDetected(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): CoordinatorSnapshot =
        mutex.withLock {
            lastBatteryPercent = batteryPercent
            if (tier2Active()) {
                val transition = dutyFsm.onStepDetected(nowNanos, batteryPercent, thermalSeverity)
                handleDutyAction(transition, nowNanos, batteryPercent / PERCENT)
            }
            snapshotLocked()
        }

    /** Feed one multi-anchor ranging epoch; the burst collector calls this for every sample. */
    suspend fun onUwbEpoch(
        sample: RangingEpochSample,
        batteryPercent: Int = lastBatteryPercent,
    ): CoordinatorSnapshot =
        mutex.withLock {
            if (!tier2Active()) {
                return@withLock snapshotLocked()
            }
            val publisher = batchPublisher
            if (publisher != null) {
                publisher.enqueue(sample)
                if (publisher.shouldFlush(sample.measuredAtElapsedNanos)) {
                    publisher.flush(
                        nowElapsedNanos = sample.measuredAtElapsedNanos,
                        wearState = wearStateWire(),
                        motionState = MotionStateWire.MOVING,
                        batteryFraction = batteryPercent / PERCENT,
                    )
                }
            }
            val transition = dutyFsm.onRangingEpochObserved(sample.measuredAtElapsedNanos, sample.ranges.size)
            handleDutyAction(transition, sample.measuredAtElapsedNanos, batteryPercent / PERCENT)
            snapshotLocked()
        }

    /**
     * 1 Hz pulse from the service: advances burst and cooldown timers when no
     * step or ranging event arrives, and flushes epochs left in the publisher.
     */
    suspend fun onTick(
        nowNanos: Long,
        batteryPercent: Int,
        thermalSeverity: ThermalSeverity = ThermalSeverity.NONE,
    ): CoordinatorSnapshot =
        mutex.withLock {
            lastBatteryPercent = batteryPercent
            if (tier2Active()) {
                val transition = dutyFsm.tick(nowNanos, batteryPercent, thermalSeverity)
                handleDutyAction(transition, nowNanos, batteryPercent / PERCENT)
                val publisher = batchPublisher
                if (publisher != null && publisher.shouldFlush(nowNanos)) {
                    publisher.flush(
                        nowElapsedNanos = nowNanos,
                        wearState = wearStateWire(),
                        motionState = transition.motionState,
                        batteryFraction = batteryPercent / PERCENT,
                    )
                }
            }
            snapshotLocked()
        }

    suspend fun snapshot(): CoordinatorSnapshot = mutex.withLock { snapshotLocked() }

    /** Stop any running burst and flush buffered epochs before this coordinator is discarded. */
    suspend fun shutdown(nowNanos: Long) {
        mutex.withLock {
            stopBurst(nowNanos, wearStateWire(), MotionStateWire.UNSPECIFIED, lastBatteryPercent / PERCENT)
        }
    }

    /** True when Tier 2 is configured and the ranging flow will be collected. */
    fun tier2Active(): Boolean = tier2Configured()

    private fun tier2Configured(): Boolean =
        mode == RuntimeMode.BLE_PLUS_UWB_BURST &&
            uwbAnchors.isNotEmpty() &&
            uwbEngine != null

    private fun wearStateWire(): WearStateWire = if (onBody) WearStateWire.ON_BODY else WearStateWire.OFF_BODY

    private fun snapshotLocked(): CoordinatorSnapshot {
        val currentId = identifierStore.current()
        val transition = beaconFsm.evaluate(beaconInputs(currentId))
        return CoordinatorSnapshot(
            mode = mode,
            beaconState = transition.state,
            tileStatus = transition.tileStatus,
            dutyCycleState = dutyFsm.state,
            hardwareKey = currentId?.hardwareKey,
        )
    }

    private fun beaconInputs(currentId: BeaconIdentifier?): BeaconInputs =
        BeaconInputs(
            paired = currentId != null,
            onBody = onBody,
            bluetoothEnabled = bluetoothEnabled,
            screenOn = screenOn,
        )

    private fun reconcileBle() {
        val currentId = identifierStore.current()
        val transition = beaconFsm.evaluate(beaconInputs(currentId))
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
        when (transition.action) {
            DutyCycleAction.NONE -> Unit
            DutyCycleAction.START_UWB_BURST -> startBurst(nowNanos)
            DutyCycleAction.STOP_UWB_BURST ->
                stopBurst(nowNanos, wearStateWire(), transition.motionState, batteryFraction)
            DutyCycleAction.STOP_ALL_RADIOS -> {
                stopBurst(nowNanos, WearStateWire.OFF_BODY, MotionStateWire.STILL, batteryFraction)
                bleSink.stopAdvertising()
            }
        }
    }

    private fun startBurst(nowNanos: Long) {
        val engine = uwbEngine ?: return
        val launchScope = scope ?: return
        burstJob?.cancel()
        val generation = ++burstGeneration
        batchPublisher?.markBurstStarted(nowNanos)
        wakeHold?.acquire(burstHoldTimeoutMs())
        burstJob = launchScope.launch { runBurst(engine, generation) }
    }

    private suspend fun runBurst(
        engine: UwbRangingEngine,
        generation: Long,
    ) {
        try {
            if (!engine.isAvailable()) {
                return
            }
            val first = uwbAnchors.first()
            val request =
                UwbBurstRequest(
                    sessionId = nextSessionId(),
                    anchors = uwbAnchors,
                    channel = first.channel,
                    preambleIndex = first.preambleIndex,
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
            val wakeSent =
                mutex.withLock {
                    if (burstStillWanted(generation)) {
                        bleSink.broadcastWakeFrame(wakeFrame, PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS)
                        true
                    } else {
                        false
                    }
                }
            if (!wakeSent) {
                return
            }
            delay(PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS)
            val ranging = mutex.withLock { if (burstStillWanted(generation)) engine.startRanging(request) else null }
            ranging?.collect { sample -> onUwbEpoch(sample) }
        } catch (err: CancellationException) {
            throw err
        } catch (
            @Suppress("TooGenericExceptionCaught") err: Exception,
        ) {
            onBurstError(err)
        }
    }

    private fun burstStillWanted(generation: Long): Boolean =
        generation == burstGeneration && dutyFsm.state == DutyCycleState.UWB_RANGING_BURST

    private suspend fun stopBurst(
        nowNanos: Long,
        wearState: WearStateWire,
        motionState: MotionStateWire,
        batteryFraction: Double?,
    ) {
        val engine = uwbEngine ?: return
        val job = burstJob
        burstJob = null
        burstGeneration += 1
        try {
            engine.stopRanging()
        } catch (err: CancellationException) {
            throw err
        } catch (
            @Suppress("TooGenericExceptionCaught") err: Exception,
        ) {
            onBurstError(err)
        }
        batchPublisher?.flush(
            nowElapsedNanos = nowNanos,
            wearState = wearState,
            motionState = motionState,
            batteryFraction = batteryFraction,
        )
        if (job != null) {
            wakeHold?.release()
            job.cancel()
        }
    }

    private fun burstHoldTimeoutMs(): Long =
        (fsmConfig.maxBurstDurationS * MILLIS_PER_SECOND).toLong() +
            PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS +
            WAKE_HOLD_MARGIN_MS

    companion object {
        private const val FULL_BATTERY_PERCENT: Int = 100
        private const val PERCENT: Double = 100.0
        private const val MILLIS_PER_SECOND: Double = 1_000.0
        private const val WAKE_HOLD_MARGIN_MS: Long = 3_000L
        private val sessionRandom = SecureRandom()

        /**
         * Random positive FiRa session ID per burst, so two watches in one
         * household never present the same session to the shared anchors.
         */
        fun nextSessionId(): Int = sessionRandom.nextInt(Int.MAX_VALUE - 1) + 1
    }
}
