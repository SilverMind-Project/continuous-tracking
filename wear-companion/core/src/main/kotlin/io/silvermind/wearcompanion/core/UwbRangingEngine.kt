package io.silvermind.wearcompanion.core

import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.emitAll
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

/** Single anchor measurement emitted inside one UWB ranging epoch. */
data class AnchorRangeReading(
    val anchorId: String,
    val uwbAddressHex: String,
    val distanceM: Double,
    val azimuthDeg: Double? = null,
    val elevationDeg: Double? = null,
    val losConfidence: Double? = null,
    val technology: RangingTechnologyWire = RangingTechnologyWire.UWB,
) {
    init {
        require(anchorId.isNotBlank()) { "anchorId must be non-blank" }
        require(distanceM.isFinite() && distanceM >= 0.0) { "distanceM must be finite and >= 0" }
    }
}

/** One multi-anchor UWB ranging epoch stamped on the monotonic clock. */
data class RangingEpochSample(
    val epochSeq: Long,
    val measuredAtElapsedNanos: Long,
    val ranges: List<AnchorRangeReading>,
) {
    init {
        require(epochSeq >= 0L) { "epochSeq must be >= 0" }
        require(ranges.isNotEmpty()) { "RangingEpochSample must contain at least one range" }
    }
}

/** Parameters passed when opening a controller ranging burst. */
data class UwbBurstRequest(
    val sessionId: Int,
    val anchors: List<UwbAnchorEntry>,
    val channel: Int = UwbAnchorEntry.DEFAULT_UWB_CHANNEL,
    val preambleIndex: Int = UwbAnchorEntry.DEFAULT_PREAMBLE_INDEX,
    val burstDurationS: Int = 12,
    val updateRateHz: Int = 1,
)

/** Local controller address and channel returned by the UWB HAL when preparing a burst. */
data class PreparedUwbSession(
    val sessionId: Int,
    val localControllerAddress: Int,
    val channel: Int,
    val preambleIndex: Int,
)

/**
 * Strategy interface isolating Google Play services (`GmsNearbyUwbEngine`) and
 * Jetpack (`JetpackUwbEngine`) UWB APIs from all coordination and FSM logic.
 */
interface UwbRangingEngine {
    val backendName: String

    suspend fun isAvailable(): Boolean

    suspend fun prepareBurst(request: UwbBurstRequest): PreparedUwbSession

    fun startRanging(request: UwbBurstRequest): Flow<RangingEpochSample>

    suspend fun stopRanging()
}

/**
 * Deterministic [UwbRangingEngine] peer for unit tests.
 *
 * Like the real engines, [startRanging] returns a cold flow: ranging counts as
 * active only while a collector is subscribed, so a caller that drops the
 * returned flow without collecting it is visible in [isRangingActive].
 */
class FakeUwbRangingEngine(
    private val available: Boolean = true,
    private val localAddress: Int = 0x1001,
    private val prepareFailure: Exception? = null,
    /** Simulates a HAL call that ignores cancellation for this long while preparing. */
    private val nonCancellablePrepareMs: Long = 0L,
) : UwbRangingEngine {
    override val backendName: String = "fake_uwb"

    private val stream = MutableSharedFlow<RangingEpochSample>(extraBufferCapacity = 64)
    var preparedCount: Int = 0
        private set
    var startedCount: Int = 0
        private set
    var stoppedCount: Int = 0
        private set
    var activeCollectors: Int = 0
        private set
    var lastRequest: UwbBurstRequest? = null
        private set

    val isRangingActive: Boolean
        get() = activeCollectors > 0

    override suspend fun isAvailable(): Boolean = available

    override suspend fun prepareBurst(request: UwbBurstRequest): PreparedUwbSession {
        preparedCount += 1
        if (nonCancellablePrepareMs > 0L) {
            withContext(NonCancellable) { delay(nonCancellablePrepareMs) }
        }
        prepareFailure?.let { throw it }
        return PreparedUwbSession(
            sessionId = request.sessionId,
            localControllerAddress = localAddress,
            channel = request.channel,
            preambleIndex = request.preambleIndex,
        )
    }

    override fun startRanging(request: UwbBurstRequest): Flow<RangingEpochSample> {
        startedCount += 1
        lastRequest = request
        return flow {
            activeCollectors += 1
            try {
                emitAll(stream)
            } finally {
                activeCollectors -= 1
            }
        }
    }

    /** Deliver [sample] to every active collector; dropped when nobody collects. */
    suspend fun emitEpoch(sample: RangingEpochSample) {
        stream.emit(sample)
    }

    override suspend fun stopRanging() {
        stoppedCount += 1
    }
}
