package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import android.os.SystemClock
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.uwb.RangingParameters
import com.google.android.gms.nearby.uwb.RangingPosition
import com.google.android.gms.nearby.uwb.RangingSessionCallback
import com.google.android.gms.nearby.uwb.UwbAddress
import com.google.android.gms.nearby.uwb.UwbComplexChannel
import com.google.android.gms.nearby.uwb.UwbDevice
import io.silvermind.wearcompanion.core.AnchorRangeReading
import io.silvermind.wearcompanion.core.EpochSequence
import io.silvermind.wearcompanion.core.PeerRangeReading
import io.silvermind.wearcompanion.core.PreparedUwbSession
import io.silvermind.wearcompanion.core.RangingEpochSample
import io.silvermind.wearcompanion.core.UwbBurstRequest
import io.silvermind.wearcompanion.core.UwbRangingEngine
import io.silvermind.wearcompanion.core.coalesceRounds
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Tier 2 [UwbRangingEngine] implementation backed by Google Play services
 * `com.google.android.gms.nearby.uwb.UwbClient` (`CONFIG_MULTICAST_DS_TWR`).
 *
 * Play services calls `onRangingResult` once per peer anchor per round; the
 * per-peer callbacks are coalesced into multi-anchor epochs in `:core`.
 */
class GmsNearbyUwbEngine(
    context: Context,
    private val sessionKeyInfo: ByteArray,
) : UwbRangingEngine {
    override val backendName: String = "gms_nearby_uwb"

    private val client = Nearby.getUwbControllerClient(context)
    private val epochSequence = EpochSequence()

    @Volatile
    private var activeCallback: RangingSessionCallback? = null

    override suspend fun isAvailable(): Boolean =
        suspendCancellableCoroutine { cont ->
            client.isAvailable()
                .addOnSuccessListener { available -> cont.resume(available) }
                .addOnFailureListener { cont.resume(false) }
        }

    override suspend fun prepareBurst(request: UwbBurstRequest): PreparedUwbSession {
        val localAddr =
            suspendCancellableCoroutine<UwbAddress> { cont ->
                client.localAddress
                    .addOnSuccessListener { cont.resume(it) }
                    .addOnFailureListener { cont.resumeWithException(it) }
            }
        return PreparedUwbSession(
            sessionId = request.sessionId,
            localControllerAddress = shortAddress(localAddr.address),
            channel = request.channel,
            preambleIndex = request.preambleIndex,
        )
    }

    // Coalesce rounds off the service's serial thread: a blocking MQTT reconnect
    // there must not let the round timer split one round into two epochs.
    override fun startRanging(request: UwbBurstRequest): Flow<RangingEpochSample> =
        peerReadings(request).coalesceRounds(
            expectedAnchorCount = request.anchors.size,
            clockNanos = SystemClock::elapsedRealtimeNanos,
            sequence = epochSequence,
        ).flowOn(Dispatchers.Default)

    override suspend fun stopRanging() {
        activeCallback?.let { stopCallback(it) }
    }

    private fun peerReadings(request: UwbBurstRequest): Flow<PeerRangeReading> =
        callbackFlow {
            val peerByHex = request.anchors.associateBy { it.uwbAddressHex.uppercase() }
            val complexChannel =
                UwbComplexChannel.Builder()
                    .setChannel(request.channel)
                    .setPreambleIndex(request.preambleIndex)
                    .build()
            val builder =
                RangingParameters.Builder()
                    .setSessionId(request.sessionId)
                    .setSessionKeyInfo(sessionKeyInfo)
                    .setComplexChannel(complexChannel)
                    .setRangingUpdateRate(RangingParameters.RangingUpdateRate.AUTOMATIC)
                    .setUwbConfigId(MULTICAST_DS_TWR_CONFIG_ID)
            for (anchor in request.anchors) {
                builder.addPeerDevice(UwbDevice.createForAddress(anchor.addressBytes()))
            }
            val params = builder.build()

            val callback =
                object : RangingSessionCallback {
                    override fun onRangingInitialized(device: UwbDevice) = Unit

                    override fun onRangingResult(
                        device: UwbDevice,
                        position: RangingPosition,
                    ) {
                        val addrBytes = device.address.address
                        if (addrBytes.size < 2) {
                            return
                        }
                        val hex = "%02X:%02X".format(addrBytes[0], addrBytes[1])
                        val anchor = peerByHex[hex] ?: return
                        val distance = position.distance?.value?.toDouble() ?: return
                        val reading =
                            AnchorRangeReading(
                                anchorId = anchor.anchorId,
                                uwbAddressHex = hex,
                                distanceM = distance,
                                azimuthDeg = position.azimuth?.value?.toDouble(),
                                elevationDeg = position.elevation?.value?.toDouble(),
                            )
                        trySend(PeerRangeReading(reading, SystemClock.elapsedRealtimeNanos()))
                    }

                    override fun onRangingSuspended(
                        device: UwbDevice,
                        reason: Int,
                    ) = Unit
                }

            activeCallback = callback
            client.startRanging(params, callback)
                .addOnFailureListener { err -> close(err) }
            awaitClose { stopCallback(callback) }
        }

    /** Stop a session once: both [stopRanging] and flow cancellation can reach here. */
    private fun stopCallback(callback: RangingSessionCallback) {
        synchronized(this) {
            if (activeCallback !== callback) {
                return
            }
            activeCallback = null
        }
        client.stopRanging(callback)
    }

    private companion object {
        /**
         * Play services numbers FiRa configurations instead of naming them;
         * `CONFIG_ID_2` is multicast DS-TWR with Static STS, the same value as
         * Jetpack's `RangingParameters.CONFIG_MULTICAST_DS_TWR`.
         */
        const val MULTICAST_DS_TWR_CONFIG_ID: Int = RangingParameters.UwbConfigId.CONFIG_ID_2
    }

    private fun shortAddress(bytes: ByteArray): Int =
        if (bytes.size >= 2) {
            ((bytes[0].toInt() and 0xFF) shl 8) or (bytes[1].toInt() and 0xFF)
        } else {
            0
        }
}
