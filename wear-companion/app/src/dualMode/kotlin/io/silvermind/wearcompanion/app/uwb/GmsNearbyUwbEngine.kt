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
import io.silvermind.wearcompanion.core.PreparedUwbSession
import io.silvermind.wearcompanion.core.RangingEpochSample
import io.silvermind.wearcompanion.core.UwbBurstRequest
import io.silvermind.wearcompanion.core.UwbRangingEngine
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Tier 2 [UwbRangingEngine] implementation backed by Google Play services
 * `com.google.android.gms.nearby.uwb.UwbClient` (`CONFIG_MULTICAST_DS_TWR`).
 */
class GmsNearbyUwbEngine(
    context: Context,
    private val sessionKeyInfo: ByteArray,
) : UwbRangingEngine {
    override val backendName: String = "gms_nearby_uwb"

    private val client = Nearby.getUwbControllerClient(context)
    private var activeCallback: RangingSessionCallback? = null
    private var epochSeq: Long = 0L

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
        val bytes = localAddr.address
        val shortAddr =
            if (bytes.size >= 2) {
                ((bytes[0].toInt() and 0xFF) shl 8) or (bytes[1].toInt() and 0xFF)
            } else {
                0
            }
        return PreparedUwbSession(
            sessionId = request.sessionId,
            localControllerAddress = shortAddr,
            channel = request.channel,
            preambleIndex = request.preambleIndex,
        )
    }

    override fun startRanging(request: UwbBurstRequest): Flow<RangingEpochSample> =
        callbackFlow {
            val peerByHex = request.anchors.associateBy { it.uwbAddressHex.uppercase() }
            val peers =
                request.anchors.map { anchor ->
                    UwbDevice.createForAddress(anchor.addressBytes())
                }
            val complexChannel =
                UwbComplexChannel.Builder()
                    .setChannel(request.channel)
                    .setPreambleIndex(request.preambleIndex)
                    .build()
            val params =
                RangingParameters.Builder()
                    .setSessionId(request.sessionId)
                    .setSessionKeyInfo(sessionKeyInfo)
                    .setComplexChannel(complexChannel)
                    .setPeerDevices(peers)
                    .setRangingUpdateRate(RangingParameters.RangingUpdateRate.AUTOMATIC)
                    .setUwbConfigId(RangingParameters.UwbConfigId.CONFIG_MULTICAST_DS_TWR)
                    .build()

            val callback =
                object : RangingSessionCallback {
                    override fun onRangingInitialized(device: UwbDevice) = Unit

                    override fun onRangingResult(device: UwbDevice, position: RangingPosition) {
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
                        epochSeq += 1L
                        trySend(
                            RangingEpochSample(
                                epochSeq = epochSeq,
                                measuredAtElapsedNanos = SystemClock.elapsedRealtimeNanos(),
                                ranges = listOf(reading),
                            ),
                        )
                    }

                    override fun onRangingSuspended(device: UwbDevice, reason: Int) = Unit
                }

            activeCallback = callback
            client.startRanging(params, callback)
            awaitClose {
                client.stopRanging(callback)
                activeCallback = null
            }
        }

    override suspend fun stopRanging() {
        val cb = activeCallback ?: return
        activeCallback = null
        client.stopRanging(cb)
    }
}
