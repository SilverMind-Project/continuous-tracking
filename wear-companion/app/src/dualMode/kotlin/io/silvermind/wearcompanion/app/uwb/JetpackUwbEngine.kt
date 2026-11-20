package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import android.os.SystemClock
import androidx.core.uwb.RangingParameters
import androidx.core.uwb.RangingResult
import androidx.core.uwb.UwbAddress
import androidx.core.uwb.UwbComplexChannel
import androidx.core.uwb.UwbControllerSessionScope
import androidx.core.uwb.UwbDevice
import androidx.core.uwb.UwbManager
import io.silvermind.wearcompanion.core.AnchorRangeReading
import io.silvermind.wearcompanion.core.PreparedUwbSession
import io.silvermind.wearcompanion.core.RangingEpochSample
import io.silvermind.wearcompanion.core.UwbBurstRequest
import io.silvermind.wearcompanion.core.UwbRangingEngine
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * Tier 2 [UwbRangingEngine] implementation backed by Jetpack `androidx.core.uwb`.
 */
class JetpackUwbEngine(
    private val context: Context,
    private val sessionKeyInfo: ByteArray,
) : UwbRangingEngine {
    override val backendName: String = "jetpack_uwb"

    private var activeScope: UwbControllerSessionScope? = null
    private var epochSeq: Long = 0L

    override suspend fun isAvailable(): Boolean =
        runCatching {
            UwbManager.createInstance(context)
            true
        }.getOrDefault(false)

    override suspend fun prepareBurst(request: UwbBurstRequest): PreparedUwbSession {
        val manager = UwbManager.createInstance(context)
        val scope = manager.controllerSessionScope()
        activeScope = scope
        val bytes = scope.localAddress.address
        val shortAddr =
            if (bytes.size >= 2) {
                ((bytes[0].toInt() and 0xFF) shl 8) or (bytes[1].toInt() and 0xFF)
            } else {
                0
            }
        return PreparedUwbSession(
            sessionId = request.sessionId,
            localControllerAddress = shortAddr,
            channel = scope.uwbComplexChannel.channel,
            preambleIndex = scope.uwbComplexChannel.preambleIndex,
        )
    }

    override fun startRanging(request: UwbBurstRequest): Flow<RangingEpochSample> =
        flow {
            val scope = activeScope ?: UwbManager.createInstance(context).controllerSessionScope()
            val peerByHex = request.anchors.associateBy { it.uwbAddressHex.uppercase() }
            val peers = request.anchors.map { UwbDevice(UwbAddress(it.addressBytes())) }
            val params =
                RangingParameters(
                    uwbConfigType = RangingParameters.CONFIG_MULTICAST_DS_TWR,
                    sessionId = request.sessionId,
                    subSessionId = 0,
                    sessionKeyInfo = sessionKeyInfo,
                    subSessionKeyInfo = null,
                    complexChannel = UwbComplexChannel(request.channel, request.preambleIndex),
                    peerDevices = peers,
                    updateRateType = RangingParameters.RANGING_UPDATE_RATE_AUTOMATIC,
                )
            scope.prepareSession(params).collect { result ->
                if (result is RangingResult.RangingResultPosition) {
                    val addrBytes = result.device.address.address
                    if (addrBytes.size < 2) {
                        return@collect
                    }
                    val hex = "%02X:%02X".format(addrBytes[0], addrBytes[1])
                    val anchor = peerByHex[hex] ?: return@collect
                    val distance = result.position.distance?.value?.toDouble() ?: return@collect
                    epochSeq += 1L
                    emit(
                        RangingEpochSample(
                            epochSeq = epochSeq,
                            measuredAtElapsedNanos = SystemClock.elapsedRealtimeNanos(),
                            ranges =
                                listOf(
                                    AnchorRangeReading(
                                        anchorId = anchor.anchorId,
                                        uwbAddressHex = hex,
                                        distanceM = distance,
                                        azimuthDeg = result.position.azimuth?.value?.toDouble(),
                                        elevationDeg = result.position.elevation?.value?.toDouble(),
                                    ),
                                ),
                        ),
                    )
                }
            }
        }

    override suspend fun stopRanging() {
        activeScope = null
    }
}
