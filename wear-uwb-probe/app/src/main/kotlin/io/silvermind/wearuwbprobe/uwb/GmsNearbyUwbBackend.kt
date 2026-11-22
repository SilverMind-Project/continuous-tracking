package io.silvermind.wearuwbprobe.uwb

import android.content.Context
import android.content.pm.PackageManager
import android.os.SystemClock
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.uwb.RangingParameters
import com.google.android.gms.nearby.uwb.RangingPosition
import com.google.android.gms.nearby.uwb.RangingSessionCallback
import com.google.android.gms.nearby.uwb.UwbAddress
import com.google.android.gms.nearby.uwb.UwbComplexChannel
import com.google.android.gms.nearby.uwb.UwbDevice
import com.google.android.gms.tasks.Task
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * Direct Google Play services Nearby UWB probe backend (`com.google.android.gms.nearby.uwb.UwbClient`).
 *
 * Captures raw `ApiException.statusCode` values (such as 42000 SERVICE_NOT_ENABLED or
 * 42002 MISSING_PERMISSION_UWB_RANGING), GMS measurement confidence levels, HAL-assigned
 * `localAddress`/`complexChannel`, and tests optional dynamic `addControlee`.
 */
class GmsNearbyUwbBackend(
    private val context: Context,
) : UwbProbeBackend {
    override val name: String = BACKEND_NAME

    private val client = Nearby.getUwbControllerClient(context)

    override suspend fun queryCapabilities(): ProbeCapabilities {
        val hasFeature = context.packageManager.hasSystemFeature(PackageManager.FEATURE_UWB)
        return try {
            val available = client.isAvailable().awaitTask()
            val caps = client.rangingCapabilities.awaitTask()
            val localAddr = runCatching { client.localAddress.awaitTask() }.getOrNull()
            val complexChannel = runCatching { client.complexChannel.awaitTask() }.getOrNull()

            val supportedConfigs = caps.supportedConfigIds.toList()
            val supportedChannels = caps.supportedChannels.toList()
            val localHex = localAddr?.address?.toShortAddressHex()

            ProbeCapabilities(
                backendName = name,
                featureFlagPresent = hasFeature,
                isAvailable = available,
                supportsDistance = caps.supportsDistance(),
                supportsAzimuth = caps.supportsAzimuthalAngle(),
                supportsElevation = caps.supportsElevationAngle(),
                minRangingIntervalMs = caps.minRangingInterval,
                supportedChannels = supportedChannels,
                supportedConfigIds = supportedConfigs,
                supportsMulticastDsTwr =
                    supportedConfigs.contains(RangingParameters.UwbConfigId.CONFIG_MULTICAST_DS_TWR),
                localAddressHex = localHex,
                assignedChannel = complexChannel?.channel,
                assignedPreambleIndex = complexChannel?.preambleIndex,
            )
        } catch (err: ApiException) {
            ProbeCapabilities(
                backendName = name,
                featureFlagPresent = hasFeature,
                isAvailable = false,
                supportsDistance = false,
                supportsAzimuth = false,
                supportsElevation = false,
                rawStatusCode = err.statusCode,
                errorDetail = err.message ?: "ApiException(${err.statusCode})",
            )
        } catch (err: SecurityException) {
            ProbeCapabilities(
                backendName = name,
                featureFlagPresent = hasFeature,
                isAvailable = false,
                supportsDistance = false,
                supportsAzimuth = false,
                supportsElevation = false,
                errorDetail = "SecurityException: ${err.message}",
            )
        }
    }

    override suspend fun openControllerSession(
        sessionId: Int,
        preferredChannel: Int,
        preferredPreambleIndex: Int,
    ): PreparedControllerSession? {
        return try {
            val localAddr = client.localAddress.awaitTask()
            val complexChannel = runCatching { client.complexChannel.awaitTask() }.getOrNull()
            val bytes = localAddr.address
            if (bytes.size < 2) {
                return null
            }
            val shortAddr = ((bytes[0].toInt() and 0xFF) shl 8) or (bytes[1].toInt() and 0xFF)
            PreparedControllerSession(
                backendName = name,
                sessionId = sessionId,
                localAddressHex = bytes.toShortAddressHex() ?: "00:00",
                localControllerAddress = shortAddr,
                channel = complexChannel?.channel ?: preferredChannel,
                preambleIndex = complexChannel?.preambleIndex ?: preferredPreambleIndex,
            )
        } catch (_: ApiException) {
            null
        } catch (_: SecurityException) {
            null
        }
    }

    override suspend fun runMulticastSession(
        config: MulticastSessionConfig,
        durationMs: Long,
        onSample: (ProbeRangingSample) -> Unit,
    ): SessionSummary {
        val startedAtMs = System.currentTimeMillis()
        val startedNanos = SystemClock.elapsedRealtimeNanos()
        val seedPeers =
            if (config.useDynamicAddControlee && config.peers.size > 1) {
                listOf(config.peers.first())
            } else {
                config.peers
            }
        val extraPeers =
            if (config.useDynamicAddControlee && config.peers.size > 1) {
                config.peers.drop(1)
            } else {
                emptyList()
            }

        val complexChannel =
            UwbComplexChannel.Builder()
                .setChannel(config.channel)
                .setPreambleIndex(config.preambleIndex)
                .build()
        val params =
            RangingParameters.Builder()
                .setSessionId(config.sessionId)
                .setSessionKeyInfo(config.sessionKeyInfo)
                .setComplexChannel(complexChannel)
                .setPeerDevices(seedPeers.map { UwbDevice.createForAddress(it.addressBytes()) })
                .setRangingUpdateRate(RangingParameters.RangingUpdateRate.AUTOMATIC)
                .setUwbConfigId(config.uwbConfigId)
                .build()

        val countsByPeer = linkedMapOf<String, Int>()
        val epochBuckets = linkedMapOf<Long, MutableSet<String>>()
        var samplesCollected = 0
        var droppedSessions = 0
        var firstRangeMs: Long? = null
        var firstThreeAnchorMs: Long? = null

        val callback =
            object : RangingSessionCallback {
                override fun onRangingInitialized(device: UwbDevice) = Unit

                override fun onRangingResult(
                    device: UwbDevice,
                    position: RangingPosition,
                ) {
                    val peerHex = device.address.address.toShortAddressHex() ?: return
                    val nowNanos = SystemClock.elapsedRealtimeNanos()
                    val nowUnixMs = System.currentTimeMillis()
                    val relMs = ((nowNanos - startedNanos) / NANOS_PER_MS).coerceAtLeast(0L)
                    val sample =
                        ProbeRangingSample(
                            backendName = name,
                            elapsedRealtimeNanos = nowNanos,
                            wallClockUnixMs = nowUnixMs,
                            peerAddressHex = peerHex,
                            distanceM = position.distance?.value,
                            distanceConfidence = position.distance?.confidence?.toFloat(),
                            azimuthDeg = position.azimuth?.value,
                            azimuthConfidence = position.azimuth?.confidence?.toFloat(),
                            elevationDeg = position.elevation?.value,
                            elevationConfidence = position.elevation?.confidence?.toFloat(),
                            rssiDbm = readOptionalRssi(position),
                        )
                    synchronized(countsByPeer) {
                        samplesCollected += 1
                        countsByPeer[peerHex] = (countsByPeer[peerHex] ?: 0) + 1
                        if (firstRangeMs == null) {
                            firstRangeMs = relMs
                        }
                        val bucket = nowNanos / EPOCH_BUCKET_NANOS
                        val peersInBucket = epochBuckets.getOrPut(bucket) { linkedSetOf() }
                        peersInBucket.add(peerHex)
                        if (firstThreeAnchorMs == null && peersInBucket.size >= MIN_SOLVABLE_ANCHORS) {
                            firstThreeAnchorMs = relMs
                        }
                    }
                    onSample(sample)
                }

                override fun onRangingSuspended(
                    device: UwbDevice,
                    reason: Int,
                ) {
                    synchronized(countsByPeer) {
                        droppedSessions += 1
                    }
                }
            }

        var rawStatus: Int? = null
        var errorDetail: String? = null
        try {
            client.startRanging(params, callback).awaitTask()
            for (peer in extraPeers) {
                client.addControlee(UwbAddress(peer.addressBytes())).awaitTask()
            }
            delay(durationMs)
        } catch (err: ApiException) {
            rawStatus = err.statusCode
            errorDetail = err.message ?: "ApiException(${err.statusCode})"
            droppedSessions += 1
        } catch (err: SecurityException) {
            errorDetail = "SecurityException: ${err.message}"
            droppedSessions += 1
        } finally {
            runCatching { client.stopRanging(callback).awaitTask() }
        }

        val endedAtMs = System.currentTimeMillis()
        val elapsedS = ((endedAtMs - startedAtMs).coerceAtLeast(1L)) / 1000.0
        val rates = synchronized(countsByPeer) { countsByPeer.mapValues { (_, c) -> c.toDouble() / elapsedS } }

        return SessionSummary(
            backendName = name,
            sessionId = config.sessionId,
            startedAtUnixMs = startedAtMs,
            endedAtUnixMs = endedAtMs,
            samplesCollected = samplesCollected,
            distinctPeersAcquired = rates.size,
            timeToFirstRangeMs = firstRangeMs,
            timeToFirstThreeAnchorEpochMs = firstThreeAnchorMs,
            effectiveRateHzByPeer = rates,
            droppedSessions = droppedSessions,
            rawStatusCode = rawStatus,
            errorDetail = errorDetail,
        )
    }

    private fun readOptionalRssi(position: RangingPosition): Int? =
        runCatching {
            val method = position.javaClass.methods.firstOrNull { it.name == "getRssi" && it.parameterCount == 0 }
            (method?.invoke(position) as? Number)?.toInt()
        }.getOrNull()

    private fun ByteArray.toShortAddressHex(): String? =
        if (size >= 2) "%02X:%02X".format(this[0], this[1]) else null

    private suspend fun <T> Task<T>.awaitTask(): T =
        suspendCancellableCoroutine { cont ->
            addOnSuccessListener { result -> cont.resume(result) }
            addOnFailureListener { err -> cont.resumeWithException(err) }
        }

    companion object {
        const val BACKEND_NAME: String = "gms_nearby_uwb"
        private const val NANOS_PER_MS: Long = 1_000_000L
        private const val EPOCH_BUCKET_NANOS: Long = 250_000_000L
        private const val MIN_SOLVABLE_ANCHORS: Int = 3
    }
}
