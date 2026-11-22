package io.silvermind.wearuwbprobe.uwb

import android.content.Context
import android.content.pm.PackageManager
import android.os.SystemClock
import androidx.core.uwb.RangingParameters
import androidx.core.uwb.RangingResult
import androidx.core.uwb.UwbAddress
import androidx.core.uwb.UwbComplexChannel
import androidx.core.uwb.UwbControllerSessionScope
import androidx.core.uwb.UwbDevice
import androidx.core.uwb.UwbManager
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Jetpack `androidx.core.uwb.UwbManager` probe backend.
 *
 * Evaluates whether the public Jetpack wrapper allows opening a controller session on
 * Wear OS or rejects it before delegating to Play services.
 */
class JetpackUwbBackend(
    private val context: Context,
) : UwbProbeBackend {
    override val name: String = BACKEND_NAME

    private var cachedScope: UwbControllerSessionScope? = null

    override suspend fun queryCapabilities(): ProbeCapabilities {
        val hasFeature = context.packageManager.hasSystemFeature(PackageManager.FEATURE_UWB)
        return try {
            val manager = UwbManager.createInstance(context)
            val scope = manager.controllerSessionScope()
            cachedScope = scope
            val caps = scope.rangingCapabilities
            val localHex = scope.localAddress.address.toShortAddressHex()
            val supportedConfigs = caps.supportedConfigIds.toList()

            ProbeCapabilities(
                backendName = name,
                featureFlagPresent = hasFeature,
                isAvailable = true,
                supportsDistance = caps.isDistanceSupported,
                supportsAzimuth = caps.isAzimuthalAngleSupported,
                supportsElevation = caps.isElevationAngleSupported,
                minRangingIntervalMs = caps.minRangingInterval,
                supportedChannels = caps.supportedChannels.toList(),
                supportedConfigIds = supportedConfigs,
                supportsMulticastDsTwr =
                    supportedConfigs.contains(RangingParameters.CONFIG_MULTICAST_DS_TWR),
                localAddressHex = localHex,
                assignedChannel = scope.uwbComplexChannel.channel,
                assignedPreambleIndex = scope.uwbComplexChannel.preambleIndex,
            )
        } catch (err: Exception) {
            ProbeCapabilities(
                backendName = name,
                featureFlagPresent = hasFeature,
                isAvailable = false,
                supportsDistance = false,
                supportsAzimuth = false,
                supportsElevation = false,
                errorDetail = "${err.javaClass.simpleName}: ${err.message}",
            )
        }
    }

    override suspend fun openControllerSession(
        sessionId: Int,
        preferredChannel: Int,
        preferredPreambleIndex: Int,
    ): PreparedControllerSession? {
        return try {
            val scope = cachedScope ?: UwbManager.createInstance(context).controllerSessionScope()
            cachedScope = scope
            val bytes = scope.localAddress.address
            if (bytes.size < 2) {
                return null
            }
            val shortAddr = ((bytes[0].toInt() and 0xFF) shl 8) or (bytes[1].toInt() and 0xFF)
            PreparedControllerSession(
                backendName = name,
                sessionId = sessionId,
                localAddressHex = bytes.toShortAddressHex() ?: "00:00",
                localControllerAddress = shortAddr,
                channel = scope.uwbComplexChannel.channel,
                preambleIndex = scope.uwbComplexChannel.preambleIndex,
            )
        } catch (_: Exception) {
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
        val countsByPeer = linkedMapOf<String, Int>()
        val epochBuckets = linkedMapOf<Long, MutableSet<String>>()
        var samplesCollected = 0
        var droppedSessions = 0
        var firstRangeMs: Long? = null
        var firstThreeAnchorMs: Long? = null
        var errorDetail: String? = null

        try {
            val scope = cachedScope ?: UwbManager.createInstance(context).controllerSessionScope()
            cachedScope = scope
            val peers = config.peers.map { UwbDevice(UwbAddress(it.addressBytes())) }
            val params =
                RangingParameters(
                    uwbConfigType = config.uwbConfigId,
                    sessionId = config.sessionId,
                    subSessionId = 0,
                    sessionKeyInfo = config.sessionKeyInfo,
                    subSessionKeyInfo = null,
                    complexChannel = UwbComplexChannel(config.channel, config.preambleIndex),
                    peerDevices = peers,
                    updateRateType = RangingParameters.RANGING_UPDATE_RATE_AUTOMATIC,
                )

            withTimeoutOrNull(durationMs) {
                scope.prepareSession(params).collect { result ->
                    when (result) {
                        is RangingResult.RangingResultPosition -> {
                            val peerHex = result.device.address.address.toShortAddressHex() ?: return@collect
                            val nowNanos = SystemClock.elapsedRealtimeNanos()
                            val relMs = ((nowNanos - startedNanos) / NANOS_PER_MS).coerceAtLeast(0L)
                            val sample =
                                ProbeRangingSample(
                                    backendName = name,
                                    elapsedRealtimeNanos = nowNanos,
                                    wallClockUnixMs = System.currentTimeMillis(),
                                    peerAddressHex = peerHex,
                                    distanceM = result.position.distance?.value,
                                    azimuthDeg = result.position.azimuth?.value,
                                    elevationDeg = result.position.elevation?.value,
                                )
                            samplesCollected += 1
                            countsByPeer[peerHex] = (countsByPeer[peerHex] ?: 0) + 1
                            if (firstRangeMs == null) {
                                firstRangeMs = relMs
                            }
                            val bucket = nowNanos / EPOCH_BUCKET_NANOS
                            val bucketPeers = epochBuckets.getOrPut(bucket) { linkedSetOf() }
                            bucketPeers.add(peerHex)
                            if (firstThreeAnchorMs == null && bucketPeers.size >= MIN_SOLVABLE_ANCHORS) {
                                firstThreeAnchorMs = relMs
                            }
                            onSample(sample)
                        }
                        is RangingResult.RangingResultPeerDisconnected -> {
                            droppedSessions += 1
                        }
                    }
                }
            }
        } catch (err: Exception) {
            errorDetail = "${err.javaClass.simpleName}: ${err.message}"
            droppedSessions += 1
        }

        val endedAtMs = System.currentTimeMillis()
        val elapsedS = ((endedAtMs - startedAtMs).coerceAtLeast(1L)) / 1000.0
        val rates = countsByPeer.mapValues { (_, count) -> count.toDouble() / elapsedS }

        return SessionSummary(
            backendName = name,
            sessionId = config.sessionId,
            startedAtUnixMs = startedAtMs,
            endedAtUnixMs = endedAtMs,
            samplesCollected = samplesCollected,
            distinctPeersAcquired = countsByPeer.size,
            timeToFirstRangeMs = firstRangeMs,
            timeToFirstThreeAnchorEpochMs = firstThreeAnchorMs,
            effectiveRateHzByPeer = rates,
            droppedSessions = droppedSessions,
            errorDetail = errorDetail,
        )
    }

    private fun ByteArray.toShortAddressHex(): String? =
        if (size >= 2) "%02X:%02X".format(this[0], this[1]) else null

    companion object {
        const val BACKEND_NAME: String = "jetpack_uwb"
        private const val NANOS_PER_MS: Long = 1_000_000L
        private const val EPOCH_BUCKET_NANOS: Long = 250_000_000L
        private const val MIN_SOLVABLE_ANCHORS: Int = 3
    }
}
