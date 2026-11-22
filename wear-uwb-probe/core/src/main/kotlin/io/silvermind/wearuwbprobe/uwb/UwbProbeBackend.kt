package io.silvermind.wearuwbprobe.uwb

import kotlinx.serialization.Serializable

/** Capability snapshot reported by a single UWB/ranging provider backend on the watch. */
@Serializable
data class ProbeCapabilities(
    val backendName: String,
    val featureFlagPresent: Boolean,
    val isAvailable: Boolean,
    val supportsDistance: Boolean,
    val supportsAzimuth: Boolean,
    val supportsElevation: Boolean,
    val minRangingIntervalMs: Int? = null,
    val supportedChannels: List<Int> = emptyList(),
    val supportedConfigIds: List<Int> = emptyList(),
    val supportsMulticastDsTwr: Boolean = false,
    val rawStatusCode: Int? = null,
    val errorDetail: String? = null,
    val localAddressHex: String? = null,
    val assignedChannel: Int? = null,
    val assignedPreambleIndex: Int? = null,
    val platformTechnologies: List<String> = emptyList(),
) {
    /** True when the backend opened a controller session and returned local address and channel. */
    fun passesControllerGate(): Boolean =
        isAvailable &&
            !localAddressHex.isNullOrBlank() &&
            assignedChannel != null &&
            assignedPreambleIndex != null &&
            rawStatusCode == null
}

/** Single per-peer ranging measurement captured from a UWB backend. */
@Serializable
data class ProbeRangingSample(
    val backendName: String,
    val elapsedRealtimeNanos: Long,
    val wallClockUnixMs: Long,
    val peerAddressHex: String,
    val distanceM: Float? = null,
    val distanceConfidence: Float? = null,
    val azimuthDeg: Float? = null,
    val azimuthConfidence: Float? = null,
    val elevationDeg: Float? = null,
    val elevationConfidence: Float? = null,
    val rssiDbm: Int? = null,
)

/** Static FiRa short-address anchor descriptor for bench sessions. */
@Serializable
data class AnchorPeerSpec(
    val anchorId: String,
    val uwbAddressHex: String,
    val channel: Int = DEFAULT_CHANNEL,
    val preambleIndex: Int = DEFAULT_PREAMBLE_INDEX,
) {
    init {
        require(anchorId.isNotBlank()) { "anchorId must be non-blank" }
        require(ADDRESS_REGEX.matches(uwbAddressHex.trim())) {
            "uwbAddressHex must match XX:XX hexadecimal format, got '$uwbAddressHex'"
        }
        require(channel in VALID_CHANNELS) { "channel must be 5 or 9, got $channel" }
        require(preambleIndex in MIN_PREAMBLE..MAX_PREAMBLE) {
            "preambleIndex must be in $MIN_PREAMBLE..$MAX_PREAMBLE, got $preambleIndex"
        }
    }

    /** Normalized uppercase `XX:XX` address string. */
    fun normalizedAddressHex(): String = uwbAddressHex.trim().uppercase()

    /** 2-byte FiRa short address decoded from `XX:XX`. */
    fun addressBytes(): ByteArray {
        val parts = uwbAddressHex.trim().split(":")
        return byteArrayOf(
            parts[0].toInt(16).toByte(),
            parts[1].toInt(16).toByte(),
        )
    }

    companion object {
        const val DEFAULT_CHANNEL: Int = 9
        const val DEFAULT_PREAMBLE_INDEX: Int = 10
        const val MIN_PREAMBLE: Int = 9
        const val MAX_PREAMBLE: Int = 32
        val VALID_CHANNELS: Set<Int> = setOf(5, 9)
        private val ADDRESS_REGEX = Regex("^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$")
    }
}

/** Configuration passed to [UwbProbeBackend.runMulticastSession]. */
data class MulticastSessionConfig(
    val sessionId: Int,
    val uwbConfigId: Int = CONFIG_MULTICAST_DS_TWR,
    val channel: Int = AnchorPeerSpec.DEFAULT_CHANNEL,
    val preambleIndex: Int = AnchorPeerSpec.DEFAULT_PREAMBLE_INDEX,
    val sessionKeyInfo: ByteArray,
    val peers: List<AnchorPeerSpec>,
    val useDynamicAddControlee: Boolean = false,
) {
    init {
        require(sessionKeyInfo.size == SESSION_KEY_INFO_BYTES) {
            "sessionKeyInfo must be $SESSION_KEY_INFO_BYTES bytes, got ${sessionKeyInfo.size}"
        }
        require(peers.isNotEmpty()) { "peers must be non-empty" }
        require(channel in AnchorPeerSpec.VALID_CHANNELS) { "channel must be 5 or 9, got $channel" }
        require(preambleIndex in AnchorPeerSpec.MIN_PREAMBLE..AnchorPeerSpec.MAX_PREAMBLE) {
            "preambleIndex out of FiRa range: $preambleIndex"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MulticastSessionConfig) return false
        return sessionId == other.sessionId &&
            uwbConfigId == other.uwbConfigId &&
            channel == other.channel &&
            preambleIndex == other.preambleIndex &&
            sessionKeyInfo.contentEquals(other.sessionKeyInfo) &&
            peers == other.peers &&
            useDynamicAddControlee == other.useDynamicAddControlee
    }

    override fun hashCode(): Int {
        var result = sessionId
        result = 31 * result + uwbConfigId
        result = 31 * result + channel
        result = 31 * result + preambleIndex
        result = 31 * result + sessionKeyInfo.contentHashCode()
        result = 31 * result + peers.hashCode()
        result = 31 * result + useDynamicAddControlee.hashCode()
        return result
    }

    companion object {
        const val CONFIG_UNICAST_DS_TWR: Int = 1
        const val CONFIG_MULTICAST_DS_TWR: Int = 2
        const val SESSION_KEY_INFO_BYTES: Int = 8
    }
}

/** HAL-assigned controller parameters returned before OOB wake negotiation. */
data class PreparedControllerSession(
    val backendName: String,
    val sessionId: Int,
    val localAddressHex: String,
    val localControllerAddress: Int,
    val channel: Int,
    val preambleIndex: Int,
)

/** Summary metrics produced by one multicast ranging session run. */
@Serializable
data class SessionSummary(
    val backendName: String,
    val sessionId: Int,
    val startedAtUnixMs: Long,
    val endedAtUnixMs: Long,
    val samplesCollected: Int,
    val distinctPeersAcquired: Int,
    val timeToFirstRangeMs: Long? = null,
    val timeToFirstThreeAnchorEpochMs: Long? = null,
    val effectiveRateHzByPeer: Map<String, Double> = emptyMap(),
    val droppedSessions: Int = 0,
    val rawStatusCode: Int? = null,
    val errorDetail: String? = null,
)

/** Strategy interface isolating Wear OS UWB provider APIs from probe orchestration and tests. */
interface UwbProbeBackend {
    val name: String

    suspend fun queryCapabilities(): ProbeCapabilities

    suspend fun openControllerSession(
        sessionId: Int,
        preferredChannel: Int = AnchorPeerSpec.DEFAULT_CHANNEL,
        preferredPreambleIndex: Int = AnchorPeerSpec.DEFAULT_PREAMBLE_INDEX,
    ): PreparedControllerSession?

    suspend fun runMulticastSession(
        config: MulticastSessionConfig,
        durationMs: Long,
        onSample: (ProbeRangingSample) -> Unit,
    ): SessionSummary
}

/** Deterministic in-memory [UwbProbeBackend] test double for JVM unit tests. */
class FakeUwbProbeBackend(
    override val name: String = "fake_uwb",
    private val capabilities: ProbeCapabilities =
        ProbeCapabilities(
            backendName = name,
            featureFlagPresent = true,
            isAvailable = true,
            supportsDistance = true,
            supportsAzimuth = true,
            supportsElevation = false,
            minRangingIntervalMs = 200,
            supportedChannels = listOf(9),
            supportedConfigIds = listOf(1, 2),
            supportsMulticastDsTwr = true,
            localAddressHex = "AA:BB",
            assignedChannel = 9,
            assignedPreambleIndex = 10,
        ),
    private val scriptedSamples: List<ProbeRangingSample> = emptyList(),
    private val sessionErrorStatusCode: Int? = null,
    private val sessionErrorDetail: String? = null,
) : UwbProbeBackend {
    val recordedConfigs: MutableList<MulticastSessionConfig> = mutableListOf()

    override suspend fun queryCapabilities(): ProbeCapabilities = capabilities

    override suspend fun openControllerSession(
        sessionId: Int,
        preferredChannel: Int,
        preferredPreambleIndex: Int,
    ): PreparedControllerSession? {
        if (!capabilities.passesControllerGate()) {
            return null
        }
        val hex = capabilities.localAddressHex ?: return null
        val parts = hex.split(":")
        val shortAddr = ((parts[0].toInt(16) and 0xFF) shl 8) or (parts[1].toInt(16) and 0xFF)
        return PreparedControllerSession(
            backendName = name,
            sessionId = sessionId,
            localAddressHex = hex,
            localControllerAddress = shortAddr,
            channel = capabilities.assignedChannel ?: preferredChannel,
            preambleIndex = capabilities.assignedPreambleIndex ?: preferredPreambleIndex,
        )
    }

    override suspend fun runMulticastSession(
        config: MulticastSessionConfig,
        durationMs: Long,
        onSample: (ProbeRangingSample) -> Unit,
    ): SessionSummary {
        recordedConfigs.add(config)
        val startMs = scriptedSamples.firstOrNull()?.wallClockUnixMs ?: 1_000L
        if (sessionErrorStatusCode != null) {
            return SessionSummary(
                backendName = name,
                sessionId = config.sessionId,
                startedAtUnixMs = startMs,
                endedAtUnixMs = startMs + durationMs,
                samplesCollected = 0,
                distinctPeersAcquired = 0,
                droppedSessions = 1,
                rawStatusCode = sessionErrorStatusCode,
                errorDetail = sessionErrorDetail,
            )
        }

        val countsByPeer = linkedMapOf<String, Int>()
        val epochBuckets = linkedMapOf<Long, MutableSet<String>>()
        var firstRangeMs: Long? = null
        var firstThreeAnchorMs: Long? = null

        for (sample in scriptedSamples) {
            onSample(sample)
            countsByPeer[sample.peerAddressHex] = (countsByPeer[sample.peerAddressHex] ?: 0) + 1
            val relMs = (sample.wallClockUnixMs - startMs).coerceAtLeast(0L)
            if (firstRangeMs == null) {
                firstRangeMs = relMs
            }
            val bucket = sample.elapsedRealtimeNanos / EPOCH_BUCKET_NANOS
            val bucketPeers = epochBuckets.getOrPut(bucket) { linkedSetOf() }
            bucketPeers.add(sample.peerAddressHex)
            if (firstThreeAnchorMs == null && bucketPeers.size >= MIN_SOLVABLE_ANCHORS) {
                firstThreeAnchorMs = relMs
            }
        }

        val endMs = scriptedSamples.lastOrNull()?.wallClockUnixMs ?: (startMs + durationMs)
        val elapsedS = ((endMs - startMs).coerceAtLeast(durationMs).coerceAtLeast(1L)) / 1000.0
        val rates = countsByPeer.mapValues { (_, count) -> count.toDouble() / elapsedS }

        return SessionSummary(
            backendName = name,
            sessionId = config.sessionId,
            startedAtUnixMs = startMs,
            endedAtUnixMs = endMs,
            samplesCollected = scriptedSamples.size,
            distinctPeersAcquired = countsByPeer.size,
            timeToFirstRangeMs = firstRangeMs,
            timeToFirstThreeAnchorEpochMs = firstThreeAnchorMs,
            effectiveRateHzByPeer = rates,
        )
    }

    companion object {
        private const val EPOCH_BUCKET_NANOS: Long = 250_000_000L
        private const val MIN_SOLVABLE_ANCHORS: Int = 3
    }
}
