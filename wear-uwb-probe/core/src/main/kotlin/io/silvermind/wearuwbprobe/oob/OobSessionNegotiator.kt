package io.silvermind.wearuwbprobe.oob

import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import kotlinx.serialization.Serializable

/** Out-of-band session setup strategy compared in bench latency trials. */
@Serializable
enum class OobStrategy {
    PRESHARED_STS_BROADCAST,
    BLE_GATT_SEQUENTIAL,
}

/** Input parameters for out-of-band UWB session negotiation. */
data class OobNegotiationRequest(
    val sessionId: Int,
    val controllerAddress: Int,
    val channel: Int,
    val preambleIndex: Int,
    val burstDurationS: Int,
    val peers: List<AnchorPeerSpec>,
    val sessionKeyInfo: ByteArray,
) {
    init {
        require(controllerAddress in 0..0xFFFF) { "controllerAddress must fit in uint16" }
        require(channel in AnchorPeerSpec.VALID_CHANNELS) { "channel must be 5 or 9, got $channel" }
        require(preambleIndex in AnchorPeerSpec.MIN_PREAMBLE..AnchorPeerSpec.MAX_PREAMBLE) {
            "preambleIndex out of FiRa range: $preambleIndex"
        }
        require(burstDurationS in 1..60) { "burstDurationS must be in 1..60, got $burstDurationS" }
        require(peers.isNotEmpty()) { "peers must be non-empty" }
        require(sessionKeyInfo.size == SESSION_KEY_INFO_BYTES) {
            "sessionKeyInfo must be $SESSION_KEY_INFO_BYTES bytes"
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is OobNegotiationRequest) return false
        return sessionId == other.sessionId &&
            controllerAddress == other.controllerAddress &&
            channel == other.channel &&
            preambleIndex == other.preambleIndex &&
            burstDurationS == other.burstDurationS &&
            peers == other.peers &&
            sessionKeyInfo.contentEquals(other.sessionKeyInfo)
    }

    override fun hashCode(): Int {
        var result = sessionId
        result = 31 * result + controllerAddress
        result = 31 * result + channel
        result = 31 * result + preambleIndex
        result = 31 * result + burstDurationS
        result = 31 * result + peers.hashCode()
        result = 31 * result + sessionKeyInfo.contentHashCode()
        return result
    }

    companion object {
        private const val SESSION_KEY_INFO_BYTES: Int = 8
    }
}

/** Outcome of one out-of-band session negotiation attempt. */
@Serializable
data class OobNegotiationResult(
    val strategy: OobStrategy,
    val succeeded: Boolean,
    val setupLatencyMs: Long,
    val peersNegotiated: List<String>,
    val peersFailed: List<String> = emptyList(),
    val wakePayloadHex: String? = null,
    val errorDetail: String? = null,
)

/** Common interface for zero-GATT broadcast wake and sequential BLE GATT OOB negotiation. */
interface OobSessionNegotiator {
    val strategy: OobStrategy

    suspend fun negotiate(request: OobNegotiationRequest): OobNegotiationResult
}
