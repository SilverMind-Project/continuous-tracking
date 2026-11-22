package io.silvermind.wearuwbprobe.oob

import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Result of one sequential BLE GATT OOB parameter exchange with a single FiRa anchor. */
data class PeerGattExchangeOutcome(
    val succeeded: Boolean,
    val latencyMs: Long,
    val errorDetail: String? = null,
)

/** Hardware boundary for performing a FiRa BLE GATT OOB write to one anchor peer. */
interface FiRaBleGattPeerClient {
    suspend fun exchangeSessionParameters(
        peer: AnchorPeerSpec,
        configPayload: ByteArray,
    ): PeerGattExchangeOutcome
}

/**
 * Standard sequential 1-by-1 FiRa BLE GATT OOB negotiator used as the baseline comparison
 * against [PreSharedStsNegotiator] during latency benchmarking.
 */
class BleGattOobNegotiator(
    private val gattClient: FiRaBleGattPeerClient,
) : OobSessionNegotiator {
    override val strategy: OobStrategy = OobStrategy.BLE_GATT_SEQUENTIAL

    override suspend fun negotiate(request: OobNegotiationRequest): OobNegotiationResult {
        val payload = encodeGattSessionConfig(request)
        val negotiated = mutableListOf<String>()
        val failed = mutableListOf<String>()
        val errors = mutableListOf<String>()
        var totalLatencyMs = 0L

        for (peer in request.peers) {
            val addrHex = peer.normalizedAddressHex()
            val outcome = gattClient.exchangeSessionParameters(peer, payload)
            totalLatencyMs += outcome.latencyMs.coerceAtLeast(0L)
            if (outcome.succeeded) {
                negotiated.add(addrHex)
            } else {
                failed.add(addrHex)
                if (!outcome.errorDetail.isNullOrBlank()) {
                    errors.add("$addrHex:${outcome.errorDetail}")
                }
            }
        }

        return OobNegotiationResult(
            strategy = strategy,
            succeeded = negotiated.isNotEmpty() && failed.isEmpty(),
            setupLatencyMs = totalLatencyMs,
            peersNegotiated = negotiated,
            peersFailed = failed,
            wakePayloadHex = payload.joinToString(separator = "") { "%02X".format(it) },
            errorDetail = errors.takeIf { it.isNotEmpty() }?.joinToString(separator = "; "),
        )
    }

    companion object {
        /** FiRa BLE OOB service UUID used by standard Qorvo/NXP anchor firmware demos. */
        const val FIRA_OOB_SERVICE_UUID: String = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
        const val FIRA_OOB_RX_CHAR_UUID: String = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
        const val FIRA_OOB_TX_CHAR_UUID: String = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
        private const val GATT_CONFIG_OPCODE: Byte = 0x0B
        private const val GATT_PAYLOAD_BYTES: Int = 18

        /**
         * Encodes a compact 18-byte big-endian GATT session configuration payload:
         * `[opcode=0x0B, sessionId:u32, controllerAddr:u16, channel:u8, preamble:u8, burstS:u8, keyInfo:8B]`.
         */
        fun encodeGattSessionConfig(request: OobNegotiationRequest): ByteArray {
            val buf = ByteBuffer.allocate(GATT_PAYLOAD_BYTES).order(ByteOrder.BIG_ENDIAN)
            buf.put(GATT_CONFIG_OPCODE)
            buf.putInt(request.sessionId)
            buf.putShort(request.controllerAddress.toShort())
            buf.put(request.channel.toByte())
            buf.put(request.preambleIndex.toByte())
            buf.put(request.burstDurationS.toByte())
            buf.put(request.sessionKeyInfo)
            return buf.array()
        }
    }
}
