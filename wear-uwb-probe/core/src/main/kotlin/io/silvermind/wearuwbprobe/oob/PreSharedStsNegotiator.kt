package io.silvermind.wearuwbprobe.oob

import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Parameters carried in the 300 ms zero-GATT BLE broadcast wake payload. */
data class UwbWakeFrame(
    val sessionId: Int,
    val controllerAddress: Int,
    val channel: Int,
    val preambleIndex: Int,
    val burstDurationS: Int,
    val protocolVersion: Int = PreSharedStsNegotiator.WAKE_PROTOCOL_VERSION,
) {
    init {
        require(protocolVersion in 1..255) { "protocolVersion must fit in uint8" }
        require(controllerAddress in 0..0xFFFF) { "controllerAddress must fit in uint16" }
        require(channel in AnchorPeerSpec.VALID_CHANNELS) { "channel must be 5 or 9, got $channel" }
        require(preambleIndex in AnchorPeerSpec.MIN_PREAMBLE..AnchorPeerSpec.MAX_PREAMBLE) {
            "preambleIndex out of FiRa range: $preambleIndex"
        }
        require(burstDurationS in 1..60) { "burstDurationS must be in 1..60, got $burstDurationS" }
    }
}

/** Hardware boundary for transmitting a non-connectable manufacturer-specific BLE wake burst. */
interface BleWakeAdvertiser {
    /** Advertises [payload] under [manufacturerId] for [durationMs] and returns measured duration in ms. */
    suspend fun broadcastWakePayload(
        manufacturerId: Int,
        payload: ByteArray,
        durationMs: Long,
    ): Long
}

/**
 * Zero-GATT Pre-Shared Static STS broadcast-wake negotiator.
 *
 * Encodes the exact 10-byte big-endian manufacturer-specific BLE wake frame shared with
 * the companion app and USB-powered FiRa room anchors:
 * - byte 0: protocol version (`0x01`)
 * - bytes 1..4: uint32 sessionId
 * - bytes 5..6: uint16 controllerAddress
 * - byte 7: uint8 channel (`5` or `9`)
 * - byte 8: uint8 preambleIndex (`9..32`)
 * - byte 9: uint8 burstDurationS (`1..60`)
 */
class PreSharedStsNegotiator(
    val staticStsVendorId: ByteArray,
    val staticStsIv: ByteArray,
    private val advertiser: BleWakeAdvertiser,
    private val broadcastDurationMs: Long = WAKE_BROADCAST_DURATION_MS,
) : OobSessionNegotiator {
    init {
        require(staticStsVendorId.size == VENDOR_ID_BYTES) {
            "staticStsVendorId must be $VENDOR_ID_BYTES bytes"
        }
        require(staticStsIv.size == STATIC_STS_IV_BYTES) {
            "staticStsIv must be $STATIC_STS_IV_BYTES bytes"
        }
        require(broadcastDurationMs > 0L) { "broadcastDurationMs must be > 0" }
    }

    override val strategy: OobStrategy = OobStrategy.PRESHARED_STS_BROADCAST

    /** Full 8-byte FiRa Static STS session key (`vendorId || staticStsIv`). */
    fun sessionKeyInfo(): ByteArray = staticStsVendorId + staticStsIv

    override suspend fun negotiate(request: OobNegotiationRequest): OobNegotiationResult {
        val frame =
            UwbWakeFrame(
                sessionId = request.sessionId,
                controllerAddress = request.controllerAddress,
                channel = request.channel,
                preambleIndex = request.preambleIndex,
                burstDurationS = request.burstDurationS,
            )
        val payload = encodeWakePayload(frame)
        val payloadHex = payload.joinToString(separator = "") { "%02X".format(it) }
        return try {
            val elapsedMs =
                advertiser.broadcastWakePayload(
                    manufacturerId = MANUFACTURER_ID,
                    payload = payload,
                    durationMs = broadcastDurationMs,
                )
            OobNegotiationResult(
                strategy = strategy,
                succeeded = true,
                setupLatencyMs = elapsedMs,
                peersNegotiated = request.peers.map { it.normalizedAddressHex() },
                wakePayloadHex = payloadHex,
            )
        } catch (err: IllegalStateException) {
            OobNegotiationResult(
                strategy = strategy,
                succeeded = false,
                setupLatencyMs = 0L,
                peersNegotiated = emptyList(),
                peersFailed = request.peers.map { it.normalizedAddressHex() },
                wakePayloadHex = payloadHex,
                errorDetail = err.message ?: "BLE wake broadcast failed",
            )
        }
    }

    companion object {
        const val MANUFACTURER_ID: Int = 0xFFFF
        const val WAKE_PROTOCOL_VERSION: Int = 1
        const val WAKE_PAYLOAD_BYTES: Int = 10
        const val WAKE_BROADCAST_DURATION_MS: Long = 300L
        const val VENDOR_ID_BYTES: Int = 2
        const val STATIC_STS_IV_BYTES: Int = 6

        fun encodeWakePayload(frame: UwbWakeFrame): ByteArray {
            val buf = ByteBuffer.allocate(WAKE_PAYLOAD_BYTES).order(ByteOrder.BIG_ENDIAN)
            buf.put(frame.protocolVersion.toByte())
            buf.putInt(frame.sessionId)
            buf.putShort(frame.controllerAddress.toShort())
            buf.put(frame.channel.toByte())
            buf.put(frame.preambleIndex.toByte())
            buf.put(frame.burstDurationS.toByte())
            return buf.array()
        }

        fun decodeWakePayload(payload: ByteArray): UwbWakeFrame {
            require(payload.size == WAKE_PAYLOAD_BYTES) {
                "Wake payload must be $WAKE_PAYLOAD_BYTES bytes, got ${payload.size}"
            }
            val buf = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
            val version = buf.get().toInt() and 0xFF
            require(version == WAKE_PROTOCOL_VERSION) {
                "Unsupported wake protocol version: $version"
            }
            val sessionId = buf.int
            val controllerAddress = buf.short.toInt() and 0xFFFF
            val channel = buf.get().toInt() and 0xFF
            val preambleIndex = buf.get().toInt() and 0xFF
            val burstDurationS = buf.get().toInt() and 0xFF
            return UwbWakeFrame(
                sessionId = sessionId,
                controllerAddress = controllerAddress,
                channel = channel,
                preambleIndex = preambleIndex,
                burstDurationS = burstDurationS,
                protocolVersion = version,
            )
        }
    }
}
