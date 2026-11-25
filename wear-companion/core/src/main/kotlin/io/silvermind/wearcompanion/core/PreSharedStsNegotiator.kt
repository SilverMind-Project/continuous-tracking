package io.silvermind.wearcompanion.core

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One household UWB FiRa anchor synced from Cognitive Companion. */
data class UwbAnchorEntry(
    val anchorId: String,
    val uwbAddressHex: String,
    val channel: Int = DEFAULT_UWB_CHANNEL,
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

    /** 2-byte FiRa short address decoded from `XX:XX`. */
    fun addressBytes(): ByteArray {
        val parts = uwbAddressHex.trim().split(":")
        return byteArrayOf(
            parts[0].toInt(16).toByte(),
            parts[1].toInt(16).toByte(),
        )
    }

    companion object {
        const val DEFAULT_UWB_CHANNEL: Int = 9
        const val DEFAULT_PREAMBLE_INDEX: Int = 10
        const val MIN_PREAMBLE: Int = 9
        const val MAX_PREAMBLE: Int = 32
        val VALID_CHANNELS: Set<Int> = setOf(5, 9)
        private val ADDRESS_REGEX = Regex("^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$")
    }
}

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
        require(channel in UwbAnchorEntry.VALID_CHANNELS) { "channel must be 5 or 9, got $channel" }
        require(preambleIndex in UwbAnchorEntry.MIN_PREAMBLE..UwbAnchorEntry.MAX_PREAMBLE) {
            "preambleIndex out of FiRa range: $preambleIndex"
        }
        require(burstDurationS in 1..60) { "burstDurationS must be in 1..60, got $burstDurationS" }
    }
}

/**
 * Zero-GATT Pre-Shared Static STS broadcast-wake encoder and anchor directory.
 *
 * Encodes an exact 12-byte big-endian manufacturer-specific BLE wake frame,
 * advertised under [WAKE_MANUFACTURER_ID] so it never collides with the
 * Tier 1 iBeacon frame (Apple `0x004C`):
 * - bytes 0..1: magic `0x53 0x4D` ("SM"), because `0xFFFF` is the shared
 *   Bluetooth SIG test ID that other development devices also advertise
 * - byte 2: protocol version (`0x01`)
 * - bytes 3..6: uint32 sessionId
 * - bytes 7..8: uint16 controllerAddress
 * - byte 9: uint8 channel (`5` or `9`)
 * - byte 10: uint8 preambleIndex (`9..32`)
 * - byte 11: uint8 burstDurationS (`1..60`)
 */
class PreSharedStsNegotiator(
    val staticStsVendorId: ByteArray,
    val staticStsIv: ByteArray,
) {
    init {
        require(staticStsVendorId.size == VENDOR_ID_BYTES) {
            "staticStsVendorId must be $VENDOR_ID_BYTES bytes"
        }
        require(staticStsIv.size == STATIC_STS_IV_BYTES) {
            "staticStsIv must be $STATIC_STS_IV_BYTES bytes"
        }
    }

    /** Full 8-byte FiRa Static STS session key (`vendorId || staticStsIv`). */
    fun sessionKeyInfo(): ByteArray = staticStsVendorId + staticStsIv

    companion object {
        const val WAKE_PROTOCOL_VERSION: Int = 1
        const val WAKE_PAYLOAD_BYTES: Int = 12

        /** Bluetooth SIG test company ID; never Apple's iBeacon `0x004C`. */
        const val WAKE_MANUFACTURER_ID: Int = 0xFFFF
        const val WAKE_MAGIC_0: Byte = 0x53
        const val WAKE_MAGIC_1: Byte = 0x4D
        const val WAKE_BROADCAST_DURATION_MS: Long = 300L
        const val VENDOR_ID_BYTES: Int = 2
        const val STATIC_STS_IV_BYTES: Int = 6

        fun encodeWakePayload(frame: UwbWakeFrame): ByteArray {
            val buf = ByteBuffer.allocate(WAKE_PAYLOAD_BYTES).order(ByteOrder.BIG_ENDIAN)
            buf.put(WAKE_MAGIC_0)
            buf.put(WAKE_MAGIC_1)
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
            require(buf.get() == WAKE_MAGIC_0 && buf.get() == WAKE_MAGIC_1) {
                "Wake payload magic mismatch"
            }
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
