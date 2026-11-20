package io.silvermind.wearcompanion.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

/**
 * Standard 23-byte iBeacon manufacturer-specific payload carried under Apple
 * company identifier `0x004C` in a non-connectable BLE advertisement.
 *
 * Layout (big-endian):
 * - bytes 0..1: type `0x02`, remaining length `0x15` (21 bytes)
 * - bytes 2..17: 16-byte proximity UUID
 * - bytes 18..19: uint16 major
 * - bytes 20..21: uint16 minor
 * - byte 22: int8 calibrated RSSI at 1 m (txPowerDbm)
 */
object IBeaconFrameEncoder {
    const val APPLE_COMPANY_ID: Int = 0x004C
    const val IBEACON_TYPE: Byte = 0x02
    const val IBEACON_LENGTH: Byte = 0x15
    const val PAYLOAD_SIZE_BYTES: Int = 23
    const val DEFAULT_TX_POWER_DBM: Int = -59

    data class DecodedIBeacon(
        val identifier: BeaconIdentifier,
        val txPowerDbm: Int,
    )

    fun encode(
        identifier: BeaconIdentifier,
        txPowerDbm: Int = DEFAULT_TX_POWER_DBM,
    ): ByteArray {
        require(txPowerDbm in Byte.MIN_VALUE..Byte.MAX_VALUE) {
            "txPowerDbm must fit in signed int8, got $txPowerDbm"
        }
        val buf = ByteBuffer.allocate(PAYLOAD_SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
        buf.put(IBEACON_TYPE)
        buf.put(IBEACON_LENGTH)
        buf.putLong(identifier.uuid.mostSignificantBits)
        buf.putLong(identifier.uuid.leastSignificantBits)
        buf.putShort(identifier.major.toShort())
        buf.putShort(identifier.minor.toShort())
        buf.put(txPowerDbm.toByte())
        return buf.array()
    }

    fun decode(payload: ByteArray): DecodedIBeacon {
        require(payload.size == PAYLOAD_SIZE_BYTES) {
            "iBeacon payload must be $PAYLOAD_SIZE_BYTES bytes, got ${payload.size}"
        }
        require(payload[0] == IBEACON_TYPE && payload[1] == IBEACON_LENGTH) {
            "Invalid iBeacon header bytes: 0x%02x 0x%02x".format(payload[0], payload[1])
        }
        val buf = ByteBuffer.wrap(payload).order(ByteOrder.BIG_ENDIAN)
        buf.position(2)
        val msb = buf.long
        val lsb = buf.long
        val major = buf.short.toInt() and 0xFFFF
        val minor = buf.short.toInt() and 0xFFFF
        val txPower = buf.get().toInt()
        return DecodedIBeacon(
            identifier = BeaconIdentifier(uuid = UUID(msb, lsb), major = major, minor = minor),
            txPowerDbm = txPower,
        )
    }
}
