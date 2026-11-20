package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.util.UUID

class IBeaconFrameEncoderTest {
    @Test
    fun `encoded bytes match standard 23-byte iBeacon layout for a known identifier`() {
        val uuid = UUID.fromString("e2c56db5-dffb-48d2-b060-d0f5a71096e0")
        val id = BeaconIdentifier(uuid = uuid, major = 0x1234, minor = 0x5678)
        val encoded = IBeaconFrameEncoder.encode(id, txPowerDbm = -59)

        val expected =
            byteArrayOf(
                0x02,
                0x15,
                0xe2.toByte(),
                0xc5.toByte(),
                0x6d,
                0xb5.toByte(),
                0xdf.toByte(),
                0xfb.toByte(),
                0x48,
                0xd2.toByte(),
                0xb0.toByte(),
                0x60,
                0xd0.toByte(),
                0xf5.toByte(),
                0xa7.toByte(),
                0x10,
                0x96.toByte(),
                0xe0.toByte(),
                0x12,
                0x34,
                0x56,
                0x78,
                (-59).toByte(),
            )
        assertEquals(IBeaconFrameEncoder.PAYLOAD_SIZE_BYTES, encoded.size)
        assertArrayEquals(expected, encoded)

        val decoded = IBeaconFrameEncoder.decode(encoded)
        assertEquals(id, decoded.identifier)
        assertEquals(-59, decoded.txPowerDbm)
    }
}
