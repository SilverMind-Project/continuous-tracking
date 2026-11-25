package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class PreSharedStsNegotiatorTest {
    @Test
    fun `wake payload encodes and decodes exact 12-byte big-endian frame with magic prefix`() {
        val frame =
            UwbWakeFrame(
                sessionId = 0x12345678,
                controllerAddress = 0xABCD,
                channel = 9,
                preambleIndex = 10,
                burstDurationS = 12,
            )
        val encoded = PreSharedStsNegotiator.encodeWakePayload(frame)
        assertEquals(PreSharedStsNegotiator.WAKE_PAYLOAD_BYTES, encoded.size)

        assertArrayEquals(
            byteArrayOf(0x53, 0x4D, 0x01, 0x12, 0x34, 0x56, 0x78, 0xAB.toByte(), 0xCD.toByte(), 9, 10, 12),
            encoded,
        )

        val decoded = PreSharedStsNegotiator.decodeWakePayload(encoded)
        assertEquals(frame, decoded)
        assertEquals(0xFFFF, PreSharedStsNegotiator.WAKE_MANUFACTURER_ID)
    }

    @Test
    fun `decoder rejects frames without the magic prefix or with the wrong length`() {
        val frame =
            UwbWakeFrame(sessionId = 7, controllerAddress = 1, channel = 9, preambleIndex = 10, burstDurationS = 12)
        val encoded = PreSharedStsNegotiator.encodeWakePayload(frame)
        val wrongMagic = encoded.copyOf().also { it[0] = 0x00 }
        assertThrows(IllegalArgumentException::class.java) { PreSharedStsNegotiator.decodeWakePayload(wrongMagic) }
        assertThrows(IllegalArgumentException::class.java) {
            PreSharedStsNegotiator.decodeWakePayload(encoded.copyOfRange(2, encoded.size))
        }
    }

    @Test
    fun `rejects invalid channel, preamble, and anchor address formats`() {
        assertThrows(IllegalArgumentException::class.java) {
            UwbWakeFrame(
                sessionId = 1,
                controllerAddress = 1,
                channel = 7,
                preambleIndex = 10,
                burstDurationS = 12,
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            UwbAnchorEntry(anchorId = "a1", uwbAddressHex = "ZZZZ")
        }
    }
}
