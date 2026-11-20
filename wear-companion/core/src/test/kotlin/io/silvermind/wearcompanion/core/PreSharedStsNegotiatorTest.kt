package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class PreSharedStsNegotiatorTest {
    @Test
    fun `wake payload encodes and decodes exact 10-byte big-endian frame`() {
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

        val decoded = PreSharedStsNegotiator.decodeWakePayload(encoded)
        assertEquals(frame, decoded)
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
