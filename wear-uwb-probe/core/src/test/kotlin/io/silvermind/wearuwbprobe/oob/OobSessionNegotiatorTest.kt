package io.silvermind.wearuwbprobe.oob

import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class OobSessionNegotiatorTest {
    private val vendorId = byteArrayOf(0x53, 0x4D)
    private val staticIv = byteArrayOf(0x01, 0x02, 0x03, 0x04, 0x05, 0x06)
    private val peers =
        listOf(
            AnchorPeerSpec("anchor-1", "00:01"),
            AnchorPeerSpec("anchor-2", "00:02"),
            AnchorPeerSpec("anchor-3", "00:03"),
            AnchorPeerSpec("anchor-4", "00:04"),
        )

    @Test
    fun `PreSharedStsNegotiator encodes exact 10-byte big-endian 0x01 wake payload`() =
        runTest {
            var capturedManufacturerId = -1
            var capturedPayload: ByteArray = byteArrayOf()
            val fakeAdvertiser =
                object : BleWakeAdvertiser {
                    override suspend fun broadcastWakePayload(
                        manufacturerId: Int,
                        payload: ByteArray,
                        durationMs: Long,
                    ): Long {
                        capturedManufacturerId = manufacturerId
                        capturedPayload = payload
                        return 210L
                    }
                }

            val negotiator = PreSharedStsNegotiator(vendorId, staticIv, fakeAdvertiser)
            val request =
                OobNegotiationRequest(
                    sessionId = 0x11223344,
                    controllerAddress = 0xAABB,
                    channel = 9,
                    preambleIndex = 10,
                    burstDurationS = 12,
                    peers = peers,
                    sessionKeyInfo = negotiator.sessionKeyInfo(),
                )

            val result = negotiator.negotiate(request)
            assertTrue(result.succeeded)
            assertEquals(OobStrategy.PRESHARED_STS_BROADCAST, result.strategy)
            assertEquals(210L, result.setupLatencyMs)
            assertEquals(4, result.peersNegotiated.size)
            assertEquals(PreSharedStsNegotiator.MANUFACTURER_ID, capturedManufacturerId)
            assertEquals(PreSharedStsNegotiator.WAKE_PAYLOAD_BYTES, capturedPayload.size)

            val expectedBytes =
                byteArrayOf(
                    0x01,
                    0x11,
                    0x22,
                    0x33,
                    0x44,
                    0xAA.toByte(),
                    0xBB.toByte(),
                    0x09,
                    0x0A,
                    0x0C,
                )
            assertArrayEquals(expectedBytes, capturedPayload)

            val decoded = PreSharedStsNegotiator.decodeWakePayload(capturedPayload)
            assertEquals(1, decoded.protocolVersion)
            assertEquals(0x11223344, decoded.sessionId)
            assertEquals(0xAABB, decoded.controllerAddress)
            assertEquals(9, decoded.channel)
            assertEquals(10, decoded.preambleIndex)
            assertEquals(12, decoded.burstDurationS)
        }

    @Test
    fun `BleGattOobNegotiator aggregates sequential per-peer latency and reports partial failures`() =
        runTest {
            val fakeClient =
                object : FiRaBleGattPeerClient {
                    override suspend fun exchangeSessionParameters(
                        peer: AnchorPeerSpec,
                        configPayload: ByteArray,
                    ): PeerGattExchangeOutcome {
                        assertEquals(18, configPayload.size)
                        return if (peer.anchorId == "anchor-3") {
                            PeerGattExchangeOutcome(succeeded = false, latencyMs = 950L, errorDetail = "timeout")
                        } else {
                            PeerGattExchangeOutcome(succeeded = true, latencyMs = 620L)
                        }
                    }
                }

            val negotiator = BleGattOobNegotiator(fakeClient)
            val request =
                OobNegotiationRequest(
                    sessionId = 42,
                    controllerAddress = 0x1234,
                    channel = 9,
                    preambleIndex = 10,
                    burstDurationS = 12,
                    peers = peers,
                    sessionKeyInfo = vendorId + staticIv,
                )

            val result = negotiator.negotiate(request)
            assertFalse(result.succeeded)
            assertEquals(OobStrategy.BLE_GATT_SEQUENTIAL, result.strategy)
            assertEquals(620L * 3 + 950L, result.setupLatencyMs)
            assertEquals(listOf("00:01", "00:02", "00:04"), result.peersNegotiated)
            assertEquals(listOf("00:03"), result.peersFailed)
        }
}
