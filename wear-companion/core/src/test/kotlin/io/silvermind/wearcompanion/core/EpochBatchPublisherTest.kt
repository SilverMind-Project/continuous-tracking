package io.silvermind.wearcompanion.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class EpochBatchPublisherTest {
    private val bootUnixNs: Long = 1_794_650_000_000_000_000L

    private fun sample(
        seq: Long,
        elapsedSec: Long,
    ): RangingEpochSample =
        RangingEpochSample(
            epochSeq = seq,
            measuredAtElapsedNanos = elapsedSec * 1_000_000_000L,
            ranges =
                listOf(
                    AnchorRangeReading(
                        anchorId = "living-room-uwb-1",
                        uwbAddressHex = "00:01",
                        distanceM = 2.41,
                        azimuthDeg = -14.2,
                        elevationDeg = 8.5,
                        losConfidence = 0.94,
                    ),
                ),
        )

    @Test
    fun `preserves monotonic measured_at_unix_ns across micro-batches and emits valid device_ranging v1 json`() {
        val sent = mutableListOf<Pair<String, String>>()
        val publisher =
            EpochBatchPublisher(
                tagId = "watch-sriram-pw5",
                bootUnixNs = bootUnixNs,
                sender = { topic, payload ->
                    sent.add(topic to payload)
                    true
                },
            )

        publisher.markBurstStarted(0L)
        for (i in 1L..3L) {
            publisher.enqueue(sample(i, i))
        }
        assertTrue(publisher.shouldFlush(3_000_000_000L))
        val batch1 =
            publisher.flush(
                nowElapsedNanos = 3_000_000_000L,
                wearState = WearStateWire.ON_BODY,
                motionState = MotionStateWire.MOVING,
                batteryFraction = 0.78,
            )
        assertNotNull(batch1)
        assertEquals(0, publisher.bufferedCount())

        // Enqueue two samples with identical elapsedNanos -> timestamps remain strictly monotonic.
        publisher.enqueue(sample(4, 4))
        publisher.enqueue(sample(5, 4))
        val batch2 =
            publisher.flush(
                nowElapsedNanos = 6_000_000_000L,
                wearState = WearStateWire.ON_BODY,
                motionState = MotionStateWire.MOVING,
                batteryFraction = 0.77,
            )
        assertNotNull(batch2)
        assertEquals(2, sent.size)
        assertEquals("rf/devices/watch-sriram-pw5/ranges", sent[0].first)

        val root1 = Json.parseToJsonElement(sent[0].second).jsonObject
        val root2 = Json.parseToJsonElement(sent[1].second).jsonObject
        assertEquals("device_ranging.v1", root1["schema"]?.jsonPrimitive?.content)
        assertEquals("watch-sriram-pw5", root1["tag_id"]?.jsonPrimitive?.content)
        assertEquals("on_body", root1["wear_state"]?.jsonPrimitive?.content)
        assertEquals("moving", root1["motion_state"]?.jsonPrimitive?.content)
        assertEquals(0.78, root1["battery_fraction"]?.jsonPrimitive?.double)

        val allTimestamps =
            (root1["epochs"]!!.jsonArray + root2["epochs"]!!.jsonArray).map {
                it.jsonObject["measured_at_unix_ns"]!!.jsonPrimitive.long
            }
        assertEquals(5, allTimestamps.size)
        for (idx in 1 until allTimestamps.size) {
            assertTrue(allTimestamps[idx] > allTimestamps[idx - 1])
        }
    }

    @Test
    fun `drops oldest epochs cleanly when wifi is unreachable beyond 60-epoch ring capacity`() {
        var wifiUp = false
        val sent = mutableListOf<String>()
        val publisher =
            EpochBatchPublisher(
                tagId = "watch-pw5",
                bootUnixNs = bootUnixNs,
                sender = { _, payload ->
                    if (!wifiUp) {
                        false
                    } else {
                        sent.add(payload)
                        true
                    }
                },
                maxBufferedEpochs = 60,
            )

        for (seq in 1L..75L) {
            publisher.enqueue(sample(seq, seq))
            val failed =
                publisher.flush(
                    nowElapsedNanos = seq * 1_000_000_000L,
                    wearState = WearStateWire.ON_BODY,
                    motionState = MotionStateWire.MOVING,
                )
            assertNull(failed)
        }
        assertEquals(60, publisher.bufferedCount())
        assertEquals(15, publisher.droppedOldestCount)

        wifiUp = true
        val recovered =
            publisher.flush(
                nowElapsedNanos = 76_000_000_000L,
                wearState = WearStateWire.ON_BODY,
                motionState = MotionStateWire.MOVING,
            )
        assertNotNull(recovered)
        val epochs = Json.parseToJsonElement(sent.single()).jsonObject["epochs"]!!.jsonArray
        assertEquals(60, epochs.size)
        assertEquals(16L, epochs.first().jsonObject["epoch_seq"]!!.jsonPrimitive.long)
        assertEquals(75L, epochs.last().jsonObject["epoch_seq"]!!.jsonPrimitive.long)
    }
}
