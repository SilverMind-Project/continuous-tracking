package io.silvermind.wearcompanion.core

import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

private class RecordingBleSink : BleAdvertiserSink {
    val beaconPayloads = mutableListOf<ByteArray>()
    val wakePayloads = mutableListOf<Pair<ByteArray, Long>>()
    var advertising: Boolean = false
        private set

    override fun startIBeacon(payload: ByteArray) {
        advertising = true
        beaconPayloads.add(payload)
    }

    override fun stopAdvertising() {
        advertising = false
    }

    override fun broadcastWakeFrame(
        payload: ByteArray,
        durationMs: Long,
    ) {
        wakePayloads.add(payload to durationMs)
    }
}

class CompanionCoordinatorTest {
    private fun sec(s: Double): Long = (s * 1_000_000_000.0).toLong()

    @Test
    fun `ble-only default mode advertises on-body and never wakes uwb engine`() =
        runTest {
            val sink = RecordingBleSink()
            val fakeEngine = FakeUwbRangingEngine()
            val coordinator =
                CompanionCoordinator(
                    identifierStore = InMemoryIdentifierStore(),
                    bleSink = sink,
                    mode = RuntimeMode.BLE_ONLY,
                    uwbEngine = fakeEngine,
                )

            coordinator.pair()
            coordinator.updateBodyState(sec(0.0), isOnBody = true)
            assertTrue(sink.advertising)
            assertEquals(BeaconState.ADVERTISING, coordinator.snapshot().beaconState)

            coordinator.onLinearAcceleration(sec(1.0), 0.5)
            coordinator.onLinearAcceleration(sec(1.2), 3.5)
            coordinator.onStepDetected(sec(1.0), batteryPercent = 90)
            coordinator.onStepDetected(sec(1.5), batteryPercent = 90)
            coordinator.onStepDetected(sec(2.0), batteryPercent = 90)
            assertEquals(0, fakeEngine.startedCount)
            assertTrue(sink.wakePayloads.isEmpty())

            coordinator.updateBodyState(sec(5.0), isOnBody = false)
            assertFalse(sink.advertising)
            assertEquals(BeaconState.IDLE, coordinator.snapshot().beaconState)
        }

    @Test
    fun `dual-mode broadcasts 10-byte wake frame, runs burst, and flushes device_ranging v1 batches`() =
        runTest {
            val sink = RecordingBleSink()
            val fakeEngine = FakeUwbRangingEngine()
            val published = mutableListOf<String>()
            val batchPublisher =
                EpochBatchPublisher(
                    tagId = "watch-pw5",
                    bootUnixNs = 1_794_650_000_000_000_000L,
                    sender = { _, payload ->
                        published.add(payload)
                        true
                    },
                )
            val anchors =
                listOf(
                    UwbAnchorEntry("a1", "00:01"),
                    UwbAnchorEntry("a2", "00:02"),
                    UwbAnchorEntry("a3", "00:03"),
                    UwbAnchorEntry("a4", "00:04"),
                )
            val coordinator =
                CompanionCoordinator(
                    identifierStore = InMemoryIdentifierStore(),
                    bleSink = sink,
                    mode = RuntimeMode.BLE_PLUS_UWB_BURST,
                    uwbAnchors = anchors,
                    uwbEngine = fakeEngine,
                    batchPublisher = batchPublisher,
                )

            coordinator.pair()
            coordinator.updateBodyState(sec(0.0), isOnBody = true)
            for (i in 0..5) {
                coordinator.onLinearAcceleration(sec(0.2 * i), if (i % 2 == 0) 0.5 else 3.5)
            }
            coordinator.onStepDetected(sec(0.5), batteryPercent = 85)
            coordinator.onStepDetected(sec(1.0), batteryPercent = 85)
            val burstSnap = coordinator.onStepDetected(sec(1.5), batteryPercent = 85)

            assertEquals(DutyCycleState.UWB_RANGING_BURST, burstSnap.dutyCycleState)
            assertEquals(1, fakeEngine.startedCount)
            assertEquals(1, sink.wakePayloads.size)
            assertEquals(PreSharedStsNegotiator.WAKE_PAYLOAD_BYTES, sink.wakePayloads.single().first.size)

            for (seq in 1L..10L) {
                val sample =
                    RangingEpochSample(
                        epochSeq = seq,
                        measuredAtElapsedNanos = sec(1.5 + seq.toDouble()),
                        ranges =
                            anchors.map { a ->
                                AnchorRangeReading(a.anchorId, a.uwbAddressHex, distanceM = 2.5)
                            },
                    )
                coordinator.onUwbEpoch(sample, batteryPercent = 84)
            }

            assertEquals(DutyCycleState.BURST_COOLDOWN, coordinator.snapshot().dutyCycleState)
            assertEquals(1, fakeEngine.stoppedCount)
            assertTrue(published.size >= 3)
        }
}
