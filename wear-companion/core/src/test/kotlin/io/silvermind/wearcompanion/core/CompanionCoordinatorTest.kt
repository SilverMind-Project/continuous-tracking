package io.silvermind.wearcompanion.core

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertThrows
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

private class RecordingWakeHold : BurstWakeHold {
    var acquired: Int = 0
    var released: Int = 0
    var lastTimeoutMs: Long = 0L

    override fun acquire(timeoutMs: Long) {
        acquired += 1
        lastTimeoutMs = timeoutMs
    }

    override fun release() {
        released += 1
    }
}

private val defaultAnchors =
    listOf(
        UwbAnchorEntry("a1", "00:01"),
        UwbAnchorEntry("a2", "00:02"),
        UwbAnchorEntry("a3", "00:03"),
        UwbAnchorEntry("a4", "00:04"),
    )

private class Tier2Fixture(
    scope: CoroutineScope,
    anchors: List<UwbAnchorEntry> = defaultAnchors,
    val engine: FakeUwbRangingEngine = FakeUwbRangingEngine(),
) {
    val sink = RecordingBleSink()
    val wakeHold = RecordingWakeHold()
    val published = mutableListOf<String>()
    val errors = mutableListOf<Throwable>()
    val publisher =
        EpochBatchPublisher(
            tagId = "watch-pw5",
            bootUnixNs = 1_794_650_000_000_000_000L,
            sender = { _, payload ->
                published.add(payload)
                true
            },
        )
    val coordinator =
        CompanionCoordinator(
            identifierStore = InMemoryIdentifierStore(),
            bleSink = sink,
            mode = RuntimeMode.BLE_PLUS_UWB_BURST,
            uwbAnchors = anchors,
            uwbEngine = engine,
            batchPublisher = publisher,
            scope = scope,
            wakeHold = wakeHold,
            onBurstError = { errors.add(it) },
        )

    fun publishedEpochCount(): Int = published.sumOf(::epochCount)
}

private fun epochCount(payload: String): Int = Json.parseToJsonElement(payload).jsonObject["epochs"]!!.jsonArray.size

private fun sec(s: Double): Long = (s * 1_000_000_000.0).toLong()

private fun fourAnchorSample(
    seq: Long,
    atSec: Double,
    anchors: List<UwbAnchorEntry> = defaultAnchors,
): RangingEpochSample =
    RangingEpochSample(
        epochSeq = seq,
        measuredAtElapsedNanos = sec(atSec),
        ranges = anchors.map { AnchorRangeReading(it.anchorId, it.uwbAddressHex, distanceM = 2.5) },
    )

/** Puts the wearer on-body and walks three steps with in-band arm swing, which starts a burst at 1.5 s. */
private suspend fun walkIntoBurst(coordinator: CompanionCoordinator): CoordinatorSnapshot {
    coordinator.pair()
    coordinator.updateBodyState(sec(0.0), isOnBody = true)
    for (i in 0..5) {
        coordinator.onLinearAcceleration(sec(0.2 * i), if (i % 2 == 0) 0.5 else 3.5)
    }
    coordinator.onStepDetected(sec(0.5), batteryPercent = 85)
    coordinator.onStepDetected(sec(1.0), batteryPercent = 85)
    return coordinator.onStepDetected(sec(1.5), batteryPercent = 85)
}

@OptIn(ExperimentalCoroutinesApi::class)
private fun TestScope.passWakeWindow() {
    runCurrent()
    advanceTimeBy(PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS + 1)
    runCurrent()
}

@OptIn(ExperimentalCoroutinesApi::class)
class CompanionCoordinatorTest {
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
            assertFalse(coordinator.tier2Active())

            coordinator.updateBodyState(sec(5.0), isOnBody = false)
            assertFalse(sink.advertising)
            assertEquals(BeaconState.IDLE, coordinator.snapshot().beaconState)
        }

    @Test
    fun `start burst collects the cold ranging flow and publishes four batches holding ten epochs`() =
        runTest {
            val f = Tier2Fixture(backgroundScope)

            val burstSnap = walkIntoBurst(f.coordinator)
            assertEquals(DutyCycleState.UWB_RANGING_BURST, burstSnap.dutyCycleState)
            runCurrent()
            assertEquals(1, f.engine.preparedCount)
            assertEquals(1, f.sink.wakePayloads.size)
            // Ranging waits for the wake broadcast to reach the anchors.
            assertEquals(0, f.engine.startedCount)

            passWakeWindow()
            assertEquals(1, f.engine.startedCount)
            assertTrue(f.engine.isRangingActive, "the coordinator must collect the cold flow")

            for (seq in 1L..10L) {
                f.engine.emitEpoch(fourAnchorSample(seq, 1.5 + seq.toDouble()))
                runCurrent()
            }

            assertEquals(DutyCycleState.BURST_COOLDOWN, f.coordinator.snapshot().dutyCycleState)
            assertEquals(1, f.engine.stoppedCount)
            assertFalse(f.engine.isRangingActive, "stopping the burst cancels the collector")
            assertEquals(4, f.published.size)
            assertEquals(10, f.publishedEpochCount())
            assertEquals(1, f.wakeHold.acquired)
            assertEquals(1, f.wakeHold.released)
            assertEquals(15_300L, f.wakeHold.lastTimeoutMs)
            assertTrue(f.errors.isEmpty())
        }

    @Test
    fun `wake frame carries the burst session and the anchors shared channel and preamble`() =
        runTest {
            val anchors = listOf(UwbAnchorEntry("k1", "00:11", channel = 5, preambleIndex = 11))
            val f = Tier2Fixture(backgroundScope, anchors = anchors)

            walkIntoBurst(f.coordinator)
            passWakeWindow()

            val (payload, durationMs) = f.sink.wakePayloads.single()
            assertEquals(PreSharedStsNegotiator.WAKE_PAYLOAD_BYTES, payload.size)
            assertEquals(PreSharedStsNegotiator.WAKE_BROADCAST_DURATION_MS, durationMs)
            val frame = PreSharedStsNegotiator.decodeWakePayload(payload)
            val request = f.engine.lastRequest!!
            assertEquals(request.sessionId, frame.sessionId)
            assertTrue(request.sessionId > 0)
            assertEquals(5, request.channel)
            assertEquals(11, request.preambleIndex)
            assertEquals(5, frame.channel)
            assertEquals(11, frame.preambleIndex)
        }

    @Test
    fun `off-body during the wake delay never opens a ranging session`() =
        runTest {
            val f = Tier2Fixture(backgroundScope)

            walkIntoBurst(f.coordinator)
            runCurrent()
            assertEquals(1, f.sink.wakePayloads.size)

            f.coordinator.updateBodyState(sec(1.6), isOnBody = false)
            passWakeWindow()

            assertEquals(0, f.engine.startedCount)
            assertFalse(f.engine.isRangingActive)
            assertEquals(DutyCycleState.OFF_BODY, f.coordinator.snapshot().dutyCycleState)
            assertEquals(1, f.wakeHold.released)
        }

    @Test
    fun `a burst stopped while the engine ignores cancellation never broadcasts its wake frame`() =
        runTest {
            val f = Tier2Fixture(backgroundScope, engine = FakeUwbRangingEngine(nonCancellablePrepareMs = 100L))

            walkIntoBurst(f.coordinator)
            runCurrent()
            f.coordinator.updateBodyState(sec(1.6), isOnBody = false)
            advanceTimeBy(200L)
            passWakeWindow()

            assertTrue(f.sink.wakePayloads.isEmpty())
            assertEquals(0, f.engine.startedCount)
        }

    @Test
    fun `tick ends a burst when steps and ranging epochs stop`() =
        runTest {
            val f = Tier2Fixture(backgroundScope)
            walkIntoBurst(f.coordinator)
            passWakeWindow()
            assertTrue(f.engine.isRangingActive)

            val mid = f.coordinator.onTick(sec(8.0), batteryPercent = 84)
            assertEquals(DutyCycleState.UWB_RANGING_BURST, mid.dutyCycleState)

            val end = f.coordinator.onTick(sec(13.6), batteryPercent = 84)
            runCurrent()

            assertEquals(DutyCycleState.BURST_COOLDOWN, end.dutyCycleState)
            assertFalse(f.engine.isRangingActive)
            assertEquals(1, f.engine.stoppedCount)
            assertEquals(1, f.wakeHold.released)

            // The 10 s stop-walk reset returns to still on later ticks without any step event.
            val reset = f.coordinator.onTick(sec(24.0), batteryPercent = 84)
            assertEquals(DutyCycleState.ON_BODY_STILL, reset.dutyCycleState)
        }

    @Test
    fun `tick flushes buffered epochs once the flush interval passes`() =
        runTest {
            val f = Tier2Fixture(backgroundScope)
            walkIntoBurst(f.coordinator)
            passWakeWindow()

            f.engine.emitEpoch(fourAnchorSample(1, 2.5))
            f.engine.emitEpoch(fourAnchorSample(2, 3.5))
            runCurrent()
            assertTrue(f.published.isEmpty())

            f.coordinator.onTick(sec(4.6), batteryPercent = 84)

            assertEquals(1, f.published.size)
            assertEquals(2, f.publishedEpochCount())
        }

    @Test
    fun `a failing engine reports the error and the burst still ends on tick`() =
        runTest {
            val engine = FakeUwbRangingEngine(prepareFailure = IllegalStateException("uwb hal unavailable"))
            val f = Tier2Fixture(backgroundScope, engine = engine)

            walkIntoBurst(f.coordinator)
            passWakeWindow()

            assertEquals(1, f.errors.size)
            assertTrue(f.sink.wakePayloads.isEmpty())
            assertEquals(0, engine.startedCount)
            val end = f.coordinator.onTick(sec(13.6), batteryPercent = 84)
            assertEquals(DutyCycleState.BURST_COOLDOWN, end.dutyCycleState)
            assertEquals(1, f.wakeHold.released)
        }

    @Test
    fun `shutdown stops an active burst and flushes buffered epochs`() =
        runTest {
            val f = Tier2Fixture(backgroundScope)
            walkIntoBurst(f.coordinator)
            passWakeWindow()
            f.engine.emitEpoch(fourAnchorSample(1, 2.5))
            runCurrent()

            f.coordinator.shutdown(sec(3.0))
            runCurrent()

            assertFalse(f.engine.isRangingActive)
            assertEquals(1, f.published.size)
        }

    @Test
    fun `tier 2 without a collection scope is rejected at construction`() {
        assertThrows(IllegalArgumentException::class.java) {
            CompanionCoordinator(
                identifierStore = InMemoryIdentifierStore(),
                bleSink = RecordingBleSink(),
                mode = RuntimeMode.BLE_PLUS_UWB_BURST,
                uwbAnchors = defaultAnchors,
                uwbEngine = FakeUwbRangingEngine(),
            )
        }
    }

    @Test
    fun `anchors on different channels are rejected at construction`() =
        runTest {
            assertThrows(IllegalArgumentException::class.java) {
                CompanionCoordinator(
                    identifierStore = InMemoryIdentifierStore(),
                    bleSink = RecordingBleSink(),
                    mode = RuntimeMode.BLE_PLUS_UWB_BURST,
                    uwbAnchors =
                        listOf(
                            UwbAnchorEntry("a1", "00:01", channel = 9),
                            UwbAnchorEntry("a2", "00:02", channel = 5),
                        ),
                    uwbEngine = FakeUwbRangingEngine(),
                    scope = backgroundScope,
                )
            }
        }

    @Test
    fun `session ids are positive`() {
        repeat(1_000) { assertTrue(CompanionCoordinator.nextSessionId() > 0) }
    }
}
