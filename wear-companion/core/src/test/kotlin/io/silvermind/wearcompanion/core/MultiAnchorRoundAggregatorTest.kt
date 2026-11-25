package io.silvermind.wearcompanion.core

import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.consumeAsFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.currentTime
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

private const val MS: Long = 1_000_000L

private fun reading(anchor: String): AnchorRangeReading =
    AnchorRangeReading(anchorId = anchor, uwbAddressHex = "00:0${anchor.last()}", distanceM = 2.0)

class MultiAnchorRoundAggregatorTest {
    @Test
    fun `coalesces four peer callbacks into a single epoch with one epoch seq`() {
        val aggregator = MultiAnchorRoundAggregator(expectedAnchorCount = 4)

        assertTrue(aggregator.onPeerReading(reading("a1"), 0L).isEmpty())
        assertTrue(aggregator.onPeerReading(reading("a2"), 5 * MS).isEmpty())
        assertTrue(aggregator.onPeerReading(reading("a3"), 10 * MS).isEmpty())
        val emitted = aggregator.onPeerReading(reading("a4"), 15 * MS)

        val sample = emitted.single()
        assertEquals(1L, sample.epochSeq)
        assertEquals(listOf("a1", "a2", "a3", "a4"), sample.ranges.map { it.anchorId })
        assertEquals(15 * MS, sample.measuredAtElapsedNanos)
        assertNull(aggregator.flushAll())
    }

    @Test
    fun `flushes a three-anchor partial round when the fourth anchor is body shadowed`() {
        val aggregator = MultiAnchorRoundAggregator(expectedAnchorCount = 4)
        aggregator.onPeerReading(reading("a1"), 0L)
        aggregator.onPeerReading(reading("a2"), 4 * MS)
        aggregator.onPeerReading(reading("a3"), 8 * MS)

        assertNull(aggregator.flushExpired(149 * MS))
        val sample = aggregator.flushExpired(150 * MS)

        assertNotNull(sample)
        assertEquals(1L, sample!!.epochSeq)
        assertEquals(3, sample.ranges.size)
    }

    @Test
    fun `duplicate anchor closes the previous round and opens the next epoch seq`() {
        val aggregator = MultiAnchorRoundAggregator(expectedAnchorCount = 4)
        aggregator.onPeerReading(reading("a1"), 0L)
        aggregator.onPeerReading(reading("a2"), 3 * MS)
        aggregator.onPeerReading(reading("a3"), 6 * MS)

        val emitted = aggregator.onPeerReading(reading("a1"), 100 * MS)

        assertEquals(1L, emitted.single().epochSeq)
        assertEquals(3, emitted.single().ranges.size)
        val next = aggregator.flushAll()
        assertEquals(2L, next!!.epochSeq)
        assertEquals(listOf("a1"), next.ranges.map { it.anchorId })
    }

    @Test
    fun `a late reading past the round window opens a new round`() {
        val aggregator = MultiAnchorRoundAggregator(expectedAnchorCount = 4)
        aggregator.onPeerReading(reading("a1"), 0L)
        aggregator.onPeerReading(reading("a2"), 5 * MS)

        val emitted = aggregator.onPeerReading(reading("a3"), 160 * MS)

        assertEquals(listOf("a1", "a2"), emitted.single().ranges.map { it.anchorId })
        assertEquals(listOf("a3"), aggregator.flushAll()!!.ranges.map { it.anchorId })
    }

    @Test
    fun `epoch sequence stays monotonic across aggregators sharing one sequence`() {
        val sequence = EpochSequence()
        val burst1 = MultiAnchorRoundAggregator(expectedAnchorCount = 1, sequence = sequence)
        val burst2 = MultiAnchorRoundAggregator(expectedAnchorCount = 1, sequence = sequence)

        val first = burst1.onPeerReading(reading("a1"), 0L).single()
        val second = burst2.onPeerReading(reading("a1"), 0L).single()

        assertEquals(1L, first.epochSeq)
        assertEquals(2L, second.epochSeq)
    }

    @Test
    fun `coalesceRounds groups callbacks and emits the trailing round when upstream completes`() =
        runTest {
            val peers =
                flowOf(
                    PeerRangeReading(reading("a1"), 0L),
                    PeerRangeReading(reading("a2"), 2 * MS),
                    PeerRangeReading(reading("a3"), 4 * MS),
                    PeerRangeReading(reading("a1"), 200 * MS),
                    PeerRangeReading(reading("a2"), 202 * MS),
                )

            val samples = peers.coalesceRounds(expectedAnchorCount = 3, clockNanos = { 0L }).toList()

            assertEquals(listOf(1L, 2L), samples.map { it.epochSeq })
            assertEquals(listOf(3, 2), samples.map { it.ranges.size })
        }

    @OptIn(ExperimentalCoroutinesApi::class)
    @Test
    fun `coalesceRounds timer emits a partial round about one window after it opened`() =
        runTest {
            val peers = Channel<PeerRangeReading>(Channel.UNLIMITED)
            val received = mutableListOf<RangingEpochSample>()
            val clock = virtualClock()
            val job =
                launch {
                    peers.consumeAsFlow()
                        .coalesceRounds(expectedAnchorCount = 4, clockNanos = clock)
                        .collect { received.add(it) }
                }
            runCurrent()
            peers.send(PeerRangeReading(reading("a1"), clock()))
            peers.send(PeerRangeReading(reading("a2"), clock()))
            peers.send(PeerRangeReading(reading("a3"), clock()))
            runCurrent()
            assertTrue(received.isEmpty())

            advanceTimeBy(149)
            runCurrent()
            assertTrue(received.isEmpty())

            advanceTimeBy(60)
            runCurrent()
            assertEquals(3, received.single().ranges.size)
            job.cancel()
        }

    @OptIn(ExperimentalCoroutinesApi::class)
    private fun TestScope.virtualClock(): () -> Long = { currentTime * MS }
}
