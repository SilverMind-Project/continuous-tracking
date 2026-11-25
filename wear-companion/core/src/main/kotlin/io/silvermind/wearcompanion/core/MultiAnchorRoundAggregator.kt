package io.silvermind.wearcompanion.core

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/** One per-peer UWB ranging callback, stamped on the monotonic clock when it arrived. */
data class PeerRangeReading(
    val reading: AnchorRangeReading,
    val measuredAtElapsedNanos: Long,
)

/**
 * Monotonic epoch sequence shared by every burst of one ranging engine, so
 * `epoch_seq` never restarts at 1 between bursts of the same service lifetime.
 */
class EpochSequence(
    first: Long = 1L,
) {
    init {
        require(first >= 0L) { "first epoch sequence must be >= 0" }
    }

    private var next: Long = first

    @Synchronized
    fun next(): Long = next++
}

/**
 * Coalesces per-peer Android UWB ranging callbacks that belong to the same
 * multicast DS-TWR round into one multi-anchor [RangingEpochSample] sharing
 * one [RangingEpochSample.epochSeq].
 *
 * Android delivers `onRangingResult(device, position)` once per peer anchor per
 * round. Numbering each callback as its own epoch splits every round into
 * 1-anchor epochs, which `rf-ingress` can only turn into room fallbacks.
 *
 * A round closes and is emitted when:
 * 1. all [expectedAnchorCount] distinct anchors have reported, or
 * 2. a reading arrives for an anchor already in the open round (the radio has
 *    moved on to the next round), or
 * 3. a reading arrives [roundWindowNanos] or later after the round opened.
 *
 * A trailing partial round (for example 3 of 4 anchors while the wearer's body
 * shadows the fourth) is emitted by [flushExpired] once the window has passed,
 * or by [flushAll] when ranging ends.
 */
class MultiAnchorRoundAggregator(
    private val expectedAnchorCount: Int,
    val roundWindowNanos: Long = DEFAULT_ROUND_WINDOW_NANOS,
    private val sequence: EpochSequence = EpochSequence(),
) {
    init {
        require(expectedAnchorCount >= 1) { "expectedAnchorCount must be >= 1" }
        require(roundWindowNanos > 0L) { "roundWindowNanos must be > 0" }
    }

    private var roundStartNanos: Long? = null
    private var latestNanos: Long = 0L
    private val currentByAnchor = LinkedHashMap<String, AnchorRangeReading>()

    @Synchronized
    fun onPeerReading(
        reading: AnchorRangeReading,
        measuredAtElapsedNanos: Long,
    ): List<RangingEpochSample> {
        val emitted = mutableListOf<RangingEpochSample>()
        val start = roundStartNanos
        if (
            start != null &&
            (
                measuredAtElapsedNanos - start >= roundWindowNanos ||
                    currentByAnchor.containsKey(reading.anchorId)
            )
        ) {
            freezeCurrent()?.let { emitted.add(it) }
        }
        if (roundStartNanos == null) {
            roundStartNanos = measuredAtElapsedNanos
            latestNanos = measuredAtElapsedNanos
        }
        if (measuredAtElapsedNanos > latestNanos) {
            latestNanos = measuredAtElapsedNanos
        }
        currentByAnchor[reading.anchorId] = reading

        if (currentByAnchor.size >= expectedAnchorCount) {
            freezeCurrent()?.let { emitted.add(it) }
        }
        return emitted
    }

    /** Emit the open round when [roundWindowNanos] has passed since it opened. */
    @Synchronized
    fun flushExpired(nowElapsedNanos: Long): RangingEpochSample? {
        val start = roundStartNanos ?: return null
        if (nowElapsedNanos - start >= roundWindowNanos) {
            return freezeCurrent()
        }
        return null
    }

    /** Emit whatever round is open, regardless of its age. */
    @Synchronized
    fun flushAll(): RangingEpochSample? = freezeCurrent()

    private fun freezeCurrent(): RangingEpochSample? {
        if (currentByAnchor.isEmpty()) {
            roundStartNanos = null
            return null
        }
        val ranges = currentByAnchor.values.toList()
        val ts = latestNanos
        currentByAnchor.clear()
        roundStartNanos = null
        return RangingEpochSample(
            epochSeq = sequence.next(),
            measuredAtElapsedNanos = ts,
            ranges = ranges,
        )
    }

    companion object {
        const val DEFAULT_ROUND_WINDOW_NANOS: Long = 150_000_000L
        const val DEFAULT_FLUSH_CHECK_INTERVAL_MS: Long = 50L
    }
}

/**
 * Turns a stream of per-peer callbacks into multi-anchor ranging epochs.
 *
 * Besides the closing rules of [MultiAnchorRoundAggregator], a timer checks the
 * open round every [flushCheckIntervalMs] against [clockNanos], so a partial
 * round is emitted about [roundWindowNanos] after it opened even when no later
 * callback arrives. When the upstream completes, the open round is emitted.
 * When the collector is cancelled (burst stopped), the open round is dropped.
 */
fun Flow<PeerRangeReading>.coalesceRounds(
    expectedAnchorCount: Int,
    clockNanos: () -> Long,
    sequence: EpochSequence = EpochSequence(),
    roundWindowNanos: Long = MultiAnchorRoundAggregator.DEFAULT_ROUND_WINDOW_NANOS,
    flushCheckIntervalMs: Long = MultiAnchorRoundAggregator.DEFAULT_FLUSH_CHECK_INTERVAL_MS,
): Flow<RangingEpochSample> {
    require(flushCheckIntervalMs > 0L) { "flushCheckIntervalMs must be > 0" }
    val upstream = this
    return channelFlow {
        val aggregator = MultiAnchorRoundAggregator(expectedAnchorCount, roundWindowNanos, sequence)
        val timer =
            launch {
                while (isActive) {
                    delay(flushCheckIntervalMs)
                    aggregator.flushExpired(clockNanos())?.let { send(it) }
                }
            }
        try {
            upstream.collect { peer ->
                for (sample in aggregator.onPeerReading(peer.reading, peer.measuredAtElapsedNanos)) {
                    send(sample)
                }
            }
        } finally {
            timer.cancel()
        }
        aggregator.flushAll()?.let { send(it) }
    }
}
