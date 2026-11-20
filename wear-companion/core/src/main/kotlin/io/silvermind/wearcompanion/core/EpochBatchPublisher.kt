package io.silvermind.wearcompanion.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/** Transport boundary delivering serialized `device_ranging.v1` JSON micro-batches. */
fun interface BatchMqttSender {
    /** Return true when the broker accepted the payload; false keeps epochs buffered. */
    fun publish(
        topic: String,
        payloadJson: String,
    ): Boolean
}

@Serializable
data class WireRangeEntry(
    @SerialName("anchor_id")
    val anchorId: String,
    @SerialName("uwb_address_hex")
    val uwbAddressHex: String,
    @SerialName("distance_m")
    val distanceM: Double,
    @SerialName("azimuth_deg")
    val azimuthDeg: Double? = null,
    @SerialName("elevation_deg")
    val elevationDeg: Double? = null,
    @SerialName("los_confidence")
    val losConfidence: Double? = null,
)

@Serializable
data class WireEpochEntry(
    @SerialName("epoch_seq")
    val epochSeq: Long,
    @SerialName("measured_at_unix_ns")
    val measuredAtUnixNs: Long,
    @SerialName("ranges")
    val ranges: List<WireRangeEntry>,
)

@Serializable
data class DeviceRangingV1Batch(
    @SerialName("schema")
    val schema: String = SCHEMA_ID,
    @SerialName("tag_id")
    val tagId: String,
    @SerialName("sent_at_unix_ns")
    val sentAtUnixNs: Long,
    @SerialName("wear_state")
    val wearState: String,
    @SerialName("motion_state")
    val motionState: String,
    @SerialName("battery_fraction")
    val batteryFraction: Double? = null,
    @SerialName("epochs")
    val epochs: List<WireEpochEntry>,
) {
    companion object {
        const val SCHEMA_ID: String = "device_ranging.v1"
    }
}

/**
 * Store-and-forward micro-batch publisher with monotonic clock anchoring.
 *
 * Keeps at most [maxBufferedEpochs] epochs in an in-memory ring buffer,
 * dropping the oldest epoch cleanly if Wi-Fi/MQTT stays unreachable. Converts
 * monotonic `elapsedRealtimeNanos` into strictly monotonic `measured_at_unix_ns`
 * using a fixed `bootUnixNs` anchor, and stamps top-level `sent_at_unix_ns` on
 * flush so rf-ingress decouples micro-batching delay from clock skew.
 */
class EpochBatchPublisher(
    private val tagId: String,
    private val bootUnixNs: Long,
    private val sender: BatchMqttSender,
    private val maxBufferedEpochs: Int = DEFAULT_MAX_BUFFERED_EPOCHS,
    private val flushIntervalS: Double = DEFAULT_FLUSH_INTERVAL_S,
) {
    init {
        require(tagId.isNotBlank()) { "tagId must be non-blank" }
        require(bootUnixNs > 0L) { "bootUnixNs must be positive" }
        require(maxBufferedEpochs >= 1) { "maxBufferedEpochs must be >= 1" }
        require(flushIntervalS > 0.0) { "flushIntervalS must be > 0" }
    }

    val topic: String = "rf/devices/$tagId/ranges"

    private val ringBuffer: ArrayDeque<WireEpochEntry> = ArrayDeque()
    private var lastMeasuredUnixNs: Long = 0L
    private var lastFlushElapsedNanos: Long? = null

    var droppedOldestCount: Int = 0
        private set

    fun bufferedCount(): Int = ringBuffer.size

    fun enqueue(sample: RangingEpochSample) {
        val rawUnixNs = bootUnixNs + sample.measuredAtElapsedNanos
        // Enforce monotonic unix_ns even if two callbacks share an identical tick.
        val monotonicUnixNs = if (rawUnixNs <= lastMeasuredUnixNs) lastMeasuredUnixNs + 1L else rawUnixNs
        lastMeasuredUnixNs = monotonicUnixNs

        if (ringBuffer.size >= maxBufferedEpochs) {
            ringBuffer.removeFirst()
            droppedOldestCount += 1
        }
        ringBuffer.addLast(
            WireEpochEntry(
                epochSeq = sample.epochSeq,
                measuredAtUnixNs = monotonicUnixNs,
                ranges =
                    sample.ranges.map { reading ->
                        WireRangeEntry(
                            anchorId = reading.anchorId,
                            uwbAddressHex = reading.uwbAddressHex,
                            distanceM = reading.distanceM,
                            azimuthDeg = reading.azimuthDeg,
                            elevationDeg = reading.elevationDeg,
                            losConfidence = reading.losConfidence,
                        )
                    },
            ),
        )
    }

    fun shouldFlush(nowElapsedNanos: Long): Boolean {
        if (ringBuffer.isEmpty()) {
            return false
        }
        val prev = lastFlushElapsedNanos ?: return false
        val elapsedS = (nowElapsedNanos - prev).toDouble() / NANOS_PER_SECOND
        return elapsedS >= flushIntervalS
    }

    fun markBurstStarted(nowElapsedNanos: Long) {
        lastFlushElapsedNanos = nowElapsedNanos
    }

    /**
     * Flush all buffered epochs as one `device_ranging.v1` JSON message.
     *
     * Returns the published JSON payload when sent, or `null` if the buffer was
     * empty or the MQTT transport failed (leaving epochs buffered for retry).
     */
    fun flush(
        nowElapsedNanos: Long,
        wearState: WearStateWire,
        motionState: MotionStateWire,
        batteryFraction: Double? = null,
    ): String? {
        if (ringBuffer.isEmpty()) {
            return null
        }
        val sentAtUnixNs = bootUnixNs + nowElapsedNanos
        val batch =
            DeviceRangingV1Batch(
                tagId = tagId,
                sentAtUnixNs = sentAtUnixNs,
                wearState = wearState.wireValue,
                motionState = motionState.wireValue,
                batteryFraction = batteryFraction,
                epochs = ringBuffer.toList(),
            )
        val jsonPayload = jsonFormat.encodeToString(batch)
        val ok = sender.publish(topic, jsonPayload)
        if (!ok) {
            return null
        }
        ringBuffer.clear()
        lastFlushElapsedNanos = nowElapsedNanos
        return jsonPayload
    }

    companion object {
        const val DEFAULT_MAX_BUFFERED_EPOCHS: Int = 60
        const val DEFAULT_FLUSH_INTERVAL_S: Double = 3.0
        private const val NANOS_PER_SECOND: Double = 1_000_000_000.0
        private val jsonFormat =
            Json {
                encodeDefaults = true
                explicitNulls = false
            }
    }
}
