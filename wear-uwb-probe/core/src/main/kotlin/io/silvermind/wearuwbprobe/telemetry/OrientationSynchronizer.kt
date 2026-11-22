package io.silvermind.wearuwbprobe.telemetry

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sin
import kotlin.math.sqrt

/** Unit quaternion representing rotation `q_{world <- watch}` in East-North-Up (ENU) coordinates. */
@Serializable
data class Quaternion(
    val w: Double,
    val x: Double,
    val y: Double,
    val z: Double,
) {
    /** Returns a unit-length copy of this quaternion (defaults to identity if degenerate). */
    fun normalized(): Quaternion {
        val norm = sqrt(w * w + x * x + y * y + z * z)
        if (!norm.isFinite() || norm < NORM_EPSILON) {
            return IDENTITY
        }
        return Quaternion(w / norm, x / norm, y / norm, z / norm)
    }

    /**
     * Shortest-path spherical linear interpolation (SLERP) from `this` (`t = 0`) to [other] (`t = 1`).
     */
    fun slerp(
        other: Quaternion,
        t: Double,
    ): Quaternion {
        val alpha = t.coerceIn(0.0, 1.0)
        val q0 = normalized()
        var q1 = other.normalized()
        var dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z
        if (dot < 0.0) {
            dot = -dot
            q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        }
        if (dot > SLERP_LINEAR_THRESHOLD) {
            return Quaternion(
                w = q0.w + alpha * (q1.w - q0.w),
                x = q0.x + alpha * (q1.x - q0.x),
                y = q0.y + alpha * (q1.y - q0.y),
                z = q0.z + alpha * (q1.z - q0.z),
            ).normalized()
        }
        val theta = acos(dot.coerceIn(-1.0, 1.0))
        val sinTheta = sin(theta)
        val w0 = sin((1.0 - alpha) * theta) / sinTheta
        val w1 = sin(alpha * theta) / sinTheta
        return Quaternion(
            w = w0 * q0.w + w1 * q1.w,
            x = w0 * q0.x + w1 * q1.x,
            y = w0 * q0.y + w1 * q1.y,
            z = w0 * q0.z + w1 * q1.z,
        ).normalized()
    }

    /** Rotates a 3D vector `(vx, vy, vz)` from the watch frame into the world ENU frame. */
    fun rotateVector(
        vx: Double,
        vy: Double,
        vz: Double,
    ): Triple<Double, Double, Double> {
        val q = normalized()
        val tx = 2.0 * (q.y * vz - q.z * vy)
        val ty = 2.0 * (q.z * vx - q.x * vz)
        val tz = 2.0 * (q.x * vy - q.y * vx)
        val rx = vx + q.w * tx + (q.y * tz - q.z * ty)
        val ry = vy + q.w * ty + (q.z * tx - q.x * tz)
        val rz = vz + q.w * tz + (q.x * ty - q.y * tx)
        return Triple(rx, ry, rz)
    }

    companion object {
        val IDENTITY: Quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)
        private const val NORM_EPSILON: Double = 1e-12
        private const val SLERP_LINEAR_THRESHOLD: Double = 0.9995

        /**
         * Builds a unit quaternion from Android's `Sensor.TYPE_ROTATION_VECTOR` or
         * `Sensor.TYPE_GAME_ROTATION_VECTOR` float array `[x, y, z]` or `[x, y, z, w]`.
         */
        fun fromAndroidRotationVector(values: FloatArray): Quaternion {
            require(values.size >= 3) { "Rotation vector requires at least 3 elements, got ${values.size}" }
            val x = values[0].toDouble()
            val y = values[1].toDouble()
            val z = values[2].toDouble()
            val w =
                if (values.size >= 4) {
                    values[3].toDouble()
                } else {
                    val sq = 1.0 - (x * x + y * y + z * z)
                    if (sq > 0.0) sqrt(sq) else 0.0
                }
            return Quaternion(w = w, x = x, y = y, z = z).normalized()
        }
    }
}

/** 3D gravity acceleration sample in m/s^2 in the watch sensor frame. */
@Serializable
data class GravityVectorMps2(
    val xMps2: Double,
    val yMps2: Double,
    val zMps2: Double,
)

/** Raw timestamped IMU orientation snapshot buffered by [OrientationSynchronizer]. */
data class OrientationSample(
    val elapsedRealtimeNanos: Long,
    val worldQuaternion: Quaternion? = null,
    val gameQuaternion: Quaternion? = null,
    val gravityMps2: GravityVectorMps2? = null,
)

/** Orientation state SLERP-interpolated to a UWB ranging sample's monotonic timestamp. */
@Serializable
data class InterpolatedOrientation(
    val targetElapsedRealtimeNanos: Long,
    val worldQuaternion: Quaternion? = null,
    val gameQuaternion: Quaternion? = null,
    val gravityMps2: GravityVectorMps2? = null,
    val nearestImuDeltaMs: Double? = null,
    val extrapolated: Boolean = false,
)

/**
 * Buffers timestamped `ROTATION_VECTOR`, `GAME_ROTATION_VECTOR`, and `GRAVITY` IMU samples
 * and SLERP-interpolates them to each UWB ranging measurement's `elapsedRealtimeNanos`.
 */
class OrientationSynchronizer(
    private val maxSamplesPerStream: Int = DEFAULT_MAX_SAMPLES,
) {
    private data class TimedQuaternion(
        val elapsedRealtimeNanos: Long,
        val q: Quaternion,
    )

    private data class TimedGravity(
        val elapsedRealtimeNanos: Long,
        val g: GravityVectorMps2,
    )

    private val worldBuffer: ArrayDeque<TimedQuaternion> = ArrayDeque()
    private val gameBuffer: ArrayDeque<TimedQuaternion> = ArrayDeque()
    private val gravityBuffer: ArrayDeque<TimedGravity> = ArrayDeque()

    init {
        require(maxSamplesPerStream >= 2) { "maxSamplesPerStream must be >= 2" }
    }

    fun recordWorldRotation(
        elapsedRealtimeNanos: Long,
        quaternion: Quaternion,
    ) {
        appendQuaternion(worldBuffer, elapsedRealtimeNanos, quaternion.normalized())
    }

    fun recordGameRotation(
        elapsedRealtimeNanos: Long,
        quaternion: Quaternion,
    ) {
        appendQuaternion(gameBuffer, elapsedRealtimeNanos, quaternion.normalized())
    }

    fun recordGravity(
        elapsedRealtimeNanos: Long,
        gravityMps2: GravityVectorMps2,
    ) {
        if (gravityBuffer.isNotEmpty() && elapsedRealtimeNanos < gravityBuffer.last().elapsedRealtimeNanos) {
            return
        }
        gravityBuffer.addLast(TimedGravity(elapsedRealtimeNanos, gravityMps2))
        while (gravityBuffer.size > maxSamplesPerStream) {
            gravityBuffer.removeFirst()
        }
    }

    fun recordSample(sample: OrientationSample) {
        sample.worldQuaternion?.let { recordWorldRotation(sample.elapsedRealtimeNanos, it) }
        sample.gameQuaternion?.let { recordGameRotation(sample.elapsedRealtimeNanos, it) }
        sample.gravityMps2?.let { recordGravity(sample.elapsedRealtimeNanos, it) }
    }

    fun interpolateAt(targetElapsedRealtimeNanos: Long): InterpolatedOrientation {
        val worldResult = interpolateQuaternion(worldBuffer, targetElapsedRealtimeNanos)
        val gameResult = interpolateQuaternion(gameBuffer, targetElapsedRealtimeNanos)
        val gravityResult = interpolateGravity(gravityBuffer, targetElapsedRealtimeNanos)

        val deltas = listOfNotNull(worldResult?.nearestDeltaMs, gameResult?.nearestDeltaMs)
        val extrapolated =
            (worldResult?.extrapolated == true) ||
                (worldResult == null && gameResult?.extrapolated == true)

        return InterpolatedOrientation(
            targetElapsedRealtimeNanos = targetElapsedRealtimeNanos,
            worldQuaternion = worldResult?.value,
            gameQuaternion = gameResult?.value,
            gravityMps2 = gravityResult,
            nearestImuDeltaMs = deltas.minOrNull(),
            extrapolated = extrapolated,
        )
    }

    fun clear() {
        worldBuffer.clear()
        gameBuffer.clear()
        gravityBuffer.clear()
    }

    private fun appendQuaternion(
        buffer: ArrayDeque<TimedQuaternion>,
        elapsedRealtimeNanos: Long,
        q: Quaternion,
    ) {
        if (buffer.isNotEmpty() && elapsedRealtimeNanos < buffer.last().elapsedRealtimeNanos) {
            return
        }
        buffer.addLast(TimedQuaternion(elapsedRealtimeNanos, q))
        while (buffer.size > maxSamplesPerStream) {
            buffer.removeFirst()
        }
    }

    private data class QuatInterpResult(
        val value: Quaternion,
        val nearestDeltaMs: Double,
        val extrapolated: Boolean,
    )

    private fun interpolateQuaternion(
        buffer: ArrayDeque<TimedQuaternion>,
        targetNanos: Long,
    ): QuatInterpResult? {
        if (buffer.isEmpty()) {
            return null
        }
        if (targetNanos <= buffer.first().elapsedRealtimeNanos) {
            val first = buffer.first()
            return QuatInterpResult(
                value = first.q,
                nearestDeltaMs = abs(targetNanos - first.elapsedRealtimeNanos) / NANOS_PER_MS,
                extrapolated = targetNanos < first.elapsedRealtimeNanos,
            )
        }
        if (targetNanos >= buffer.last().elapsedRealtimeNanos) {
            val last = buffer.last()
            return QuatInterpResult(
                value = last.q,
                nearestDeltaMs = abs(targetNanos - last.elapsedRealtimeNanos) / NANOS_PER_MS,
                extrapolated = targetNanos > last.elapsedRealtimeNanos,
            )
        }
        for (i in 0 until buffer.size - 1) {
            val a = buffer[i]
            val b = buffer[i + 1]
            if (targetNanos in a.elapsedRealtimeNanos..b.elapsedRealtimeNanos) {
                val span = b.elapsedRealtimeNanos - a.elapsedRealtimeNanos
                val alpha = if (span <= 0L) 0.0 else (targetNanos - a.elapsedRealtimeNanos).toDouble() / span.toDouble()
                val nearest =
                    minOf(
                        abs(targetNanos - a.elapsedRealtimeNanos),
                        abs(b.elapsedRealtimeNanos - targetNanos),
                    ) / NANOS_PER_MS
                return QuatInterpResult(
                    value = a.q.slerp(b.q, alpha),
                    nearestDeltaMs = nearest,
                    extrapolated = false,
                )
            }
        }
        return null
    }

    private fun interpolateGravity(
        buffer: ArrayDeque<TimedGravity>,
        targetNanos: Long,
    ): GravityVectorMps2? {
        if (buffer.isEmpty()) {
            return null
        }
        if (targetNanos <= buffer.first().elapsedRealtimeNanos) {
            return buffer.first().g
        }
        if (targetNanos >= buffer.last().elapsedRealtimeNanos) {
            return buffer.last().g
        }
        for (i in 0 until buffer.size - 1) {
            val a = buffer[i]
            val b = buffer[i + 1]
            if (targetNanos in a.elapsedRealtimeNanos..b.elapsedRealtimeNanos) {
                val span = b.elapsedRealtimeNanos - a.elapsedRealtimeNanos
                val alpha = if (span <= 0L) 0.0 else (targetNanos - a.elapsedRealtimeNanos).toDouble() / span.toDouble()
                return GravityVectorMps2(
                    xMps2 = a.g.xMps2 + alpha * (b.g.xMps2 - a.g.xMps2),
                    yMps2 = a.g.yMps2 + alpha * (b.g.yMps2 - a.g.yMps2),
                    zMps2 = a.g.zMps2 + alpha * (b.g.zMps2 - a.g.zMps2),
                )
            }
        }
        return null
    }

    companion object {
        const val DEFAULT_MAX_SAMPLES: Int = 512
        private const val NANOS_PER_MS: Double = 1_000_000.0
    }
}
