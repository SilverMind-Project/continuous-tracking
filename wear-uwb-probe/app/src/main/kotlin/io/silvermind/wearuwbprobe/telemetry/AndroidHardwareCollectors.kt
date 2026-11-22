package io.silvermind.wearuwbprobe.telemetry

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import io.silvermind.wearuwbprobe.fsm.ThermalSeverity
import io.silvermind.wearuwbprobe.runner.ProbeClock
import kotlin.math.sqrt

/** Android monotonic + UTC wall clock for [io.silvermind.wearuwbprobe.runner.ProbeExperimentRunner]. */
class AndroidProbeClock : ProbeClock {
    override fun elapsedRealtimeNanos(): Long = SystemClock.elapsedRealtimeNanos()

    override fun wallClockUnixMs(): Long = System.currentTimeMillis()
}

/**
 * Reads high-resolution coulomb-counter battery properties (`BATTERY_PROPERTY_CHARGE_COUNTER`
 * in uAh, `BATTERY_PROPERTY_CURRENT_NOW` in uA, `BATTERY_PROPERTY_CAPACITY` in %), voltage,
 * temperature, and `PowerManager.currentThermalStatus`.
 */
class AndroidBatteryReader(
    private val context: Context,
) : BatterySnapshotReader {
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    override fun readSnapshot(): RawBatterySnapshot {
        val nowNanos = SystemClock.elapsedRealtimeNanos()
        val nowUnixMs = System.currentTimeMillis()
        val capacity =
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                .coerceIn(0, 100)
        val rawCharge = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        val chargeUah = if (rawCharge == Long.MIN_VALUE || rawCharge < 0L) null else rawCharge
        val rawCurrent = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val currentUa = if (rawCurrent == Long.MIN_VALUE) null else rawCurrent

        val stickyIntent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val voltageMv = stickyIntent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)?.takeIf { it > 0 }
        val tempDeciC = stickyIntent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
            ?.takeIf { it != Int.MIN_VALUE }

        val thermal =
            when (powerManager.currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> ThermalSeverity.NONE
                PowerManager.THERMAL_STATUS_LIGHT -> ThermalSeverity.LIGHT
                PowerManager.THERMAL_STATUS_MODERATE -> ThermalSeverity.MODERATE
                PowerManager.THERMAL_STATUS_SEVERE -> ThermalSeverity.SEVERE
                else -> ThermalSeverity.CRITICAL
            }

        return RawBatterySnapshot(
            elapsedRealtimeNanos = nowNanos,
            wallClockUnixMs = nowUnixMs,
            capacityPercent = capacity,
            chargeCounterUah = chargeUah,
            currentNowUa = currentUa,
            voltageMv = voltageMv,
            temperatureDeciC = tempDeciC,
            thermalStatus = thermal,
        )
    }
}

/** Callback sink receiving locomotion and wear-state events from Wear OS sensors. */
interface LocomotionSensorListener {
    fun onWearStateChanged(
        elapsedRealtimeNanos: Long,
        onBody: Boolean,
    )

    fun onStepDetected(elapsedRealtimeNanos: Long)

    fun onLinearAcceleration(
        elapsedRealtimeNanos: Long,
        magnitudeMps2: Double,
    )
}

/**
 * Subscribes to `ROTATION_VECTOR`, `GAME_ROTATION_VECTOR`, `GRAVITY`, `STEP_DETECTOR`,
 * `LINEAR_ACCELERATION`, and `LOW_LATENCY_OFFBODY_DETECT` for Stages P3 and P4.
 */
class AndroidSensorCollector(
    context: Context,
    private val orientationSynchronizer: OrientationSynchronizer,
    private val locomotionListener: LocomotionSensorListener? = null,
) : SensorEventListener {
    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var started: Boolean = false

    fun start(
        enableOrientation: Boolean,
        enableLocomotion: Boolean,
    ) {
        if (started) {
            return
        }
        started = true
        if (enableOrientation) {
            registerIfPresent(Sensor.TYPE_ROTATION_VECTOR, SensorManager.SENSOR_DELAY_GAME)
            registerIfPresent(Sensor.TYPE_GAME_ROTATION_VECTOR, SensorManager.SENSOR_DELAY_GAME)
            registerIfPresent(Sensor.TYPE_GRAVITY, SensorManager.SENSOR_DELAY_GAME)
        }
        if (enableLocomotion) {
            registerIfPresent(Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT, SensorManager.SENSOR_DELAY_NORMAL)
            registerIfPresent(Sensor.TYPE_STEP_DETECTOR, SensorManager.SENSOR_DELAY_NORMAL)
            registerIfPresent(Sensor.TYPE_LINEAR_ACCELERATION, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    fun stop() {
        if (!started) {
            return
        }
        started = false
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR -> {
                if (event.values.size >= 3) {
                    orientationSynchronizer.recordWorldRotation(
                        event.timestamp,
                        Quaternion.fromAndroidRotationVector(event.values),
                    )
                }
            }
            Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                if (event.values.size >= 3) {
                    orientationSynchronizer.recordGameRotation(
                        event.timestamp,
                        Quaternion.fromAndroidRotationVector(event.values),
                    )
                }
            }
            Sensor.TYPE_GRAVITY -> {
                if (event.values.size >= 3) {
                    orientationSynchronizer.recordGravity(
                        event.timestamp,
                        GravityVectorMps2(
                            xMps2 = event.values[0].toDouble(),
                            yMps2 = event.values[1].toDouble(),
                            zMps2 = event.values[2].toDouble(),
                        ),
                    )
                }
            }
            Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT -> {
                if (event.values.isNotEmpty()) {
                    locomotionListener?.onWearStateChanged(event.timestamp, event.values[0] >= 1.0f)
                }
            }
            Sensor.TYPE_STEP_DETECTOR -> {
                locomotionListener?.onStepDetected(event.timestamp)
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                if (event.values.size >= 3) {
                    val x = event.values[0].toDouble()
                    val y = event.values[1].toDouble()
                    val z = event.values[2].toDouble()
                    locomotionListener?.onLinearAcceleration(event.timestamp, sqrt(x * x + y * y + z * z))
                }
            }
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int,
    ) = Unit

    private fun registerIfPresent(
        sensorType: Int,
        samplingPeriodUs: Int,
    ) {
        val sensor = sensorManager.getDefaultSensor(sensorType) ?: return
        sensorManager.registerListener(this, sensor, samplingPeriodUs)
    }
}
