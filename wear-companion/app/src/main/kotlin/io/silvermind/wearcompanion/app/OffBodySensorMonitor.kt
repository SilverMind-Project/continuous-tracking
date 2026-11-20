package io.silvermind.wearcompanion.app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.sqrt

/** Listener callbacks for Wear OS off-body and optional Tier 2 locomotion sensors. */
interface WatchSensorCallbacks {
    fun onBodyStateChanged(nowNanos: Long, onBody: Boolean)

    fun onStepDetected(nowNanos: Long)

    fun onLinearAcceleration(nowNanos: Long, magnitudeMps2: Double)
}

/**
 * Registers `Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT` always, and registers
 * `Sensor.TYPE_STEP_DETECTOR` + `Sensor.TYPE_LINEAR_ACCELERATION` only when
 * Tier 2 locomotion gating is enabled.
 */
class OffBodySensorMonitor(
    context: Context,
    private val callbacks: WatchSensorCallbacks,
    private val enableLocomotionSensors: Boolean,
) : SensorEventListener {
    private val sensorManager: SensorManager =
        context.getSystemService(SensorManager::class.java)

    fun start() {
        sensorManager.getDefaultSensor(Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT)?.let { sensor ->
            sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
        if (enableLocomotionSensors) {
            sensorManager.getDefaultSensor(Sensor.TYPE_STEP_DETECTOR)?.let { sensor ->
                sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
            }
            sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)?.let { sensor ->
                sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
            }
        }
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_LOW_LATENCY_OFFBODY_DETECT -> {
                val isOnBody = event.values.isNotEmpty() && event.values[0] != 0.0f
                callbacks.onBodyStateChanged(event.timestamp, isOnBody)
            }
            Sensor.TYPE_STEP_DETECTOR -> {
                callbacks.onStepDetected(event.timestamp)
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                if (event.values.size >= 3) {
                    val x = event.values[0].toDouble()
                    val y = event.values[1].toDouble()
                    val z = event.values[2].toDouble()
                    callbacks.onLinearAcceleration(event.timestamp, sqrt(x * x + y * y + z * z))
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Sensor accuracy hints do not alter binary on-body or step transitions.
    }
}
