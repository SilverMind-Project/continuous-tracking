package io.silvermind.wearuwbprobe.ui

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import io.silvermind.wearuwbprobe.R
import io.silvermind.wearuwbprobe.fsm.DutyCycleAction
import io.silvermind.wearuwbprobe.fsm.LocomotionDutyCycleFsm
import io.silvermind.wearuwbprobe.oob.AndroidBleGattPeerClient
import io.silvermind.wearuwbprobe.oob.AndroidBleWakeAdvertiser
import io.silvermind.wearuwbprobe.oob.BleGattOobNegotiator
import io.silvermind.wearuwbprobe.oob.OobStrategy
import io.silvermind.wearuwbprobe.oob.PreSharedStsNegotiator
import io.silvermind.wearuwbprobe.runner.ProbeCommand
import io.silvermind.wearuwbprobe.runner.ProbeExperimentRunner
import io.silvermind.wearuwbprobe.runner.ProbeStage
import io.silvermind.wearuwbprobe.telemetry.AndroidBatteryReader
import io.silvermind.wearuwbprobe.telemetry.AndroidProbeClock
import io.silvermind.wearuwbprobe.telemetry.AndroidSensorCollector
import io.silvermind.wearuwbprobe.telemetry.BatteryProfiler
import io.silvermind.wearuwbprobe.telemetry.BatteryRegime
import io.silvermind.wearuwbprobe.telemetry.LocomotionSensorListener
import io.silvermind.wearuwbprobe.telemetry.OrientationSynchronizer
import io.silvermind.wearuwbprobe.telemetry.ProbeTelemetryRecorder
import io.silvermind.wearuwbprobe.uwb.GmsNearbyUwbBackend
import io.silvermind.wearuwbprobe.uwb.JetpackUwbBackend
import io.silvermind.wearuwbprobe.uwb.PlatformRangingBackend
import io.silvermind.wearuwbprobe.uwb.UwbProbeBackend
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Connected-device foreground service executing bench Stages P1 to P4 so UWB, BLE OOB,
 * IMU quaternions, and coulomb-counter battery profiling continue with the screen off.
 */
class ProbeForegroundService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var activeStageJob: Job? = null
    private var sensorCollector: AndroidSensorCollector? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        ensureNotificationChannel()
        startForeground(
            NOTIFICATION_ID,
            buildNotification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
        )

        val extras = extractStringExtras(intent)
        val command = ProbeCommand.parse(extras)
        if (command == null) {
            Log.w(TAG, "Ignoring invalid probe command extras")
            return START_NOT_STICKY
        }

        if (command.stage == ProbeStage.STOP) {
            stopActiveStage()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        stopActiveStage()
        activeStageJob = serviceScope.launch { executeStage(command) }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopActiveStage()
        serviceScope.cancel()
        super.onDestroy()
    }

    private suspend fun executeStage(command: ProbeCommand) {
        val outputDir = (getExternalFilesDir(null) ?: filesDir).toPath()
        val recorder = ProbeTelemetryRecorder(outputDir)
        val orientationSync = OrientationSynchronizer()
        val batteryProfiler = BatteryProfiler()
        val clock = AndroidProbeClock()
        val batteryReader = AndroidBatteryReader(this)

        val gmsBackend = GmsNearbyUwbBackend(this)
        val jetpackBackend = JetpackUwbBackend(this)
        val platformBackend = PlatformRangingBackend(this)
        val allBackends: List<UwbProbeBackend> = listOf(gmsBackend, jetpackBackend, platformBackend)

        val stsNegotiator =
            PreSharedStsNegotiator(
                staticStsVendorId = ProbeExperimentRunner.DEFAULT_STATIC_STS_KEY_INFO.copyOfRange(0, 2),
                staticStsIv = ProbeExperimentRunner.DEFAULT_STATIC_STS_KEY_INFO.copyOfRange(2, 8),
                advertiser = AndroidBleWakeAdvertiser(this),
            )
        val gattNegotiator = BleGattOobNegotiator(AndroidBleGattPeerClient(this))

        val runner =
            ProbeExperimentRunner(
                backends = allBackends,
                negotiators =
                    mapOf(
                        OobStrategy.PRESHARED_STS_BROADCAST to stsNegotiator,
                        OobStrategy.BLE_GATT_SEQUENTIAL to gattNegotiator,
                    ),
                recorder = recorder,
                orientationSynchronizer = orientationSync,
                batteryProfiler = batteryProfiler,
                clock = clock,
            )

        val primaryBackend = selectPrimaryBackend(command.backendFilter, gmsBackend, jetpackBackend)
        when (command.stage) {
            ProbeStage.P1 -> {
                val results =
                    runner.runStageP1(
                        backendFilter = command.backendFilter,
                        screenOffForegroundService = command.screenOffTest,
                    )
                Log.i(TAG, "Stage P1 completed (${results.size} backends audited) -> $outputDir")
            }
            ProbeStage.P2 -> {
                val outcomes =
                    runner.runStageP2(
                        backend = primaryBackend,
                        strategy = command.oobStrategy,
                        peers = command.peers,
                        trials = command.trials,
                        sessionDurationMs = command.durationSeconds * 1_000L,
                    )
                Log.i(TAG, "Stage P2 completed (${outcomes.size} trials) -> $outputDir")
            }
            ProbeStage.P3 -> {
                val collector = AndroidSensorCollector(this, orientationSync)
                sensorCollector = collector
                collector.start(enableOrientation = true, enableLocomotion = false)
                try {
                    runner.runStageP3(
                        backend = primaryBackend,
                        strategy = command.oobStrategy,
                        peers = command.peers,
                        pointId = command.pointId,
                        facingDir = command.facingDir,
                        durationMs = command.durationSeconds * 1_000L,
                        wristRotationTrial = command.wristRotationTrial,
                    )
                } finally {
                    collector.stop()
                    sensorCollector = null
                }
                Log.i(TAG, "Stage P3 completed (${command.pointId}/${command.facingDir}) -> $outputDir")
            }
            ProbeStage.P4 -> {
                runStageP4Loop(command, runner, primaryBackend, batteryReader, orientationSync)
                Log.i(TAG, "Stage P4 completed (regime=${command.regime}) -> $outputDir")
            }
            ProbeStage.STOP -> Unit
        }
    }

    private suspend fun runStageP4Loop(
        command: ProbeCommand,
        runner: ProbeExperimentRunner,
        backend: UwbProbeBackend,
        batteryReader: AndroidBatteryReader,
        orientationSync: OrientationSynchronizer,
    ) {
        val fsm: LocomotionDutyCycleFsm = runner.createDutyCycleFsm()
        val regime = command.regime
        val startMs = System.currentTimeMillis()
        val endMs = startMs + command.durationSeconds * 1_000L

        val listener =
            object : LocomotionSensorListener {
                override fun onWearStateChanged(
                    elapsedRealtimeNanos: Long,
                    onBody: Boolean,
                ) {
                    val prev = fsm.state
                    val tr = if (onBody) fsm.onOnBody(elapsedRealtimeNanos) else fsm.onOffBody(elapsedRealtimeNanos)
                    runner.recordFsmTransition(regime, prev, tr)
                }

                override fun onStepDetected(elapsedRealtimeNanos: Long) {
                    val snap = batteryReader.readSnapshot()
                    val prev = fsm.state
                    val tr = fsm.onStepDetected(elapsedRealtimeNanos, snap.capacityPercent, snap.thermalStatus)
                    runner.recordFsmTransition(regime, prev, tr)
                    if (regime == BatteryRegime.C_MOTION_DUTY_CYCLED && tr.action == DutyCycleAction.START_UWB_BURST) {
                        serviceScope.launch {
                            acquireBurstWakeLock()
                            runner.recordBatteryCheckpoint(regime, "burst_start", batteryReader, fsm)
                            try {
                                runner.runStageP2(
                                    backend = backend,
                                    strategy = command.oobStrategy,
                                    peers = command.peers,
                                    trials = 1,
                                    sessionDurationMs = 12_000L,
                                )
                            } finally {
                                runner.recordBatteryCheckpoint(regime, "burst_end", batteryReader, fsm)
                                releaseBurstWakeLock()
                            }
                        }
                    }
                }

                override fun onLinearAcceleration(
                    elapsedRealtimeNanos: Long,
                    magnitudeMps2: Double,
                ) {
                    fsm.onLinearAcceleration(elapsedRealtimeNanos, magnitudeMps2)
                }
            }

        val collector = AndroidSensorCollector(this, orientationSync, listener)
        sensorCollector = collector
        collector.start(enableOrientation = false, enableLocomotion = true)
        runner.recordBatteryCheckpoint(regime, "regime_start", batteryReader, fsm)

        try {
            if (regime == BatteryRegime.B_CONTINUOUS_UWB) {
                val sampler =
                    serviceScope.launch {
                        while (isActive && System.currentTimeMillis() < endMs) {
                            delay(BatteryProfiler.SAMPLE_INTERVAL_MS)
                            runner.recordBatteryCheckpoint(regime, "periodic_10s", batteryReader, fsm)
                        }
                    }
                try {
                    runner.runStageP2(
                        backend = backend,
                        strategy = command.oobStrategy,
                        peers = command.peers,
                        trials = 1,
                        sessionDurationMs = command.durationSeconds * 1_000L,
                    )
                } finally {
                    sampler.cancel()
                }
            } else {
                while (serviceScope.isActive && System.currentTimeMillis() < endMs) {
                    delay(BatteryProfiler.SAMPLE_INTERVAL_MS)
                    val snap = batteryReader.readSnapshot()
                    val prev = fsm.state
                    val tr = fsm.tick(snap.elapsedRealtimeNanos, snap.capacityPercent, snap.thermalStatus)
                    runner.recordFsmTransition(regime, prev, tr)
                    runner.recordBatteryCheckpoint(regime, "periodic_10s", batteryReader, fsm)
                }
            }
        } finally {
            runner.recordBatteryCheckpoint(regime, "regime_end", batteryReader, fsm)
            collector.stop()
            sensorCollector = null
            releaseBurstWakeLock()
        }
    }

    private fun selectPrimaryBackend(
        filter: String,
        gmsBackend: UwbProbeBackend,
        jetpackBackend: UwbProbeBackend,
    ): UwbProbeBackend =
        when (filter) {
            JetpackUwbBackend.BACKEND_NAME -> jetpackBackend
            else -> gmsBackend
        }

    private fun stopActiveStage() {
        activeStageJob?.cancel()
        activeStageJob = null
        sensorCollector?.stop()
        sensorCollector = null
        releaseBurstWakeLock()
    }

    private fun acquireBurstWakeLock() {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val lock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$TAG:uwb_burst")
        lock.acquire(MAX_BURST_WAKELOCK_MS)
        wakeLock = lock
    }

    private fun releaseBurstWakeLock() {
        wakeLock?.let { lock ->
            if (lock.isHeld) {
                lock.release()
            }
        }
        wakeLock = null
    }

    private fun ensureNotificationChannel() {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel =
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.probe_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            )
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification =
        Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.probe_notification_title))
            .setContentText(getString(R.string.probe_notification_body))
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setOngoing(true)
            .build()

    private fun extractStringExtras(intent: Intent?): Map<String, String?> {
        val bundle = intent?.extras ?: return emptyMap()
        val result = linkedMapOf<String, String?>()
        for (key in bundle.keySet()) {
            val value = bundle.get(key)
            result[key] = value?.toString()
        }
        return result
    }

    companion object {
        private const val TAG: String = "WearUwbProbeService"
        private const val CHANNEL_ID: String = "wear_uwb_probe_channel"
        private const val NOTIFICATION_ID: Int = 2501
        private const val MAX_BURST_WAKELOCK_MS: Long = 15_000L
    }
}
