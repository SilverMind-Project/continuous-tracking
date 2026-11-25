package io.silvermind.wearcompanion.app

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.os.BatteryManager
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import io.silvermind.wearcompanion.app.uwb.Tier2Provider
import io.silvermind.wearcompanion.app.uwb.Tier2RuntimeConfig
import io.silvermind.wearcompanion.core.CompanionCoordinator
import io.silvermind.wearcompanion.core.CoordinatorSnapshot
import io.silvermind.wearcompanion.core.FileIdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.ThermalSeverity
import io.silvermind.wearcompanion.core.TileStatus
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Foreground service (`connectedDevice`) owning Tier 1 on-body BLE beacon
 * advertising and optional Tier 2 motion-duty-cycled UWB bursts.
 *
 * All coordinator work runs on one serial dispatcher, so sensor samples keep
 * their order. It is an IO dispatcher because a micro-batch flush makes a
 * blocking LAN MQTT call. A Tier 2 reload detaches the running coordinator
 * first, so no sensor event can start a burst on a coordinator being replaced,
 * and every reload goes through one conflated channel so two never interleave.
 */
class CompanionForegroundService : Service(), WatchSensorCallbacks {
    private val serviceScope =
        CoroutineScope(
            SupervisorJob() +
                Dispatchers.IO.limitedParallelism(1) +
                CoroutineExceptionHandler { _, err -> Log.e(TAG, "Coordinator task failed: ${err::class.java.simpleName}") },
        )
    private lateinit var idStore: FileIdentifierStore
    private lateinit var bleTransmitter: BleBeaconTransmitter

    @Volatile
    private var coordinator: CompanionCoordinator? = null
    @Volatile
    private var tier2: Tier2RuntimeConfig? = null
    private var sensorMonitor: OffBodySensorMonitor? = null
    private var tickerJob: Job? = null

    @Volatile
    private var lastOnBody: Boolean? = null

    private val bluetoothReceiver =
        object : BroadcastReceiver() {
            override fun onReceive(
                context: Context,
                intent: Intent,
            ) {
                if (intent.action == BluetoothAdapter.ACTION_STATE_CHANGED) {
                    val state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
                    val usable = state == BluetoothAdapter.STATE_ON && canAdvertise()
                    serviceScope.launch { publish(coordinator?.updateBluetooth(usable)) }
                }
            }
        }

    override fun onCreate() {
        super.onCreate()
        startForeground(
            NOTIFICATION_ID,
            buildNotification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
        )
        idStore = FileIdentifierStore(filesDir.toPath().resolve(IDENTIFIER_FILE_NAME))
        bleTransmitter = BleBeaconTransmitter(this)
        registerReceiver(bluetoothReceiver, IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED))
        serviceScope.launch {
            startTier(Tier2Provider.buildConfig(this@CompanionForegroundService, idStore))
            // receive() rather than consumeEach: the channel outlives this service
            // instance and must not be closed when its scope is cancelled.
            while (true) {
                reloadRequests.receive()
                reloadTier2()
            }
        }
        running = true
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int,
    ): Int {
        when (intent?.action) {
            // An identifier reset changes the default UWB tag_id, so it rebuilds Tier 2 too.
            ACTION_REFRESH, ACTION_RELOAD_TIER2 -> reloadRequests.trySend(Unit)
        }
        return START_STICKY
    }

    override fun onDestroy() {
        running = false
        sensorMonitor?.stop()
        unregisterReceiver(bluetoothReceiver)
        bleTransmitter.stopAdvertising()
        serviceScope.cancel()
        tier2?.let { config ->
            config.wakeHold?.release()
            config.close()
        }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onBodyStateChanged(
        nowNanos: Long,
        onBody: Boolean,
    ) {
        lastOnBody = onBody
        serviceScope.launch {
            publish(coordinator?.updateBodyState(nowNanos, onBody, currentBatteryPercent() / PERCENT))
        }
    }

    override fun onStepDetected(nowNanos: Long) {
        serviceScope.launch {
            publish(
                coordinator?.onStepDetected(
                    nowNanos = nowNanos,
                    batteryPercent = currentBatteryPercent(),
                    thermalSeverity = currentThermalSeverity(),
                ),
            )
        }
    }

    override fun onLinearAcceleration(
        nowNanos: Long,
        magnitudeMps2: Double,
    ) {
        serviceScope.launch { coordinator?.onLinearAcceleration(nowNanos, magnitudeMps2) }
    }

    private fun publish(snapshot: CoordinatorSnapshot?) {
        if (snapshot != null) {
            lastTileStatus = snapshot.tileStatus
        }
    }

    /** Build and start a coordinator for [config]; runs on the serial service scope. */
    private suspend fun startTier(config: Tier2RuntimeConfig) {
        Log.i(TAG, config.status)
        val tier2Mode = config.mode == RuntimeMode.BLE_PLUS_UWB_BURST
        val fresh =
            CompanionCoordinator(
                identifierStore = idStore,
                bleSink = bleTransmitter,
                mode = config.mode,
                uwbAnchors = config.anchors,
                uwbEngine = config.engine,
                batchPublisher = config.batchPublisher,
                scope = if (tier2Mode) serviceScope else null,
                wakeHold = config.wakeHold,
                onBurstError = { err -> Log.w(TAG, "UWB burst failed: ${err::class.java.simpleName}") },
            )
        tier2 = config
        val btManager = getSystemService(BluetoothManager::class.java)
        publish(fresh.updateBluetooth(btManager?.adapter?.isEnabled == true && canAdvertise()))
        lastOnBody?.let { publish(fresh.updateBodyState(SystemClock.elapsedRealtimeNanos(), it)) }
        coordinator = fresh

        val stepsPermitted = RequiredPermissions.isGranted(this, Manifest.permission.ACTIVITY_RECOGNITION)
        if (fresh.tier2Active() && !stepsPermitted) {
            Log.w(TAG, "ACTIVITY_RECOGNITION not granted: no step events, so no UWB bursts")
        }
        sensorMonitor =
            OffBodySensorMonitor(
                context = this,
                callbacks = this,
                enableLocomotionSensors = fresh.tier2Active() && stepsPermitted,
            ).also { it.start() }
        if (fresh.tier2Active()) {
            tickerJob = serviceScope.launch { tick(fresh) }
        }
    }

    private suspend fun tick(target: CompanionCoordinator) {
        while (serviceScope.isActive) {
            delay(TICK_INTERVAL_MS)
            publish(
                target.onTick(
                    nowNanos = SystemClock.elapsedRealtimeNanos(),
                    batteryPercent = currentBatteryPercent(),
                    thermalSeverity = currentThermalSeverity(),
                ),
            )
        }
    }

    /** Tear down the running coordinator, then start a new one from the current Tier 2 file. */
    private suspend fun reloadTier2() {
        val old = coordinator
        coordinator = null
        tickerJob?.cancel()
        tickerJob = null
        sensorMonitor?.stop()
        sensorMonitor = null
        old?.shutdown(SystemClock.elapsedRealtimeNanos())
        tier2?.let { config ->
            config.wakeHold?.release()
            config.close()
        }
        tier2 = null
        startTier(Tier2Provider.buildConfig(this, idStore))
    }

    /** Without the runtime grant the advertiser throws, so the tile must not claim to advertise. */
    private fun canAdvertise(): Boolean {
        val granted = RequiredPermissions.isGranted(this, Manifest.permission.BLUETOOTH_ADVERTISE)
        if (!granted) {
            Log.w(TAG, "BLUETOOTH_ADVERTISE not granted: beacon off until granted in the app")
        }
        return granted
    }

    private fun currentBatteryPercent(): Int {
        val manager = getSystemService(BatteryManager::class.java)
        val pct = manager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: FULL_BATTERY_PERCENT
        return pct.coerceIn(0, FULL_BATTERY_PERCENT)
    }

    private fun currentThermalSeverity(): ThermalSeverity {
        val pm = getSystemService(PowerManager::class.java) ?: return ThermalSeverity.NONE
        return when (pm.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> ThermalSeverity.NONE
            PowerManager.THERMAL_STATUS_LIGHT -> ThermalSeverity.LIGHT
            PowerManager.THERMAL_STATUS_MODERATE -> ThermalSeverity.MODERATE
            PowerManager.THERMAL_STATUS_SEVERE -> ThermalSeverity.SEVERE
            else -> ThermalSeverity.CRITICAL
        }
    }

    private fun buildNotification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        val channel =
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.notification_channel_name),
                NotificationManager.IMPORTANCE_LOW,
            )
        manager.createNotificationChannel(channel)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(getString(R.string.notification_text))
            .setOngoing(true)
            .build()
    }

    companion object {
        const val IDENTIFIER_FILE_NAME: String = "beacon_identifier.txt"
        const val ACTION_REFRESH: String = "io.silvermind.wearcompanion.REFRESH"
        const val ACTION_RELOAD_TIER2: String = "io.silvermind.wearcompanion.RELOAD_TIER2_SERVICE"
        private const val TAG: String = "CompanionService"
        private const val CHANNEL_ID: String = "silvermind_beacon_channel"
        private const val NOTIFICATION_ID: Int = 1001
        private const val TICK_INTERVAL_MS: Long = 1_000L
        private const val FULL_BATTERY_PERCENT: Int = 100
        private const val PERCENT: Double = 100.0

        // Conflated: requests made while a reload runs collapse into one more reload,
        // and a request made before the consumer starts is kept, not dropped.
        private val reloadRequests = Channel<Unit>(Channel.CONFLATED)

        @Volatile
        private var running: Boolean = false

        @Volatile
        var lastTileStatus: TileStatus = TileStatus.UNPAIRED
            private set

        /**
         * Ask the running service to rebuild Tier 2 from its config file, in
         * process, without a background foreground-service start. Returns false
         * when the service is not running (it reads the file when it starts).
         */
        fun requestTier2Reload(): Boolean {
            if (!running) {
                return false
            }
            return reloadRequests.trySend(Unit).isSuccess
        }

        fun bootUnixAnchorNanos(): Long {
            val nowUnixNs = System.currentTimeMillis() * 1_000_000L
            return nowUnixNs - SystemClock.elapsedRealtimeNanos()
        }
    }
}
