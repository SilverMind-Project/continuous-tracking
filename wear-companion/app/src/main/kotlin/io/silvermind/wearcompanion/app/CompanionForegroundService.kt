package io.silvermind.wearcompanion.app

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
import io.silvermind.wearcompanion.app.uwb.Tier2Provider
import io.silvermind.wearcompanion.core.CompanionCoordinator
import io.silvermind.wearcompanion.core.FileIdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.ThermalSeverity
import io.silvermind.wearcompanion.core.TileStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Foreground service (`connectedDevice`) owning Tier 1 on-body BLE beacon
 * advertising and optional Tier 2 motion-duty-cycled UWB bursts.
 */
class CompanionForegroundService : Service(), WatchSensorCallbacks {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private lateinit var coordinator: CompanionCoordinator
    private lateinit var sensorMonitor: OffBodySensorMonitor
    private lateinit var bleTransmitter: BleBeaconTransmitter

    private val bluetoothReceiver =
        object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                if (intent.action == BluetoothAdapter.ACTION_STATE_CHANGED) {
                    val state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
                    val enabled = state == BluetoothAdapter.STATE_ON
                    val snap = coordinator.updateBluetooth(enabled)
                    lastTileStatus = snap.tileStatus
                }
            }
        }

    override fun onCreate() {
        super.onCreate()
        val idStore = FileIdentifierStore(filesDir.toPath().resolve(IDENTIFIER_FILE_NAME))
        bleTransmitter = BleBeaconTransmitter(this)
        val tier2Config = Tier2Provider.buildConfig(this, idStore)
        coordinator =
            CompanionCoordinator(
                identifierStore = idStore,
                bleSink = bleTransmitter,
                mode = tier2Config.mode,
                uwbAnchors = tier2Config.anchors,
                uwbEngine = tier2Config.engine,
                batchPublisher = tier2Config.batchPublisher,
            )
        val btManager = getSystemService(BluetoothManager::class.java)
        coordinator.updateBluetooth(btManager?.adapter?.isEnabled == true)
        lastTileStatus = coordinator.snapshot().tileStatus

        sensorMonitor =
            OffBodySensorMonitor(
                context = this,
                callbacks = this,
                enableLocomotionSensors = tier2Config.mode == RuntimeMode.BLE_PLUS_UWB_BURST,
            )
        registerReceiver(bluetoothReceiver, IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED))
        sensorMonitor.start()
        startForeground(
            NOTIFICATION_ID,
            buildNotification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE,
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_REFRESH) {
            lastTileStatus = coordinator.snapshot().tileStatus
        }
        return START_STICKY
    }

    override fun onDestroy() {
        sensorMonitor.stop()
        unregisterReceiver(bluetoothReceiver)
        bleTransmitter.stopAdvertising()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onBodyStateChanged(nowNanos: Long, onBody: Boolean) {
        serviceScope.launch {
            val snap = coordinator.updateBodyState(nowNanos, onBody, currentBatteryPercent() / 100.0)
            lastTileStatus = snap.tileStatus
        }
    }

    override fun onStepDetected(nowNanos: Long) {
        serviceScope.launch {
            val snap =
                coordinator.onStepDetected(
                    nowNanos = nowNanos,
                    batteryPercent = currentBatteryPercent(),
                    thermalSeverity = currentThermalSeverity(),
                )
            lastTileStatus = snap.tileStatus
        }
    }

    override fun onLinearAcceleration(nowNanos: Long, magnitudeMps2: Double) {
        coordinator.onLinearAcceleration(nowNanos, magnitudeMps2)
    }

    private fun currentBatteryPercent(): Int {
        val manager = getSystemService(BatteryManager::class.java)
        val pct = manager?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: 100
        return pct.coerceIn(0, 100)
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
        private const val CHANNEL_ID: String = "silvermind_beacon_channel"
        private const val NOTIFICATION_ID: Int = 1001

        @Volatile
        var lastTileStatus: TileStatus = TileStatus.UNPAIRED
            private set

        fun bootUnixAnchorNanos(): Long {
            val nowUnixNs = System.currentTimeMillis() * 1_000_000L
            return nowUnixNs - SystemClock.elapsedRealtimeNanos()
        }
    }
}
