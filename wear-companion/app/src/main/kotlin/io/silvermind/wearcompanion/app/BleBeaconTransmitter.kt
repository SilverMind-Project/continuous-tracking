package io.silvermind.wearcompanion.app

import android.annotation.SuppressLint
import android.bluetooth.BluetoothManager
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.bluetooth.le.BluetoothLeAdvertiser
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.silvermind.wearcompanion.core.BleAdvertiserSink
import io.silvermind.wearcompanion.core.IBeaconFrameEncoder
import io.silvermind.wearcompanion.core.PreSharedStsNegotiator

/**
 * Wraps [BluetoothLeAdvertiser] for non-connectable Tier 1 iBeacon frames and
 * short low-latency Tier 2 UWB wake broadcasts.
 *
 * iBeacon frames use Apple's company ID (`0x004C`), which is what room receivers
 * expect. Wake frames use [PreSharedStsNegotiator.WAKE_MANUFACTURER_ID] so a
 * receiver never parses one as an iBeacon. Calls arrive from the coordinator's
 * coroutine and from the main-thread restore timer, so every method is synchronized.
 */
class BleBeaconTransmitter(
    private val context: Context,
) : BleAdvertiserSink {
    private val handler = Handler(Looper.getMainLooper())
    private var currentBeaconPayload: ByteArray? = null
    private var activeCallback: AdvertiseCallback? = null
    private var wakeGeneration: Long = 0L

    private fun advertiser(): BluetoothLeAdvertiser? {
        val manager = context.getSystemService(BluetoothManager::class.java)
        val adapter = manager?.adapter ?: return null
        if (!adapter.isEnabled) {
            return null
        }
        return adapter.bluetoothLeAdvertiser
    }

    @Synchronized
    override fun startIBeacon(payload: ByteArray) {
        currentBeaconPayload = payload.copyOf()
        wakeGeneration += 1
        startAdvertisement(
            payload = payload,
            manufacturerId = IBeaconFrameEncoder.APPLE_COMPANY_ID,
            mode = AdvertiseSettings.ADVERTISE_MODE_BALANCED,
            txPower = AdvertiseSettings.ADVERTISE_TX_POWER_LOW,
            timeoutMs = 0,
        )
    }

    @Synchronized
    override fun stopAdvertising() {
        currentBeaconPayload = null
        wakeGeneration += 1
        stopActive()
    }

    @Synchronized
    override fun broadcastWakeFrame(
        payload: ByteArray,
        durationMs: Long,
    ) {
        val generation = ++wakeGeneration
        startAdvertisement(
            payload = payload,
            manufacturerId = PreSharedStsNegotiator.WAKE_MANUFACTURER_ID,
            mode = AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY,
            txPower = AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM,
            // The controller's own timeout is a backstop if the restore below never runs.
            timeoutMs = durationMs.coerceIn(1L, MAX_ADVERTISE_TIMEOUT_MS).toInt(),
        )
        handler.postDelayed({ restoreAfterWake(generation) }, durationMs)
    }

    @Synchronized
    private fun restoreAfterWake(generation: Long) {
        if (generation != wakeGeneration) {
            // A newer beacon start, stop, or wake already replaced this wake frame.
            return
        }
        val beacon = currentBeaconPayload
        if (beacon != null) {
            startIBeacon(beacon)
        } else {
            stopActive()
        }
    }

    @SuppressLint("MissingPermission")
    private fun stopActive() {
        val adv = advertiser()
        val cb = activeCallback
        activeCallback = null
        if (adv != null && cb != null) {
            guarded { adv.stopAdvertising(cb) }
        }
    }

    @SuppressLint("MissingPermission")
    private fun startAdvertisement(
        payload: ByteArray,
        manufacturerId: Int,
        mode: Int,
        txPower: Int,
        timeoutMs: Int,
    ) {
        val adv = advertiser() ?: return
        val existing = activeCallback
        activeCallback = null
        if (existing != null) {
            guarded { adv.stopAdvertising(existing) }
        }
        val settings =
            AdvertiseSettings.Builder()
                .setAdvertiseMode(mode)
                .setTxPowerLevel(txPower)
                .setConnectable(false)
                .setTimeout(timeoutMs)
                .build()
        val data =
            AdvertiseData.Builder()
                .setIncludeDeviceName(false)
                .setIncludeTxPowerLevel(false)
                .addManufacturerData(manufacturerId, payload)
                .build()
        val callback = object : AdvertiseCallback() {}
        if (guarded { adv.startAdvertising(settings, data, callback) }) {
            activeCallback = callback
        }
    }

    /**
     * Runs a Bluetooth call that throws `SecurityException` without the runtime
     * `BLUETOOTH_ADVERTISE` grant. The wake restore runs on the main thread, where
     * an uncaught exception would kill the process, so failures are logged instead.
     */
    private inline fun guarded(call: () -> Unit): Boolean =
        try {
            call()
            true
        } catch (err: SecurityException) {
            Log.w(TAG, "BLE advertising refused: ${err::class.java.simpleName}")
            false
        }

    private companion object {
        const val TAG: String = "BleBeaconTransmitter"

        /** `AdvertiseSettings.Builder.setTimeout` accepts at most 180 000 ms. */
        const val MAX_ADVERTISE_TIMEOUT_MS: Long = 180_000L
    }
}
