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
import io.silvermind.wearcompanion.core.BleAdvertiserSink
import io.silvermind.wearcompanion.core.IBeaconFrameEncoder

/**
 * Wraps [BluetoothLeAdvertiser] for non-connectable Tier 1 iBeacon frames and
 * optional 300 ms low-latency Tier 2 UWB wake broadcasts.
 */
class BleBeaconTransmitter(
    private val context: Context,
) : BleAdvertiserSink {
    private val handler = Handler(Looper.getMainLooper())
    private var currentBeaconPayload: ByteArray? = null
    private var activeCallback: AdvertiseCallback? = null

    private fun advertiser(): BluetoothLeAdvertiser? {
        val manager = context.getSystemService(BluetoothManager::class.java)
        val adapter = manager?.adapter ?: return null
        if (!adapter.isEnabled) {
            return null
        }
        return adapter.bluetoothLeAdvertiser
    }

    @SuppressLint("MissingPermission")
    override fun startIBeacon(payload: ByteArray) {
        currentBeaconPayload = payload.copyOf()
        startAdvertisement(
            payload = payload,
            mode = AdvertiseSettings.ADVERTISE_MODE_BALANCED,
            txPower = AdvertiseSettings.ADVERTISE_TX_POWER_LOW,
        )
    }

    @SuppressLint("MissingPermission")
    override fun stopAdvertising() {
        currentBeaconPayload = null
        val adv = advertiser()
        val cb = activeCallback
        if (adv != null && cb != null) {
            adv.stopAdvertising(cb)
        }
        activeCallback = null
    }

    @SuppressLint("MissingPermission")
    override fun broadcastWakeFrame(payload: ByteArray, durationMs: Long) {
        val savedBeacon = currentBeaconPayload?.copyOf()
        startAdvertisement(
            payload = payload,
            mode = AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY,
            txPower = AdvertiseSettings.ADVERTISE_TX_POWER_MEDIUM,
        )
        handler.postDelayed(
            {
                if (savedBeacon != null && currentBeaconPayload != null) {
                    startIBeacon(savedBeacon)
                }
            },
            durationMs,
        )
    }

    @SuppressLint("MissingPermission")
    private fun startAdvertisement(
        payload: ByteArray,
        mode: Int,
        txPower: Int,
    ) {
        val adv = advertiser() ?: return
        val existing = activeCallback
        if (existing != null) {
            adv.stopAdvertising(existing)
        }
        val settings =
            AdvertiseSettings.Builder()
                .setAdvertiseMode(mode)
                .setTxPowerLevel(txPower)
                .setConnectable(false)
                .setTimeout(0)
                .build()
        val data =
            AdvertiseData.Builder()
                .setIncludeDeviceName(false)
                .setIncludeTxPowerLevel(false)
                .addManufacturerData(IBeaconFrameEncoder.APPLE_COMPANY_ID, payload)
                .build()
        val callback = object : AdvertiseCallback() {}
        activeCallback = callback
        adv.startAdvertising(settings, data, callback)
    }
}
