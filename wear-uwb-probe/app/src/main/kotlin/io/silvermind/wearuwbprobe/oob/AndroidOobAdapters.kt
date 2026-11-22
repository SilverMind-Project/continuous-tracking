package io.silvermind.wearuwbprobe.oob

import android.annotation.SuppressLint
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.AdvertiseCallback
import android.bluetooth.le.AdvertiseData
import android.bluetooth.le.AdvertiseSettings
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanFilter
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.ParcelUuid
import android.os.SystemClock
import io.silvermind.wearuwbprobe.uwb.AnchorPeerSpec
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeoutOrNull
import java.util.UUID

/**
 * Android BLE low-latency non-connectable broadcaster for the 10-byte Pre-Shared Static STS
 * wake payload (`0xFFFF` manufacturer ID, 300 ms burst).
 */
class AndroidBleWakeAdvertiser(
    private val context: Context,
) : BleWakeAdvertiser {
    @SuppressLint("MissingPermission")
    override suspend fun broadcastWakePayload(
        manufacturerId: Int,
        payload: ByteArray,
        durationMs: Long,
    ): Long {
        val manager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
        val advertiser =
            manager?.adapter?.bluetoothLeAdvertiser
                ?: throw IllegalStateException("BluetoothLeAdvertiser unavailable")

        val startMs = SystemClock.elapsedRealtime()
        val started = CompletableDeferred<Unit>()
        val settings =
            AdvertiseSettings.Builder()
                .setAdvertiseMode(AdvertiseSettings.ADVERTISE_MODE_LOW_LATENCY)
                .setTxPowerLevel(AdvertiseSettings.ADVERTISE_TX_POWER_HIGH)
                .setConnectable(false)
                .setTimeout(durationMs.toInt().coerceIn(100, 5_000))
                .build()
        val data =
            AdvertiseData.Builder()
                .setIncludeDeviceName(false)
                .addManufacturerData(manufacturerId, payload)
                .build()

        val callback =
            object : AdvertiseCallback() {
                override fun onStartSuccess(settingsInEffect: AdvertiseSettings) {
                    started.complete(Unit)
                }

                override fun onStartFailure(errorCode: Int) {
                    started.completeExceptionally(
                        IllegalStateException("BLE advertise failed with code $errorCode"),
                    )
                }
            }

        advertiser.startAdvertising(settings, data, callback)
        try {
            started.await()
            delay(durationMs)
        } finally {
            runCatching { advertiser.stopAdvertising(callback) }
        }
        return (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L)
    }
}

/**
 * Sequential FiRa BLE GATT OOB baseline client used to measure GATT setup latency against
 * fixed room anchors in bench latency trials.
 */
class AndroidBleGattPeerClient(
    private val context: Context,
    private val perPeerTimeoutMs: Long = 3_000L,
) : FiRaBleGattPeerClient {
    @SuppressLint("MissingPermission")
    override suspend fun exchangeSessionParameters(
        peer: AnchorPeerSpec,
        configPayload: ByteArray,
    ): PeerGattExchangeOutcome {
        val startMs = SystemClock.elapsedRealtime()
        val manager = context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager
        val scanner =
            manager?.adapter?.bluetoothLeScanner
                ?: return PeerGattExchangeOutcome(
                    succeeded = false,
                    latencyMs = 0L,
                    errorDetail = "BluetoothLeScanner unavailable",
                )

        val serviceUuid = UUID.fromString(BleGattOobNegotiator.FIRA_OOB_SERVICE_UUID)
        val rxUuid = UUID.fromString(BleGattOobNegotiator.FIRA_OOB_RX_CHAR_UUID)
        val foundDevice = CompletableDeferred<ScanResult?>()
        val scanCallback =
            object : ScanCallback() {
                override fun onScanResult(
                    callbackType: Int,
                    result: ScanResult,
                ) {
                    val name = result.device?.name.orEmpty()
                    if (name.contains(peer.anchorId, ignoreCase = true) ||
                        name.contains(peer.normalizedAddressHex().replace(":", ""), ignoreCase = true)
                    ) {
                        foundDevice.complete(result)
                    }
                }

                override fun onScanFailed(errorCode: Int) {
                    foundDevice.complete(null)
                }
            }

        val filters = listOf(ScanFilter.Builder().setServiceUuid(ParcelUuid(serviceUuid)).build())
        val scanSettings = ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build()
        scanner.startScan(filters, scanSettings, scanCallback)
        val scanResult =
            try {
                withTimeoutOrNull(perPeerTimeoutMs / 2L) { foundDevice.await() }
            } finally {
                runCatching { scanner.stopScan(scanCallback) }
            }

        if (scanResult == null) {
            return PeerGattExchangeOutcome(
                succeeded = false,
                latencyMs = SystemClock.elapsedRealtime() - startMs,
                errorDetail = "Peer ${peer.anchorId} not found via BLE scan",
            )
        }

        val writeOutcome = CompletableDeferred<Boolean>()
        var activeGatt: BluetoothGatt? = null
        val gattCallback =
            object : BluetoothGattCallback() {
                override fun onConnectionStateChange(
                    gatt: BluetoothGatt,
                    status: Int,
                    newState: Int,
                ) {
                    if (newState == BluetoothProfile.STATE_CONNECTED && status == BluetoothGatt.GATT_SUCCESS) {
                        gatt.discoverServices()
                    } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                        writeOutcome.complete(false)
                    }
                }

                override fun onServicesDiscovered(
                    gatt: BluetoothGatt,
                    status: Int,
                ) {
                    val characteristic = gatt.getService(serviceUuid)?.getCharacteristic(rxUuid)
                    if (status != BluetoothGatt.GATT_SUCCESS || characteristic == null) {
                        writeOutcome.complete(false)
                        return
                    }
                    val writeStatus =
                        gatt.writeCharacteristic(
                            characteristic,
                            configPayload,
                            BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT,
                        )
                    if (writeStatus != BluetoothStatusCodes.SUCCESS) {
                        writeOutcome.complete(false)
                    }
                }

                override fun onCharacteristicWrite(
                    gatt: BluetoothGatt,
                    characteristic: BluetoothGattCharacteristic,
                    status: Int,
                ) {
                    writeOutcome.complete(status == BluetoothGatt.GATT_SUCCESS)
                }
            }

        activeGatt = scanResult.device.connectGatt(context, false, gattCallback)
        val ok =
            try {
                withTimeoutOrNull(perPeerTimeoutMs) { writeOutcome.await() } ?: false
            } finally {
                runCatching { activeGatt?.close() }
            }

        return PeerGattExchangeOutcome(
            succeeded = ok,
            latencyMs = (SystemClock.elapsedRealtime() - startMs).coerceAtLeast(0L),
            errorDetail = if (ok) null else "GATT parameter write failed or timed out",
        )
    }
}
