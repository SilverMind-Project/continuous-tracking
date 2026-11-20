package io.silvermind.wearcompanion.core

/**
 * Runtime operating tier for the Wear OS companion app.
 *
 * [BLE_ONLY] is always the default so a watch requires zero Wi-Fi/UWB hardware
 * unless UWB anchors are explicitly configured and enabled.
 */
enum class RuntimeMode {
    BLE_ONLY,
    BLE_PLUS_UWB_BURST,
}

/** Wire vocabulary mirroring cts-contracts WearState. */
enum class WearStateWire(val wireValue: String) {
    UNSPECIFIED("unspecified"),
    ON_BODY("on_body"),
    OFF_BODY("off_body"),
}

/** Wire vocabulary mirroring cts-contracts MotionState. */
enum class MotionStateWire(val wireValue: String) {
    UNSPECIFIED("unspecified"),
    MOVING("moving"),
    STILL("still"),
}

/** Wire vocabulary mirroring cts-contracts RangingTechnology. */
enum class RangingTechnologyWire(val wireValue: String) {
    UWB("uwb"),
    BLE_CHANNEL_SOUNDING("ble_channel_sounding"),
    BLE_RSSI("ble_rssi"),
}
