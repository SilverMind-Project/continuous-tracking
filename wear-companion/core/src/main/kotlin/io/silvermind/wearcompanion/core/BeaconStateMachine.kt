package io.silvermind.wearcompanion.core

/** Tier 1 lifecycle state for the on-body BLE beacon advertiser. */
enum class BeaconState {
    UNPAIRED,
    IDLE,
    ADVERTISING,
}

/** Human-readable status rendered on the Wear OS status tile. */
enum class TileStatus {
    UNPAIRED,
    ADVERTISING,
    IDLE_NOT_WORN,
    BLUETOOTH_OFF,
}

/** Immutable sensor and pairing inputs evaluated by [BeaconStateMachine]. */
data class BeaconInputs(
    val paired: Boolean,
    val onBody: Boolean,
    val bluetoothEnabled: Boolean,
    val screenOn: Boolean = true,
)

/** Resulting state and concrete advertising command from a transition step. */
data class BeaconTransition(
    val state: BeaconState,
    val shouldAdvertise: Boolean,
    val tileStatus: TileStatus,
)

/**
 * Pure Tier 1 state machine governing BLE beacon advertising.
 *
 * Advertising runs strictly when paired, worn on-body, and Bluetooth is on.
 * Turning the screen off never stops advertising; removing the watch or
 * disabling Bluetooth stops advertising immediately with no fallback timer.
 */
class BeaconStateMachine {
    var state: BeaconState = BeaconState.UNPAIRED
        private set

    fun evaluate(inputs: BeaconInputs): BeaconTransition {
        val nextState =
            when {
                !inputs.paired -> BeaconState.UNPAIRED
                inputs.onBody && inputs.bluetoothEnabled -> BeaconState.ADVERTISING
                else -> BeaconState.IDLE
            }
        state = nextState
        val tile =
            when {
                !inputs.paired -> TileStatus.UNPAIRED
                !inputs.bluetoothEnabled -> TileStatus.BLUETOOTH_OFF
                inputs.onBody -> TileStatus.ADVERTISING
                else -> TileStatus.IDLE_NOT_WORN
            }
        return BeaconTransition(
            state = nextState,
            shouldAdvertise = nextState == BeaconState.ADVERTISING,
            tileStatus = tile,
        )
    }
}
