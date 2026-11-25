package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import io.silvermind.wearcompanion.core.BurstWakeHold
import io.silvermind.wearcompanion.core.EpochBatchPublisher
import io.silvermind.wearcompanion.core.IdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.UwbAnchorEntry
import io.silvermind.wearcompanion.core.UwbRangingEngine

/**
 * Runtime Tier 2 dependency bundle (always empty in the zero-network `bleOnly`
 * flavor). Its shape must match the `dualMode` copy, because main-source-set
 * code compiles against whichever flavor is built.
 */
data class Tier2RuntimeConfig(
    val mode: RuntimeMode,
    val status: String,
    val anchors: List<UwbAnchorEntry> = emptyList(),
    val engine: UwbRangingEngine? = null,
    val batchPublisher: EpochBatchPublisher? = null,
    val wakeHold: BurstWakeHold? = null,
    val close: () -> Unit = {},
)

/** Compile-time no-op Tier 2 provider for the `bleOnly` product flavor. */
object Tier2Provider {
    @Suppress("UNUSED_PARAMETER")
    fun buildConfig(
        context: Context,
        identifierStore: IdentifierStore,
    ): Tier2RuntimeConfig = Tier2RuntimeConfig(mode = RuntimeMode.BLE_ONLY, status = "bleOnly build")
}
