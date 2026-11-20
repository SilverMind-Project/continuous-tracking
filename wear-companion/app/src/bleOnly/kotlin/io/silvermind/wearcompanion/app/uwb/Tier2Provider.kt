package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import io.silvermind.wearcompanion.core.EpochBatchPublisher
import io.silvermind.wearcompanion.core.IdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.UwbAnchorEntry
import io.silvermind.wearcompanion.core.UwbRangingEngine

/** Runtime Tier 2 dependency bundle (empty in the zero-network `bleOnly` flavor). */
data class Tier2RuntimeConfig(
    val mode: RuntimeMode,
    val anchors: List<UwbAnchorEntry> = emptyList(),
    val engine: UwbRangingEngine? = null,
    val batchPublisher: EpochBatchPublisher? = null,
)

/** Compile-time no-op Tier 2 provider for the `bleOnly` product flavor. */
object Tier2Provider {
    @Suppress("UNUSED_PARAMETER")
    fun buildConfig(context: Context, identifierStore: IdentifierStore): Tier2RuntimeConfig =
        Tier2RuntimeConfig(mode = RuntimeMode.BLE_ONLY)
}
