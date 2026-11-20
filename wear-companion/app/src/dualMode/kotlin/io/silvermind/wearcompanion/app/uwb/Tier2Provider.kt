package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import io.silvermind.wearcompanion.app.CompanionForegroundService
import io.silvermind.wearcompanion.core.EpochBatchPublisher
import io.silvermind.wearcompanion.core.IdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.UwbAnchorEntry
import io.silvermind.wearcompanion.core.UwbRangingEngine

/** Runtime Tier 2 dependency bundle for the `dualMode` product flavor. */
data class Tier2RuntimeConfig(
    val mode: RuntimeMode,
    val anchors: List<UwbAnchorEntry> = emptyList(),
    val engine: UwbRangingEngine? = null,
    val batchPublisher: EpochBatchPublisher? = null,
)

/**
 * Dual-mode Tier 2 provider.
 *
 * Defaults strictly to [RuntimeMode.BLE_ONLY] unless an operator explicitly
 * configures household UWB anchors, a Static STS key, and a local MQTT broker
 * after the physical Wear OS UWB probe passes.
 */
object Tier2Provider {
    fun buildConfig(
        context: Context,
        identifierStore: IdentifierStore,
        anchors: List<UwbAnchorEntry> = emptyList(),
        brokerUrl: String? = null,
        sessionKeyInfo: ByteArray? = null,
        preferJetpackBackend: Boolean = false,
    ): Tier2RuntimeConfig {
        if (anchors.isEmpty() || brokerUrl.isNullOrBlank() || sessionKeyInfo == null) {
            return Tier2RuntimeConfig(mode = RuntimeMode.BLE_ONLY)
        }
        val tagId = identifierStore.loadOrCreate().hardwareKey
        val engine: UwbRangingEngine =
            if (preferJetpackBackend) {
                JetpackUwbEngine(context, sessionKeyInfo)
            } else {
                GmsNearbyUwbEngine(context, sessionKeyInfo)
            }
        val transport = MqttBatchTransport(brokerUrl = brokerUrl, clientId = "wear-$tagId")
        val publisher =
            EpochBatchPublisher(
                tagId = tagId,
                bootUnixNs = CompanionForegroundService.bootUnixAnchorNanos(),
                sender = transport,
            )
        return Tier2RuntimeConfig(
            mode = RuntimeMode.BLE_PLUS_UWB_BURST,
            anchors = anchors,
            engine = engine,
            batchPublisher = publisher,
        )
    }
}
