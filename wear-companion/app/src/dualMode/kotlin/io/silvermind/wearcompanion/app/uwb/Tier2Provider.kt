package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import io.silvermind.wearcompanion.app.CompanionForegroundService
import io.silvermind.wearcompanion.core.BurstWakeHold
import io.silvermind.wearcompanion.core.EpochBatchPublisher
import io.silvermind.wearcompanion.core.FileTier2ConfigStore
import io.silvermind.wearcompanion.core.IdentifierStore
import io.silvermind.wearcompanion.core.RuntimeMode
import io.silvermind.wearcompanion.core.Tier2ConfigState
import io.silvermind.wearcompanion.core.UwbAnchorEntry
import io.silvermind.wearcompanion.core.UwbRangingEngine

/**
 * Runtime Tier 2 dependency bundle for the `dualMode` product flavor.
 *
 * [status] is a short description safe to log (never a key or password), and
 * [close] releases the MQTT connection when the service stops or reloads.
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

/**
 * Dual-mode Tier 2 provider.
 *
 * Stays in [RuntimeMode.BLE_ONLY] unless `tier2_config.json` in app-private
 * storage is present, enabled, and valid. The file is written only by
 * [Tier2ProvisioningReceiver] (ADB shell) after the physical watch probe passes.
 */
object Tier2Provider {
    fun configFile(context: Context) = context.filesDir.toPath().resolve(FileTier2ConfigStore.DEFAULT_FILE_NAME)

    fun buildConfig(
        context: Context,
        identifierStore: IdentifierStore,
    ): Tier2RuntimeConfig {
        val settings =
            when (val state = FileTier2ConfigStore(configFile(context)).load()) {
                Tier2ConfigState.Missing -> return Tier2RuntimeConfig(RuntimeMode.BLE_ONLY, "tier2 not provisioned")
                Tier2ConfigState.Disabled -> return Tier2RuntimeConfig(RuntimeMode.BLE_ONLY, "tier2 disabled")
                is Tier2ConfigState.Invalid ->
                    return Tier2RuntimeConfig(RuntimeMode.BLE_ONLY, "tier2 config invalid: ${state.reason}")
                is Tier2ConfigState.Enabled -> state.settings
            }
        val tagId = settings.tagId ?: identifierStore.loadOrCreate().hardwareKey
        val engine: UwbRangingEngine =
            if (settings.preferJetpackBackend) {
                JetpackUwbEngine(context, settings.sessionKeyInfo)
            } else {
                GmsNearbyUwbEngine(context, settings.sessionKeyInfo)
            }
        val transport =
            MqttBatchTransport(
                brokerUrl = settings.brokerUrl,
                clientId = "wear-$tagId",
                username = settings.mqttUsername,
                password = settings.mqttPassword,
            )
        val publisher =
            EpochBatchPublisher(
                tagId = tagId,
                bootUnixNs = CompanionForegroundService.bootUnixAnchorNanos(),
                sender = transport,
            )
        return Tier2RuntimeConfig(
            mode = RuntimeMode.BLE_PLUS_UWB_BURST,
            status = "tier2 enabled: ${settings.anchors.size} anchors via ${engine.backendName}",
            anchors = settings.anchors,
            engine = engine,
            batchPublisher = publisher,
            wakeHold = PowerManagerBurstWakeHold(context),
            close = { transport.close() },
        )
    }
}
