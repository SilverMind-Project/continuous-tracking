package io.silvermind.wearcompanion.app.uwb

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import io.silvermind.wearcompanion.app.CompanionForegroundService
import io.silvermind.wearcompanion.core.FileTier2ConfigStore
import io.silvermind.wearcompanion.core.Tier2ConfigCodec
import io.silvermind.wearcompanion.core.Tier2ConfigState

/**
 * ADB-only Tier 2 provisioning entry point (`dualMode` flavor).
 *
 * The manifest protects this receiver with `android.permission.DUMP`, which
 * the ADB shell holds and installed apps cannot obtain, so no other app on the
 * watch can point it at a different broker or replace the STS key. Usage:
 *
 * ```
 * adb shell am broadcast \
 *   -n io.silvermind.wearcompanion/io.silvermind.wearcompanion.app.uwb.Tier2ProvisioningReceiver \
 *   -a io.silvermind.wearcompanion.PROVISION_TIER2 --es config_b64 <base64 tier2_config.json>
 * ```
 *
 * `CLEAR_TIER2` deletes the file (back to BLE only) and `RELOAD_TIER2` re-reads
 * it. The broadcast result line reports the outcome; secrets are never logged.
 */
class Tier2ProvisioningReceiver : BroadcastReceiver() {
    override fun onReceive(
        context: Context,
        intent: Intent,
    ) {
        val store = FileTier2ConfigStore(Tier2Provider.configFile(context))
        val (ok, message) =
            when (intent.action) {
                ACTION_PROVISION -> provision(store, intent.getStringExtra(EXTRA_CONFIG_B64))
                ACTION_CLEAR -> true to if (store.clear()) "tier2 config cleared" else "no tier2 config to clear"
                ACTION_RELOAD -> true to describe(store.load())
                else -> return
            }
        val reload = if (ok) requestReload(context) else "service not reloaded"
        val result = "$message; $reload"
        if (ok) Log.i(TAG, result) else Log.w(TAG, result)
        if (isOrderedBroadcast) {
            setResult(if (ok) Activity.RESULT_OK else Activity.RESULT_CANCELED, result, null)
        }
    }

    private fun provision(
        store: FileTier2ConfigStore,
        encoded: String?,
    ): Pair<Boolean, String> {
        if (encoded.isNullOrBlank()) {
            return false to "missing --es $EXTRA_CONFIG_B64"
        }
        return try {
            true to describe(store.save(Tier2ConfigCodec.decodeBase64(encoded)))
        } catch (err: IllegalArgumentException) {
            false to "rejected: ${err.message}"
        }
    }

    private fun describe(state: Tier2ConfigState): String =
        when (state) {
            Tier2ConfigState.Missing -> "no tier2 config"
            Tier2ConfigState.Disabled -> "tier2 config saved, disabled"
            is Tier2ConfigState.Invalid -> "tier2 config invalid: ${state.reason}"
            is Tier2ConfigState.Enabled -> "tier2 config valid: ${state.settings.anchors.size} anchors"
        }

    private fun requestReload(context: Context): String {
        if (CompanionForegroundService.requestTier2Reload()) {
            return "running service reloading"
        }
        return runCatching {
            context.startForegroundService(Intent(context, CompanionForegroundService::class.java))
            "service starting"
        }.getOrElse { "service not running; open the app to start it (${it::class.java.simpleName})" }
    }

    companion object {
        const val ACTION_PROVISION: String = "io.silvermind.wearcompanion.PROVISION_TIER2"
        const val ACTION_CLEAR: String = "io.silvermind.wearcompanion.CLEAR_TIER2"
        const val ACTION_RELOAD: String = "io.silvermind.wearcompanion.RELOAD_TIER2"
        const val EXTRA_CONFIG_B64: String = "config_b64"
        private const val TAG: String = "Tier2Provisioning"
    }
}
