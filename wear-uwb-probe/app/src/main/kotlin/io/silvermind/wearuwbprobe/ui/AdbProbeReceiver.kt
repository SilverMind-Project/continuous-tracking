package io.silvermind.wearuwbprobe.ui

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import io.silvermind.wearuwbprobe.runner.ProbeCommand

/**
 * ADB broadcast entry point allowing hands-free stage execution with the watch screen off:
 * `adb shell am broadcast -a io.silvermind.wearuwbprobe.RUN_STAGE --es stage P1`.
 */
class AdbProbeReceiver : BroadcastReceiver() {
    override fun onReceive(
        context: Context,
        intent: Intent?,
    ) {
        if (intent == null || intent.action != ProbeCommand.ACTION_RUN_STAGE) {
            return
        }
        val bundle = intent.extras
        val extrasMap =
            buildMap<String, String?> {
                if (bundle != null) {
                    for (key in bundle.keySet()) {
                        put(key, bundle.get(key)?.toString())
                    }
                }
            }
        val validated = ProbeCommand.parse(extrasMap)
        if (validated == null) {
            Log.w(TAG, "Rejected malformed RUN_STAGE broadcast")
            return
        }

        val serviceIntent =
            Intent(context, ProbeForegroundService::class.java).apply {
                for ((k, v) in extrasMap) {
                    if (v != null) {
                        putExtra(k, v)
                    }
                }
            }
        context.startForegroundService(serviceIntent)
    }

    companion object {
        private const val TAG: String = "AdbProbeReceiver"
    }
}
