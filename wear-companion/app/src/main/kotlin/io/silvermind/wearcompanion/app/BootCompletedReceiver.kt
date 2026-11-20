package io.silvermind.wearcompanion.app

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/** Restarts [CompanionForegroundService] after watch reboot so worn watches resume advertising. */
class BootCompletedReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            context.startForegroundService(Intent(context, CompanionForegroundService::class.java))
        }
    }
}
