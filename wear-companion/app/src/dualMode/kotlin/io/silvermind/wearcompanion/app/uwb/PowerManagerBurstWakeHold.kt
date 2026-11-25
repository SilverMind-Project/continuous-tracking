package io.silvermind.wearcompanion.app.uwb

import android.content.Context
import android.os.PowerManager
import io.silvermind.wearcompanion.core.BurstWakeHold

/**
 * Partial wake lock held for one UWB burst so the burst, tick, and flush timers
 * run while the screen is off. Not reference counted: one [release] ends it,
 * and the acquire timeout bounds it if a release is ever missed.
 */
class PowerManagerBurstWakeHold(
    context: Context,
) : BurstWakeHold {
    private val wakeLock: PowerManager.WakeLock =
        context.getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG)
            .apply { setReferenceCounted(false) }

    override fun acquire(timeoutMs: Long) {
        wakeLock.acquire(timeoutMs)
    }

    override fun release() {
        if (wakeLock.isHeld) {
            wakeLock.release()
        }
    }

    private companion object {
        const val WAKE_LOCK_TAG: String = "silvermind:uwb-burst"
    }
}
