package io.silvermind.wearcompanion.app

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager

/**
 * Runtime (dangerous) permissions this build needs. Declaring them in the
 * manifest is not enough on Android 12 and later: without a grant, advertising
 * and UWB ranging throw `SecurityException` and the step detector stays silent.
 */
object RequiredPermissions {
    /** Tier 1: the iBeacon advertiser and the on-body sensor. Tier 2 adds UWB and step detection. */
    fun forThisBuild(): List<String> =
        buildList {
            add(Manifest.permission.BLUETOOTH_ADVERTISE)
            add(Manifest.permission.BODY_SENSORS)
            if (BuildConfig.TIER2_UWB_ENABLED) {
                add(Manifest.permission.UWB_RANGING)
                add(Manifest.permission.ACTIVITY_RECOGNITION)
            }
        }

    fun isGranted(
        context: Context,
        permission: String,
    ): Boolean = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    fun missing(context: Context): List<String> = forThisBuild().filterNot { isGranted(context, it) }
}
