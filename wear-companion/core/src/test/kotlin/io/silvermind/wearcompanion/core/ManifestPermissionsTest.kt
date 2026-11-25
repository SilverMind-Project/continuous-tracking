package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Path

class ManifestPermissionsTest {
    private fun resolveAppManifest(relative: String): String {
        val candidates =
            listOf(
                Path.of("..", "app", "src", relative, "AndroidManifest.xml"),
                Path.of("app", "src", relative, "AndroidManifest.xml"),
            )
        val path =
            candidates.firstOrNull { Files.exists(it) }
                ?: error("Could not find app/src/$relative/AndroidManifest.xml from ${Path.of("").toAbsolutePath()}")
        return Files.readString(path)
    }

    @Test
    fun `base bleOnly manifest contains zero network or UWB permissions`() {
        val manifest = resolveAppManifest("main")

        val forbidden =
            listOf(
                "android.permission.INTERNET",
                "android.permission.ACCESS_NETWORK_STATE",
                "android.permission.CHANGE_WIFI_STATE",
                "android.permission.UWB_RANGING",
                "android.permission.WAKE_LOCK",
                "android.permission.ACTIVITY_RECOGNITION",
            )
        for (perm in forbidden) {
            assertFalse(manifest.contains(perm), "Base bleOnly manifest must not declare $perm")
        }

        val required =
            listOf(
                "android.permission.BLUETOOTH_ADVERTISE",
                "android.permission.BODY_SENSORS",
                "android.permission.FOREGROUND_SERVICE",
                "android.permission.FOREGROUND_SERVICE_CONNECTED_DEVICE",
                "android.permission.RECEIVE_BOOT_COMPLETED",
                """android:allowBackup="false"""",
                """com.google.android.wearable.standalone""",
            )
        for (item in required) {
            assertTrue(manifest.contains(item), "Base manifest must declare $item")
        }
    }

    @Test
    fun `dualMode overlay manifest adds UWB and LAN MQTT permissions`() {
        val manifest = resolveAppManifest("dualMode")

        val expected =
            listOf(
                "android.permission.UWB_RANGING",
                "android.permission.WAKE_LOCK",
                "android.permission.INTERNET",
                "android.permission.ACCESS_NETWORK_STATE",
                "android.permission.CHANGE_WIFI_STATE",
                "android.permission.ACTIVITY_RECOGNITION",
                """android:required="false"""",
            )
        for (item in expected) {
            assertTrue(manifest.contains(item), "dualMode manifest must declare $item")
        }
    }

    @Test
    fun `tier 2 provisioning receiver exists only in dualMode and only the adb shell can reach it`() {
        val base = resolveAppManifest("main")
        assertFalse(base.contains("Tier2ProvisioningReceiver"), "bleOnly must not expose Tier 2 provisioning")

        val dual = resolveAppManifest("dualMode")
        val receiver =
            Regex("<receiver[^>]*Tier2ProvisioningReceiver[^>]*>", RegexOption.DOT_MATCHES_ALL).find(dual)?.value
                ?: error("dualMode manifest must declare Tier2ProvisioningReceiver")
        assertTrue(
            receiver.contains("""android:permission="android.permission.DUMP""""),
            "the exported provisioning receiver must require android.permission.DUMP",
        )
    }
}
