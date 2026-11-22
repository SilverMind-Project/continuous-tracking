package io.silvermind.wearuwbprobe

import io.silvermind.wearuwbprobe.runner.ProbeCommand
import io.silvermind.wearuwbprobe.runner.ProbeStage
import io.silvermind.wearuwbprobe.telemetry.ProbeTelemetryRecorder
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class ManifestAndSecurityTest {
    @TempDir
    lateinit var tempDir: Path

    private fun readAppManifest(): String {
        val candidates =
            listOf(
                Path.of("..", "app", "src", "main", "AndroidManifest.xml"),
                Path.of("app", "src", "main", "AndroidManifest.xml"),
            )
        val path =
            candidates.firstOrNull { Files.exists(it) }
                ?: error("Could not locate app/src/main/AndroidManifest.xml")
        return Files.readString(path)
    }

    @Test
    fun `manifest declares required standalone probe permissions and zero network permissions`() {
        val manifest = readAppManifest()

        val required =
            listOf(
                "android.permission.UWB_RANGING",
                "android.permission.BLUETOOTH_ADVERTISE",
                "android.permission.BLUETOOTH_CONNECT",
                "android.permission.BLUETOOTH_SCAN",
                "android.permission.BODY_SENSORS",
                "android.permission.ACTIVITY_RECOGNITION",
                "android.permission.FOREGROUND_SERVICE",
                "android.permission.FOREGROUND_SERVICE_CONNECTED_DEVICE",
                "android.permission.WAKE_LOCK",
                """android:allowBackup="false"""",
                """com.google.android.wearable.standalone""",
                "io.silvermind.wearuwbprobe.RUN_STAGE",
            )
        for (item in required) {
            assertTrue(manifest.contains(item), "Probe manifest must declare $item")
        }

        // Probe writes JSONL locally for adb pull and must never request network permissions.
        assertFalse(manifest.contains("android.permission.INTERNET"))
        assertFalse(manifest.contains("android.permission.ACCESS_NETWORK_STATE"))
    }

    @Test
    fun `ProbeTelemetryRecorder rejects path traversal and non-jsonl filenames`() {
        val recorder = ProbeTelemetryRecorder(tempDir)
        assertNotNull(recorder.resolveSafeFile("p1_capabilities.jsonl"))

        val invalidNames =
            listOf(
                "../escape.jsonl",
                "sub/dir.jsonl",
                "..\\win.jsonl",
                "not_jsonl.txt",
                ".hidden.jsonl",
                "",
            )
        for (bad in invalidNames) {
            assertThrows(IllegalArgumentException::class.java) {
                recorder.resolveSafeFile(bad)
            }
        }
    }

    @Test
    fun `ProbeCommand strictly allow-lists ADB broadcast intent extras`() {
        val valid =
            ProbeCommand.parse(
                mapOf(
                    "stage" to "P3",
                    "backend" to "gms_nearby_uwb",
                    "oob" to "preshared_sts",
                    "point_id" to "P04",
                    "facing_dir" to "W",
                    "duration_s" to "60",
                    "anchors" to "00:01,00:02,00:03,00:04",
                    "wrist_rotation" to "true",
                ),
            )
        assertNotNull(valid)
        assertEquals(ProbeStage.P3, valid!!.stage)
        assertEquals("P04", valid.pointId)
        assertEquals("W", valid.facingDir)
        assertTrue(valid.wristRotationTrial)
        assertEquals(4, valid.peers.size)

        // Malicious or out-of-range inputs are rejected cleanly (return null).
        assertNull(ProbeCommand.parse(mapOf("stage" to "P9")))
        assertNull(ProbeCommand.parse(mapOf("stage" to "P1", "backend" to "arbitrary")))
        assertNull(ProbeCommand.parse(mapOf("stage" to "P3", "point_id" to "../etc/passwd")))
        assertNull(ProbeCommand.parse(mapOf("stage" to "P3", "facing_dir" to "NW")))
        assertNull(ProbeCommand.parse(mapOf("stage" to "P4", "duration_s" to "999999")))
        assertNull(ProbeCommand.parse(mapOf("stage" to "P2", "anchors" to "INVALID_ADDR")))
    }
}
