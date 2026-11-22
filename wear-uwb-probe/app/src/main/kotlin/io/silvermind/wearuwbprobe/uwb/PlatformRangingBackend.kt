package io.silvermind.wearuwbprobe.uwb

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build

/**
 * Capability-only probe for Android 16's `android.ranging.RangingManager` (API 36+) and
 * AOSP's `android.uwb.UwbManager` system service via reflection.
 *
 * Using reflection allows compiling cleanly against `compileSdk = 35` while auditing
 * Wear OS 7 / Android 16 platform ranging technologies (`UWB`, `BLE_CS`, `BLE_RSSI`,
 * `WIFI_NAN_RTT`) when installed on the physical watch.
 */
class PlatformRangingBackend(
    private val context: Context,
) : UwbProbeBackend {
    override val name: String = BACKEND_NAME

    override suspend fun queryCapabilities(): ProbeCapabilities {
        val hasFeature = context.packageManager.hasSystemFeature(PackageManager.FEATURE_UWB)
        val detectedTechs = mutableListOf<String>()
        val notes = mutableListOf<String>()

        val platformUwbService = runCatching { context.getSystemService("uwb") }.getOrNull()
        if (platformUwbService != null) {
            detectedTechs.add("PLATFORM_UWB_SERVICE")
        }

        var rangingManagerPresent = false
        if (Build.VERSION.SDK_INT >= ANDROID_16_API_LEVEL) {
            val rangingService = runCatching { context.getSystemService("ranging") }.getOrNull()
            if (rangingService != null) {
                rangingManagerPresent = true
                detectedTechs.add("ANDROID_RANGING_MANAGER")
                notes.addAll(inspectRangingManagerMethods(rangingService))
            } else {
                notes.add("getSystemService('ranging') returned null on SDK ${Build.VERSION.SDK_INT}")
            }
        } else {
            notes.add("SDK_INT=${Build.VERSION.SDK_INT} (< 36; android.ranging.RangingManager unavailable)")
        }

        return ProbeCapabilities(
            backendName = name,
            featureFlagPresent = hasFeature,
            isAvailable = platformUwbService != null || rangingManagerPresent,
            supportsDistance = hasFeature && (platformUwbService != null || rangingManagerPresent),
            supportsAzimuth = false,
            supportsElevation = false,
            errorDetail = notes.takeIf { it.isNotEmpty() }?.joinToString(separator = "; "),
            platformTechnologies = detectedTechs,
        )
    }

    override suspend fun openControllerSession(
        sessionId: Int,
        preferredChannel: Int,
        preferredPreambleIndex: Int,
    ): PreparedControllerSession? = null

    override suspend fun runMulticastSession(
        config: MulticastSessionConfig,
        durationMs: Long,
        onSample: (ProbeRangingSample) -> Unit,
    ): SessionSummary {
        val nowMs = System.currentTimeMillis()
        return SessionSummary(
            backendName = name,
            sessionId = config.sessionId,
            startedAtUnixMs = nowMs,
            endedAtUnixMs = nowMs,
            samplesCollected = 0,
            distinctPeersAcquired = 0,
            errorDetail = "PlatformRangingBackend is capability-audit only",
        )
    }

    private fun inspectRangingManagerMethods(service: Any): List<String> {
        val methods = service.javaClass.methods.map { it.name }.sorted().distinct()
        val relevant =
            methods.filter {
                it.contains("Capabil", ignoreCase = true) || it.contains("Ranging", ignoreCase = true)
            }
        return listOf("class=${service.javaClass.name}", "methods=${relevant.take(12).joinToString(",")}")
    }

    companion object {
        const val BACKEND_NAME: String = "platform_ranging"
        private const val ANDROID_16_API_LEVEL: Int = 36
    }
}
