package io.silvermind.wearuwbprobe.runner

import io.silvermind.wearuwbprobe.fsm.ThermalSeverity
import io.silvermind.wearuwbprobe.oob.BleWakeAdvertiser
import io.silvermind.wearuwbprobe.oob.OobStrategy
import io.silvermind.wearuwbprobe.oob.PreSharedStsNegotiator
import io.silvermind.wearuwbprobe.telemetry.BatteryProfiler
import io.silvermind.wearuwbprobe.telemetry.BatteryRegime
import io.silvermind.wearuwbprobe.telemetry.BatterySnapshotReader
import io.silvermind.wearuwbprobe.telemetry.OrientationSynchronizer
import io.silvermind.wearuwbprobe.telemetry.P1CapabilityRecord
import io.silvermind.wearuwbprobe.telemetry.P2OobTrialRecord
import io.silvermind.wearuwbprobe.telemetry.P2SessionSummaryRecord
import io.silvermind.wearuwbprobe.telemetry.P3RangingOrientationRecord
import io.silvermind.wearuwbprobe.telemetry.P4BatterySampleRecord
import io.silvermind.wearuwbprobe.telemetry.ProbeTelemetryRecorder
import io.silvermind.wearuwbprobe.telemetry.Quaternion
import io.silvermind.wearuwbprobe.telemetry.RawBatterySnapshot
import io.silvermind.wearuwbprobe.uwb.FakeUwbProbeBackend
import io.silvermind.wearuwbprobe.uwb.ProbeCapabilities
import io.silvermind.wearuwbprobe.uwb.ProbeRangingSample
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

class ProbeExperimentRunnerTest {
    @TempDir
    lateinit var tempDir: Path

    private data class TestFixture(
        val gmsBackend: FakeUwbProbeBackend,
        val recorder: ProbeTelemetryRecorder,
        val runner: ProbeExperimentRunner,
    )

    private fun buildFixture(): TestFixture {
        val scriptedSamples =
            listOf("00:01", "00:02", "00:03", "00:04").mapIndexed { idx, addr ->
                ProbeRangingSample(
                    backendName = "gms_nearby_uwb",
                    elapsedRealtimeNanos = 1_000_000_000L + idx * 10_000_000L,
                    wallClockUnixMs = 1_700_000_000_000L + idx * 10L,
                    peerAddressHex = addr,
                    distanceM = 2.5f + idx * 0.1f,
                    distanceConfidence = 2.0f,
                    azimuthDeg = -10.0f + idx * 5.0f,
                    elevationDeg = 4.0f,
                )
            }
        val gmsBackend = FakeUwbProbeBackend(name = "gms_nearby_uwb", scriptedSamples = scriptedSamples)
        val jetpackBackend =
            FakeUwbProbeBackend(
                name = "jetpack_uwb",
                capabilities =
                    ProbeCapabilities(
                        backendName = "jetpack_uwb",
                        featureFlagPresent = true,
                        isAvailable = false,
                        supportsDistance = false,
                        supportsAzimuth = false,
                        supportsElevation = false,
                        rawStatusCode = 42000,
                        errorDetail = "SERVICE_NOT_ENABLED",
                    ),
            )
        val fakeAdvertiser =
            object : BleWakeAdvertiser {
                override suspend fun broadcastWakePayload(
                    manufacturerId: Int,
                    payload: ByteArray,
                    durationMs: Long,
                ): Long = 180L
            }
        val stsNegotiator =
            PreSharedStsNegotiator(
                staticStsVendorId = byteArrayOf(0x53, 0x4D),
                staticStsIv = byteArrayOf(0x01, 0x02, 0x03, 0x04, 0x05, 0x06),
                advertiser = fakeAdvertiser,
            )
        val recorder = ProbeTelemetryRecorder(tempDir)
        val orientationSync = OrientationSynchronizer()
        orientationSync.recordWorldRotation(900_000_000L, Quaternion.IDENTITY)
        orientationSync.recordWorldRotation(1_200_000_000L, Quaternion.IDENTITY)
        val clock =
            object : ProbeClock {
                override fun elapsedRealtimeNanos(): Long = 1_000_000_000L

                override fun wallClockUnixMs(): Long = 1_700_000_000_000L
            }
        val runner =
            ProbeExperimentRunner(
                backends = listOf(gmsBackend, jetpackBackend),
                negotiators = mapOf(OobStrategy.PRESHARED_STS_BROADCAST to stsNegotiator),
                recorder = recorder,
                orientationSynchronizer = orientationSync,
                batteryProfiler = BatteryProfiler(),
                clock = clock,
            )
        return TestFixture(gmsBackend, recorder, runner)
    }

    @Test
    fun `Stage P1 audits all backends and persists capability JSONL records`() =
        runTest {
            val fixture = buildFixture()
            val p1 = fixture.runner.runStageP1(screenOffForegroundService = true)
            assertEquals(2, p1.size)
            assertTrue(p1[0].passesControllerGate())
            assertEquals(42000, p1[1].rawStatusCode)
            val records = fixture.recorder.readAll(ProbeTelemetryRecorder.P1_FILENAME)
            assertEquals(2, records.filterIsInstance<P1CapabilityRecord>().size)
        }

    @Test
    fun `Stage P2 executes OOB trials and multicast sessions`() =
        runTest {
            val fixture = buildFixture()
            val p2 =
                fixture.runner.runStageP2(
                    backend = fixture.gmsBackend,
                    strategy = OobStrategy.PRESHARED_STS_BROADCAST,
                    peers = ProbeCommand.DEFAULT_PEERS,
                    trials = 2,
                    sessionDurationMs = 2_000L,
                )
            assertEquals(2, p2.size)
            val records = fixture.recorder.readAll(ProbeTelemetryRecorder.P2_FILENAME)
            assertEquals(2, records.filterIsInstance<P2OobTrialRecord>().size)
            assertEquals(2, records.filterIsInstance<P2SessionSummaryRecord>().size)
        }

    @Test
    fun `Stage P3 and P4 record synchronized orientation and battery checkpoints`() =
        runTest {
            val fixture = buildFixture()
            fixture.runner.runStageP3(
                backend = fixture.gmsBackend,
                strategy = OobStrategy.PRESHARED_STS_BROADCAST,
                peers = ProbeCommand.DEFAULT_PEERS,
                pointId = "P01",
                facingDir = "N",
                durationMs = 2_000L,
                wristRotationTrial = true,
            )
            val p3Typed =
                fixture.recorder.readAll(ProbeTelemetryRecorder.P3_FILENAME)
                    .filterIsInstance<P3RangingOrientationRecord>()
            assertEquals(4, p3Typed.size)
            assertTrue(p3Typed.all { it.pointId == "P01" && it.facingDir == "N" && it.wristRotationTrial })

            val fsm = fixture.runner.createDutyCycleFsm()
            val reader =
                object : BatterySnapshotReader {
                    override fun readSnapshot(): RawBatterySnapshot =
                        RawBatterySnapshot(
                            elapsedRealtimeNanos = 1_000_000_000L,
                            wallClockUnixMs = 1_700_000_000_000L,
                            capacityPercent = 95,
                            chargeCounterUah = 370_000L,
                            currentNowUa = -15_000L,
                            voltageMv = 4150,
                            temperatureDeciC = 300,
                            thermalStatus = ThermalSeverity.NONE,
                        )
                }
            fixture.runner.recordBatteryCheckpoint(BatteryRegime.C_MOTION_DUTY_CYCLED, "regime_start", reader, fsm)
            val p4Typed =
                fixture.recorder.readAll(ProbeTelemetryRecorder.P4_FILENAME)
                    .filterIsInstance<P4BatterySampleRecord>()
            assertEquals(1, p4Typed.size)
        }
}
