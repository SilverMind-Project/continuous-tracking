package io.silvermind.wearuwbprobe.ui

import android.Manifest
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import io.silvermind.wearuwbprobe.runner.ProbeStage

/**
 * Minimal standalone Wear OS diagnostic launcher allowing manual permission grants and
 * one-tap triggers for Stages P1 to P4 alongside ADB broadcast automation.
 */
class ProbeActivity : ComponentActivity() {
    private lateinit var statusView: TextView

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val grantedCount = grants.values.count { it }
            statusView.text = "Permissions granted: $grantedCount / ${grants.size}"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val container =
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(24, 32, 24, 32)
            }

        statusView =
            TextView(this).apply {
                text = "Wear UWB Bench Probe Ready"
            }
        container.addView(statusView)

        container.addView(
            Button(this).apply {
                text = "Grant Permissions"
                setOnClickListener { permissionLauncher.launch(REQUIRED_PERMISSIONS) }
            },
        )
        container.addView(makeStageButton("Run P1 (Audit)", ProbeStage.P1))
        container.addView(makeStageButton("Run P2 (OOB/Multicast)", ProbeStage.P2))
        container.addView(makeStageButton("Run P3 (Accuracy/AoA)", ProbeStage.P3))
        container.addView(makeStageButton("Run P4 (Battery)", ProbeStage.P4))
        container.addView(makeStageButton("Stop Probe", ProbeStage.STOP))

        val scroll = ScrollView(this).apply { addView(container) }
        setContentView(scroll)
    }

    private fun makeStageButton(
        label: String,
        stage: ProbeStage,
    ): Button =
        Button(this).apply {
            text = label
            setOnClickListener {
                val serviceIntent =
                    Intent(this@ProbeActivity, ProbeForegroundService::class.java).apply {
                        putExtra("stage", stage.name)
                    }
                startForegroundService(serviceIntent)
                statusView.text = "Dispatched stage: ${stage.name}"
            }
        }

    companion object {
        private val REQUIRED_PERMISSIONS: Array<String> =
            arrayOf(
                Manifest.permission.UWB_RANGING,
                Manifest.permission.BLUETOOTH_ADVERTISE,
                Manifest.permission.BLUETOOTH_CONNECT,
                Manifest.permission.BLUETOOTH_SCAN,
                Manifest.permission.BODY_SENSORS,
                Manifest.permission.ACTIVITY_RECOGNITION,
            )
    }
}
