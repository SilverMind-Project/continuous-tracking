package io.silvermind.wearcompanion.app

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import io.silvermind.wearcompanion.core.FileIdentifierStore

/**
 * Minimal pairing screen showing the generated `uuid:major:minor` hardware key
 * for entry in Cognitive Companion, plus an explicit reset button.
 *
 * It also asks once for the runtime permissions the build needs (see
 * [RequiredPermissions]) and lists any that are still missing, because the
 * service can neither advertise nor start UWB bursts without them.
 */
class PairingActivity : ComponentActivity() {
    private lateinit var store: FileIdentifierStore
    private lateinit var keyView: TextView
    private lateinit var permissionView: TextView

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            showPermissionStatus()
            refreshService()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store = FileIdentifierStore(filesDir.toPath().resolve(CompanionForegroundService.IDENTIFIER_FILE_NAME))
        val id = store.loadOrCreate()

        val container =
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(24, 24, 24, 24)
            }
        keyView =
            TextView(this).apply {
                text = id.hardwareKey
                textSize = 12f
            }
        permissionView = TextView(this).apply { textSize = 11f }
        val resetButton =
            Button(this).apply {
                text = getString(R.string.reset_identifier)
                setOnClickListener {
                    val fresh = store.reset()
                    keyView.text = fresh.hardwareKey
                    refreshService()
                }
            }
        container.addView(keyView)
        container.addView(permissionView)
        container.addView(resetButton)
        setContentView(container)

        startForegroundService(Intent(this, CompanionForegroundService::class.java))
        val missing = RequiredPermissions.missing(this)
        if (missing.isNotEmpty()) {
            permissionRequest.launch(missing.toTypedArray())
        }
        showPermissionStatus()
    }

    private fun showPermissionStatus() {
        val missing = RequiredPermissions.missing(this)
        permissionView.text =
            if (missing.isEmpty()) {
                getString(R.string.permissions_granted)
            } else {
                getString(R.string.permissions_missing, missing.joinToString { it.substringAfterLast('.') })
            }
    }

    /** Rebuild the service so it picks up a new identifier or newly granted permissions. */
    private fun refreshService() {
        startForegroundService(
            Intent(this, CompanionForegroundService::class.java)
                .setAction(CompanionForegroundService.ACTION_REFRESH),
        )
    }
}
