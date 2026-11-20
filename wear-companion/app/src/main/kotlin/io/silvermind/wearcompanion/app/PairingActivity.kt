package io.silvermind.wearcompanion.app

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import io.silvermind.wearcompanion.core.FileIdentifierStore

/**
 * Minimal pairing screen showing the generated `uuid:major:minor` hardware key
 * for entry in Cognitive Companion, plus an explicit reset button.
 */
class PairingActivity : ComponentActivity() {
    private lateinit var store: FileIdentifierStore
    private lateinit var keyView: TextView

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
        val resetButton =
            Button(this).apply {
                text = getString(R.string.reset_identifier)
                setOnClickListener {
                    val fresh = store.reset()
                    keyView.text = fresh.hardwareKey
                    startForegroundService(
                        Intent(this@PairingActivity, CompanionForegroundService::class.java)
                            .setAction(CompanionForegroundService.ACTION_REFRESH),
                    )
                }
            }
        container.addView(keyView)
        container.addView(resetButton)
        setContentView(container)

        startForegroundService(Intent(this, CompanionForegroundService::class.java))
    }
}
