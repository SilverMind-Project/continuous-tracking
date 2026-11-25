package io.silvermind.wearcompanion.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.PosixFileAttributeView
import java.nio.file.attribute.PosixFilePermission
import java.util.Base64

/**
 * On-disk Tier 2 configuration (`tier2_config.json` in app-private storage).
 *
 * The anchors' `anchor_id` values must equal the Cognitive Companion RF node
 * `anchor_id`s exactly, because `rf-ingress` uses a non-empty `anchor_id` from
 * the watch verbatim.
 */
@Serializable
data class PersistedTier2Config(
    val enabled: Boolean = false,
    @SerialName("tag_id") val tagId: String? = null,
    @SerialName("broker_url") val brokerUrl: String = "",
    @SerialName("mqtt_username") val mqttUsername: String? = null,
    @SerialName("mqtt_password") val mqttPassword: String? = null,
    @SerialName("session_key_hex") val sessionKeyHex: String = "",
    @SerialName("prefer_jetpack_backend") val preferJetpackBackend: Boolean = false,
    val anchors: List<PersistedAnchorEntry> = emptyList(),
) {
    override fun toString(): String =
        "PersistedTier2Config(enabled=$enabled, anchors=${anchors.size}, " +
            "preferJetpackBackend=$preferJetpackBackend, credentials=<redacted>)"
}

@Serializable
data class PersistedAnchorEntry(
    @SerialName("anchor_id") val anchorId: String,
    @SerialName("uwb_address_hex") val uwbAddressHex: String,
    val channel: Int = UwbAnchorEntry.DEFAULT_UWB_CHANNEL,
    @SerialName("preamble_index") val preambleIndex: Int = UwbAnchorEntry.DEFAULT_PREAMBLE_INDEX,
)

/** Validated Tier 2 runtime settings. Never log an instance's secrets; [toString] redacts them. */
class Tier2Settings(
    val tagId: String?,
    val brokerUrl: String,
    val mqttUsername: String?,
    val mqttPassword: String?,
    val sessionKeyInfo: ByteArray,
    val preferJetpackBackend: Boolean,
    val anchors: List<UwbAnchorEntry>,
) {
    override fun toString(): String =
        "Tier2Settings(anchors=${anchors.size}, preferJetpackBackend=$preferJetpackBackend, " +
            "credentials=<redacted>)"
}

/** Result of loading or validating a Tier 2 configuration. */
sealed interface Tier2ConfigState {
    /** No configuration file exists: Tier 2 stays off. */
    data object Missing : Tier2ConfigState

    /** A configuration exists with `enabled = false`: Tier 2 stays off. */
    data object Disabled : Tier2ConfigState

    /** The configuration is unusable; [reason] never contains secrets. */
    data class Invalid(val reason: String) : Tier2ConfigState

    data class Enabled(val settings: Tier2Settings) : Tier2ConfigState
}

/** JSON codec and validation for [PersistedTier2Config]. */
object Tier2ConfigCodec {
    /** Exactly 8 bytes: both engines range with `CONFIG_MULTICAST_DS_TWR` (Static STS). */
    const val STATIC_STS_KEY_HEX_CHARS: Int = 16
    const val MAX_TAG_ID_CHARS: Int = 128

    private val json =
        Json {
            ignoreUnknownKeys = false
            encodeDefaults = true
            explicitNulls = false
        }
    private val hexRegex = Regex("^[0-9a-fA-F]+$")
    private val brokerRegex = Regex("^(tcp|ssl)://[A-Za-z0-9.-]+(:(\\d{1,5}))?$")
    private val forbiddenTagChars = setOf('/', '+', '#')

    fun encode(config: PersistedTier2Config): String = json.encodeToString(PersistedTier2Config.serializer(), config)

    /**
     * Parse JSON text. Throws [IllegalArgumentException] with a message that never
     * echoes the input, because the input can contain the STS key or MQTT password.
     */
    fun decode(text: String): PersistedTier2Config =
        try {
            json.decodeFromString(PersistedTier2Config.serializer(), text)
        } catch (_: SerializationException) {
            throw IllegalArgumentException("tier2 config is not valid tier2_config JSON")
        } catch (_: IllegalArgumentException) {
            throw IllegalArgumentException("tier2 config is not valid tier2_config JSON")
        }

    /** Decode the base64 JSON carried by the ADB provisioning broadcast. */
    fun decodeBase64(encoded: String): PersistedTier2Config {
        val bytes =
            try {
                Base64.getDecoder().decode(encoded.trim())
            } catch (_: IllegalArgumentException) {
                throw IllegalArgumentException("tier2 config is not valid base64")
            }
        return decode(String(bytes, Charsets.UTF_8))
    }

    fun validate(config: PersistedTier2Config): Tier2ConfigState {
        if (!config.enabled) {
            return Tier2ConfigState.Disabled
        }
        val problem = firstProblem(config)
        if (problem != null) {
            return Tier2ConfigState.Invalid(problem)
        }
        return Tier2ConfigState.Enabled(
            Tier2Settings(
                tagId = config.tagId?.trim(),
                brokerUrl = config.brokerUrl.trim(),
                mqttUsername = config.mqttUsername?.takeIf { it.isNotBlank() },
                mqttPassword = config.mqttPassword?.takeIf { it.isNotEmpty() },
                sessionKeyInfo = hexToBytes(config.sessionKeyHex.trim()),
                preferJetpackBackend = config.preferJetpackBackend,
                anchors = config.anchors.map(::toAnchorEntry),
            ),
        )
    }

    @Suppress("CyclomaticComplexMethod", "ReturnCount")
    private fun firstProblem(config: PersistedTier2Config): String? {
        val broker = brokerRegex.matchEntire(config.brokerUrl.trim())
        if (broker == null) {
            return "broker_url must look like tcp://host:port or ssl://host:port"
        }
        val port = broker.groupValues[3]
        if (port.isNotEmpty() && port.toInt() !in 1..MAX_PORT) {
            return "broker_url port must be in 1..$MAX_PORT"
        }
        val key = config.sessionKeyHex.trim()
        if (key.length != STATIC_STS_KEY_HEX_CHARS || !hexRegex.matches(key)) {
            return "session_key_hex must be exactly $STATIC_STS_KEY_HEX_CHARS hex characters (8-byte Static STS key)"
        }
        val tagId = config.tagId?.trim()
        if (tagId != null && !isUsableTagId(tagId)) {
            return "tag_id must be 1..$MAX_TAG_ID_CHARS characters without '/', '+', or '#'"
        }
        if (config.anchors.isEmpty()) {
            return "anchors must list at least one UWB anchor"
        }
        val anchors =
            try {
                config.anchors.map(::toAnchorEntry)
            } catch (err: IllegalArgumentException) {
                return "invalid anchor: ${err.message}"
            }
        if (anchors.map { it.anchorId }.toSet().size != anchors.size) {
            return "anchor_id values must be unique"
        }
        if (anchors.map { it.uwbAddressHex }.toSet().size != anchors.size) {
            return "uwb_address_hex values must be unique"
        }
        if (anchors.map { it.channel to it.preambleIndex }.toSet().size != 1) {
            return "all anchors must share one channel and preamble_index (one multicast session ranges them all)"
        }
        return null
    }

    private fun isUsableTagId(tagId: String): Boolean =
        tagId.isNotEmpty() && tagId.length <= MAX_TAG_ID_CHARS && tagId.none { it in forbiddenTagChars }

    private fun toAnchorEntry(entry: PersistedAnchorEntry): UwbAnchorEntry =
        UwbAnchorEntry(
            anchorId = entry.anchorId.trim(),
            uwbAddressHex = entry.uwbAddressHex.trim().uppercase(),
            channel = entry.channel,
            preambleIndex = entry.preambleIndex,
        )

    private fun hexToBytes(hex: String): ByteArray = hex.chunked(2).map { it.toInt(16).toByte() }.toByteArray()

    private const val MAX_PORT: Int = 65_535
}

/** Persistence boundary for the Tier 2 configuration. */
interface Tier2ConfigStore {
    fun load(): Tier2ConfigState

    /** Validate and persist; throws [IllegalArgumentException] and writes nothing when invalid. */
    fun save(config: PersistedTier2Config): Tier2ConfigState

    /** Delete the configuration so Tier 2 turns off; returns true when a file existed. */
    fun clear(): Boolean
}

/**
 * Single-file [Tier2ConfigStore] in app-private storage.
 *
 * Writes atomically (temporary file plus rename) with owner-only permissions
 * where the file system supports POSIX attributes, so a crash never leaves a
 * half-written key file and other users cannot read the STS key or password.
 */
class FileTier2ConfigStore(
    private val filePath: Path,
) : Tier2ConfigStore {
    override fun load(): Tier2ConfigState {
        if (!Files.isRegularFile(filePath)) {
            return Tier2ConfigState.Missing
        }
        val config =
            try {
                Tier2ConfigCodec.decode(Files.readString(filePath))
            } catch (err: IllegalArgumentException) {
                return Tier2ConfigState.Invalid(err.message ?: "unreadable tier2 config")
            } catch (_: IOException) {
                return Tier2ConfigState.Invalid("tier2 config could not be read")
            }
        return Tier2ConfigCodec.validate(config)
    }

    override fun save(config: PersistedTier2Config): Tier2ConfigState {
        val state = Tier2ConfigCodec.validate(config)
        if (state is Tier2ConfigState.Invalid) {
            throw IllegalArgumentException(state.reason)
        }
        val parent = filePath.toAbsolutePath().parent
        Files.createDirectories(parent)
        val tmp = Files.createTempFile(parent, ".tier2_config", ".tmp")
        try {
            restrictToOwner(tmp)
            Files.writeString(tmp, Tier2ConfigCodec.encode(config) + "\n")
            try {
                Files.move(tmp, filePath, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(tmp, filePath, StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            Files.deleteIfExists(tmp)
        }
        return state
    }

    override fun clear(): Boolean = Files.deleteIfExists(filePath)

    private fun restrictToOwner(path: Path) {
        val view = Files.getFileAttributeView(path, PosixFileAttributeView::class.java) ?: return
        view.setPermissions(setOf(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE))
    }

    companion object {
        const val DEFAULT_FILE_NAME: String = "tier2_config.json"
    }
}
