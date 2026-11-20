package io.silvermind.wearcompanion.core

import java.nio.file.Files
import java.nio.file.Path
import java.security.SecureRandom
import java.util.UUID

/**
 * Stable iBeacon identifier generated once on first run and registered in
 * Cognitive Companion as the device's hardware_key (`uuid:major:minor`).
 */
data class BeaconIdentifier(
    val uuid: UUID,
    val major: Int,
    val minor: Int,
) {
    init {
        require(major in 0..MAX_UINT16) { "major must be in 0..$MAX_UINT16, got $major" }
        require(minor in 0..MAX_UINT16) { "minor must be in 0..$MAX_UINT16, got $minor" }
    }

    /** Canonical hardware key entered by a caregiver in Cognitive Companion. */
    val hardwareKey: String
        get() = "${uuid.toString().lowercase()}:$major:$minor"

    companion object {
        const val MAX_UINT16: Int = 65_535
        private val secureRandom = SecureRandom()

        fun generate(random: SecureRandom = secureRandom): BeaconIdentifier {
            val major = random.nextInt(MAX_UINT16 + 1)
            val minor = random.nextInt(MAX_UINT16 + 1)
            return BeaconIdentifier(
                uuid = UUID.randomUUID(),
                major = major,
                minor = minor,
            )
        }

        fun parseHardwareKey(raw: String): BeaconIdentifier {
            val parts = raw.trim().split(":")
            require(parts.size == 3) { "Expected 'uuid:major:minor', got '$raw'" }
            val uuid = UUID.fromString(parts[0])
            val major = parts[1].toInt()
            val minor = parts[2].toInt()
            return BeaconIdentifier(uuid = uuid, major = major, minor = minor)
        }
    }
}

/** Persistence boundary for the watch's stable beacon identifier. */
interface IdentifierStore {
    fun current(): BeaconIdentifier?

    fun loadOrCreate(): BeaconIdentifier

    fun reset(): BeaconIdentifier
}

/** In-memory [IdentifierStore] for unit tests. */
class InMemoryIdentifierStore(
    private val generator: () -> BeaconIdentifier = { BeaconIdentifier.generate() },
) : IdentifierStore {
    private var stored: BeaconIdentifier? = null

    override fun current(): BeaconIdentifier? = stored

    override fun loadOrCreate(): BeaconIdentifier {
        val existing = stored
        if (existing != null) {
            return existing
        }
        val created = generator()
        stored = created
        return created
    }

    override fun reset(): BeaconIdentifier {
        val created = generator()
        stored = created
        return created
    }
}

/**
 * Single-file [IdentifierStore] backed by app-private storage.
 *
 * Writes only the identifier file (`uuid:major:minor`) and nothing else so
 * Tier 1 persists no sensor, location, or network data.
 */
class FileIdentifierStore(
    private val filePath: Path,
    private val generator: () -> BeaconIdentifier = { BeaconIdentifier.generate() },
) : IdentifierStore {
    override fun current(): BeaconIdentifier? {
        if (!Files.isRegularFile(filePath)) {
            return null
        }
        val content = Files.readString(filePath).trim()
        if (content.isEmpty()) {
            return null
        }
        return BeaconIdentifier.parseHardwareKey(content)
    }

    override fun loadOrCreate(): BeaconIdentifier {
        val existing = current()
        if (existing != null) {
            return existing
        }
        return writeNew()
    }

    override fun reset(): BeaconIdentifier = writeNew()

    private fun writeNew(): BeaconIdentifier {
        val created = generator()
        val parent = filePath.parent
        if (parent != null) {
            Files.createDirectories(parent)
        }
        Files.writeString(filePath, created.hardwareKey + "\n")
        return created
    }
}
