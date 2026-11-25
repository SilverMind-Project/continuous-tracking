package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertInstanceOf
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.attribute.PosixFilePermission
import java.util.Base64

class Tier2ConfigStoreTest {
    @TempDir
    lateinit var dir: Path

    private fun validConfig(): PersistedTier2Config =
        PersistedTier2Config(
            enabled = true,
            brokerUrl = "tcp://192.0.2.10:1883",
            mqttUsername = "watch",
            mqttPassword = "not-a-real-password",
            sessionKeyHex = "0a1b2c3d4e5f6071",
            anchors =
                listOf(
                    PersistedAnchorEntry("living-uwb-1", "00:01"),
                    PersistedAnchorEntry("living-uwb-2", "00:02"),
                    PersistedAnchorEntry("living-uwb-3", "00:03"),
                ),
        )

    @Test
    fun `round trips tier2 config and converts anchors and the static sts key`() {
        val path = dir.resolve(FileTier2ConfigStore.DEFAULT_FILE_NAME)
        val store = FileTier2ConfigStore(path)
        assertEquals(Tier2ConfigState.Missing, store.load())

        store.save(validConfig())
        val loaded = store.load()

        val enabled = assertInstanceOf(Tier2ConfigState.Enabled::class.java, loaded)
        val settings = enabled.settings
        assertEquals("tcp://192.0.2.10:1883", settings.brokerUrl)
        assertEquals(listOf("living-uwb-1", "living-uwb-2", "living-uwb-3"), settings.anchors.map { it.anchorId })
        assertEquals(9, settings.anchors.first().channel)
        assertEquals(10, settings.anchors.first().preambleIndex)
        assertArrayEquals(
            byteArrayOf(0x0a, 0x1b, 0x2c, 0x3d, 0x4e, 0x5f, 0x60, 0x71),
            settings.sessionKeyInfo,
        )
        assertEquals(null, settings.tagId)
    }

    @Test
    fun `saved file is owner-only and leaves no temporary files`() {
        val path = dir.resolve(FileTier2ConfigStore.DEFAULT_FILE_NAME)
        FileTier2ConfigStore(path).save(validConfig())

        val perms = Files.getPosixFilePermissions(path)
        assertEquals(setOf(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE), perms)
        Files.list(dir).use { files -> assertEquals(listOf(path), files.toList()) }
    }

    @Test
    fun `rejects malformed hex keys and provisioned-sts length keys without writing`() {
        val path = dir.resolve(FileTier2ConfigStore.DEFAULT_FILE_NAME)
        val store = FileTier2ConfigStore(path)
        for (badKey in listOf("0a1b2c3d4e5f607", "0a1b2c3d4e5f607z", "0a1b2c3d4e5f60710a1b2c3d4e5f6071")) {
            assertThrows(IllegalArgumentException::class.java) {
                store.save(validConfig().copy(sessionKeyHex = badKey))
            }
        }
        assertFalse(Files.exists(path))
    }

    @Test
    fun `rejects anchors that cannot share one multicast session or collide`() {
        val base = validConfig()
        val cases =
            listOf(
                base.copy(anchors = emptyList()),
                base.copy(anchors = base.anchors + PersistedAnchorEntry("kitchen-uwb-1", "00:09", channel = 5)),
                base.copy(anchors = base.anchors + PersistedAnchorEntry("living-uwb-1", "00:09")),
                base.copy(anchors = base.anchors + PersistedAnchorEntry("living-uwb-9", "00:01")),
                base.copy(anchors = base.anchors + PersistedAnchorEntry("bad", "0001")),
            )
        for (config in cases) {
            assertInstanceOf(Tier2ConfigState.Invalid::class.java, Tier2ConfigCodec.validate(config))
        }
    }

    @Test
    fun `rejects broker urls and tag ids that rf-ingress or paho cannot use`() {
        val base = validConfig()
        val badBrokers = listOf("", "mqtt://192.0.2.10:1883", "tcp://", "tcp://host:70000", "tcp://host/path")
        for (broker in badBrokers) {
            val state = Tier2ConfigCodec.validate(base.copy(brokerUrl = broker))
            assertInstanceOf(Tier2ConfigState.Invalid::class.java, state)
        }
        for (tag in listOf("", "watch/uwb", "watch+uwb", "x".repeat(129))) {
            assertInstanceOf(Tier2ConfigState.Invalid::class.java, Tier2ConfigCodec.validate(base.copy(tagId = tag)))
        }
        assertInstanceOf(
            Tier2ConfigState.Enabled::class.java,
            Tier2ConfigCodec.validate(base.copy(brokerUrl = "ssl://mqtt.home.arpa:8883", tagId = "watch-member-uwb")),
        )
    }

    @Test
    fun `disabled config turns tier 2 off without validating the rest`() {
        val store = FileTier2ConfigStore(dir.resolve("tier2.json"))
        store.save(PersistedTier2Config(enabled = false))
        assertEquals(Tier2ConfigState.Disabled, store.load())
        assertTrue(store.clear())
        assertEquals(Tier2ConfigState.Missing, store.load())
        assertFalse(store.clear())
    }

    @Test
    fun `decodes the adb base64 extra and never echoes secrets in errors or toString`() {
        val json = Tier2ConfigCodec.encode(validConfig())
        val b64 = Base64.getEncoder().encodeToString(json.toByteArray())

        val decoded = Tier2ConfigCodec.decodeBase64(b64)
        assertEquals(validConfig(), decoded)
        assertFalse(decoded.toString().contains("not-a-real-password"))
        assertFalse(decoded.toString().contains("0a1b2c3d4e5f6071"))

        val broken = json.replace("\"enabled\"", "\"enabeld\"")
        val err = assertThrows(IllegalArgumentException::class.java) { Tier2ConfigCodec.decode(broken) }
        assertFalse(err.message!!.contains("not-a-real-password"))
        assertThrows(IllegalArgumentException::class.java) { Tier2ConfigCodec.decodeBase64("%%%not-base64") }

        val settings = (Tier2ConfigCodec.validate(decoded) as Tier2ConfigState.Enabled).settings
        assertFalse(settings.toString().contains("not-a-real-password"))
    }

    @Test
    fun `unreadable file loads as invalid instead of throwing`() {
        val path = dir.resolve(FileTier2ConfigStore.DEFAULT_FILE_NAME)
        Files.writeString(path, "{\"enabled\": true, \"mqtt_password\": \"not-a-real-password\"")
        val state = FileTier2ConfigStore(path).load()
        val invalid = assertInstanceOf(Tier2ConfigState.Invalid::class.java, state)
        assertFalse(invalid.reason.contains("not-a-real-password"))
    }
}
