package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path

class IdentifierStoreTest {
    @Test
    fun `file identifier store persists across restarts and changes only on reset`(
        @TempDir tempDir: Path,
    ) {
        val path = tempDir.resolve("beacon_id.txt")
        val store1 = FileIdentifierStore(path)
        assertNull(store1.current())

        val first = store1.loadOrCreate()
        assertEquals(first, store1.loadOrCreate())

        // Simulate app restart with a fresh FileIdentifierStore instance.
        val store2 = FileIdentifierStore(path)
        assertEquals(first, store2.current())
        assertEquals(first, store2.loadOrCreate())

        // Explicit reset mints a new identifier and persists it.
        val replaced = store2.reset()
        assertNotEquals(first, replaced)
        val store3 = FileIdentifierStore(path)
        assertEquals(replaced, store3.current())
    }

    @Test
    fun `hardwareKey round-trips through parseHardwareKey`() {
        val id = BeaconIdentifier.generate()
        val parsed = BeaconIdentifier.parseHardwareKey(id.hardwareKey)
        assertEquals(id, parsed)
    }
}
