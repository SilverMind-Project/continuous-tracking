package io.silvermind.wearcompanion.core

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class BeaconStateMachineTest {
    @Test
    fun `unpaired watch never advertises regardless of body or bluetooth state`() {
        val fsm = BeaconStateMachine()
        val t1 = fsm.evaluate(BeaconInputs(paired = false, onBody = true, bluetoothEnabled = true))
        assertEquals(BeaconState.UNPAIRED, t1.state)
        assertFalse(t1.shouldAdvertise)
        assertEquals(TileStatus.UNPAIRED, t1.tileStatus)
    }

    @Test
    fun `paired watch transitions cleanly across idle, advertising, screen-off, off-body, and bluetooth-off`() {
        val fsm = BeaconStateMachine()

        // Paired but off-body -> Idle (not worn).
        val idle = fsm.evaluate(BeaconInputs(paired = true, onBody = false, bluetoothEnabled = true))
        assertEquals(BeaconState.IDLE, idle.state)
        assertFalse(idle.shouldAdvertise)
        assertEquals(TileStatus.IDLE_NOT_WORN, idle.tileStatus)

        // Placed on wrist -> Advertising.
        val adv = fsm.evaluate(BeaconInputs(paired = true, onBody = true, bluetoothEnabled = true, screenOn = true))
        assertEquals(BeaconState.ADVERTISING, adv.state)
        assertTrue(adv.shouldAdvertise)
        assertEquals(TileStatus.ADVERTISING, adv.tileStatus)

        // Screen turns off -> keeps advertising.
        val screenOff =
            fsm.evaluate(BeaconInputs(paired = true, onBody = true, bluetoothEnabled = true, screenOn = false))
        assertEquals(BeaconState.ADVERTISING, screenOff.state)
        assertTrue(screenOff.shouldAdvertise)

        // Taken off wrist -> stops immediately (Idle).
        val offBody = fsm.evaluate(BeaconInputs(paired = true, onBody = false, bluetoothEnabled = true))
        assertEquals(BeaconState.IDLE, offBody.state)
        assertFalse(offBody.shouldAdvertise)
        assertEquals(TileStatus.IDLE_NOT_WORN, offBody.tileStatus)

        // On wrist but Bluetooth disabled -> Idle (Bluetooth off).
        val btOff = fsm.evaluate(BeaconInputs(paired = true, onBody = true, bluetoothEnabled = false))
        assertEquals(BeaconState.IDLE, btOff.state)
        assertFalse(btOff.shouldAdvertise)
        assertEquals(TileStatus.BLUETOOTH_OFF, btOff.tileStatus)

        // Bluetooth re-enabled while worn -> resumes Advertising.
        val resumed = fsm.evaluate(BeaconInputs(paired = true, onBody = true, bluetoothEnabled = true))
        assertEquals(BeaconState.ADVERTISING, resumed.state)
        assertTrue(resumed.shouldAdvertise)
    }
}
