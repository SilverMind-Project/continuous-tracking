package io.silvermind.wearcompanion.app.uwb

import io.silvermind.wearcompanion.core.BatchMqttSender
import org.eclipse.paho.client.mqttv3.MqttClient
import org.eclipse.paho.client.mqttv3.MqttConnectOptions
import org.eclipse.paho.client.mqttv3.MqttMessage
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence

/**
 * LAN-only MQTT batch transport publishing `device_ranging.v1` JSON micro-batches
 * to `rf-ingress` (`rf/devices/{tag_id}/ranges`).
 */
class MqttBatchTransport(
    private val brokerUrl: String,
    private val clientId: String,
    private val username: String? = null,
    private val password: String? = null,
) : BatchMqttSender {
    private var client: MqttClient? = null

    override fun publish(topic: String, payloadJson: String): Boolean =
        runCatching {
            val active = ensureConnected()
            val message =
                MqttMessage(payloadJson.toByteArray(Charsets.UTF_8)).apply {
                    qos = 0
                    isRetained = false
                }
            active.publish(topic, message)
            true
        }.getOrDefault(false)

    fun close() {
        runCatching {
            client?.let {
                if (it.isConnected) {
                    it.disconnect()
                }
                it.close()
            }
        }
        client = null
    }

    private fun ensureConnected(): MqttClient {
        val existing = client
        if (existing != null && existing.isConnected) {
            return existing
        }
        val fresh = MqttClient(brokerUrl, clientId, MemoryPersistence())
        val opts =
            MqttConnectOptions().apply {
                isCleanSession = true
                connectionTimeout = 3
                keepAliveInterval = 30
                if (!username.isNullOrBlank()) {
                    userName = username
                }
                if (!password.isNullOrBlank()) {
                    this.password = password.toCharArray()
                }
            }
        fresh.connect(opts)
        client = fresh
        return fresh
    }
}
