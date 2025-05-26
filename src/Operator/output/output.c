#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shell.h"
#include "thread.h"
#include "msg.h"
#include "xtimer.h"
#include "net/emcute.h"
#include "net/ipv6/addr.h"

#define EMCUTE_PORT         (1883U)        // Default port for MQTT
#define TOPIC_NAME          "riot/sensors" // Topic for publishing
#define BROKER_IP           "2001:db8::1"  // Replace with your broker's IPv6 address
#define CLIENT_ID           "riot_device"  // MQTT client ID
#define SLEEP_INTERVAL      (5U)           // Interval between publishes (seconds)

#define NUMOFSUBS           (16U)
#define TOPIC_MAXLEN        (64U)

static char stack[THREAD_STACKSIZE_MAIN];
static msg_t queue[NUMOFSUBS];
static emcute_sub_t subscriptions[NUMOFSUBS];
static char topics[NUMOFSUBS][TOPIC_MAXLEN];

/**
 * @brief Connect to the MQTT broker
 */
static int mqtt_connect(const char *address, const char *client_id)
{
    sock_udp_ep_t gw = { .family = AF_INET6, .port = EMCUTE_PORT };
    if (ipv6_addr_from_str((ipv6_addr_t *)&gw.addr.ipv6, address) == NULL) {
        puts("Error: Unable to parse broker address");
        return 1;
    }

    if (emcute_con(&gw, true, client_id, NULL, 0) != EMCUTE_OK) {
        puts("Error: Unable to connect to the broker");
        return 1;
    }

    printf("Connected to broker at [%s]:%i\n", address, EMCUTE_PORT);
    return 0;
}

/**
 * @brief Publish a message to an MQTT topic
 */
static int mqtt_publish(const char *topic, const char *message)
{
    emcute_topic_t t;
    unsigned flags = EMCUTE_QOS_0;

    t.name = topic;
    if (emcute_reg(&t) != EMCUTE_OK) {
        puts("Error: Unable to register topic");
        return 1;
    }

    if (emcute_pub(&t, message, strlen(message), flags) != EMCUTE_OK) {
        puts("Error: Unable to publish message");
        return 1;
    }

    printf("Published message: \"%s\" to topic: \"%s\"\n", message, topic);
    return 0;
}

/**
 * @brief Callback for received messages
 */
static void on_msg(const emcute_topic_t *topic, void *data, size_t len)
{
    printf("Received message on topic '%s': %.*s\n", topic->name, (int)len, (char *)data);
}

/**
 * @brief RIOT main thread function
 */
static void *mqtt_thread(void *arg)
{
    (void)arg;

    if (mqtt_connect(BROKER_IP, CLIENT_ID) != 0) {
        puts("Error: MQTT connection failed");
        return NULL;
    }

    emcute_sub_t sub = {
        .cb = on_msg,
        .topic.name = TOPIC_NAME
    };
    if (emcute_sub(&sub, EMCUTE_QOS_0) != EMCUTE_OK) {
        puts("Error: Unable to subscribe to topic");
        return NULL;
    }
    printf("Subscribed to topic: %s\n", TOPIC_NAME);

    while (1) {
        static int counter = 0;
        char message[64];
        snprintf(message, sizeof(message), "Sensor data %d", counter++);

        if (mqtt_publish(TOPIC_NAME, message) != 0) {
            puts("Error: MQTT publish failed");
        }

        xtimer_sleep(SLEEP_INTERVAL);
    }

    return NULL;
}

/**
 * @brief Application entry point
 */
int main(void)
{
    puts("MQTT RIOT Application");

    msg_init_queue(queue, NUMOFSUBS);
    emcute_run(stack, sizeof(stack), EMCUTE_PORT, "emcute_app");

    thread_create(stack, sizeof(stack), THREAD_PRIORITY_MAIN - 1,
                  THREAD_CREATE_STACKTEST, mqtt_thread, NULL, "mqtt_thread");

    return 0;
}