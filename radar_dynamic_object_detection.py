#include <Arduino.h>
#include "driver/twai.h"

#define CAN_TX GPIO_NUM_17
#define CAN_RX GPIO_NUM_16

void setup()
{
    Serial.begin(115200);
    delay(2000);

    twai_general_config_t g =
        TWAI_GENERAL_CONFIG_DEFAULT(
            CAN_TX,
            CAN_RX,
            TWAI_MODE_NORMAL);

    twai_timing_config_t t =
        TWAI_TIMING_CONFIG_500KBITS();

    twai_filter_config_t f =
        TWAI_FILTER_CONFIG_ACCEPT_ALL();

    twai_driver_install(&g, &t, &f);
    twai_start();

    Serial.println("Waiting for Dynamic Objects...");
}

void loop()
{
    twai_message_t msg;

    if (twai_receive(&msg, pdMS_TO_TICKS(100)) == ESP_OK)
    {
        if (msg.identifier >= 0x510 && msg.identifier <= 0x519)
        {
            uint16_t raw =
                ((uint16_t)msg.data[1] << 8) |
                msg.data[0];

            raw &= 0x1FFF;

            float x_distance =
                raw * 0.032f - 60.8f;

            Serial.print("Object ");
            Serial.print(msg.identifier - 0x510 + 1);

            Serial.print("   Distance = ");
            Serial.print(x_distance, 2);
            Serial.println(" m");
        }
    }
}
