#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_adc/adc_continuous.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "mqtt_client.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "filter_pipeline.hpp"
#include "protocol.hpp"

static const char *TAG = "ECG_SSM_V1";

// --- Configuration Constants ---
static constexpr uint32_t ADC_READ_LEN = 256; 
static constexpr uint32_t SAMPLE_RATE_HZ = 187; // Standardized to MIT-BIH rate
static constexpr uint32_t BATCH_SIZE = CONFIG_ECG_MQTT_BATCH_SIZE;

// --- Global State ---
static EventGroupHandle_t wifi_event_group;
static constexpr int WIFI_CONNECTED_BIT = BIT0;
static bool mqtt_ready = false;
static esp_mqtt_client_handle_t mqtt_client = nullptr;
static adc_continuous_handle_t adc_handle = nullptr;
static QueueHandle_t sample_queue = nullptr;
static EcgFilterPipeline filter_pipeline(static_cast<float>(SAMPLE_RATE_HZ));

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    (void) handler_args; (void) base; (void) event_data;
    if (event_id == MQTT_EVENT_CONNECTED) {
        mqtt_ready = true;
        ESP_LOGI(TAG, "MQTT connected");
    } else if (event_id == MQTT_EVENT_DISCONNECTED) {
        mqtt_ready = false;
        ESP_LOGW(TAG, "MQTT disconnected");
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        mqtt_ready = false;
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void init_wifi() {
    wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr));

    wifi_config_t wifi_config = {};
    strncpy((char *)wifi_config.sta.ssid, CONFIG_ECG_WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy((char *)wifi_config.sta.password, CONFIG_ECG_WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1);
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
}

static void init_mqtt() {
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = CONFIG_ECG_MQTT_BROKER_URI;
    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY, mqtt_event_handler, nullptr));
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));
}

static void init_adc() {
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 1024,
        .conv_frame_size = ADC_READ_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_unit(&adc_config, &adc_handle));

    adc_continuous_config_t config = {
        .pattern_num = 1,
        .sample_freq_hz = SAMPLE_RATE_HZ,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE1,
    };
    adc_digi_pattern_config_t adc_pattern = {
        .atten = (uint8_t)CONFIG_ECG_ADC_ATTEN_DB,
        .channel = (uint8_t)CONFIG_ECG_ADC_CHANNEL,
        .unit = ADC_UNIT_1,
        .bit_width = ADC_BITWIDTH_DEFAULT,
    };
    config.adc_pattern = &adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(adc_handle, &config));
    ESP_ERROR_CHECK(adc_continuous_start(adc_handle));

    if (CONFIG_ECG_LO_PLUS_GPIO != -1) gpio_set_direction((gpio_num_t)CONFIG_ECG_LO_PLUS_GPIO, GPIO_MODE_INPUT);
    if (CONFIG_ECG_LO_MINUS_GPIO != -1) gpio_set_direction((gpio_num_t)CONFIG_ECG_LO_MINUS_GPIO, GPIO_MODE_INPUT);
}

/**
 * @brief Task to read from DMA buffers and apply digital filters
 */
static void acquisition_task(void *pvParameters) {
    uint8_t result[ADC_READ_LEN] = {0};
    uint32_t ret_num = 0;

    while (true) {
        // Lead-Off Detection Check
        bool lo_pos = (CONFIG_ECG_LO_PLUS_GPIO != -1) ? gpio_get_level((gpio_num_t)CONFIG_ECG_LO_PLUS_GPIO) : false;
        bool lo_neg = (CONFIG_ECG_LO_MINUS_GPIO != -1) ? gpio_get_level((gpio_num_t)CONFIG_ECG_LO_MINUS_GPIO) : false;
        bool leads_off = lo_pos || lo_neg;

        esp_err_t ret = adc_continuous_read(adc_handle, result, ADC_READ_LEN, &ret_num, 0);
        if (ret == ESP_OK) {
            for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
                adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i];
                float sample = (float)p->type1.data;
                
                // If leads are off, zero the sample to prevent junk classification
                float filtered = leads_off ? 0.0f : filter_pipeline.apply(sample);
                
                xQueueSend(sample_queue, &filtered, pdMS_TO_TICKS(10));
            }
        }
        vTaskDelay(pdMS_TO_TICKS(1)); 
    }
}

/**
 * @brief Task to batch samples and publish via MQTT using binary protocol
 */
static void publishing_task(void *pvParameters) {
    uint32_t sequence = 0;
    float batch[BATCH_SIZE];
    // Size: Header(6) + Data(BATCH_SIZE*4) + CRC(1)
    // Max size with BATCH_SIZE=64 is 6 + 256 + 1 = 263 bytes
    uint8_t packet_buf[256 + 16]; 

    while (true) {
        if (!mqtt_ready) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        for (int i = 0; i < BATCH_SIZE; i++) {
            if (xQueueReceive(sample_queue, &batch[i], portMAX_DELAY) != pdPASS) {
                i--; continue;
            }
        }

        // Build Packet
        ecg::EcgPacketHeader *hdr = (ecg::EcgPacketHeader *)packet_buf;
        hdr->magic = 0xA5;
        hdr->sequence = sequence++;
        hdr->batch_size = (uint8_t)BATCH_SIZE;

        memcpy(packet_buf + sizeof(ecg::EcgPacketHeader), batch, sizeof(batch));
        
        size_t data_len = sizeof(ecg::EcgPacketHeader) + sizeof(batch);
        packet_buf[data_len] = ecg::crc8(packet_buf, data_len);
        
        esp_mqtt_client_publish(mqtt_client, CONFIG_ECG_MQTT_TOPIC, (const char *)packet_buf, data_len + 1, CONFIG_ECG_MQTT_QOS, 0);
        
        if (CONFIG_ECG_LOG_EVERY_N_SAMPLES > 0 && (sequence * BATCH_SIZE) % CONFIG_ECG_LOG_EVERY_N_SAMPLES == 0) {
            ESP_LOGI(TAG, "Sent batch %lu, total samples %lu", sequence, sequence * BATCH_SIZE);
        }
    }
}

extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    sample_queue = xQueueCreate(BATCH_SIZE * 4, sizeof(float));
    init_wifi();
    init_mqtt();
    init_adc();

    xTaskCreatePinnedToCore(acquisition_task, "AcqTask", 4096, nullptr, 10, nullptr, 1);
    xTaskCreatePinnedToCore(publishing_task, "PubTask", 4096, nullptr, 5, nullptr, 0);
}
