#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "mqtt_client.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "filter_pipeline.hpp"

static const char *TAG = "ECG_MQTT_PUB";

static EventGroupHandle_t wifi_event_group;
static constexpr int WIFI_CONNECTED_BIT = BIT0;
static bool mqtt_ready = false;
static esp_mqtt_client_handle_t mqtt_client = nullptr;
static adc_oneshot_unit_handle_t adc_handle = nullptr;
static constexpr adc_unit_t ECG_ADC_UNIT = ADC_UNIT_1;
static constexpr adc_channel_t ECG_ADC_CHANNEL = static_cast<adc_channel_t>(CONFIG_ECG_ADC_CHANNEL);
static constexpr adc_atten_t ECG_ADC_ATTEN = static_cast<adc_atten_t>(CONFIG_ECG_ADC_ATTEN_DB);
static constexpr gpio_num_t ECG_LO_PLUS_PIN = static_cast<gpio_num_t>(CONFIG_ECG_LO_PLUS_GPIO);
static constexpr gpio_num_t ECG_LO_MINUS_PIN = static_cast<gpio_num_t>(CONFIG_ECG_LO_MINUS_GPIO);
static constexpr int SAMPLE_RATE_HZ = CONFIG_ECG_SAMPLE_RATE_HZ;
static constexpr TickType_t SAMPLE_DELAY_TICKS = pdMS_TO_TICKS(1000 / SAMPLE_RATE_HZ);
static constexpr int MAX_BATCH_SIZE = 64;

static EcgFilterPipeline filter_pipeline;

static const char *wifi_auth_mode_name(wifi_auth_mode_t mode) {
    switch (mode) {
        case WIFI_AUTH_OPEN:
            return "OPEN";
        case WIFI_AUTH_WEP:
            return "WEP";
        case WIFI_AUTH_WPA_PSK:
            return "WPA_PSK";
        case WIFI_AUTH_WPA2_PSK:
            return "WPA2_PSK";
        case WIFI_AUTH_WPA_WPA2_PSK:
            return "WPA_WPA2_PSK";
        case WIFI_AUTH_WPA3_PSK:
            return "WPA3_PSK";
        default:
            return "OTHER";
    }
}

static bool leads_connected() {
    if (ECG_LO_PLUS_PIN != GPIO_NUM_NC && gpio_get_level(ECG_LO_PLUS_PIN) == 1) {
        return false;
    }
    if (ECG_LO_MINUS_PIN != GPIO_NUM_NC && gpio_get_level(ECG_LO_MINUS_PIN) == 1) {
        return false;
    }
    return true;
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    (void) handler_args;
    (void) base;
    (void) event_data;

    switch (event_id) {
        case MQTT_EVENT_CONNECTED:
            mqtt_ready = true;
            ESP_LOGI(TAG, "MQTT connected");
            break;
        case MQTT_EVENT_DISCONNECTED:
            mqtt_ready = false;
            ESP_LOGW(TAG, "MQTT disconnected");
            break;
        default:
            break;
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
    (void) arg;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        mqtt_ready = false;
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "WiFi disconnected, retrying");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const auto *event = static_cast<ip_event_got_ip_t *>(event_data);
        ESP_LOGI(TAG, "WiFi connected, IP=" IPSTR, IP2STR(&event->ip_info.ip));
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
    strncpy(reinterpret_cast<char *>(wifi_config.sta.ssid), CONFIG_ECG_WIFI_SSID, sizeof(wifi_config.sta.ssid) - 1);
    strncpy(reinterpret_cast<char *>(wifi_config.sta.password), CONFIG_ECG_WIFI_PASSWORD, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.threshold.authmode = static_cast<wifi_auth_mode_t>(CONFIG_ECG_WIFI_AUTHMODE);
    wifi_config.sta.pmf_cfg.capable = true;
    wifi_config.sta.pmf_cfg.required = false;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi SSID=%s auth=%s", CONFIG_ECG_WIFI_SSID, wifi_auth_mode_name(wifi_config.sta.threshold.authmode));
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
}

static void init_mqtt() {
    esp_mqtt_client_config_t mqtt_cfg = {};
    mqtt_cfg.broker.address.uri = CONFIG_ECG_MQTT_BROKER_URI;
    mqtt_cfg.session.keepalive = CONFIG_ECG_MQTT_KEEPALIVE_SEC;
    mqtt_cfg.network.reconnect_timeout_ms = CONFIG_ECG_MQTT_RECONNECT_TIMEOUT_MS;

    mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
    ESP_ERROR_CHECK(esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY, mqtt_event_handler, nullptr));
    ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));
}

static void init_adc() {
    adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = ECG_ADC_UNIT,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&unit_cfg, &adc_handle));

    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ECG_ADC_ATTEN,
        .bitwidth = ADC_BITWIDTH_DEFAULT,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc_handle, ECG_ADC_CHANNEL, &chan_cfg));

    if (ECG_LO_PLUS_PIN != GPIO_NUM_NC) {
        gpio_set_direction(ECG_LO_PLUS_PIN, GPIO_MODE_INPUT);
    }
    if (ECG_LO_MINUS_PIN != GPIO_NUM_NC) {
        gpio_set_direction(ECG_LO_MINUS_PIN, GPIO_MODE_INPUT);
    }
}

extern "C" void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "Starting ECG MQTT publisher");
    init_wifi();
    init_mqtt();
    init_adc();

    char payload[16 * MAX_BATCH_SIZE];
    int sample = 0;
    uint32_t sample_count = 0;
    int batch_count = 0;
    int payload_len = 0;

    while (true) {
        if (!mqtt_ready) {
            vTaskDelay(pdMS_TO_TICKS(250));
            continue;
        }

        if (!leads_connected()) {
            ESP_LOGW(TAG, "AD8232 leads disconnected");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        esp_err_t adc_err = adc_oneshot_read(adc_handle, ECG_ADC_CHANNEL, &sample);
        if (adc_err != ESP_OK) {
            ESP_LOGE(TAG, "ADC read failed: %s", esp_err_to_name(adc_err));
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        float filtered_sample = filter_pipeline.apply(static_cast<float>(sample));

        int written = 0;
        if (batch_count == 0) {
            written = snprintf(payload, sizeof(payload), "%.2f", filtered_sample);
        } else {
            written = snprintf(payload + payload_len, sizeof(payload) - payload_len, ",%.2f", filtered_sample);
        }

        if (written <= 0 || (payload_len + written) >= static_cast<int>(sizeof(payload))) {
            ESP_LOGW(TAG, "Batch buffer overflow risk, flushing early");
            written = 0;
        } else {
            payload_len += written;
            batch_count++;
        }

        bool flush_batch = batch_count >= CONFIG_ECG_MQTT_BATCH_SIZE;
        if (written == 0 && batch_count == 0) {
            vTaskDelay(SAMPLE_DELAY_TICKS);
            continue;
        }

        if (flush_batch || written == 0) {
            int msg_id = esp_mqtt_client_publish(
                mqtt_client,
                CONFIG_ECG_MQTT_TOPIC,
                payload,
                payload_len,
                CONFIG_ECG_MQTT_QOS,
                0
            );
            if (msg_id < 0) {
                ESP_LOGW(TAG, "MQTT publish failed for batch ending sample=%d", sample);
            } else {
                sample_count += batch_count;
                if (CONFIG_ECG_LOG_EVERY_N_SAMPLES > 0 && (sample_count % CONFIG_ECG_LOG_EVERY_N_SAMPLES) < static_cast<uint32_t>(batch_count)) {
                    ESP_LOGI(
                        TAG,
                        "Published batch=%d total_samples=%lu topic=%s",
                        batch_count,
                        static_cast<unsigned long>(sample_count),
                        CONFIG_ECG_MQTT_TOPIC
                    );
                }
            }
            batch_count = 0;
            payload_len = 0;
        }

        vTaskDelay(SAMPLE_DELAY_TICKS);
    }
}
