#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "esp_adc/adc_continuous.h"
#include "esp_check.h"
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

// Ensure CONFIG macros are defined, or provide defaults to avoid compilation
// failure
#ifndef CONFIG_ECG_MQTT_BATCH_SIZE
#define CONFIG_ECG_MQTT_BATCH_SIZE 8
#endif

static const char *TAG = "ECG_SSM_V6";

// --- Configuration Constants ---
static constexpr uint32_t ADC_READ_LEN = 256;
static constexpr uint32_t SAMPLE_RATE_HZ = 187; // MIT-BIH rate
static constexpr uint32_t BATCH_SIZE = CONFIG_ECG_MQTT_BATCH_SIZE;

// --- Global State ---
static EventGroupHandle_t wifi_event_group;
static constexpr int WIFI_CONNECTED_BIT = BIT0;
static volatile bool mqtt_ready = false;
static esp_mqtt_client_handle_t mqtt_client = nullptr;
static adc_continuous_handle_t adc_handle = nullptr;
static QueueHandle_t sample_queue = nullptr;
static EcgFilterPipeline filter_pipeline(static_cast<float>(SAMPLE_RATE_HZ));

/**
 * @brief MQTT Event Handler
 */
static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
  (void)handler_args;
  (void)base;
  (void)event_data;

  switch (static_cast<esp_mqtt_event_id_t>(event_id)) {
  case MQTT_EVENT_CONNECTED:
    mqtt_ready = true;
    ESP_LOGI(TAG, "MQTT connected to broker");
    break;
  case MQTT_EVENT_DISCONNECTED:
    mqtt_ready = false;
    ESP_LOGW(TAG, "MQTT disconnected");
    break;
  default:
    break;
  }
}

/**
 * @brief WiFi Event Handler
 */
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    mqtt_ready = false;
    xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
    esp_wifi_connect();
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

/**
 * @brief Initialize WiFi and Wait for Connection
 */
static esp_err_t init_wifi() {
  wifi_event_group = xEventGroupCreate();
  ESP_RETURN_ON_FALSE(wifi_event_group, ESP_ERR_NO_MEM, TAG,
                      "Failed to create event group");

  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr, nullptr));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, nullptr, nullptr));

  wifi_config_t wifi_config = {};
#ifdef CONFIG_ECG_WIFI_SSID
  strncpy((char *)wifi_config.sta.ssid, CONFIG_ECG_WIFI_SSID,
          sizeof(wifi_config.sta.ssid) - 1);
#endif
#ifdef CONFIG_ECG_WIFI_PASSWORD
  strncpy((char *)wifi_config.sta.password, CONFIG_ECG_WIFI_PASSWORD,
          sizeof(wifi_config.sta.password) - 1);
#endif

  // Enable PMF (Protected Management Frames)
  // Modern phone hotspots (especially iOS 15+ and Android 13+) use WPA3/WPA2 mixed mode
  // and will aggressively drop clients after ~5 seconds if they do not advertise PMF capability.
  wifi_config.sta.pmf_cfg.capable = true;
  wifi_config.sta.pmf_cfg.required = false;
  
  // Enhance Android 15 / WPA3 mixed mode compatibility
  wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;

  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());
  ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

  ESP_LOGI(TAG, "Connecting to WiFi...");
  xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE,
                      portMAX_DELAY);
  ESP_LOGI(TAG, "WiFi Connected");

  return ESP_OK;
}

/**
 * @brief Initialize MQTT Client
 */
static esp_err_t init_mqtt() {
  esp_mqtt_client_config_t mqtt_cfg = {};
#ifdef CONFIG_ECG_MQTT_BROKER_URI
  mqtt_cfg.broker.address.uri = CONFIG_ECG_MQTT_BROKER_URI;
#endif

  mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
  ESP_RETURN_ON_FALSE(mqtt_client, ESP_FAIL, TAG, "Failed to init MQTT client");

  ESP_ERROR_CHECK(esp_mqtt_client_register_event(mqtt_client, MQTT_EVENT_ANY,
                                                 mqtt_event_handler, nullptr));
  ESP_ERROR_CHECK(esp_mqtt_client_start(mqtt_client));

  return ESP_OK;
}

/**
 * @brief Initialize Continuous ADC with DMA
 */
static esp_err_t init_adc() {
  adc_continuous_handle_cfg_t adc_config = {.max_store_buf_size = 1024,
                                            .conv_frame_size = ADC_READ_LEN,
                                            .flags = {.flush_pool = 0}};

  ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &adc_handle));

  adc_digi_pattern_config_t adc_pattern = {
      .atten = static_cast<adc_atten_t>(0),
      .channel = static_cast<adc_channel_t>(0),
      .unit = ADC_UNIT_1,
      .bit_width = static_cast<adc_bitwidth_t>(SOC_ADC_DIGI_MAX_BITWIDTH),
  };
#ifdef CONFIG_ECG_ADC_ATTEN_DB
  adc_pattern.atten = static_cast<adc_atten_t>(CONFIG_ECG_ADC_ATTEN_DB);
#endif
#ifdef CONFIG_ECG_ADC_CHANNEL
  adc_pattern.channel = static_cast<adc_channel_t>(CONFIG_ECG_ADC_CHANNEL);
#endif

  // ESP32 continuous ADC requires a minimum sampling frequency of 20000 Hz.
  // We set it to 20kHz here and decimate the data in the acquisition task to achieve 187 Hz.
  adc_continuous_config_t config = {
      .pattern_num = 1,
      .adc_pattern = &adc_pattern,
      .sample_freq_hz = 20000,
      .conv_mode = ADC_CONV_SINGLE_UNIT_1,
      .format = ADC_DIGI_OUTPUT_FORMAT_TYPE1,
  };

  ESP_ERROR_CHECK(adc_continuous_config(adc_handle, &config));
  ESP_ERROR_CHECK(adc_continuous_start(adc_handle));

  // Configure Lead-off GPIOs
#ifdef CONFIG_ECG_LO_PLUS_GPIO
  if (CONFIG_ECG_LO_PLUS_GPIO != -1) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << CONFIG_ECG_LO_PLUS_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
  }
#endif

#ifdef CONFIG_ECG_LO_MINUS_GPIO
  if (CONFIG_ECG_LO_MINUS_GPIO != -1) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << CONFIG_ECG_LO_MINUS_GPIO),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
  }
#endif

  return ESP_OK;
}

/**
 * @brief Acquisition Task: Reads DMA buffers and applies DSP
 */
static void acquisition_task(void *pvParameters) {
  uint8_t result[ADC_READ_LEN] = {0};
  uint32_t ret_num = 0;
  uint32_t decimation_counter = 0;
  const uint32_t decimation_factor = 20000 / SAMPLE_RATE_HZ;

  while (true) {
    // Lead-Off Detection Check
    bool lo_pos = false;
    bool lo_neg = false;
#ifdef CONFIG_ECG_LO_PLUS_GPIO
    lo_pos = (CONFIG_ECG_LO_PLUS_GPIO != -1)
                 ? gpio_get_level((gpio_num_t)CONFIG_ECG_LO_PLUS_GPIO)
                 : false;
#endif
#ifdef CONFIG_ECG_LO_MINUS_GPIO
    lo_neg = (CONFIG_ECG_LO_MINUS_GPIO != -1)
                 ? gpio_get_level((gpio_num_t)CONFIG_ECG_LO_MINUS_GPIO)
                 : false;
#endif
    bool leads_off = lo_pos || lo_neg;

    esp_err_t ret =
        adc_continuous_read(adc_handle, result, ADC_READ_LEN, &ret_num, 0);
    if (ret == ESP_OK) {
      for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
        adc_digi_output_data_t *p = (adc_digi_output_data_t *)&result[i];
        float sample = static_cast<float>(p->type1.data & 0xFFF);

        // Zero out sample if leads are off to avoid transient noise artifacts
        float filtered = leads_off ? 0.0f : filter_pipeline.apply(sample);

        decimation_counter++;
        if (decimation_counter >= decimation_factor) {
            decimation_counter = 0;
            if (xQueueSend(sample_queue, &filtered, pdMS_TO_TICKS(10)) != pdPASS) {
              ESP_LOGW(TAG, "Sample queue full, dropping data");
            }
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

/**
 * @brief Publishing Task: Batches samples and publishes via binary MQTT
 * protocol
 */
static void publishing_task(void *pvParameters) {
  uint32_t sequence = 0;
  float batch[BATCH_SIZE];
  uint8_t packet_buf[sizeof(ecg::EcgPacketHeader) +
                     (CONFIG_ECG_MQTT_BATCH_SIZE * 4) + 1];

  while (true) {
    if (!mqtt_ready) {
      vTaskDelay(pdMS_TO_TICKS(100));
      continue;
    }

    // Fill batch from queue
    for (int i = 0; i < BATCH_SIZE; i++) {
      if (xQueueReceive(sample_queue, &batch[i], portMAX_DELAY) != pdPASS) {
        i--; // Retry
        continue;
      }
    }

    // Construct high-integrity binary packet
    ecg::EcgPacketHeader *hdr =
        reinterpret_cast<ecg::EcgPacketHeader *>(packet_buf);
    hdr->magic = 0xA5;
    hdr->sequence = sequence++;
    hdr->batch_size = static_cast<uint8_t>(BATCH_SIZE);

    memcpy(packet_buf + sizeof(ecg::EcgPacketHeader), batch, sizeof(batch));

    size_t data_len = sizeof(ecg::EcgPacketHeader) + sizeof(batch);
    packet_buf[data_len] = ecg::crc8(packet_buf, data_len);

#ifdef CONFIG_ECG_MQTT_TOPIC
    int qos = 0;
#ifdef CONFIG_ECG_MQTT_QOS
    qos = CONFIG_ECG_MQTT_QOS;
#endif
    esp_mqtt_client_publish(mqtt_client, CONFIG_ECG_MQTT_TOPIC,
                            reinterpret_cast<const char *>(packet_buf),
                            data_len + 1, qos, 0);
#endif

#ifdef CONFIG_ECG_LOG_EVERY_N_SAMPLES
    if (CONFIG_ECG_LOG_EVERY_N_SAMPLES > 0 &&
        (sequence * BATCH_SIZE) % CONFIG_ECG_LOG_EVERY_N_SAMPLES == 0) {
      ESP_LOGI(TAG, "Stream active: Sent batch %lu (Total: %lu samples)",
               (unsigned long)sequence, (unsigned long)(sequence * BATCH_SIZE));
    }
#endif
  }
}

extern "C" void app_main(void) {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  sample_queue = xQueueCreate(BATCH_SIZE * 8, sizeof(float));
  if (!sample_queue) {
    ESP_LOGE(TAG, "Failed to create sample queue");
    return;
  }

  ESP_ERROR_CHECK(init_wifi());
  ESP_ERROR_CHECK(init_mqtt());
  ESP_ERROR_CHECK(init_adc());

  ESP_LOGI(TAG, "Launching acquisition and publishing threads...");
  xTaskCreatePinnedToCore(acquisition_task, "AcqTask", 4096, nullptr, 10,
                          nullptr, 1);
  xTaskCreatePinnedToCore(publishing_task, "PubTask", 4096, nullptr, 5, nullptr,
                          0);
}