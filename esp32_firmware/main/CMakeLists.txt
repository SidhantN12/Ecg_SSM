idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES driver esp_adc esp_event esp_netif esp_wifi mqtt nvs_flash
)
