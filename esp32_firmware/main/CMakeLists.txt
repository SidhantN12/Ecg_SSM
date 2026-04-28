idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_adc esp_event esp_netif esp_wifi mqtt nvs_flash
)
