#pragma once

#include <stdint.h>
#include <stddef.h>

namespace ecg {

/**
 * @brief Binary Packet Structure for MQTT Telemetry
 * Total size: 7 + (BatchSize * 4) bytes
 */
struct __attribute__((packed)) EcgPacketHeader {
    uint8_t magic = 0xA5;
    uint32_t sequence;
    uint8_t batch_size;
};

static_assert(sizeof(EcgPacketHeader) == 6, "EcgPacketHeader size must be exactly 6 bytes for binary compatibility");

/**
 * @brief Simple CRC-8 implementation (Polynomial: 0x07)
 */
inline uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80) {
                crc = (crc << 1) ^ 0x07;
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

} // namespace ecg
