#pragma once

#include <cmath>

/**
 * @brief Biquad Filter Coefficients
 */
struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;
};

/**
 * @brief A modular Biquad filter implementation using Direct Form II Transposed.
 * This structure is highly efficient for both software and FPGA (LUT-based) implementation.
 */
class BiquadFilter {
private:
    float z1 = 0, z2 = 0; // Delay registers

public:
    inline float process(float x, const BiquadCoeffs& c) {
        float y = c.b0 * x + z1;
        z1 = c.b1 * x - c.a1 * y + z2;
        z2 = c.b2 * x - c.a2 * y;
        return y;
    }

    void reset() {
        z1 = 0;
        z2 = 0;
    }
};

/**
 * @brief ECG Signal Processing Pipeline
 * Cascades High-Pass, Notch, and Low-Pass filters.
 * Target Sample Rate: 250 Hz (Default)
 */
class EcgFilterPipeline {
public:
    // Coefficients for 250Hz sample rate
    // HPF 0.5Hz, Notch 50Hz, LPF 100Hz
    // Note: In a production environment, these would be pre-calculated or set via Kconfig.
    
    // HPF 0.5Hz @ 250Hz sample rate (approximate)
    const BiquadCoeffs HPF_0_5HZ = {0.99115, -1.9823, 0.99115, -1.9822, 0.9824}; 
    
    // Notch 50Hz @ 250Hz sample rate
    const BiquadCoeffs NOTCH_50HZ = {0.9650, -1.1345, 0.9650, -1.1345, 0.9300};
    
    // LPF 100Hz @ 250Hz sample rate
    const BiquadCoeffs LPF_100HZ = {0.4208, 0.8416, 0.4208, 0.4419, 0.2413};

private:
    BiquadFilter hpf;
    BiquadFilter notch;
    BiquadFilter lpf;

public:
    float apply(float sample) {
        float x = hpf.process(sample, HPF_0_5HZ);
        x = notch.process(x, NOTCH_50HZ);
        x = lpf.process(x, LPF_100HZ);
        return x;
    }

    void reset() {
        hpf.reset();
        notch.reset();
        lpf.reset();
    }
};
