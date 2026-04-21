#pragma once

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Biquad Filter Coefficients
 */
struct BiquadCoeffs {
    float b0, b1, b2, a1, a2;
};

/**
 * @brief A modular Biquad filter implementation using Direct Form II Transposed.
 */
class BiquadFilter {
private:
    float z1 = 0, z2 = 0; 

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

    /**
     * @brief Calculate Low-Pass Filter coefficients
     */
    static BiquadCoeffs calculate_lpf(float fc, float fs, float Q = 0.7071) {
        float omega = 2.0f * M_PI * fc / fs;
        float sn = sinf(omega);
        float cs = cosf(omega);
        float alpha = sn / (2.0f * Q);
        
        float a0 = 1.0f + alpha;
        return {
            ((1.0f - cs) / 2.0f) / a0,
            (1.0f - cs) / a0,
            ((1.0f - cs) / 2.0f) / a0,
            (-2.0f * cs) / a0,
            (1.0f - alpha) / a0
        };
    }

    /**
     * @brief Calculate High-Pass Filter coefficients
     */
    static BiquadCoeffs calculate_hpf(float fc, float fs, float Q = 0.7071) {
        float omega = 2.0f * M_PI * fc / fs;
        float sn = sinf(omega);
        float cs = cosf(omega);
        float alpha = sn / (2.0f * Q);
        
        float a0 = 1.0f + alpha;
        return {
            ((1.0f + cs) / 2.0f) / a0,
            -(1.0f + cs) / a0,
            ((1.0f + cs) / 2.0f) / a0,
            (-2.0f * cs) / a0,
            (1.0f - alpha) / a0
        };
    }

    /**
     * @brief Calculate Notch Filter coefficients (50/60 Hz)
     */
    static BiquadCoeffs calculate_notch(float fn, float fs, float BW = 2.0) {
        float omega = 2.0f * M_PI * fn / fs;
        float sn = sinf(omega);
        float cs = cosf(omega);
        float alpha = sn * sinhf(logf(2.0f) / 2.0f * BW * omega / sn);
        
        float a0 = 1.0f + alpha;
        return {
            1.0f / a0,
            -2.0f * cs / a0,
            1.0f / a0,
            -2.0f * cs / a0,
            (1.0f - alpha) / a0
        };
    }
};

/**
 * @brief ECG Signal Processing Pipeline
 * Cascades High-Pass, Notch, and Low-Pass filters.
 */
class EcgFilterPipeline {
private:
    BiquadFilter hpf;
    BiquadFilter notch;
    BiquadFilter lpf;
    
    BiquadCoeffs c_hpf;
    BiquadCoeffs c_notch;
    BiquadCoeffs c_lpf;

public:
    EcgFilterPipeline(float fs = 250.0f) {
        setup(fs);
    }

    void setup(float fs) {
        c_hpf = BiquadFilter::calculate_hpf(0.5f, fs);
        c_notch = BiquadFilter::calculate_notch(50.0f, fs);
        c_lpf = BiquadFilter::calculate_lpf(100.0f, fs);
        reset();
    }

    float apply(float sample) {
        float x = hpf.process(sample, c_hpf);
        x = notch.process(x, c_notch);
        x = lpf.process(x, c_lpf);
        return x;
    }

    void reset() {
        hpf.reset();
        notch.reset();
        lpf.reset();
    }
};
