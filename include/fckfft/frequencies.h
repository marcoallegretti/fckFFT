#ifndef FCKFFT_FREQUENCIES_H
#define FCKFFT_FREQUENCIES_H

#include <cmath>
#include <cstddef>

namespace fckfft {

inline float midiToHz(float note, float tuning = 440.0f)
{
    return tuning * std::exp2((note - 69.0f) / 12.0f);
}

inline float hzToMidi(float frequency, float tuning = 440.0f)
{
    return 69.0f + 12.0f * std::log2(frequency / tuning);
}

inline void logUniformFrequencies(float minFrequency, float binsPerOctave, std::size_t count, float* out)
{
    for (std::size_t i = 0; i < count; ++i)
        out[i] = minFrequency * std::exp2(static_cast<float>(i) / binsPerOctave);
}

inline void logSpacedFrequencies(float minFrequency, float maxFrequency, std::size_t count, float* out)
{
    if (count == 1)
    {
        out[0] = minFrequency;
        return;
    }
    const double ratio = std::log(static_cast<double>(maxFrequency) / minFrequency) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = static_cast<float>(minFrequency * std::exp(ratio * static_cast<double>(i)));
    out[count - 1] = maxFrequency;
}

inline void linearSpacedFrequencies(float minFrequency, float maxFrequency, std::size_t count, float* out)
{
    if (count == 1)
    {
        out[0] = minFrequency;
        return;
    }
    const double step = (static_cast<double>(maxFrequency) - minFrequency) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = static_cast<float>(minFrequency + step * static_cast<double>(i));
    out[count - 1] = maxFrequency;
}

inline float hzToMel(float frequency, bool htk = false)
{
    if (htk)
        return static_cast<float>(2595.0 * std::log10(1.0 + frequency / 700.0));
    const double fSp = 200.0 / 3.0;
    const double minLogHz = 1000.0;
    const double minLogMel = minLogHz / fSp;
    const double logStep = std::log(6.4) / 27.0;
    if (frequency < minLogHz)
        return static_cast<float>(frequency / fSp);
    return static_cast<float>(minLogMel + std::log(frequency / minLogHz) / logStep);
}

inline float melToHz(float mel, bool htk = false)
{
    if (htk)
        return static_cast<float>(700.0 * (std::pow(10.0, mel / 2595.0) - 1.0));
    const double fSp = 200.0 / 3.0;
    const double minLogHz = 1000.0;
    const double minLogMel = minLogHz / fSp;
    const double logStep = std::log(6.4) / 27.0;
    if (mel < minLogMel)
        return static_cast<float>(fSp * mel);
    return static_cast<float>(minLogHz * std::exp(logStep * (mel - minLogMel)));
}

inline void melFrequencies(float minFrequency, float maxFrequency, std::size_t count, float* out, bool htk = false)
{
    if (count == 1)
    {
        out[0] = minFrequency;
        return;
    }
    const double minMel = hzToMel(minFrequency, htk);
    const double maxMel = hzToMel(maxFrequency, htk);
    const double step = (maxMel - minMel) / static_cast<double>(count - 1);
    for (std::size_t i = 0; i < count; ++i)
        out[i] = melToHz(static_cast<float>(minMel + step * static_cast<double>(i)), htk);
    out[0] = minFrequency;
    out[count - 1] = maxFrequency;
}

}

#endif
