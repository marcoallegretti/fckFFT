#ifndef FCKFFT_DYNAMICS_H
#define FCKFFT_DYNAMICS_H

#include <cmath>
#include <cstddef>

#include "simd.h"

namespace fckfft {
inline namespace FCKFFT_ABI {

namespace detail {

inline bool positiveFinite(float x)
{
    return finite(x) && x > 0.0f;
}

}

inline float alphaFromTimeConstant(float timeConstant, float sampleRate)
{
    if (!detail::positiveFinite(sampleRate) || !detail::finite(timeConstant) || timeConstant < 0.0f)
        return detail::notANumber();
    return static_cast<float>(-std::expm1(-1.0 / (static_cast<double>(sampleRate) * timeConstant)));
}

inline float timeConstantFromAlpha(float alpha, float sampleRate)
{
    if (!detail::positiveFinite(sampleRate) || !detail::positiveFinite(alpha) || alpha > 1.0f)
        return detail::notANumber();
    return static_cast<float>(-1.0 / (static_cast<double>(sampleRate) * std::log1p(-static_cast<double>(alpha))));
}

inline float alphaForBandwidth(float bandwidth, float sampleRate)
{
    if (!detail::positiveFinite(sampleRate) || !detail::positiveFinite(bandwidth) || bandwidth > sampleRate)
        return detail::notANumber();
    const double c = std::cos(3.14159265358979323846 * static_cast<double>(bandwidth) / static_cast<double>(sampleRate));
    const double q = 1.0 - std::sqrt(2.0), l = 2.0 * c - 2.0, k = 2.0 - 2.0 * c;
    return static_cast<float>((-l - std::sqrt(l * l - 4.0 * q * k)) / (2.0 * q));
}

inline float alphaForQ(float frequency, float q, float sampleRate)
{
    if (!detail::positiveFinite(q) || !detail::finite(frequency))
        return detail::notANumber();
    return alphaForBandwidth(frequency / q, sampleRate);
}

inline float heuristicAlpha(float frequency, float sampleRate, float k = 1.0f, float n = 1.0f)
{
    if (!detail::positiveFinite(sampleRate) || !detail::positiveFinite(k) || !detail::finite(n)
        || !detail::finite(frequency) || frequency < 0.0f)
        return detail::notANumber();
    const double f = frequency > 1e-6f ? frequency : 1e-6;
    const double scale = std::pow(std::log1p(f) / std::log(10.0), static_cast<double>(n));
    return static_cast<float>(-std::expm1(-f / (static_cast<double>(sampleRate) * k * scale)));
}

inline float heuristicGamma(float frequency, float sampleRate, float k = 1.0f, float n = 1.0f)
{
    return 0.5f * heuristicAlpha(frequency, sampleRate, k, n);
}

inline void heuristicAlphas(const float* frequencies, std::size_t count, float sampleRate, float* alphas,
                            float k = 1.0f, float n = 1.0f)
{
    for (std::size_t i = 0; i < count; ++i)
        alphas[i] = heuristicAlpha(frequencies[i], sampleRate, k, n);
}

inline void heuristicGammas(const float* frequencies, std::size_t count, float sampleRate, float* gammas,
                            float k = 1.0f, float n = 1.0f)
{
    for (std::size_t i = 0; i < count; ++i)
        gammas[i] = heuristicGamma(frequencies[i], sampleRate, k, n);
}

}
}

#endif
