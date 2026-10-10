#ifndef FCKFFT_DYNAMICS_H
#define FCKFFT_DYNAMICS_H

#include <cmath>
#include <cstddef>

namespace fckfft {

inline float alphaFromTimeConstant(float timeConstant, float sampleRate)
{
    return static_cast<float>(-std::expm1(-1.0 / (static_cast<double>(sampleRate) * timeConstant)));
}

inline float timeConstantFromAlpha(float alpha, float sampleRate)
{
    return static_cast<float>(-1.0 / (static_cast<double>(sampleRate) * std::log1p(-static_cast<double>(alpha))));
}

inline float heuristicAlpha(float frequency, float sampleRate, float k = 1.0f, float n = 1.0f)
{
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

#endif
