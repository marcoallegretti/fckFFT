#ifndef FCKFFT_TESTS_SIGNALS_H
#define FCKFFT_TESTS_SIGNALS_H

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <fckfft/dynamics.h>

namespace signals {

constexpr double kPi = 3.14159265358979323846;

template <class Freq, class Amp>
std::vector<float> render(std::size_t count, double sampleRate, Freq freq, Amp amp)
{
    std::vector<float> out(count);
    double phase = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = static_cast<float>(amp(i) * std::sin(phase));
        phase += 2.0 * kPi * freq(i) / sampleRate;
        if (phase > 2.0 * kPi)
            phase -= 2.0 * kPi;
    }
    return out;
}

inline std::vector<float> tone(std::size_t count, double sampleRate, double frequency, double amplitude = 1.0)
{
    return render(
        count, sampleRate, [frequency](std::size_t) { return frequency; }, [amplitude](std::size_t) { return amplitude; });
}

inline std::vector<float> noise(std::size_t count, std::uint32_t seed = 1u, float amplitude = 1.0f)
{
    std::vector<float> out(count);
    for (float& v : out)
    {
        seed = seed * 1664525u + 1013904223u;
        v = amplitude * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f);
    }
    return out;
}

struct TrackingModel
{
    int rule;
    double sampleRate, ratio, accPower = 0.0001;
    std::vector<double> alpha, beta, gamma;
    std::vector<std::complex<double>> r, rr, previous, z, w, naturalW, d;

    TrackingModel(const std::vector<float>& frequencies, double sr, int ruleIndex, double thresholdDb,
                  float fixedAlpha = 0.0f)
        : rule(ruleIndex), sampleRate(sr), ratio(std::pow(10.0, thresholdDb / 10.0))
    {
        for (float f : frequencies)
        {
            const double a = fixedAlpha > 0.0f ? fixedAlpha : fckfft::heuristicAlpha(f, static_cast<float>(sr));
            alpha.push_back(a);
            beta.push_back(a);
            gamma.push_back(0.5 * a);
            naturalW.push_back(std::polar(1.0, -2.0 * kPi * f / sr));
        }
        const std::size_t n = frequencies.size();
        r.assign(n, 0.0);
        rr.assign(n, 0.0);
        previous.assign(n, 0.0);
        z.assign(n, 1.0);
        d.assign(n, 0.0);
        w = naturalW;
    }

    void step(double s)
    {
        double peak = 0.0;
        for (std::size_t k = 0; k < r.size(); ++k)
        {
            r[k] = (1.0 - alpha[k]) * r[k] + alpha[k] * s * z[k];
            rr[k] = (1.0 - beta[k]) * rr[k] + beta[k] * r[k];
            peak = std::max(peak, std::norm(rr[k]));
        }
        accPower = peak;
        const double threshold = accPower * ratio;
        for (std::size_t k = 0; k < r.size(); ++k)
        {
            d[k] = std::conj(rr[k]) * previous[k];
            previous[k] = rr[k];
            std::complex<double> next = w[k];
            if (rule == 1)
                next = std::polar(1.0, std::arg(w[k]) + gamma[k] * std::arg(d[k]));
            else if (rule == 2 || rule == 3)
            {
                const double m = std::abs(d[k]);
                if (m * m > 1e-20)
                {
                    const std::complex<double> u = d[k] / m;
                    const std::complex<double> c = rule == 2 ? std::complex<double>(1.0 - gamma[k] + gamma[k] * u.real(),
                                                                                    gamma[k] * u.imag())
                                                             : std::complex<double>(1.0, gamma[k] * u.imag());
                    next = c * w[k];
                    next /= std::abs(next);
                }
            }
            if (rule != 0)
                w[k] = std::norm(rr[k]) >= threshold ? next : naturalW[k];
            z[k] *= w[k];
            z[k] /= std::abs(z[k]);
        }
    }

    double resonantFrequency(std::size_t k) const { return -std::arg(w[k]) * sampleRate / (2.0 * kPi); }
    double magnitude(std::size_t k) const { return std::abs(rr[k]); }
};

}

#endif
