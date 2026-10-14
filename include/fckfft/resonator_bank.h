#ifndef FCKFFT_RESONATOR_BANK_H
#define FCKFFT_RESONATOR_BANK_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "dynamics.h"
#include "simd.h"

namespace fckfft {
inline namespace FCKFFT_ABI {

namespace detail {

constexpr double kTwoPi = 6.283185307179586476925286766559;

inline float mulAdd(float a, float b, float c)
{
#if defined(__FP_FAST_FMAF)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

inline double mulAdd(double a, double b, double c)
{
#if defined(__FP_FAST_FMA)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
}

inline bool inUnitInterval(float x)
{
    return finite(x) && x > 0.0f && x <= 1.0f;
}

inline bool validFrequency(float f, float sampleRate)
{
    return finite(f) && f >= 0.0f && f <= 0.5f * sampleRate;
}

inline bool acceptableDecay(float requested, double effective)
{
    return effective > 0.0 && std::fabs(effective - static_cast<double>(requested)) <= 0.25 * static_cast<double>(requested);
}

inline std::uint64_t phaseIncrement(float frequency, float sampleRate)
{
    std::uint32_t fb, rb;
    std::memcpy(&fb, &frequency, sizeof fb);
    std::memcpy(&rb, &sampleRate, sizeof rb);
    const int fe = static_cast<int>((fb >> 23) & 0xFFu), re = static_cast<int>((rb >> 23) & 0xFFu);
    if (fe == 0 || re == 0)
        return 0;
    const std::uint32_t num = (fb & 0x7FFFFFu) | 0x800000u, den = (rb & 0x7FFFFFu) | 0x800000u;
    const int shift = fe - re + 64;
    if (shift < 0)
        return shift == -1 && num >= den ? 1u : 0u;
    std::uint64_t q = num / den;
    std::uint32_t r = num % den;
    for (int remaining = shift; remaining > 0;)
    {
        const int step = remaining < 8 ? remaining : 8;
        const std::uint32_t wide = r << step;
        q = (q << step) | (wide / den);
        r = wide % den;
        remaining -= step;
    }
    return q + (2 * static_cast<std::uint64_t>(r) >= den ? 1u : 0u);
}

inline void turnSinCos(std::uint64_t turns, float& s, float& c)
{
    const std::uint32_t u = static_cast<std::uint32_t>(turns >> 32);
    const std::uint32_t quadrant = (u + 0x20000000u) >> 30;
    const float x = static_cast<float>(static_cast<std::int32_t>(u - (quadrant << 30))) * 1.46291807926715968e-9f;
    const float x2 = x * x;
    const float sn =
        x * (1.0f + x2 * (-1.66666667e-1f + x2 * (8.33333333e-3f + x2 * (-1.98412698e-4f + x2 * 2.75573192e-6f))));
    const float cs =
        1.0f
        + x2 * (-0.5f + x2 * (4.16666667e-2f + x2 * (-1.38888889e-3f + x2 * (2.48015873e-5f + x2 * -2.75573192e-7f))));
    const float s0 = (quadrant & 1u) != 0 ? cs : sn;
    const float c0 = (quadrant & 1u) != 0 ? sn : cs;
    s = (quadrant & 2u) != 0 ? -s0 : s0;
    c = ((quadrant + 1u) & 2u) != 0 ? -c0 : c0;
}

inline void turnSinCos(std::uint64_t turns, double& s, double& c)
{
    const std::uint64_t quadrant = (turns + 0x2000000000000000ull) >> 62;
    const double x = static_cast<double>(static_cast<std::int64_t>(turns - (quadrant << 62))) * 3.4061215800865545e-19;
    const double x2 = x * x;
    const double sn =
        x
        * (1.0
           + x2
                 * (-1.0 / 6.0
                    + x2
                          * (1.0 / 120.0
                             + x2
                                   * (-1.0 / 5040.0
                                      + x2
                                            * (1.0 / 362880.0
                                               + x2
                                                     * (-1.0 / 39916800.0
                                                        + x2 * (1.0 / 6227020800.0 + x2 * (-1.0 / 1307674368000.0))))))));
    const double cs =
        1.0
        + x2
              * (-0.5
                 + x2
                       * (1.0 / 24.0
                          + x2
                                * (-1.0 / 720.0
                                   + x2
                                         * (1.0 / 40320.0
                                            + x2
                                                  * (-1.0 / 3628800.0
                                                     + x2 * (1.0 / 479001600.0 + x2 * (-1.0 / 87178291200.0
                                                                                       + x2 / 20922789888000.0)))))));
    const double s0 = (quadrant & 1u) != 0 ? cs : sn;
    const double c0 = (quadrant & 1u) != 0 ? sn : cs;
    s = (quadrant & 2u) != 0 ? -s0 : s0;
    c = ((quadrant + 1u) & 2u) != 0 ? -c0 : c0;
}

inline double poleDecay(float re, float im)
{
    const double x = re, y = im;
    return 1.0 - std::sqrt(mulAdd(x, x, y * y));
}

struct Pole
{
    float re;
    float im;
    double decay;
};

inline Pole pole(double s, double c, float decay)
{
    const double r = 1.0 - static_cast<double>(decay);
    const float re = static_cast<float>(r * c), im = static_cast<float>(r * s);
    return Pole{re, im, poleDecay(re, im)};
}

inline bool acceptable(const Pole& p, float decay)
{
    return inUnitInterval(decay) && acceptableDecay(decay, p.decay);
}

inline bool acceptablePoles(std::uint64_t increment, float alpha, float beta)
{
    double s, c;
    turnSinCos(increment, s, c);
    return acceptable(pole(s, c, alpha), alpha) && acceptable(pole(s, c, beta), beta);
}

inline void phasorStep(float frequency, float sampleRate, float& c, float& s)
{
    double sn, cs;
    turnSinCos(phaseIncrement(frequency, sampleRate), sn, cs);
    c = static_cast<float>(cs);
    s = static_cast<float>(-sn);
}

}

class ResonatorBank
{
public:
    using Kernel = detail::BankLayout;
    static constexpr std::size_t kMaxLanes = detail::kMaxBankLanes;
    static constexpr std::size_t kRecoveryInterval = 128;

    static constexpr std::size_t storageSize(std::size_t numResonators) noexcept
    {
        return ((numResonators + kMaxLanes - 1) / kMaxLanes) * Kernel::kFields * kMaxLanes + 8 * numResonators
               + detail::kAlignmentFloats - 1;
    }

    ResonatorBank() = default;
    ResonatorBank(const ResonatorBank&) = delete;
    ResonatorBank& operator=(const ResonatorBank&) = delete;
    ResonatorBank(ResonatorBank&& other) noexcept { take(other); }

    ResonatorBank& operator=(ResonatorBank&& other) noexcept
    {
        if (this != &other)
            take(other);
        return *this;
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
              const float* betas, std::size_t count, float sampleRate) noexcept
    {
        return initImpl(storage, storageCount, frequencies, alphas, betas, count, sampleRate);
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
              std::size_t count, float sampleRate) noexcept
    {
        return initImpl(storage, storageCount, frequencies, alphas, alphas, count, sampleRate);
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, std::size_t count,
              float sampleRate) noexcept
    {
        return initImpl(storage, storageCount, frequencies, nullptr, nullptr, count, sampleRate);
    }

    void reset() noexcept
    {
        for (std::size_t k = 0; k < paddedSize(); ++k)
        {
            field(k, Kernel::kVr) = 0.0f;
            field(k, Kernel::kVi) = 0.0f;
            field(k, Kernel::kSr) = 0.0f;
            field(k, Kernel::kSi) = 0.0f;
        }
        for (std::size_t k = 0; k < size_; ++k)
            setOffset(k, 0);
        processed_ = 0;
        sinceCheck_ = 0;
    }

    void process(const float* input, std::size_t numSamples, std::size_t stride = 1) noexcept
    {
        if (numGroups_ == 0 || numSamples == 0)
            return;
        const simd::ScopedFlushDenormals ftz;
        processed_ += numSamples;
        while (numSamples > 0)
        {
            const std::size_t due = kRecoveryInterval - sinceCheck_;
            const std::size_t n = numSamples < due ? numSamples : due;
            const bool check = n == due;
            backend_->process(groups_, numGroups_, input, n, stride, check);
            sinceCheck_ = check ? 0 : sinceCheck_ + n;
            input += n * stride;
            numSamples -= n;
        }
    }

    void process(float sample) noexcept { process(&sample, 1, 1); }

    bool isInitialized() const noexcept { return numGroups_ != 0; }
    std::size_t size() const noexcept { return size_; }
    std::size_t lanes() const noexcept { return lanes_; }
    Isa isa() const noexcept { return backend_->isa; }
    float sampleRate() const noexcept { return sampleRate_; }
    unsigned long long samplesProcessed() const noexcept { return processed_; }
    float frequency(std::size_t k) const noexcept { return info_[k]; }
    float alpha(std::size_t k) const noexcept { return info_[size_ + k]; }
    float beta(std::size_t k) const noexcept { return info_[2 * size_ + k]; }
    float timeConstant(std::size_t k) const noexcept { return timeConstantFromAlpha(alpha(k), sampleRate_); }

    FCKFFT_NOINLINE static bool validParameters(const float* frequencies, const float* alphas, const float* betas, std::size_t count,
                                float sampleRate) noexcept
    {
        if (frequencies == nullptr || count == 0 || !detail::finite(sampleRate) || !(sampleRate > 0.0f))
            return false;
        for (std::size_t k = 0; k < count; ++k)
        {
            if (!detail::validFrequency(frequencies[k], sampleRate))
                return false;
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            if (!detail::acceptablePoles(detail::phaseIncrement(frequencies[k], sampleRate), a, b))
                return false;
        }
        return true;
    }

    bool setAlpha(std::size_t k, float a) noexcept
    {
        if (k >= size_)
            return false;
        double s, c;
        detail::turnSinCos(increment(k), s, c);
        const detail::Pole p = detail::pole(s, c, a);
        if (!detail::acceptable(p, a))
            return false;
        const float ratio = static_cast<float>(effectiveAlpha(k) / p.decay);
        info_[size_ + k] = a;
        field(k, Kernel::kCr) = p.re;
        field(k, Kernel::kCi) = p.im;
        info_[3 * size_ + k] = static_cast<float>(p.decay * effectiveBeta(k));
        field(k, Kernel::kVr) *= ratio;
        field(k, Kernel::kVi) *= ratio;
        field(k, Kernel::kSr) *= ratio;
        field(k, Kernel::kSi) *= ratio;
        return true;
    }

    bool setBeta(std::size_t k, float b) noexcept
    {
        if (k >= size_)
            return false;
        double s, c;
        detail::turnSinCos(increment(k), s, c);
        const detail::Pole p = detail::pole(s, c, b);
        if (!detail::acceptable(p, b))
            return false;
        const float ratio = static_cast<float>(effectiveBeta(k) / p.decay);
        info_[2 * size_ + k] = b;
        field(k, Kernel::kDr) = p.re;
        field(k, Kernel::kDi) = p.im;
        info_[3 * size_ + k] = static_cast<float>(effectiveAlpha(k) * p.decay);
        field(k, Kernel::kSr) *= ratio;
        field(k, Kernel::kSi) *= ratio;
        return true;
    }

    bool setFrequency(std::size_t k, float f) noexcept
    {
        if (k >= size_ || !detail::validFrequency(f, sampleRate_))
            return false;
        const std::uint64_t inc = detail::phaseIncrement(f, sampleRate_);
        double s, c;
        detail::turnSinCos(inc, s, c);
        const detail::Pole pa = detail::pole(s, c, alpha(k)), pb = detail::pole(s, c, beta(k));
        if (!detail::acceptable(pa, alpha(k)) || !detail::acceptable(pb, beta(k)))
            return false;
        const float v = static_cast<float>(effectiveAlpha(k) / pa.decay);
        const float w = static_cast<float>(info_[3 * size_ + k] / (pa.decay * pb.decay));
        setOffset(k, offset(k) + (increment(k) - inc) * (processed_ - 1));
        info_[k] = f;
        writeCoefficients(k, inc, pa, pb);
        field(k, Kernel::kVr) *= v;
        field(k, Kernel::kVi) *= v;
        field(k, Kernel::kSr) *= w;
        field(k, Kernel::kSi) *= w;
        return true;
    }

    Complex value(std::size_t k) const noexcept { return readValue(k, field(k, Kernel::kSr), field(k, Kernel::kSi)); }

    float power(std::size_t k) const noexcept { return readPower(k, field(k, Kernel::kSr), field(k, Kernel::kSi)); }

    float magnitude(std::size_t k) const noexcept
    {
        return static_cast<float>(std::sqrt(static_cast<double>(power(k))));
    }

    float phase(std::size_t k) const noexcept
    {
        const Complex v = value(k);
        return std::atan2(v.im, v.re);
    }

    void values(Complex* out) const noexcept
    {
        forEach([&](std::size_t k, const float* g, std::size_t l) {
            out[k] = readValue(k, g[Kernel::kSr * lanes_ + l], g[Kernel::kSi * lanes_ + l]);
        });
    }

    void powers(float* out) const noexcept
    {
        forEach([&](std::size_t k, const float* g, std::size_t l) {
            out[k] = readPower(k, g[Kernel::kSr * lanes_ + l], g[Kernel::kSi * lanes_ + l]);
        });
    }

    void magnitudes(float* out) const noexcept
    {
        forEach([&](std::size_t k, const float* g, std::size_t l) {
            out[k] = static_cast<float>(
                std::sqrt(static_cast<double>(readPower(k, g[Kernel::kSr * lanes_ + l], g[Kernel::kSi * lanes_ + l]))));
        });
    }

    void phases(float* out) const noexcept
    {
        forEach([&](std::size_t k, const float* g, std::size_t l) {
            const Complex v = readValue(k, g[Kernel::kSr * lanes_ + l], g[Kernel::kSi * lanes_ + l]);
            out[k] = std::atan2(v.im, v.re);
        });
    }

private:
    void take(ResonatorBank& other) noexcept
    {
        backend_ = other.backend_;
        lanes_ = other.lanes_;
        groups_ = other.groups_;
        info_ = other.info_;
        size_ = other.size_;
        numGroups_ = other.numGroups_;
        processed_ = other.processed_;
        sinceCheck_ = other.sinceCheck_;
        sampleRate_ = other.sampleRate_;
        other.groups_ = nullptr;
        other.info_ = nullptr;
        other.size_ = 0;
        other.numGroups_ = 0;
        other.processed_ = 0;
        other.sinceCheck_ = 0;
    }

    std::size_t paddedSize() const noexcept { return numGroups_ * lanes_; }

    template <class F>
    void forEach(F&& f) const noexcept
    {
        std::size_t k = 0;
        for (std::size_t g = 0; g < numGroups_; ++g)
        {
            const float* base = groups_ + g * Kernel::kFields * lanes_;
            for (std::size_t l = 0; l < lanes_ && k < size_; ++l, ++k)
                f(k, base, l);
        }
    }

    std::uint64_t increment(std::size_t k) const noexcept
    {
        std::uint64_t value;
        std::memcpy(&value, info_ + 4 * size_ + 2 * k, sizeof value);
        return value;
    }

    std::uint64_t offset(std::size_t k) const noexcept
    {
        std::uint64_t value;
        std::memcpy(&value, info_ + 6 * size_ + 2 * k, sizeof value);
        return value;
    }

    void setOffset(std::size_t k, std::uint64_t value) noexcept
    {
        std::memcpy(info_ + 6 * size_ + 2 * k, &value, sizeof value);
    }

    Complex readValue(std::size_t k, float sr, float si) const noexcept
    {
        float s, c;
        detail::turnSinCos(increment(k) * (processed_ - 1) + offset(k), s, c);
        const float scale = info_[3 * size_ + k];
        const float re = scale * sr, im = scale * si;
        return Complex{detail::mulAdd(re, c, im * s), detail::mulAdd(im, c, -(re * s))};
    }

    float readPower(std::size_t k, float sr, float si) const noexcept
    {
        const double scale = info_[3 * size_ + k];
        const double re = scale * sr, im = scale * si;
        return static_cast<float>(detail::mulAdd(re, re, im * im));
    }

    float& field(std::size_t k, int f) noexcept
    {
        return groups_[((k / lanes_) * Kernel::kFields + static_cast<std::size_t>(f)) * lanes_ + k % lanes_];
    }

    const float& field(std::size_t k, int f) const noexcept
    {
        return groups_[((k / lanes_) * Kernel::kFields + static_cast<std::size_t>(f)) * lanes_ + k % lanes_];
    }

    double effectiveAlpha(std::size_t k) const noexcept
    {
        return detail::poleDecay(field(k, Kernel::kCr), field(k, Kernel::kCi));
    }

    double effectiveBeta(std::size_t k) const noexcept
    {
        return detail::poleDecay(field(k, Kernel::kDr), field(k, Kernel::kDi));
    }

    void writeCoefficients(std::size_t k, std::uint64_t inc, const detail::Pole& pa, const detail::Pole& pb) noexcept
    {
        std::memcpy(info_ + 4 * size_ + 2 * k, &inc, sizeof inc);
        field(k, Kernel::kCr) = pa.re;
        field(k, Kernel::kCi) = pa.im;
        field(k, Kernel::kDr) = pb.re;
        field(k, Kernel::kDi) = pb.im;
        info_[3 * size_ + k] = static_cast<float>(pa.decay * pb.decay);
    }

protected:
    FCKFFT_NOINLINE bool initImpl(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
                  const float* betas, std::size_t count, float sampleRate, bool validated = false) noexcept
    {
        if (storage == nullptr || count == 0 || storageCount < storageSize(count)
            || (!validated && !validParameters(frequencies, alphas, betas, count, sampleRate)))
            return false;
        const detail::BankBackend& backend = detail::selectBankBackend();
        const std::size_t numGroups = (count + backend.lanes - 1) / backend.lanes;
        backend_ = &backend;
        lanes_ = backend.lanes;
        groups_ = detail::alignStorage(storage);
        info_ = groups_ + numGroups * Kernel::kFields * lanes_;
        size_ = count;
        numGroups_ = numGroups;
        sampleRate_ = sampleRate;
        for (std::size_t k = 0; k < paddedSize(); ++k)
            for (int f = 0; f < Kernel::kFields; ++f)
                field(k, f) = 0.0f;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            const std::uint64_t inc = detail::phaseIncrement(frequencies[k], sampleRate);
            double s, c;
            detail::turnSinCos(inc, s, c);
            info_[k] = frequencies[k];
            info_[count + k] = a;
            info_[2 * count + k] = b;
            const detail::Pole pa = detail::pole(s, c, a);
            writeCoefficients(k, inc, pa, detail::pole(s, c, b));
        }
        reset();
        return true;
    }

private:
    const detail::BankBackend* backend_ = &detail::baseline::kBankBackend;
    std::size_t lanes_ = detail::baseline::kBankBackend.lanes;
    float* groups_ = nullptr;
    float* info_ = nullptr;
    std::size_t size_ = 0;
    std::size_t numGroups_ = 0;
    unsigned long long processed_ = 0;
    std::size_t sinceCheck_ = 0;
    float sampleRate_ = 0.0f;
};

namespace detail {

template <std::size_t Floats>
class InlineStorage
{
public:
    InlineStorage() = default;
    InlineStorage(InlineStorage&&) = delete;
    InlineStorage& operator=(InlineStorage&&) = delete;

    static bool fits(std::size_t floats) noexcept { return floats <= Floats; }
    float* acquire(std::size_t floats) noexcept { return fits(floats) ? data_ : nullptr; }

private:
    float data_[Floats];
};

class VectorStorage
{
public:
    static bool fits(std::size_t) noexcept { return true; }
    float* acquire(std::size_t floats)
    {
        data_.assign(floats, 0.0f);
        return data_.data();
    }

private:
    std::vector<float> data_;
};

}

template <class Storage>
class OwningResonatorBank : private ResonatorBank
{
public:
    using ResonatorBank::kMaxLanes;
    using ResonatorBank::kRecoveryInterval;
    using ResonatorBank::storageSize;
    using ResonatorBank::validParameters;
    using ResonatorBank::reset;
    using ResonatorBank::process;
    using ResonatorBank::isInitialized;
    using ResonatorBank::size;
    using ResonatorBank::lanes;
    using ResonatorBank::isa;
    using ResonatorBank::sampleRate;
    using ResonatorBank::samplesProcessed;
    using ResonatorBank::frequency;
    using ResonatorBank::alpha;
    using ResonatorBank::beta;
    using ResonatorBank::timeConstant;
    using ResonatorBank::setAlpha;
    using ResonatorBank::setBeta;
    using ResonatorBank::setFrequency;
    using ResonatorBank::value;
    using ResonatorBank::power;
    using ResonatorBank::magnitude;
    using ResonatorBank::phase;
    using ResonatorBank::values;
    using ResonatorBank::powers;
    using ResonatorBank::magnitudes;
    using ResonatorBank::phases;

    bool init(const float* frequencies, const float* alphas, const float* betas, std::size_t count,
              float sampleRate) noexcept(noexcept(std::declval<Storage&>().acquire(0)))
    {
        if (!Storage::fits(storageSize(count)) || !validParameters(frequencies, alphas, betas, count, sampleRate))
            return false;
        float* storage = storage_.acquire(storageSize(count));
        return storage != nullptr
               && ResonatorBank::initImpl(storage, storageSize(count), frequencies, alphas, betas, count, sampleRate,
                                          true);
    }

    bool init(const float* frequencies, const float* alphas, std::size_t count,
              float sampleRate) noexcept(noexcept(std::declval<Storage&>().acquire(0)))
    {
        return init(frequencies, alphas, alphas, count, sampleRate);
    }

    bool init(const float* frequencies, std::size_t count,
              float sampleRate) noexcept(noexcept(std::declval<Storage&>().acquire(0)))
    {
        return init(frequencies, nullptr, nullptr, count, sampleRate);
    }

private:
    Storage storage_;
};

template <std::size_t MaxResonators>
class StaticResonatorBank : public OwningResonatorBank<detail::InlineStorage<ResonatorBank::storageSize(MaxResonators)>>
{
public:
    static constexpr std::size_t kCapacity = MaxResonators;
};

class DynamicResonatorBank : public OwningResonatorBank<detail::VectorStorage>
{
};

template <class Bank>
std::size_t analyze(Bank& bank, const float* signal, std::size_t numSamples, std::size_t hop, Complex* frames) noexcept
{
    if (hop == 0 || !bank.isInitialized())
        return 0;
    const std::size_t numFrames = numSamples / hop;
    for (std::size_t f = 0; f < numFrames; ++f)
    {
        bank.process(signal + f * hop, hop);
        bank.values(frames + f * bank.size());
    }
    bank.process(signal + numFrames * hop, numSamples - numFrames * hop);
    return numFrames;
}

}
}

#endif
