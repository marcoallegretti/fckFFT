#ifndef FCKFFT_RESONATOR_BANK_H
#define FCKFFT_RESONATOR_BANK_H

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "dynamics.h"
#include "simd.h"

namespace fckfft {
inline namespace FCKFFT_ABI {

namespace detail {

constexpr double kTwoPi = 6.283185307179586476925286766559;

inline bool inUnitInterval(float x)
{
    return x > 0.0f && x <= 1.0f;
}

inline bool validFrequency(float f, float sampleRate)
{
    return f >= 0.0f && f <= 0.5f * sampleRate;
}

inline bool acceptableDecay(float requested, double effective)
{
    return effective > 0.0 && std::fabs(effective - static_cast<double>(requested)) <= 0.25 * static_cast<double>(requested);
}

inline void poleCoefficients(float frequency, float sampleRate, float decay, float& re, float& im)
{
    const double omega = kTwoPi * static_cast<double>(frequency) / static_cast<double>(sampleRate);
    const double r = 1.0 - static_cast<double>(decay);
    re = static_cast<float>(r * std::cos(omega));
    im = static_cast<float>(r * std::sin(omega));
}

inline double poleDecay(float re, float im)
{
    return 1.0 - std::hypot(static_cast<double>(re), static_cast<double>(im));
}

inline bool acceptablePole(float frequency, float sampleRate, float decay)
{
    float re, im;
    poleCoefficients(frequency, sampleRate, decay, re, im);
    return inUnitInterval(decay) && acceptableDecay(decay, poleDecay(re, im));
}

inline void phasorStep(float frequency, float sampleRate, float& c, float& s)
{
    const double omega = -kTwoPi * static_cast<double>(frequency) / static_cast<double>(sampleRate);
    c = static_cast<float>(std::cos(omega));
    s = static_cast<float>(std::sin(omega));
}

inline void rotation(double frequency, double sampleRate, unsigned long long samples, double& c, double& s)
{
    if (samples == 0)
    {
        c = 1.0;
        s = 0.0;
        return;
    }
    const double turns = std::fmod(frequency * static_cast<double>(samples - 1), sampleRate) / sampleRate;
    const float theta = static_cast<float>(-kTwoPi * (turns > 0.5 ? turns - 1.0 : turns));
    c = std::cos(theta);
    s = std::sin(theta);
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
        return ((numResonators + kMaxLanes - 1) / kMaxLanes) * Kernel::kFields * kMaxLanes + 4 * numResonators;
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
            backend_->bank(groups_, numGroups_, input, n, stride, check);
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

    static bool validParameters(const float* frequencies, const float* alphas, const float* betas, std::size_t count,
                                float sampleRate) noexcept
    {
        if (frequencies == nullptr || count == 0 || !(sampleRate > 0.0f) || !std::isfinite(sampleRate))
            return false;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            if (!detail::validFrequency(frequencies[k], sampleRate) || !detail::acceptablePole(frequencies[k], sampleRate, a)
                || !detail::acceptablePole(frequencies[k], sampleRate, b))
                return false;
        }
        return true;
    }

    bool setAlpha(std::size_t k, float a) noexcept
    {
        if (k >= size_ || !detail::acceptablePole(frequency(k), sampleRate_, a))
            return false;
        const double before = effectiveAlpha(k);
        info_[size_ + k] = a;
        writeCoefficients(k);
        const float ratio = static_cast<float>(before / effectiveAlpha(k));
        field(k, Kernel::kVr) *= ratio;
        field(k, Kernel::kVi) *= ratio;
        field(k, Kernel::kSr) *= ratio;
        field(k, Kernel::kSi) *= ratio;
        return true;
    }

    bool setBeta(std::size_t k, float b) noexcept
    {
        if (k >= size_ || !detail::acceptablePole(frequency(k), sampleRate_, b))
            return false;
        const double before = effectiveBeta(k);
        info_[2 * size_ + k] = b;
        writeCoefficients(k);
        const float ratio = static_cast<float>(before / effectiveBeta(k));
        field(k, Kernel::kSr) *= ratio;
        field(k, Kernel::kSi) *= ratio;
        return true;
    }

    bool setFrequency(std::size_t k, float f) noexcept
    {
        if (k >= size_ || !detail::validFrequency(f, sampleRate_) || !detail::acceptablePole(f, sampleRate_, alpha(k))
            || !detail::acceptablePole(f, sampleRate_, beta(k)))
            return false;
        info_[k] = f;
        writeCoefficients(k);
        return true;
    }

    Complex value(std::size_t k) const noexcept
    {
        double c, s;
        detail::rotation(info_[k], sampleRate_, processed_, c, s);
        const double scale = info_[3 * size_ + k];
        const double re = scale * field(k, Kernel::kSr), im = scale * field(k, Kernel::kSi);
        return Complex{static_cast<float>(re * c - im * s), static_cast<float>(re * s + im * c)};
    }

    float power(std::size_t k) const noexcept
    {
        const double scale = info_[3 * size_ + k];
        const double re = scale * field(k, Kernel::kSr), im = scale * field(k, Kernel::kSi);
        return static_cast<float>(re * re + im * im);
    }

    float magnitude(std::size_t k) const noexcept
    {
        const double scale = info_[3 * size_ + k];
        return static_cast<float>(scale * std::hypot(static_cast<double>(field(k, Kernel::kSr)),
                                                     static_cast<double>(field(k, Kernel::kSi))));
    }

    float phase(std::size_t k) const noexcept
    {
        const Complex v = value(k);
        return std::atan2(v.im, v.re);
    }

    void values(Complex* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = value(k);
    }

    void powers(float* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = power(k);
    }

    void magnitudes(float* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = magnitude(k);
    }

    void phases(float* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = phase(k);
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

    void writeCoefficients(std::size_t k) noexcept
    {
        detail::poleCoefficients(frequency(k), sampleRate_, alpha(k), field(k, Kernel::kCr), field(k, Kernel::kCi));
        detail::poleCoefficients(frequency(k), sampleRate_, beta(k), field(k, Kernel::kDr), field(k, Kernel::kDi));
        info_[3 * size_ + k] = static_cast<float>(effectiveAlpha(k) * effectiveBeta(k));
    }

    bool initImpl(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
                  const float* betas, std::size_t count, float sampleRate) noexcept
    {
        if (storage == nullptr || count == 0 || storageCount < storageSize(count)
            || !validParameters(frequencies, alphas, betas, count, sampleRate))
            return false;
        const detail::Backend& backend = detail::selectBackend();
        const std::size_t numGroups = (count + backend.bankLanes - 1) / backend.bankLanes;
        backend_ = &backend;
        lanes_ = backend.bankLanes;
        groups_ = storage;
        info_ = storage + numGroups * Kernel::kFields * lanes_;
        size_ = count;
        numGroups_ = numGroups;
        sampleRate_ = sampleRate;
        for (std::size_t k = 0; k < paddedSize(); ++k)
            for (int f = 0; f < Kernel::kFields; ++f)
                field(k, f) = 0.0f;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            info_[k] = frequencies[k];
            info_[count + k] = a;
            info_[2 * count + k] = betas != nullptr ? betas[k] : a;
            writeCoefficients(k);
        }
        reset();
        return true;
    }

    const detail::Backend* backend_ = &detail::baseline::kBackend;
    std::size_t lanes_ = detail::baseline::kBackend.bankLanes;
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
               && ResonatorBank::init(storage, storageSize(count), frequencies, alphas, betas, count, sampleRate);
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
    if (hop == 0)
        return 0;
    const std::size_t numFrames = numSamples / hop;
    for (std::size_t f = 0; f < numFrames; ++f)
    {
        bank.process(signal + f * hop, hop);
        bank.values(frames + f * bank.size());
    }
    return numFrames;
}

}
}

#endif
