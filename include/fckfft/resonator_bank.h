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

struct Complex
{
    float re;
    float im;
};

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

template <class Ops>
struct BankKernel
{
    using V = typename Ops::V;
    enum Field
    {
        kCr,
        kCi,
        kDr,
        kDi,
        kVr,
        kVi,
        kSr,
        kSi,
        kFields
    };
    static constexpr std::size_t kWidth = Ops::width;
    static constexpr std::size_t kPacks = 3;
    static constexpr std::size_t kLanes = kPacks * kWidth;
    static constexpr std::size_t kGroupFloats = kFields * kLanes;

    struct Pack
    {
        V cr[kPacks], ci[kPacks], dr[kPacks], di[kPacks], vr[kPacks], vi[kPacks], sr[kPacks], si[kPacks];
    };

    static FCKFFT_INLINE void loadOne(Pack& p, std::size_t j, const float* g)
    {
        p.cr[j] = Ops::load(g + kCr * kLanes);
        p.ci[j] = Ops::load(g + kCi * kLanes);
        p.dr[j] = Ops::load(g + kDr * kLanes);
        p.di[j] = Ops::load(g + kDi * kLanes);
        p.vr[j] = Ops::load(g + kVr * kLanes);
        p.vi[j] = Ops::load(g + kVi * kLanes);
        p.sr[j] = Ops::load(g + kSr * kLanes);
        p.si[j] = Ops::load(g + kSi * kLanes);
    }

    static FCKFFT_INLINE void storeOne(const Pack& p, std::size_t j, float* g)
    {
        Ops::store(g + kVr * kLanes, p.vr[j]);
        Ops::store(g + kVi * kLanes, p.vi[j]);
        Ops::store(g + kSr * kLanes, p.sr[j]);
        Ops::store(g + kSi * kLanes, p.si[j]);
    }

    static FCKFFT_INLINE void load(Pack& p, const float* g)
    {
        loadOne(p, 0, g);
        loadOne(p, 1, g + kWidth);
        loadOne(p, 2, g + 2 * kWidth);
    }

    static FCKFFT_INLINE void store(const Pack& p, float* g)
    {
        storeOne(p, 0, g);
        storeOne(p, 1, g + kWidth);
        storeOne(p, 2, g + 2 * kWidth);
    }

    static FCKFFT_INLINE void stepOne(Pack& p, std::size_t j, V x, std::false_type)
    {
        const V t = Ops::nmadd(p.ci[j], p.vi[j], x);
        p.vi[j] = Ops::madd(p.cr[j], p.vi[j], Ops::mul(p.ci[j], p.vr[j]));
        p.vr[j] = Ops::madd(p.cr[j], p.vr[j], t);
        const V u = Ops::nmadd(p.di[j], p.si[j], p.vr[j]);
        p.si[j] = Ops::madd(p.dr[j], p.si[j], Ops::madd(p.di[j], p.sr[j], p.vi[j]));
        p.sr[j] = Ops::madd(p.dr[j], p.sr[j], u);
    }

    static FCKFFT_INLINE void stepOne(Pack& p, std::size_t j, V x, std::true_type)
    {
        const V vi = Ops::madd(p.cr[j], p.vi[j], Ops::mul(p.ci[j], p.vr[j]));
        const V vr = Ops::sub(Ops::mul(p.cr[j], p.vr[j]), Ops::mul(p.ci[j], p.vi[j]));
        p.vr[j] = Ops::add(vr, x);
        p.vi[j] = vi;
        const V si = Ops::madd(p.dr[j], p.si[j], Ops::mul(p.di[j], p.sr[j]));
        const V sr = Ops::sub(Ops::mul(p.dr[j], p.sr[j]), Ops::mul(p.di[j], p.si[j]));
        p.sr[j] = Ops::add(sr, p.vr[j]);
        p.si[j] = Ops::add(si, p.vi[j]);
    }

    static FCKFFT_INLINE void step(Pack& p, V x)
    {
        const std::integral_constant<bool, kWidth == 1> scalar{};
        stepOne(p, 0, x, scalar);
        stepOne(p, 1, x, scalar);
        stepOne(p, 2, x, scalar);
    }

    static FCKFFT_INLINE void sanitizeOne(Pack& p, std::size_t j)
    {
        const V zero = Ops::set(0.0f);
        V t = Ops::mul(p.vr[j], zero);
        t = Ops::madd(p.vi[j], zero, t);
        t = Ops::madd(p.sr[j], zero, t);
        t = Ops::madd(p.si[j], zero, t);
        p.vr[j] = Ops::selectGe(t, zero, p.vr[j], zero);
        p.vi[j] = Ops::selectGe(t, zero, p.vi[j], zero);
        p.sr[j] = Ops::selectGe(t, zero, p.sr[j], zero);
        p.si[j] = Ops::selectGe(t, zero, p.si[j], zero);
    }

    static FCKFFT_INLINE void sanitize(Pack& p)
    {
        sanitizeOne(p, 0);
        sanitizeOne(p, 1);
        sanitizeOne(p, 2);
    }

    static void process(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool sanitizeAtEnd)
    {
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            float* base = groups + g * kGroupFloats;
            Pack a;
            load(a, base);
            const float* in = x;
            std::size_t i = 0;
            for (; i + 2 <= n; i += 2)
            {
                step(a, Ops::set(in[0]));
                step(a, Ops::set(in[stride]));
                in += 2 * stride;
            }
            if (i < n)
                step(a, Ops::set(in[0]));
            if (sanitizeAtEnd)
                sanitize(a);
            store(a, base);
        }
    }
};

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
    using Kernel = detail::BankKernel<simd::NativeOps>;
    static constexpr std::size_t kLanes = Kernel::kLanes;
    static constexpr std::size_t kRecoveryInterval = 128;

    static constexpr std::size_t storageSize(std::size_t numResonators) noexcept
    {
        return ((numResonators + kLanes - 1) / kLanes) * Kernel::kGroupFloats + 4 * numResonators;
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
            Kernel::process(groups_, numGroups_, input, n, stride, check);
            sinceCheck_ = check ? 0 : sinceCheck_ + n;
            input += n * stride;
            numSamples -= n;
        }
    }

    void process(float sample) noexcept { process(&sample, 1, 1); }

    bool isInitialized() const noexcept { return numGroups_ != 0; }
    std::size_t size() const noexcept { return size_; }
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

    std::size_t paddedSize() const noexcept { return numGroups_ * kLanes; }

    float& field(std::size_t k, int f) noexcept
    {
        return groups_[(k / kLanes) * Kernel::kGroupFloats + static_cast<std::size_t>(f) * kLanes + k % kLanes];
    }

    const float& field(std::size_t k, int f) const noexcept
    {
        return groups_[(k / kLanes) * Kernel::kGroupFloats + static_cast<std::size_t>(f) * kLanes + k % kLanes];
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
        const std::size_t numGroups = (count + kLanes - 1) / kLanes;
        groups_ = storage;
        info_ = storage + numGroups * Kernel::kGroupFloats;
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
    using ResonatorBank::kLanes;
    using ResonatorBank::kRecoveryInterval;
    using ResonatorBank::storageSize;
    using ResonatorBank::validParameters;
    using ResonatorBank::reset;
    using ResonatorBank::process;
    using ResonatorBank::isInitialized;
    using ResonatorBank::size;
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

#endif
