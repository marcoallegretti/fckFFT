#ifndef FCKFFT_RESONATOR_BANK_H
#define FCKFFT_RESONATOR_BANK_H

#include <cmath>
#include <cstddef>
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
        kOa,
        kOb,
        kAb,
        kWr,
        kWi,
        kZr,
        kZi,
        kUr,
        kUi,
        kSr,
        kSi,
        kFields
    };
    static constexpr std::size_t kWidth = Ops::width;
    static constexpr std::size_t kLanes = 2 * kWidth;
    static constexpr std::size_t kGroupFloats = kFields * kLanes;

    struct Pack
    {
        V oa, ob, ab, wr, wi, zr, zi, ur, ui, sr, si;
    };

    static FCKFFT_INLINE void load(Pack& p, const float* g)
    {
        p.oa = Ops::load(g + kOa * kLanes);
        p.ob = Ops::load(g + kOb * kLanes);
        p.ab = Ops::load(g + kAb * kLanes);
        p.wr = Ops::load(g + kWr * kLanes);
        p.wi = Ops::load(g + kWi * kLanes);
        p.zr = Ops::load(g + kZr * kLanes);
        p.zi = Ops::load(g + kZi * kLanes);
        p.ur = Ops::load(g + kUr * kLanes);
        p.ui = Ops::load(g + kUi * kLanes);
        p.sr = Ops::load(g + kSr * kLanes);
        p.si = Ops::load(g + kSi * kLanes);
    }

    static FCKFFT_INLINE void store(const Pack& p, float* g)
    {
        Ops::store(g + kZr * kLanes, p.zr);
        Ops::store(g + kZi * kLanes, p.zi);
        Ops::store(g + kUr * kLanes, p.ur);
        Ops::store(g + kUi * kLanes, p.ui);
        Ops::store(g + kSr * kLanes, p.sr);
        Ops::store(g + kSi * kLanes, p.si);
    }

    static FCKFFT_INLINE void step(Pack& p, V s)
    {
        p.ur = Ops::madd(p.oa, p.ur, Ops::mul(s, p.zr));
        p.ui = Ops::madd(p.oa, p.ui, Ops::mul(s, p.zi));
        const V zr = Ops::nmadd(p.zi, p.wi, Ops::mul(p.zr, p.wr));
        p.zi = Ops::madd(p.zr, p.wi, Ops::mul(p.zi, p.wr));
        p.zr = zr;
        p.sr = Ops::madd(p.ob, p.sr, Ops::mul(p.ab, p.ur));
        p.si = Ops::madd(p.ob, p.si, Ops::mul(p.ab, p.ui));
    }

    static FCKFFT_INLINE void normalize(Pack& p)
    {
        const V m = Ops::madd(p.zr, p.zr, Ops::mul(p.zi, p.zi));
        const V k = Ops::nmadd(Ops::set(0.5f), m, Ops::set(1.5f));
        p.zr = Ops::mul(p.zr, k);
        p.zi = Ops::mul(p.zi, k);
        const V zero = Ops::set(0.0f);
        V t = Ops::mul(p.zr, zero);
        t = Ops::madd(p.zi, zero, t);
        t = Ops::madd(p.ur, zero, t);
        t = Ops::madd(p.ui, zero, t);
        t = Ops::madd(p.sr, zero, t);
        t = Ops::madd(p.si, zero, t);
        p.zr = Ops::selectGe(t, zero, p.zr, Ops::set(1.0f));
        p.zi = Ops::selectGe(t, zero, p.zi, zero);
        p.ur = Ops::selectGe(t, zero, p.ur, zero);
        p.ui = Ops::selectGe(t, zero, p.ui, zero);
        p.sr = Ops::selectGe(t, zero, p.sr, zero);
        p.si = Ops::selectGe(t, zero, p.si, zero);
    }

    static void process(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool renormalize)
    {
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            float* base = groups + g * kGroupFloats;
            Pack a, b;
            load(a, base);
            load(b, base + kWidth);
            const float* in = x;
            std::size_t i = 0;
            for (; i + 2 <= n; i += 2)
            {
                const V s0 = Ops::set(in[0]);
                const V s1 = Ops::set(in[stride]);
                in += 2 * stride;
                step(a, s0);
                step(b, s0);
                step(a, s1);
                step(b, s1);
            }
            if (i < n)
            {
                const V s0 = Ops::set(in[0]);
                step(a, s0);
                step(b, s0);
            }
            if (renormalize)
            {
                normalize(a);
                normalize(b);
            }
            store(a, base);
            store(b, base + kWidth);
        }
    }
};

}

class ResonatorBank
{
public:
    using Kernel = detail::BankKernel<simd::NativeOps>;
    static constexpr std::size_t kLanes = Kernel::kLanes;
    static constexpr std::size_t kNormalizeInterval = 128;

    static constexpr std::size_t storageSize(std::size_t numResonators)
    {
        return ((numResonators + kLanes - 1) / kLanes) * Kernel::kGroupFloats + 3 * numResonators;
    }

    ResonatorBank() = default;
    ResonatorBank(const ResonatorBank&) = delete;
    ResonatorBank& operator=(const ResonatorBank&) = delete;

    bool init(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
              const float* betas, std::size_t count, float sampleRate)
    {
        return initImpl(storage, storageCount, frequencies, alphas, betas, count, sampleRate);
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
              std::size_t count, float sampleRate)
    {
        return initImpl(storage, storageCount, frequencies, alphas, alphas, count, sampleRate);
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, std::size_t count,
              float sampleRate)
    {
        return initImpl(storage, storageCount, frequencies, nullptr, nullptr, count, sampleRate);
    }

    void reset()
    {
        for (std::size_t k = 0; k < paddedSize(); ++k)
        {
            field(k, Kernel::kZr) = 1.0f;
            field(k, Kernel::kZi) = 0.0f;
            field(k, Kernel::kUr) = 0.0f;
            field(k, Kernel::kUi) = 0.0f;
            field(k, Kernel::kSr) = 0.0f;
            field(k, Kernel::kSi) = 0.0f;
        }
        sinceNormalize_ = 0;
    }

    void process(const float* input, std::size_t numSamples, std::size_t stride = 1)
    {
        if (numGroups_ == 0 || numSamples == 0)
            return;
        const simd::ScopedFlushDenormals ftz;
        while (numSamples > 0)
        {
            const std::size_t due = kNormalizeInterval - sinceNormalize_;
            const std::size_t n = numSamples < due ? numSamples : due;
            const bool renormalize = n == due;
            Kernel::process(groups_, numGroups_, input, n, stride, renormalize);
            sinceNormalize_ = renormalize ? 0 : sinceNormalize_ + n;
            input += n * stride;
            numSamples -= n;
        }
    }

    void process(float sample) { process(&sample, 1, 1); }

    bool isInitialized() const { return numGroups_ != 0; }
    std::size_t size() const { return size_; }
    float sampleRate() const { return sampleRate_; }
    float frequency(std::size_t k) const { return info_[k]; }
    float alpha(std::size_t k) const { return info_[size_ + k]; }
    float beta(std::size_t k) const { return info_[2 * size_ + k]; }
    float timeConstant(std::size_t k) const { return timeConstantFromAlpha(alpha(k), sampleRate_); }

    bool setAlpha(std::size_t k, float a)
    {
        if (k >= size_ || !detail::inUnitInterval(a))
            return false;
        const float old = alpha(k);
        field(k, Kernel::kUr) *= old / a;
        field(k, Kernel::kUi) *= old / a;
        info_[size_ + k] = a;
        writeCoefficients(k);
        return true;
    }

    bool setBeta(std::size_t k, float b)
    {
        if (k >= size_ || !detail::inUnitInterval(b))
            return false;
        info_[2 * size_ + k] = b;
        writeCoefficients(k);
        return true;
    }

    bool setFrequency(std::size_t k, float f)
    {
        if (k >= size_ || !detail::validFrequency(f, sampleRate_))
            return false;
        info_[k] = f;
        detail::phasorStep(f, sampleRate_, field(k, Kernel::kWr), field(k, Kernel::kWi));
        return true;
    }

    Complex value(std::size_t k) const { return Complex{field(k, Kernel::kSr), field(k, Kernel::kSi)}; }

    float power(std::size_t k) const
    {
        const float re = field(k, Kernel::kSr), im = field(k, Kernel::kSi);
        return re * re + im * im;
    }

    float magnitude(std::size_t k) const { return std::sqrt(power(k)); }
    float phase(std::size_t k) const { return std::atan2(field(k, Kernel::kSi), field(k, Kernel::kSr)); }

    void values(Complex* out) const
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = value(k);
    }

    void powers(float* out) const
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = power(k);
    }

    void magnitudes(float* out) const
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = magnitude(k);
    }

    void phases(float* out) const
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = phase(k);
    }

private:
    std::size_t paddedSize() const { return numGroups_ * kLanes; }

    float& field(std::size_t k, int f)
    {
        return groups_[(k / kLanes) * Kernel::kGroupFloats + static_cast<std::size_t>(f) * kLanes + k % kLanes];
    }

    const float& field(std::size_t k, int f) const
    {
        return groups_[(k / kLanes) * Kernel::kGroupFloats + static_cast<std::size_t>(f) * kLanes + k % kLanes];
    }

    void writeCoefficients(std::size_t k)
    {
        const float a = alpha(k), b = beta(k);
        field(k, Kernel::kOa) = 1.0f - a;
        field(k, Kernel::kOb) = 1.0f - b;
        field(k, Kernel::kAb) = a * b;
    }

    bool initImpl(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
                  const float* betas, std::size_t count, float sampleRate)
    {
        numGroups_ = 0;
        size_ = 0;
        if (storage == nullptr || frequencies == nullptr || count == 0 || storageCount < storageSize(count)
            || !(sampleRate > 0.0f) || !std::isfinite(sampleRate))
            return false;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            if (!detail::validFrequency(frequencies[k], sampleRate) || !detail::inUnitInterval(a)
                || !detail::inUnitInterval(b))
                return false;
        }
        const std::size_t numGroups = (count + kLanes - 1) / kLanes;
        groups_ = storage;
        info_ = storage + numGroups * Kernel::kGroupFloats;
        size_ = count;
        numGroups_ = numGroups;
        sampleRate_ = sampleRate;
        for (std::size_t k = 0; k < paddedSize(); ++k)
        {
            field(k, Kernel::kOa) = 0.0f;
            field(k, Kernel::kOb) = 1.0f;
            field(k, Kernel::kAb) = 0.0f;
            field(k, Kernel::kWr) = 1.0f;
            field(k, Kernel::kWi) = 0.0f;
        }
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            info_[k] = frequencies[k];
            info_[count + k] = a;
            info_[2 * count + k] = betas != nullptr ? betas[k] : a;
            writeCoefficients(k);
            detail::phasorStep(frequencies[k], sampleRate, field(k, Kernel::kWr), field(k, Kernel::kWi));
        }
        reset();
        return true;
    }

    float* groups_ = nullptr;
    float* info_ = nullptr;
    std::size_t size_ = 0;
    std::size_t numGroups_ = 0;
    std::size_t sinceNormalize_ = 0;
    float sampleRate_ = 0.0f;
};

template <std::size_t MaxResonators>
class StaticResonatorBank : public ResonatorBank
{
public:
    static constexpr std::size_t kCapacity = MaxResonators;

    bool init(const float* frequencies, const float* alphas, const float* betas, std::size_t count, float sampleRate)
    {
        return count <= MaxResonators
               && ResonatorBank::init(storage_, kStorage, frequencies, alphas, betas, count, sampleRate);
    }

    bool init(const float* frequencies, const float* alphas, std::size_t count, float sampleRate)
    {
        return count <= MaxResonators && ResonatorBank::init(storage_, kStorage, frequencies, alphas, count, sampleRate);
    }

    bool init(const float* frequencies, std::size_t count, float sampleRate)
    {
        return count <= MaxResonators && ResonatorBank::init(storage_, kStorage, frequencies, count, sampleRate);
    }

private:
    static constexpr std::size_t kStorage = storageSize(MaxResonators);
    float storage_[kStorage];
};

class DynamicResonatorBank : public ResonatorBank
{
public:
    bool init(const float* frequencies, const float* alphas, const float* betas, std::size_t count, float sampleRate)
    {
        storage_.assign(storageSize(count), 0.0f);
        return ResonatorBank::init(storage_.data(), storage_.size(), frequencies, alphas, betas, count, sampleRate);
    }

    bool init(const float* frequencies, const float* alphas, std::size_t count, float sampleRate)
    {
        return init(frequencies, alphas, alphas, count, sampleRate);
    }

    bool init(const float* frequencies, std::size_t count, float sampleRate)
    {
        return init(frequencies, nullptr, nullptr, count, sampleRate);
    }

private:
    std::vector<float> storage_;
};

inline std::size_t analyze(ResonatorBank& bank, const float* signal, std::size_t numSamples, std::size_t hop,
                           Complex* frames)
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
