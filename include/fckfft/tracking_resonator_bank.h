#ifndef FCKFFT_TRACKING_RESONATOR_BANK_H
#define FCKFFT_TRACKING_RESONATOR_BANK_H

#include <cmath>
#include <cstddef>
#include <vector>

#include "dynamics.h"
#include "resonator_bank.h"
#include "simd.h"

namespace fckfft {

enum class TrackingRule
{
    none,
    ewma,
    normalizedChord,
    tangent
};

namespace detail {

template <class Ops>
struct TrackingKernel
{
    using V = typename Ops::V;
    enum Field
    {
        kOa,
        kOb,
        kAb,
        kGm,
        kOg,
        kNwr,
        kNwi,
        kZr,
        kZi,
        kUr,
        kUi,
        kSr,
        kSi,
        kPr,
        kPi,
        kWr,
        kWi,
        kQr,
        kQi,
        kPw,
        kFields
    };
    static constexpr std::size_t kWidth = Ops::width;
    static constexpr std::size_t kLanes = 2 * kWidth;
    static constexpr std::size_t kGroupFloats = kFields * kLanes;

    static FCKFFT_INLINE V ld(const float* g, int f) { return Ops::load(g + static_cast<std::size_t>(f) * kLanes); }
    static FCKFFT_INLINE void st(float* g, int f, V v) { Ops::store(g + static_cast<std::size_t>(f) * kLanes, v); }

    static FCKFFT_INLINE V accumulate(float* g, V s, V zr, V zi)
    {
        const V oa = ld(g, kOa), ob = ld(g, kOb), ab = ld(g, kAb);
        const V ur = Ops::madd(oa, ld(g, kUr), Ops::mul(s, zr));
        const V ui = Ops::madd(oa, ld(g, kUi), Ops::mul(s, zi));
        const V sr = Ops::madd(ob, ld(g, kSr), Ops::mul(ab, ur));
        const V si = Ops::madd(ob, ld(g, kSi), Ops::mul(ab, ui));
        const V pw = Ops::madd(sr, sr, Ops::mul(si, si));
        st(g, kUr, ur);
        st(g, kUi, ui);
        st(g, kSr, sr);
        st(g, kSi, si);
        st(g, kPw, pw);
        return pw;
    }

    static FCKFFT_INLINE V accumulate(float* g, V s) { return accumulate(g, s, ld(g, kZr), ld(g, kZi)); }

    static FCKFFT_INLINE void renormalize(V& re, V& im)
    {
        const V m = Ops::madd(re, re, Ops::mul(im, im));
        const V k = Ops::nmadd(Ops::set(0.5f), m, Ops::set(1.5f));
        re = Ops::mul(re, k);
        im = Ops::mul(im, k);
    }

    template <TrackingRule Rule>
    static FCKFFT_INLINE void track(float* g, V threshold, bool renorm, V& zr, V& zi)
    {
        const V sr = ld(g, kSr), si = ld(g, kSi);
        const V pr = ld(g, kPr), pi = ld(g, kPi);
        const V dr = Ops::madd(sr, pr, Ops::mul(si, pi));
        const V di = Ops::nmadd(si, pr, Ops::mul(sr, pi));
        const V gm = ld(g, kGm), og = ld(g, kOg);
        st(g, kPr, sr);
        st(g, kPi, si);
        st(g, kQr, Ops::madd(og, ld(g, kQr), Ops::mul(gm, dr)));
        st(g, kQi, Ops::madd(og, ld(g, kQi), Ops::mul(gm, di)));
        V wr = ld(g, kWr), wi = ld(g, kWi);
        if (Rule != TrackingRule::none)
        {
            V cr, ci;
            if (Rule == TrackingRule::ewma)
            {
                simd::sincos<Ops>(Ops::mul(gm, simd::atan2<Ops>(di, dr)), ci, cr);
            }
            else
            {
                const V dd = Ops::madd(dr, dr, Ops::mul(di, di));
                const V valid = Ops::set(1e-20f);
                const V gi = Ops::mul(gm, Ops::rsqrt(Ops::max(dd, valid)));
                if (Rule == TrackingRule::normalizedChord)
                    cr = Ops::madd(gi, dr, og);
                else
                    cr = Ops::set(1.0f);
                ci = Ops::mul(gi, di);
                cr = Ops::selectGe(dd, valid, cr, Ops::set(1.0f));
                ci = Ops::selectGe(dd, valid, ci, Ops::set(0.0f));
            }
            V tr = Ops::nmadd(wi, ci, Ops::mul(wr, cr));
            V ti = Ops::madd(wr, ci, Ops::mul(wi, cr));
            if (Rule != TrackingRule::ewma)
                renormalize(tr, ti);
            const V pw = ld(g, kPw);
            wr = Ops::selectGe(pw, threshold, tr, ld(g, kNwr));
            wi = Ops::selectGe(pw, threshold, ti, ld(g, kNwi));
        }
        const V z0 = ld(g, kZr), z1 = ld(g, kZi);
        zr = Ops::nmadd(z1, wi, Ops::mul(z0, wr));
        zi = Ops::madd(z0, wi, Ops::mul(z1, wr));
        if (renorm)
        {
            renormalize(zr, zi);
            renormalize(wr, wi);
        }
        st(g, kZr, zr);
        st(g, kZi, zi);
        st(g, kWr, wr);
        st(g, kWi, wi);
    }

    static void sanitize(float* groups, std::size_t numGroups)
    {
        const V zero = Ops::set(0.0f);
        for (std::size_t g = 0; g < numGroups; ++g)
            for (std::size_t half = 0; half < 2; ++half)
            {
                float* base = groups + g * kGroupFloats + half * kWidth;
                V t = zero;
                for (int f = kZr; f < kFields; ++f)
                    t = Ops::madd(ld(base, f), zero, t);
                float lanes[kWidth];
                Ops::store(lanes, t);
                bool finite = true;
                for (std::size_t l = 0; l < kWidth; ++l)
                    finite = finite && lanes[l] == 0.0f;
                if (finite)
                    continue;
                for (int f = kZr; f < kFields; ++f)
                {
                    V reset = zero;
                    if (f == kZr)
                        reset = Ops::set(1.0f);
                    else if (f == kWr)
                        reset = ld(base, kNwr);
                    else if (f == kWi)
                        reset = ld(base, kNwi);
                    st(base, f, Ops::selectGe(t, zero, ld(base, f), reset));
                }
            }
    }

    template <TrackingRule Rule>
    static void process(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool renormalizeAtEnd, float sigma, float ratio, float& accPower)
    {
        const float omSigma = 1.0f - sigma;
        V peak = Ops::set(0.0f);
        {
            const V s = Ops::set(x[0]);
            for (std::size_t g = 0; g < numGroups; ++g)
            {
                float* base = groups + g * kGroupFloats;
                const V pa = accumulate(base, s);
                const V pb = accumulate(base + kWidth, s);
                peak = Ops::max(peak, Ops::max(pa, pb));
            }
        }
        for (std::size_t i = 1; i < n; ++i)
        {
            accPower = omSigma * accPower + sigma * Ops::reduceMax(peak);
            const V threshold = Ops::set(accPower * ratio);
            const V s = Ops::set(x[i * stride]);
            peak = Ops::set(0.0f);
            for (std::size_t g = 0; g < numGroups; ++g)
            {
                float* base = groups + g * kGroupFloats;
                V zra, zia, zrb, zib;
                track<Rule>(base, threshold, false, zra, zia);
                track<Rule>(base + kWidth, threshold, false, zrb, zib);
                const V pa = accumulate(base, s, zra, zia);
                const V pb = accumulate(base + kWidth, s, zrb, zib);
                peak = Ops::max(peak, Ops::max(pa, pb));
            }
        }
        accPower = omSigma * accPower + sigma * Ops::reduceMax(peak);
        const V threshold = Ops::set(accPower * ratio);
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            float* base = groups + g * kGroupFloats;
            V zr, zi;
            track<Rule>(base, threshold, renormalizeAtEnd, zr, zi);
            track<Rule>(base + kWidth, threshold, renormalizeAtEnd, zr, zi);
        }
    }
};

}

class TrackingResonatorBank
{
public:
    using Kernel = detail::TrackingKernel<simd::NativeOps>;
    static constexpr std::size_t kLanes = Kernel::kLanes;
    static constexpr std::size_t kNormalizeInterval = 32;
    static constexpr std::size_t kSanitizeEvery = 4;
    static constexpr float kInitialAccumulatedPower = 0.0001f;

    static constexpr std::size_t storageSize(std::size_t numResonators) noexcept
    {
        return ((numResonators + kLanes - 1) / kLanes) * Kernel::kGroupFloats + 4 * numResonators;
    }

    TrackingResonatorBank() = default;
    TrackingResonatorBank(const TrackingResonatorBank&) = delete;
    TrackingResonatorBank& operator=(const TrackingResonatorBank&) = delete;
    TrackingResonatorBank(TrackingResonatorBank&& other) noexcept { take(other); }

    TrackingResonatorBank& operator=(TrackingResonatorBank&& other) noexcept
    {
        if (this != &other)
            take(other);
        return *this;
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
              const float* betas, const float* gammas, std::size_t count, float sampleRate,
              TrackingRule rule = TrackingRule::ewma, float thresholdDb = -60.0f) noexcept
    {
        return initImpl(storage, storageCount, frequencies, alphas, betas, gammas, count, sampleRate, rule,
                        thresholdDb);
    }

    bool init(float* storage, std::size_t storageCount, const float* frequencies, std::size_t count, float sampleRate,
              TrackingRule rule = TrackingRule::ewma, float thresholdDb = -60.0f) noexcept
    {
        return initImpl(storage, storageCount, frequencies, nullptr, nullptr, nullptr, count, sampleRate, rule,
                        thresholdDb);
    }

    void reset() noexcept
    {
        for (std::size_t k = 0; k < paddedSize(); ++k)
        {
            field(k, Kernel::kZr) = 1.0f;
            field(k, Kernel::kZi) = 0.0f;
            field(k, Kernel::kUr) = 0.0f;
            field(k, Kernel::kUi) = 0.0f;
            field(k, Kernel::kSr) = 0.0f;
            field(k, Kernel::kSi) = 0.0f;
            field(k, Kernel::kPr) = 0.0f;
            field(k, Kernel::kPi) = 0.0f;
            field(k, Kernel::kQr) = 0.0f;
            field(k, Kernel::kQi) = 0.0f;
            field(k, Kernel::kPw) = 0.0f;
            field(k, Kernel::kWr) = field(k, Kernel::kNwr);
            field(k, Kernel::kWi) = field(k, Kernel::kNwi);
        }
        accPower_ = kInitialAccumulatedPower;
        sinceNormalize_ = 0;
        renormalizations_ = 0;
    }

    void process(const float* input, std::size_t numSamples, std::size_t stride = 1) noexcept
    {
        if (numGroups_ == 0 || numSamples == 0)
            return;
        const simd::ScopedFlushDenormals ftz;
        while (numSamples > 0)
        {
            const std::size_t due = kNormalizeInterval - sinceNormalize_;
            const std::size_t n = numSamples < due ? numSamples : due;
            const bool renormalize = n == due;
            processChunk(input, n, stride, renormalize);
            if (renormalize && ++renormalizations_ % kSanitizeEvery == 0)
            {
                Kernel::sanitize(groups_, numGroups_);
                if (!std::isfinite(accPower_))
                    accPower_ = kInitialAccumulatedPower;
            }
            sinceNormalize_ = renormalize ? 0 : sinceNormalize_ + n;
            input += n * stride;
            numSamples -= n;
        }
    }

    void process(float sample) noexcept { process(&sample, 1, 1); }

    bool isInitialized() const noexcept { return numGroups_ != 0; }
    std::size_t size() const noexcept { return size_; }
    float sampleRate() const noexcept { return sampleRate_; }
    TrackingRule rule() const noexcept { return rule_; }
    float thresholdDb() const noexcept { return static_cast<float>(10.0 * std::log10(static_cast<double>(ratio_))); }
    float accumulatedPower() const noexcept { return accPower_; }
    float naturalFrequency(std::size_t k) const noexcept { return info_[k]; }
    float alpha(std::size_t k) const noexcept { return info_[size_ + k]; }
    float beta(std::size_t k) const noexcept { return info_[2 * size_ + k]; }
    float gamma(std::size_t k) const noexcept { return info_[3 * size_ + k]; }

    void setRule(TrackingRule rule) noexcept
    {
        rule_ = rule;
        if (rule == TrackingRule::none)
            for (std::size_t k = 0; k < size_; ++k)
            {
                field(k, Kernel::kWr) = field(k, Kernel::kNwr);
                field(k, Kernel::kWi) = field(k, Kernel::kNwi);
            }
    }

    void setThresholdDb(float db) noexcept { ratio_ = static_cast<float>(std::pow(10.0, static_cast<double>(db) / 10.0)); }

    void setPowerTimeConstant(float seconds) noexcept
    {
        sigma_ = seconds > 0.0f ? alphaFromTimeConstant(seconds, sampleRate_) : 1.0f;
    }

    bool setGamma(std::size_t k, float g) noexcept
    {
        if (k >= size_ || !(g >= 0.0f && g <= 1.0f))
            return false;
        info_[3 * size_ + k] = g;
        field(k, Kernel::kGm) = g;
        field(k, Kernel::kOg) = 1.0f - g;
        return true;
    }

    Complex value(std::size_t k) const noexcept { return Complex{field(k, Kernel::kSr), field(k, Kernel::kSi)}; }

    float power(std::size_t k) const noexcept
    {
        const float re = field(k, Kernel::kSr), im = field(k, Kernel::kSi);
        return re * re + im * im;
    }

    float magnitude(std::size_t k) const noexcept { return std::sqrt(power(k)); }
    float phase(std::size_t k) const noexcept { return std::atan2(field(k, Kernel::kSi), field(k, Kernel::kSr)); }
    float deltaPhase(std::size_t k) const noexcept { return -std::atan2(field(k, Kernel::kQi), field(k, Kernel::kQr)); }

    float resonantFrequency(std::size_t k) const noexcept
    {
        return static_cast<float>(-std::atan2(field(k, Kernel::kWi), field(k, Kernel::kWr)) * sampleRate_
                                  / detail::kTwoPi);
    }

    float instantaneousFrequency(std::size_t k) const noexcept
    {
        return resonantFrequency(k) + static_cast<float>(deltaPhase(k) * sampleRate_ / detail::kTwoPi);
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

    void resonantFrequencies(float* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = resonantFrequency(k);
    }

    void instantaneousFrequencies(float* out) const noexcept
    {
        for (std::size_t k = 0; k < size_; ++k)
            out[k] = instantaneousFrequency(k);
    }

private:
    void take(TrackingResonatorBank& other) noexcept
    {
        groups_ = other.groups_;
        info_ = other.info_;
        size_ = other.size_;
        numGroups_ = other.numGroups_;
        sinceNormalize_ = other.sinceNormalize_;
        renormalizations_ = other.renormalizations_;
        sampleRate_ = other.sampleRate_;
        sigma_ = other.sigma_;
        ratio_ = other.ratio_;
        accPower_ = other.accPower_;
        rule_ = other.rule_;
        other.groups_ = nullptr;
        other.info_ = nullptr;
        other.size_ = 0;
        other.numGroups_ = 0;
        other.sinceNormalize_ = 0;
        other.renormalizations_ = 0;
    }

    void processChunk(const float* input, std::size_t n, std::size_t stride, bool renormalize) noexcept
    {
        switch (rule_)
        {
        case TrackingRule::none:
            Kernel::process<TrackingRule::none>(groups_, numGroups_, input, n, stride, renormalize, sigma_, ratio_,
                                                accPower_);
            break;
        case TrackingRule::ewma:
            Kernel::process<TrackingRule::ewma>(groups_, numGroups_, input, n, stride, renormalize, sigma_, ratio_,
                                                accPower_);
            break;
        case TrackingRule::normalizedChord:
            Kernel::process<TrackingRule::normalizedChord>(groups_, numGroups_, input, n, stride, renormalize, sigma_,
                                                           ratio_, accPower_);
            break;
        case TrackingRule::tangent:
            Kernel::process<TrackingRule::tangent>(groups_, numGroups_, input, n, stride, renormalize, sigma_, ratio_,
                                                   accPower_);
            break;
        }
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

    bool initImpl(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
                  const float* betas, const float* gammas, std::size_t count, float sampleRate, TrackingRule rule,
                  float thresholdDb) noexcept
    {
        numGroups_ = 0;
        size_ = 0;
        if (storage == nullptr || frequencies == nullptr || count == 0 || storageCount < storageSize(count)
            || !(sampleRate > 0.0f) || !std::isfinite(sampleRate) || !std::isfinite(thresholdDb))
            return false;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            const float g = gammas != nullptr ? gammas[k] : 0.5f * a;
            if (!detail::validFrequency(frequencies[k], sampleRate) || !detail::inUnitInterval(a)
                || !detail::inUnitInterval(b) || !(g >= 0.0f && g <= 1.0f))
                return false;
        }
        const std::size_t numGroups = (count + kLanes - 1) / kLanes;
        groups_ = storage;
        info_ = storage + numGroups * Kernel::kGroupFloats;
        size_ = count;
        numGroups_ = numGroups;
        sampleRate_ = sampleRate;
        rule_ = rule;
        sigma_ = 1.0f;
        setThresholdDb(thresholdDb);
        for (std::size_t k = 0; k < paddedSize(); ++k)
        {
            field(k, Kernel::kOa) = 0.0f;
            field(k, Kernel::kOb) = 1.0f;
            field(k, Kernel::kAb) = 0.0f;
            field(k, Kernel::kGm) = 0.0f;
            field(k, Kernel::kOg) = 1.0f;
            field(k, Kernel::kNwr) = 1.0f;
            field(k, Kernel::kNwi) = 0.0f;
        }
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            const float g = gammas != nullptr ? gammas[k] : 0.5f * a;
            info_[k] = frequencies[k];
            info_[count + k] = a;
            info_[2 * count + k] = b;
            info_[3 * count + k] = g;
            field(k, Kernel::kOa) = 1.0f - a;
            field(k, Kernel::kOb) = 1.0f - b;
            field(k, Kernel::kAb) = a * b;
            field(k, Kernel::kGm) = g;
            field(k, Kernel::kOg) = 1.0f - g;
            detail::phasorStep(frequencies[k], sampleRate, field(k, Kernel::kNwr), field(k, Kernel::kNwi));
        }
        reset();
        return true;
    }

    float* groups_ = nullptr;
    float* info_ = nullptr;
    std::size_t size_ = 0;
    std::size_t numGroups_ = 0;
    std::size_t sinceNormalize_ = 0;
    std::size_t renormalizations_ = 0;
    float sampleRate_ = 0.0f;
    float sigma_ = 1.0f;
    float ratio_ = 1e-6f;
    float accPower_ = kInitialAccumulatedPower;
    TrackingRule rule_ = TrackingRule::ewma;
};

template <std::size_t MaxResonators>
class StaticTrackingResonatorBank : public TrackingResonatorBank
{
public:
    static constexpr std::size_t kCapacity = MaxResonators;

    StaticTrackingResonatorBank() = default;
    StaticTrackingResonatorBank(StaticTrackingResonatorBank&&) = delete;
    StaticTrackingResonatorBank& operator=(StaticTrackingResonatorBank&&) = delete;

    bool init(const float* frequencies, const float* alphas, const float* betas, const float* gammas,
              std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma, float thresholdDb = -60.0f) noexcept
    {
        return count <= MaxResonators
               && TrackingResonatorBank::init(storage_, kStorage, frequencies, alphas, betas, gammas, count,
                                              sampleRate, rule, thresholdDb);
    }

    bool init(const float* frequencies, std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma,
              float thresholdDb = -60.0f) noexcept
    {
        return count <= MaxResonators
               && TrackingResonatorBank::init(storage_, kStorage, frequencies, count, sampleRate, rule, thresholdDb);
    }

private:
    static constexpr std::size_t kStorage = storageSize(MaxResonators);
    float storage_[kStorage];
};

class DynamicTrackingResonatorBank : public TrackingResonatorBank
{
public:
    bool init(const float* frequencies, const float* alphas, const float* betas, const float* gammas,
              std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma, float thresholdDb = -60.0f)
    {
        storage_.assign(storageSize(count), 0.0f);
        return TrackingResonatorBank::init(storage_.data(), storage_.size(), frequencies, alphas, betas, gammas, count,
                                           sampleRate, rule, thresholdDb);
    }

    bool init(const float* frequencies, std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma,
              float thresholdDb = -60.0f)
    {
        return init(frequencies, nullptr, nullptr, nullptr, count, sampleRate, rule, thresholdDb);
    }

private:
    std::vector<float> storage_;
};

}

#endif
