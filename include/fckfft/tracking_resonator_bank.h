#ifndef FCKFFT_TRACKING_RESONATOR_BANK_H
#define FCKFFT_TRACKING_RESONATOR_BANK_H

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "dynamics.h"
#include "resonator_bank.h"
#include "simd.h"

namespace fckfft {
inline namespace FCKFFT_ABI {

class TrackingResonatorBank
{
public:
    using Kernel = detail::TrackingLayout;
    static constexpr std::size_t kMaxLanes = detail::kMaxTrackingLanes;
    static constexpr std::size_t kNormalizeInterval = 32;
    static constexpr std::size_t kSanitizeEvery = 4;
    static constexpr float kInitialAccumulatedPower = 0.0001f;

    static constexpr std::size_t storageSize(std::size_t numResonators) noexcept
    {
        return ((numResonators + kMaxLanes - 1) / kMaxLanes) * Kernel::kFields * kMaxLanes + 4 * numResonators;
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
                backend_->sanitize(groups_, numGroups_);
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
    std::size_t lanes() const noexcept { return lanes_; }
    Isa isa() const noexcept { return backend_->isa; }
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

    static bool validParameters(const float* frequencies, const float* alphas, const float* betas, const float* gammas,
                                std::size_t count, float sampleRate, float thresholdDb) noexcept
    {
        if (frequencies == nullptr || count == 0 || !(sampleRate > 0.0f) || !std::isfinite(sampleRate)
            || !std::isfinite(thresholdDb))
            return false;
        for (std::size_t k = 0; k < count; ++k)
        {
            const float a = alphas != nullptr ? alphas[k] : heuristicAlpha(frequencies[k], sampleRate);
            const float b = betas != nullptr ? betas[k] : a;
            const float g = gammas != nullptr ? gammas[k] : 0.5f * a;
            if (!detail::validFrequency(frequencies[k], sampleRate) || !acceptableDecay(a) || !acceptableDecay(b)
                || !(g >= 0.0f && g <= 1.0f))
                return false;
        }
        return true;
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
        backend_ = other.backend_;
        lanes_ = other.lanes_;
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
        backend_->process[static_cast<int>(rule_)](groups_, numGroups_, input, n, stride, renormalize, sigma_, ratio_,
                                                   accPower_);
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

    static bool acceptableDecay(float decay) noexcept
    {
        return detail::inUnitInterval(decay)
               && detail::acceptableDecay(decay, 1.0 - static_cast<double>(1.0f - decay));
    }

    bool initImpl(float* storage, std::size_t storageCount, const float* frequencies, const float* alphas,
                  const float* betas, const float* gammas, std::size_t count, float sampleRate, TrackingRule rule,
                  float thresholdDb) noexcept
    {
        if (storage == nullptr || count == 0 || storageCount < storageSize(count)
            || !validParameters(frequencies, alphas, betas, gammas, count, sampleRate, thresholdDb))
            return false;
        const detail::TrackingBackend& backend = detail::selectTrackingBackend();
        const std::size_t numGroups = (count + backend.lanes - 1) / backend.lanes;
        backend_ = &backend;
        lanes_ = backend.lanes;
        groups_ = storage;
        info_ = storage + numGroups * Kernel::kFields * lanes_;
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

    const detail::TrackingBackend* backend_ = &detail::baseline::kTrackingBackend;
    std::size_t lanes_ = detail::baseline::kTrackingBackend.lanes;
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

template <class Storage>
class OwningTrackingResonatorBank : private TrackingResonatorBank
{
public:
    using TrackingResonatorBank::kMaxLanes;
    using TrackingResonatorBank::kNormalizeInterval;
    using TrackingResonatorBank::kInitialAccumulatedPower;
    using TrackingResonatorBank::storageSize;
    using TrackingResonatorBank::validParameters;
    using TrackingResonatorBank::reset;
    using TrackingResonatorBank::process;
    using TrackingResonatorBank::isInitialized;
    using TrackingResonatorBank::size;
    using TrackingResonatorBank::lanes;
    using TrackingResonatorBank::isa;
    using TrackingResonatorBank::sampleRate;
    using TrackingResonatorBank::rule;
    using TrackingResonatorBank::thresholdDb;
    using TrackingResonatorBank::accumulatedPower;
    using TrackingResonatorBank::naturalFrequency;
    using TrackingResonatorBank::alpha;
    using TrackingResonatorBank::beta;
    using TrackingResonatorBank::gamma;
    using TrackingResonatorBank::setRule;
    using TrackingResonatorBank::setThresholdDb;
    using TrackingResonatorBank::setPowerTimeConstant;
    using TrackingResonatorBank::setGamma;
    using TrackingResonatorBank::value;
    using TrackingResonatorBank::power;
    using TrackingResonatorBank::magnitude;
    using TrackingResonatorBank::phase;
    using TrackingResonatorBank::deltaPhase;
    using TrackingResonatorBank::resonantFrequency;
    using TrackingResonatorBank::instantaneousFrequency;
    using TrackingResonatorBank::values;
    using TrackingResonatorBank::powers;
    using TrackingResonatorBank::magnitudes;
    using TrackingResonatorBank::phases;
    using TrackingResonatorBank::resonantFrequencies;
    using TrackingResonatorBank::instantaneousFrequencies;

    bool init(const float* frequencies, const float* alphas, const float* betas, const float* gammas,
              std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma,
              float thresholdDb = -60.0f) noexcept(noexcept(std::declval<Storage&>().acquire(0)))
    {
        if (!Storage::fits(storageSize(count))
            || !validParameters(frequencies, alphas, betas, gammas, count, sampleRate, thresholdDb))
            return false;
        float* storage = storage_.acquire(storageSize(count));
        return storage != nullptr
               && TrackingResonatorBank::init(storage, storageSize(count), frequencies, alphas, betas, gammas, count,
                                              sampleRate, rule, thresholdDb);
    }

    bool init(const float* frequencies, std::size_t count, float sampleRate, TrackingRule rule = TrackingRule::ewma,
              float thresholdDb = -60.0f) noexcept(noexcept(std::declval<Storage&>().acquire(0)))
    {
        return init(frequencies, nullptr, nullptr, nullptr, count, sampleRate, rule, thresholdDb);
    }

private:
    Storage storage_;
};

template <std::size_t MaxResonators>
class StaticTrackingResonatorBank
    : public OwningTrackingResonatorBank<detail::InlineStorage<TrackingResonatorBank::storageSize(MaxResonators)>>
{
public:
    static constexpr std::size_t kCapacity = MaxResonators;
};

class DynamicTrackingResonatorBank : public OwningTrackingResonatorBank<detail::VectorStorage>
{
};

}
}

#endif
