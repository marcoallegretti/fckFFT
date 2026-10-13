#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>

#include <fckfft/frequencies.h>
#include <fckfft/tracking_resonator_bank.h>

#include "check.h"
#include "signals.h"

using fckfft::TrackingRule;

static const TrackingRule kTrackingRules[3] = {TrackingRule::ewma, TrackingRule::normalizedChord,
                                               TrackingRule::tangent};

static std::vector<float> semitones(float first, std::size_t count)
{
    std::vector<float> f(count);
    fckfft::logUniformFrequencies(first, 12.0f, count, f.data());
    return f;
}

template <class Bank>
static std::size_t strongest(const Bank& bank)
{
    std::size_t best = 0;
    for (std::size_t k = 1; k < bank.size(); ++k)
        if (bank.power(k) > bank.power(best))
            best = k;
    return best;
}

TEST_CASE(tracking_rejects_invalid_arguments)
{
    std::vector<float> storage(fckfft::TrackingResonatorBank::storageSize(2));
    const float f[2] = {100.0f, 200.0f};
    const float a[2] = {0.01f, 0.01f};
    const float badGamma[2] = {0.01f, -0.1f};
    fckfft::TrackingResonatorBank bank;
    CHECK(!bank.init(storage.data(), storage.size() - 1, f, 2, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), f, a, a, badGamma, 2, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), f, 2, 48000.0f, TrackingRule::ewma, std::nanf("")));
    CHECK(bank.init(storage.data(), storage.size(), f, a, nullptr, nullptr, 2, 48000.0f));
    CHECK_NEAR(bank.beta(1), 0.01f, 0.0);
    CHECK_NEAR(bank.gamma(1), 0.005f, 0.0);
    CHECK(!bank.setGamma(2, 0.1f));
    CHECK(!bank.setGamma(0, 1.5f));
    CHECK(bank.setGamma(0, 0.0f));
    fckfft::StaticTrackingResonatorBank<1> small;
    CHECK(!small.init(f, 2, 48000.0f));
}

TEST_CASE(tracking_converges_to_detuned_tone)
{
    const std::vector<float> natural = {400.0f, 440.0f, 480.0f};
    const std::vector<float> x = signals::tone(48000, 48000.0, 452.0);
    for (TrackingRule rule : kTrackingRules)
    {
        fckfft::DynamicTrackingResonatorBank bank;
        CHECK(bank.init(natural.data(), natural.size(), 48000.0f, rule));
        bank.process(x.data(), x.size());
        CHECK_NEAR(bank.resonantFrequency(1), 452.0, 0.05);
        CHECK_NEAR(bank.instantaneousFrequency(1), 452.0, 0.5);
        CHECK_NEAR(bank.magnitude(1), 0.5, 0.01);
        CHECK_NEAR(bank.naturalFrequency(1), 440.0, 0.0);
    }
}

TEST_CASE(tracking_follows_frequency_step)
{
    const std::vector<float> natural = semitones(392.0f, 5);
    const std::vector<float> x = signals::render(
        96000, 48000.0, [](std::size_t i) { return i < 48000 ? 440.0 : 455.0; }, [](std::size_t) { return 1.0; });
    for (TrackingRule rule : kTrackingRules)
    {
        fckfft::DynamicTrackingResonatorBank bank;
        CHECK(bank.init(natural.data(), natural.size(), 48000.0f, rule));
        bank.process(x.data(), 48000);
        CHECK_NEAR(bank.resonantFrequency(strongest(bank)), 440.0, 0.05);
        bank.process(x.data() + 48000, 48000);
        CHECK_NEAR(bank.resonantFrequency(strongest(bank)), 455.0, 0.05);
    }
}

TEST_CASE(tracking_matches_double_reference_model)
{
    const double sr = 48000.0;
    const std::vector<float> f = semitones(220.0f, 24);
    std::vector<std::vector<float>> inputs;
    inputs.push_back(signals::tone(24000, sr, 450.0));
    inputs.push_back(signals::render(
        24000, sr,
        [sr](std::size_t i) { return 440.0 + 10.0 * std::sin(2.0 * signals::kPi * 5.0 * static_cast<double>(i) / sr); },
        [](std::size_t) { return 1.0; }));
    inputs.push_back(signals::render(
        24000, sr, [](std::size_t i) { return 400.0 + 100.0 * static_cast<double>(i) / 24000.0; },
        [](std::size_t) { return 0.7; }));
    for (int rule = 0; rule < 4; ++rule)
        for (const std::vector<float>& x : inputs)
        {
            signals::TrackingModel model(f, sr, rule, -60.0);
            fckfft::DynamicTrackingResonatorBank bank;
            CHECK(bank.init(f.data(), f.size(), static_cast<float>(sr), static_cast<TrackingRule>(rule), -60.0f));
            double maxFrequency = 0.0, maxMagnitude = 0.0;
            for (std::size_t i = 0; i < x.size(); i += 48)
            {
                for (std::size_t j = 0; j < 48; ++j)
                    model.step(x[i + j]);
                bank.process(x.data() + i, 48);
                for (std::size_t k = 0; k < f.size(); ++k)
                {
                    maxMagnitude = std::max(maxMagnitude, std::fabs(bank.magnitude(k) - model.magnitude(k)));
                    if (model.magnitude(k) > 0.05)
                        maxFrequency = std::max(maxFrequency,
                                                std::fabs(bank.resonantFrequency(k) - model.resonantFrequency(k)));
                }
            }
            CHECK_NEAR(maxFrequency, 0.0, 0.05);
            CHECK_NEAR(maxMagnitude, 0.0, 5e-4);
        }
}

TEST_CASE(tracking_recovers_after_dropout)
{
    const double sr = 48000.0;
    const std::vector<float> f = semitones(220.0f, 24);
    const std::vector<float> x = signals::render(
        48000, sr, [](std::size_t) { return 466.0; }, [](std::size_t i) { return (i >= 8000 && i < 16000) ? 0.0 : 1.0; });
    for (int rule = 1; rule < 4; ++rule)
    {
        signals::TrackingModel model(f, sr, rule, -60.0);
        fckfft::DynamicTrackingResonatorBank bank;
        CHECK(bank.init(f.data(), f.size(), static_cast<float>(sr), static_cast<TrackingRule>(rule), -60.0f));
        for (float v : x)
            model.step(v);
        bank.process(x.data(), x.size());
        const std::size_t k = strongest(bank);
        CHECK(std::isfinite(bank.magnitude(k)));
        CHECK_NEAR(bank.resonantFrequency(k), 466.0, 0.05);
        CHECK_NEAR(bank.resonantFrequency(k), model.resonantFrequency(k), 0.05);
        CHECK_NEAR(bank.magnitude(k), model.magnitude(k), 2e-4);
    }
}

TEST_CASE(tracking_starts_from_silence_without_nan)
{
    const std::vector<float> f = semitones(100.0f, 9);
    const std::vector<float> silence(4800, 0.0f);
    const std::vector<float> x = signals::tone(4800, 48000.0, 150.0);
    for (int rule = 0; rule < 4; ++rule)
    {
        fckfft::DynamicTrackingResonatorBank bank;
        CHECK(bank.init(f.data(), f.size(), 48000.0f, static_cast<TrackingRule>(rule)));
        bank.process(silence.data(), silence.size());
        bank.process(x.data(), x.size());
        bank.process(silence.data(), silence.size());
        for (std::size_t k = 0; k < f.size(); ++k)
        {
            CHECK(std::isfinite(bank.magnitude(k)));
            CHECK(std::isfinite(bank.resonantFrequency(k)));
            CHECK(std::isfinite(bank.instantaneousFrequency(k)));
        }
    }
}

TEST_CASE(tracking_super_resolution_without_retuning)
{
    const std::vector<float> f = semitones(220.0f, 25);
    fckfft::DynamicTrackingResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f, TrackingRule::none));
    for (double target : {261.0, 452.0, 700.0})
    {
        bank.reset();
        const std::vector<float> x = signals::tone(48000, 48000.0, target);
        bank.process(x.data(), x.size());
        const std::size_t k = strongest(bank);
        CHECK_NEAR(bank.resonantFrequency(k), f[k], 1e-3 * f[k]);
        CHECK_NEAR(bank.instantaneousFrequency(k), target, 0.05);
    }
}

TEST_CASE(tracking_block_size_independent)
{
    const std::vector<float> f = semitones(300.0f, 13);
    const std::vector<float> x = signals::render(
        9600, 48000.0, [](std::size_t i) { return 420.0 + 0.004 * static_cast<double>(i); }, [](std::size_t) { return 0.8; });
    for (TrackingRule rule : kTrackingRules)
    {
        fckfft::DynamicTrackingResonatorBank a, b;
        CHECK(a.init(f.data(), f.size(), 48000.0f, rule));
        CHECK(b.init(f.data(), f.size(), 48000.0f, rule));
        a.process(x.data(), x.size());
        for (std::size_t i = 0; i < x.size(); i += 5)
            b.process(x.data() + i, std::min<std::size_t>(5, x.size() - i));
        for (std::size_t k = 0; k < f.size(); ++k)
        {
            CHECK_NEAR(a.magnitude(k), b.magnitude(k), 0.0);
            CHECK_NEAR(a.resonantFrequency(k), b.resonantFrequency(k), 0.0);
        }
    }
}

TEST_CASE(tracking_reset_and_rule_changes)
{
    const std::vector<float> f = semitones(400.0f, 4);
    const std::vector<float> x = signals::tone(24000, 48000.0, 430.0);
    fckfft::DynamicTrackingResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f, TrackingRule::normalizedChord, -40.0f));
    CHECK(bank.rule() == TrackingRule::normalizedChord);
    CHECK_NEAR(bank.thresholdDb(), -40.0, 1e-3);
    bank.process(x.data(), x.size());
    CHECK(std::fabs(bank.resonantFrequency(1) - f[1]) > 1.0f);
    bank.reset();
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_NEAR(bank.resonantFrequency(k), f[k], 1e-3 * f[k]);
        CHECK(bank.power(k) == 0.0f);
    }
    CHECK_NEAR(bank.accumulatedPower(), fckfft::TrackingResonatorBank::kInitialAccumulatedPower, 0.0);
    bank.process(x.data(), x.size());
    bank.setRule(TrackingRule::none);
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK_NEAR(bank.resonantFrequency(k), f[k], 1e-3 * f[k]);
    bank.process(x.data(), x.size());
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK_NEAR(bank.resonantFrequency(k), f[k], 1e-3 * f[k]);
}

TEST_CASE(tracking_threshold_gates_weak_resonators)
{
    const std::vector<float> f = {300.0f, 1500.0f};
    const std::vector<float> x = signals::tone(48000, 48000.0, 310.0);
    fckfft::DynamicTrackingResonatorBank strict, loose;
    CHECK(strict.init(f.data(), f.size(), 48000.0f, TrackingRule::ewma, -10.0f));
    CHECK(loose.init(f.data(), f.size(), 48000.0f, TrackingRule::ewma, -200.0f));
    strict.process(x.data(), x.size());
    loose.process(x.data(), x.size());
    CHECK_NEAR(strict.resonantFrequency(0), 310.0, 0.05);
    CHECK_NEAR(strict.resonantFrequency(1), 1500.0, 0.01);
    CHECK(std::fabs(loose.resonantFrequency(1) - 1500.0f) > 1.0f);
}

TEST_CASE(tracking_power_time_constant_smooths_threshold)
{
    const std::vector<float> f = {440.0f};
    const std::vector<float> x = signals::tone(4800, 48000.0, 440.0);
    fckfft::DynamicTrackingResonatorBank instant, smooth;
    CHECK(instant.init(f.data(), f.size(), 48000.0f));
    CHECK(smooth.init(f.data(), f.size(), 48000.0f));
    smooth.setPowerTimeConstant(1.0f);
    instant.process(x.data(), x.size());
    smooth.process(x.data(), x.size());
    CHECK_NEAR(instant.accumulatedPower(), instant.power(0), 1e-6 * instant.power(0));
    CHECK(smooth.accumulatedPower() < 0.1f * instant.accumulatedPower());
    smooth.setPowerTimeConstant(0.0f);
    smooth.process(x.data(), 1);
    CHECK_NEAR(smooth.accumulatedPower(), smooth.power(0), 1e-6 * smooth.power(0));
}

TEST_CASE(tracking_bulk_readouts)
{
    const std::vector<float> f = semitones(500.0f, 6);
    const std::vector<float> x = signals::noise(4800, 21u);
    fckfft::DynamicTrackingResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f, TrackingRule::tangent));
    bank.process(x.data(), x.size());
    std::vector<float> p(f.size()), m(f.size()), ph(f.size()), rf(f.size()), inst(f.size());
    std::vector<fckfft::Complex> v(f.size());
    bank.powers(p.data());
    bank.magnitudes(m.data());
    bank.phases(ph.data());
    bank.resonantFrequencies(rf.data());
    bank.instantaneousFrequencies(inst.data());
    bank.values(v.data());
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_NEAR(p[k], bank.power(k), 0.0);
        CHECK_NEAR(m[k], bank.magnitude(k), 0.0);
        CHECK_NEAR(ph[k], bank.phase(k), 0.0);
        CHECK_NEAR(rf[k], bank.resonantFrequency(k), 0.0);
        CHECK_NEAR(inst[k], bank.instantaneousFrequency(k), 0.0);
        CHECK_NEAR(v[k].re, bank.value(k).re, 0.0);
        CHECK_NEAR(bank.deltaPhase(k) * 48000.0 / (2.0 * signals::kPi), inst[k] - rf[k], 1e-2);
    }
}

TEST_CASE(tracking_recovers_from_non_finite_input)
{
    const std::vector<float> f = semitones(400.0f, 9);
    const float bad[2] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    const std::size_t bound =
        fckfft::TrackingResonatorBank::kNormalizeInterval * fckfft::TrackingResonatorBank::kSanitizeEvery;
    for (int rule = 0; rule < 4; ++rule)
        for (float poison : bad)
        {
            fckfft::DynamicTrackingResonatorBank bank;
            CHECK(bank.init(f.data(), f.size(), 48000.0f, static_cast<TrackingRule>(rule)));
            std::vector<float> x = signals::tone(48000, 48000.0, 452.0);
            x[1000] = poison;
            bank.process(x.data(), 1000 + bound);
            for (std::size_t k = 0; k < f.size(); ++k)
            {
                CHECK(std::isfinite(bank.power(k)));
                CHECK(std::isfinite(bank.resonantFrequency(k)));
                CHECK(std::isfinite(bank.instantaneousFrequency(k)));
            }
            CHECK(std::isfinite(bank.accumulatedPower()));
            bank.process(x.data() + 1000 + bound, x.size() - 1000 - bound);
            if (rule != 0)
                CHECK_NEAR(bank.resonantFrequency(strongest(bank)), 452.0, 0.05);
        }
}

TEST_CASE(tracking_move_and_contract)
{
    static_assert(std::is_nothrow_move_constructible<fckfft::DynamicTrackingResonatorBank>::value, "dynamic banks move without throwing");
    static_assert(std::is_nothrow_move_assignable<fckfft::DynamicTrackingResonatorBank>::value, "dynamic banks move-assign without throwing");
    static_assert(!std::is_copy_constructible<fckfft::DynamicTrackingResonatorBank>::value, "banks are not copyable");
    static_assert(!std::is_move_constructible<fckfft::StaticTrackingResonatorBank<4>>::value, "static banks own inline storage");
    static_assert(noexcept(std::declval<fckfft::DynamicTrackingResonatorBank&>().process(nullptr, 0)), "process does not throw");
    static_assert(!std::is_constructible<fckfft::TrackingResonatorBank, fckfft::StaticTrackingResonatorBank<4>&&>::value,
                  "owning banks do not convert to a storage view");
    static_assert(!std::is_constructible<fckfft::TrackingResonatorBank, fckfft::DynamicTrackingResonatorBank&&>::value,
                  "owning banks do not convert to a storage view");
    const float f[4] = {220.0f, 330.0f, 440.0f, 660.0f};
    const std::vector<float> x = signals::tone(4800, 48000.0, 440.0);
    fckfft::DynamicTrackingResonatorBank reference;
    CHECK(reference.init(f, 4, 48000.0f));
    reference.process(x.data(), x.size());
    std::vector<fckfft::DynamicTrackingResonatorBank> channels;
    for (int c = 0; c < 3; ++c)
    {
        fckfft::DynamicTrackingResonatorBank bank;
        CHECK(bank.init(f, 4, 48000.0f));
        channels.push_back(std::move(bank));
        CHECK(!bank.isInitialized());
        bank.process(x.data(), x.size());
    }
    for (auto& bank : channels)
    {
        bank.process(x.data(), x.size());
        for (std::size_t k = 0; k < 4; ++k)
            CHECK_NEAR(bank.power(k), reference.power(k), 0.0);
    }
    fckfft::DynamicTrackingResonatorBank assigned;
    assigned = std::move(channels[0]);
    CHECK(assigned.isInitialized() && !channels[0].isInitialized());
    CHECK_NEAR(assigned.power(2), reference.power(2), 0.0);
}

TEST_CASE(tracking_failed_init_keeps_configuration)
{
    std::vector<float> f(12);
    fckfft::logUniformFrequencies(110.0f, 12.0f, f.size(), f.data());
    const std::vector<float> high(13, 15000.0f);
    const std::vector<float> x = signals::tone(4800, 48000.0, 233.0);
    fckfft::DynamicTrackingResonatorBank twin, dynamic;
    fckfft::StaticTrackingResonatorBank<12> fixed;
    CHECK(twin.init(f.data(), f.size(), 48000.0f, fckfft::TrackingRule::normalizedChord));
    CHECK(dynamic.init(f.data(), f.size(), 48000.0f, fckfft::TrackingRule::normalizedChord));
    CHECK(fixed.init(f.data(), f.size(), 48000.0f, fckfft::TrackingRule::normalizedChord));
    const float tiny[12] = {1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f};
    CHECK(!dynamic.init(high.data(), 12, 22050.0f));
    CHECK(!dynamic.init(f.data(), tiny, nullptr, nullptr, f.size(), 48000.0f));
    CHECK(!fixed.init(high.data(), 13, 48000.0f));
    CHECK(!fixed.init(f.data(), tiny, nullptr, nullptr, f.size(), 48000.0f));
    twin.process(x.data(), x.size());
    dynamic.process(x.data(), x.size());
    fixed.process(x.data(), x.size());
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_NEAR(dynamic.power(k), twin.power(k), 0.0);
        CHECK_NEAR(fixed.resonantFrequency(k), twin.resonantFrequency(k), 0.0);
    }
}
