#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
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
    CHECK(!bank.init(storage.data(), storage.size(), f, 2, 48000.0f, TrackingRule::ewma, check::fromBits(0x7FC00000u)));
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

TEST_CASE(tracking_matches_model_at_wide_bandwidths)
{
    const double sr = 48000.0;
    const std::vector<float> f = semitones(400.0f, 12);
    const std::vector<float> x = signals::tone(24000, sr, 452.0, 0.8);
    for (int rule = 2; rule < 4; ++rule)
        for (float a : {0.1f, 0.3f, 0.5f})
        {
            const std::vector<float> alphas(f.size(), a), gammas(f.size(), 0.5f * a);
            signals::TrackingModel model(f, sr, rule, -60.0, a);
            fckfft::DynamicTrackingResonatorBank bank;
            CHECK(bank.init(f.data(), alphas.data(), alphas.data(), gammas.data(), f.size(), static_cast<float>(sr),
                            static_cast<TrackingRule>(rule), -60.0f));
            double maxMagnitude = 0.0;
            for (std::size_t i = 0; i < x.size(); i += 48)
            {
                for (std::size_t j = 0; j < 48; ++j)
                    model.step(x[i + j]);
                bank.process(x.data() + i, 48);
                for (std::size_t k = 0; k < f.size(); ++k)
                    maxMagnitude = std::max(maxMagnitude, std::fabs(bank.magnitude(k) - model.magnitude(k)));
            }
            CHECK_NEAR(maxMagnitude, 0.0, 1e-5);
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
        CHECK(check::finite(bank.magnitude(k)));
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
            CHECK(check::finite(bank.magnitude(k)));
            CHECK(check::finite(bank.resonantFrequency(k)));
            CHECK(check::finite(bank.instantaneousFrequency(k)));
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
            CHECK_SAME(a.magnitude(k), b.magnitude(k));
            CHECK_SAME(a.resonantFrequency(k), b.resonantFrequency(k));
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

TEST_CASE(tracking_storage_offset_does_not_change_results)
{
    const std::vector<float> f = semitones(400.0f, 9);
    const std::vector<float> x = signals::tone(4000, 48000.0, 452.0);
    fckfft::DynamicTrackingResonatorBank reference;
    CHECK(reference.init(f.data(), f.size(), 48000.0f, TrackingRule::normalizedChord));
    reference.process(x.data(), x.size());
    for (std::size_t offset = 0; offset < 16; ++offset)
    {
        std::vector<float> buffer(offset + fckfft::TrackingResonatorBank::storageSize(f.size()));
        fckfft::TrackingResonatorBank bank;
        CHECK(bank.init(buffer.data() + offset, buffer.size() - offset, f.data(), f.size(), 48000.0f,
                        TrackingRule::normalizedChord));
        bank.process(x.data(), x.size());
        for (std::size_t k = 0; k < f.size(); ++k)
        {
            CHECK_SAME(bank.power(k), reference.power(k));
            CHECK_SAME(bank.resonantFrequency(k), reference.resonantFrequency(k));
        }
    }
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
        CHECK_SAME(p[k], bank.power(k));
        CHECK_SAME(m[k], bank.magnitude(k));
        CHECK_SAME(ph[k], bank.phase(k));
        CHECK_SAME(rf[k], bank.resonantFrequency(k));
        CHECK_SAME(inst[k], bank.instantaneousFrequency(k));
        CHECK_SAME(v[k].re, bank.value(k).re);
        CHECK_NEAR(bank.deltaPhase(k) * 48000.0 / (2.0 * signals::kPi), inst[k] - rf[k], 1e-2);
    }
}

TEST_CASE(tracking_recovers_from_non_finite_input)
{
    const std::vector<float> f = semitones(400.0f, 9);
    const float bad[2] = {check::fromBits(0x7FC00000u), check::fromBits(0x7F800000u)};
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
                CHECK(check::finite(bank.power(k)));
                CHECK(check::finite(bank.resonantFrequency(k)));
                CHECK(check::finite(bank.instantaneousFrequency(k)));
            }
            CHECK(check::finite(bank.accumulatedPower()));
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
            CHECK_SAME(bank.power(k), reference.power(k));
    }
    fckfft::DynamicTrackingResonatorBank assigned;
    assigned = std::move(channels[0]);
    CHECK(assigned.isInitialized() && !channels[0].isInitialized());
    CHECK_SAME(assigned.power(2), reference.power(2));
}

TEST_CASE(tracking_stride_matches_mono)
{
    const std::vector<float> f = semitones(300.0f, 13);
    const std::vector<float> mono = signals::render(
        2401, 48000.0, [](std::size_t i) { return 420.0 + 0.01 * static_cast<double>(i); },
        [](std::size_t) { return 0.8; });
    const TrackingRule rules[4] = {TrackingRule::none, TrackingRule::ewma, TrackingRule::normalizedChord,
                                   TrackingRule::tangent};
    for (std::size_t stride : {std::size_t(2), std::size_t(3)})
    {
        std::vector<float> interleaved(stride * mono.size(), 0.25f);
        for (std::size_t i = 0; i < mono.size(); ++i)
            interleaved[stride * i + stride - 1] = mono[i];
        for (TrackingRule rule : rules)
        {
            fckfft::DynamicTrackingResonatorBank reference, strided;
            CHECK(reference.init(f.data(), f.size(), 48000.0f, rule));
            CHECK(strided.init(f.data(), f.size(), 48000.0f, rule));
            reference.process(mono.data(), mono.size());
            for (std::size_t i = 0; i < mono.size(); i += 37)
                strided.process(interleaved.data() + stride * i + stride - 1, std::min<std::size_t>(37, mono.size() - i),
                                stride);
            CHECK_SAME(strided.accumulatedPower(), reference.accumulatedPower());
            for (std::size_t k = 0; k < f.size(); ++k)
            {
                CHECK_SAME(strided.power(k), reference.power(k));
                CHECK_SAME(strided.resonantFrequency(k), reference.resonantFrequency(k));
                CHECK_SAME(strided.instantaneousFrequency(k), reference.instantaneousFrequency(k));
            }
        }
    }
}

TEST_CASE(tracking_move_keeps_configuration)
{
    const std::vector<float> f = semitones(300.0f, 13);
    const std::vector<float> x = signals::tone(4800, 48000.0, 452.0);
    auto configure = [&](fckfft::DynamicTrackingResonatorBank& bank) {
        CHECK(bank.init(f.data(), f.size(), 48000.0f, TrackingRule::tangent, -40.0f));
        bank.setPowerTimeConstant(0.05f);
        CHECK(bank.setGamma(3, 0.004f));
        bank.process(x.data(), 1000);
    };
    fckfft::DynamicTrackingResonatorBank twin, source;
    configure(twin);
    configure(source);
    fckfft::DynamicTrackingResonatorBank moved(std::move(source));
    twin.process(x.data() + 1000, x.size() - 1000);
    moved.process(x.data() + 1000, x.size() - 1000);
    CHECK(moved.rule() == TrackingRule::tangent);
    CHECK_SAME(moved.thresholdDb(), twin.thresholdDb());
    CHECK_SAME(moved.accumulatedPower(), twin.accumulatedPower());
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_SAME(moved.gamma(k), twin.gamma(k));
        CHECK_SAME(moved.power(k), twin.power(k));
        CHECK_SAME(moved.resonantFrequency(k), twin.resonantFrequency(k));
    }
}

TEST_CASE(tracking_reinit_matches_fresh_bank)
{
    const std::vector<float> first = semitones(200.0f, 21), second = semitones(330.0f, 7);
    const std::vector<float> x = signals::tone(4800, 48000.0, 452.0);
    fckfft::DynamicTrackingResonatorBank reused, fresh;
    CHECK(reused.init(first.data(), first.size(), 48000.0f, TrackingRule::ewma, -30.0f));
    reused.setPowerTimeConstant(0.2f);
    reused.process(x.data(), x.size());
    CHECK(reused.init(second.data(), second.size(), 44100.0f, TrackingRule::normalizedChord));
    CHECK(fresh.init(second.data(), second.size(), 44100.0f, TrackingRule::normalizedChord));
    CHECK(fresh.setPowerTimeConstant(0.2f));
    reused.process(x.data(), x.size());
    fresh.process(x.data(), x.size());
    CHECK(reused.size() == second.size());
    CHECK_SAME(reused.accumulatedPower(), fresh.accumulatedPower());
    for (std::size_t k = 0; k < second.size(); ++k)
    {
        CHECK_SAME(reused.power(k), fresh.power(k));
        CHECK_SAME(reused.resonantFrequency(k), fresh.resonantFrequency(k));
    }
}

TEST_CASE(tracking_setters_reject_invalid_values)
{
    const std::vector<float> f = semitones(300.0f, 5);
    fckfft::DynamicTrackingResonatorBank bank;
    CHECK(!bank.init(f.data(), f.size(), 48000.0f, static_cast<TrackingRule>(7)));
    CHECK(!bank.isInitialized());
    CHECK(bank.init(f.data(), f.size(), 48000.0f, TrackingRule::tangent, -40.0f));
    CHECK(!bank.init(f.data(), f.size(), 48000.0f, static_cast<TrackingRule>(4)));
    CHECK(bank.rule() == TrackingRule::tangent);
    CHECK(!bank.setRule(static_cast<TrackingRule>(4)));
    CHECK(bank.rule() == TrackingRule::tangent);
    CHECK(!bank.setThresholdDb(check::fromBits(0x7FC00000u)));
    CHECK(!bank.setThresholdDb(check::fromBits(0x7F800000u)));
    CHECK_NEAR(bank.thresholdDb(), -40.0, 1e-4);
    CHECK(!bank.setPowerTimeConstant(-1.0f));
    CHECK(!bank.setPowerTimeConstant(check::fromBits(0x7F800000u)));
    CHECK(!bank.setPowerTimeConstant(check::fromBits(0x7FC00000u)));
    CHECK(bank.powerTimeConstant() == 0.0f);
    CHECK(!bank.setAlpha(1, 0.0f));
    CHECK(!bank.setAlpha(5, 0.01f));
    CHECK(!bank.setBeta(1, 1.5f));
    CHECK(!bank.setNaturalFrequency(1, 30000.0f));
    CHECK(!bank.setNaturalFrequency(1, check::fromBits(0x7FC00000u)));
    CHECK_NEAR(bank.naturalFrequency(1), f[1], 0.0);
}

TEST_CASE(tracking_init_keeps_power_time_constant)
{
    const std::vector<float> f = semitones(300.0f, 9);
    const std::vector<float> x = signals::tone(4410, 44100.0, 452.0);
    fckfft::DynamicTrackingResonatorBank reused, fresh;
    CHECK(reused.setPowerTimeConstant(0.05f));
    CHECK(reused.init(f.data(), f.size(), 48000.0f, TrackingRule::ewma));
    CHECK(reused.init(f.data(), f.size(), 44100.0f, TrackingRule::ewma));
    CHECK(fresh.init(f.data(), f.size(), 44100.0f, TrackingRule::ewma));
    CHECK(fresh.setPowerTimeConstant(0.05f));
    CHECK(reused.powerTimeConstant() == 0.05f);
    reused.process(x.data(), x.size());
    fresh.process(x.data(), x.size());
    CHECK_SAME(reused.accumulatedPower(), fresh.accumulatedPower());
}

TEST_CASE(tracking_parameter_setters_reach_fresh_state)
{
    const std::vector<float> f = semitones(300.0f, 9);
    const std::vector<float> x = signals::tone(48000, 48000.0, f[4], 0.8);
    std::vector<float> alphas(f.size(), 0.01f), betas(f.size(), 0.02f);
    for (TrackingRule rule : {TrackingRule::none, TrackingRule::normalizedChord})
    {
        fckfft::DynamicTrackingResonatorBank changed, fresh;
        CHECK(changed.init(f.data(), alphas.data(), betas.data(), nullptr, f.size(), 48000.0f, rule));
        changed.process(x.data(), 4800);
        const float before = changed.magnitude(4);
        CHECK(changed.setAlpha(4, 0.004f));
        CHECK_NEAR(changed.magnitude(4), before, 0.0);
        double deviation = 0.0;
        for (std::size_t i = 4800; i < 5300; ++i)
        {
            changed.process(x[i]);
            deviation = std::max(deviation, std::fabs(static_cast<double>(changed.magnitude(4)) - before));
        }
        if (rule == TrackingRule::none)
            CHECK_NEAR(deviation, 0.0, 0.1 * before);
        CHECK(changed.setBeta(4, 0.008f));
        CHECK(changed.setNaturalFrequency(2, 330.0f));
        changed.process(x.data() + 5300, x.size() - 5300);
        alphas[4] = 0.004f;
        betas[4] = 0.008f;
        std::vector<float> natural = f;
        natural[2] = 330.0f;
        CHECK(fresh.init(natural.data(), alphas.data(), betas.data(), nullptr, f.size(), 48000.0f, rule));
        fresh.process(x.data(), x.size());
        alphas[4] = 0.01f;
        betas[4] = 0.02f;
        CHECK_NEAR(changed.naturalFrequency(2), 330.0, 0.0);
        for (std::size_t k = 0; k < f.size(); ++k)
        {
            CHECK_NEAR(changed.magnitude(k), fresh.magnitude(k), 1e-3);
            CHECK_NEAR(changed.resonantFrequency(k), fresh.resonantFrequency(k), 0.05);
        }
    }
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
        CHECK_SAME(dynamic.power(k), twin.power(k));
        CHECK_SAME(fixed.resonantFrequency(k), twin.resonantFrequency(k));
    }
}
