#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <string>
#include <vector>

#include <fckfft/frequencies.h>
#include <fckfft/resonator_bank.h>

#include "check.h"
#include "npz.h"
#include "signals.h"

#ifndef FCKFFT_TEST_DATA_DIR
#define FCKFFT_TEST_DATA_DIR "tests/data"
#endif

static std::vector<float> semitoneBins(float first, std::size_t count)
{
    std::vector<float> f(count);
    fckfft::logUniformFrequencies(first, 12.0f, count, f.data());
    return f;
}

TEST_CASE(bank_rejects_invalid_arguments)
{
    std::vector<float> storage(fckfft::ResonatorBank::storageSize(4));
    const float freqs[4] = {100.0f, 200.0f, 300.0f, 400.0f};
    const float alphas[4] = {0.01f, 0.01f, 0.01f, 0.01f};
    const float badAlphas[4] = {0.01f, 0.0f, 0.01f, 0.01f};
    const float bigAlphas[4] = {0.01f, 1.5f, 0.01f, 0.01f};
    const float badFreqs[4] = {100.0f, 30000.0f, 300.0f, 400.0f};
    const float nanFreqs[4] = {100.0f, std::nanf(""), 300.0f, 400.0f};
    fckfft::ResonatorBank bank;
    CHECK(!bank.init(nullptr, storage.size(), freqs, alphas, 4, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size() - 1, freqs, alphas, 4, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), freqs, alphas, 0, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), freqs, alphas, 4, 0.0f));
    CHECK(!bank.init(storage.data(), storage.size(), freqs, alphas, 4, std::nanf("")));
    CHECK(!bank.init(storage.data(), storage.size(), freqs, badAlphas, 4, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), freqs, bigAlphas, 4, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), badFreqs, alphas, 4, 48000.0f));
    CHECK(!bank.init(storage.data(), storage.size(), nanFreqs, alphas, 4, 48000.0f));
    CHECK(!bank.isInitialized());
    bank.process(0.5f);
    CHECK(bank.init(storage.data(), storage.size(), freqs, alphas, 4, 48000.0f));
    CHECK(bank.isInitialized());
    CHECK(bank.size() == 4);
    CHECK(!bank.setAlpha(4, 0.1f));
    CHECK(!bank.setAlpha(0, 0.0f));
    CHECK(!bank.setBeta(0, 2.0f));
    CHECK(!bank.setFrequency(0, 24001.0f));
    fckfft::StaticResonatorBank<3> small;
    CHECK(!small.init(freqs, 4, 48000.0f));
    CHECK(small.init(freqs, 3, 48000.0f));
}

TEST_CASE(bank_matches_nofft_reference)
{
    const std::string path = std::string(FCKFFT_TEST_DATA_DIR) + "/chirp_88.npz";
    npz::Array freqs, alphas, signal, ref;
    CHECK(npz::load(path, "freqs", freqs));
    CHECK(npz::load(path, "alphas", alphas));
    CHECK(npz::load(path, "signal", signal));
    CHECK(npz::load(path, "ref", ref));
    if (ref.shape.size() != 3)
        return;
    const std::size_t frames = ref.shape[0], bins = ref.shape[2], hop = 256;
    CHECK(bins == freqs.data.size());
    fckfft::DynamicResonatorBank bank;
    CHECK(bank.init(freqs.data.data(), alphas.data.data(), bins, 44100.0f));
    std::vector<fckfft::Complex> out(frames * bins);
    CHECK(fckfft::analyze(bank, signal.data.data(), signal.data.size(), hop, out.data()) == frames);
    double maxComplex = 0.0, maxMagnitude = 0.0;
    for (std::size_t f = 0; f < frames; ++f)
        for (std::size_t k = 0; k < bins; ++k)
        {
            const double re = ref.data[f * 2 * bins + k], im = ref.data[f * 2 * bins + bins + k];
            const fckfft::Complex c = out[f * bins + k];
            maxComplex = std::max(maxComplex, std::max(std::fabs(c.re - re), std::fabs(c.im - im)));
            maxMagnitude = std::max(maxMagnitude, std::fabs(std::hypot(c.re, c.im) - std::hypot(re, im)));
        }
    CHECK_NEAR(maxComplex, 0.0, 1e-3);
    CHECK_NEAR(maxMagnitude, 0.0, 1e-5);
}

TEST_CASE(bank_matched_tone_power)
{
    const float sr = 44100.0f;
    const std::vector<float> f = {220.0f, 440.0f, 880.0f};
    fckfft::DynamicResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), sr));
    const std::vector<float> x = signals::tone(2 * 44100, sr, 440.0);
    bank.process(x.data(), x.size());
    CHECK_NEAR(bank.power(1), 0.25, 0.01);
    CHECK_NEAR(bank.magnitude(1), 0.5, 0.01);
    CHECK(bank.power(1) > 10.0f * bank.power(0));
    CHECK(bank.power(1) > 10.0f * bank.power(2));
    float powers[3], mags[3], phases[3];
    fckfft::Complex values[3];
    bank.powers(powers);
    bank.magnitudes(mags);
    bank.phases(phases);
    bank.values(values);
    for (int k = 0; k < 3; ++k)
    {
        CHECK_NEAR(powers[k], bank.power(k), 0.0);
        CHECK_NEAR(mags[k], std::sqrt(powers[k]), 1e-7);
        CHECK_NEAR(phases[k], std::atan2(values[k].im, values[k].re), 1e-7);
    }
}

TEST_CASE(bank_block_size_independent)
{
    const std::vector<float> f = semitoneBins(55.0f, 61);
    const std::vector<float> x = signals::noise(9600, 7u, 0.5f);
    std::vector<fckfft::Complex> reference(f.size()), other(f.size());
    fckfft::DynamicResonatorBank a;
    CHECK(a.init(f.data(), f.size(), 48000.0f));
    a.process(x.data(), x.size());
    a.values(reference.data());
    for (std::size_t block : {std::size_t(1), std::size_t(2), std::size_t(7), std::size_t(48), std::size_t(257)})
    {
        fckfft::DynamicResonatorBank b;
        CHECK(b.init(f.data(), f.size(), 48000.0f));
        for (std::size_t i = 0; i < x.size(); i += block)
            b.process(x.data() + i, std::min(block, x.size() - i));
        b.values(other.data());
        for (std::size_t k = 0; k < f.size(); ++k)
        {
            CHECK_NEAR(other[k].re, reference[k].re, 0.0);
            CHECK_NEAR(other[k].im, reference[k].im, 0.0);
        }
    }
}

TEST_CASE(bank_stride_matches_mono)
{
    const std::vector<float> f = semitoneBins(110.0f, 37);
    const std::vector<float> left = signals::noise(4801, 3u);
    const std::vector<float> right = signals::noise(4801, 9u);
    std::vector<float> interleaved(2 * left.size());
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        interleaved[2 * i] = left[i];
        interleaved[2 * i + 1] = right[i];
    }
    fckfft::DynamicResonatorBank mono, strided;
    CHECK(mono.init(f.data(), f.size(), 48000.0f));
    CHECK(strided.init(f.data(), f.size(), 48000.0f));
    mono.process(right.data(), right.size());
    strided.process(interleaved.data() + 1, right.size(), 2);
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_NEAR(strided.value(k).re, mono.value(k).re, 0.0);
        CHECK_NEAR(strided.value(k).im, mono.value(k).im, 0.0);
    }
}

TEST_CASE(bank_padding_lanes_are_inert)
{
    std::vector<float> all(3 * fckfft::ResonatorBank::kLanes + 1);
    fckfft::logSpacedFrequencies(80.0f, 20000.0f, all.size(), all.data());
    const std::vector<float> x = signals::noise(3000, 11u);
    fckfft::DynamicResonatorBank full;
    CHECK(full.init(all.data(), all.size(), 48000.0f));
    full.process(x.data(), x.size());
    for (std::size_t n = 1; n <= all.size(); n += 3)
    {
        fckfft::DynamicResonatorBank part;
        CHECK(part.init(all.data(), n, 48000.0f));
        part.process(x.data(), x.size());
        for (std::size_t k = 0; k < n; ++k)
        {
            CHECK_NEAR(part.value(k).re, full.value(k).re, 0.0);
            CHECK_NEAR(part.value(k).im, full.value(k).im, 0.0);
        }
    }
}

TEST_CASE(bank_reset_and_setters)
{
    const std::vector<float> f = semitoneBins(200.0f, 5);
    const std::vector<float> x = signals::tone(4800, 48000.0, 300.0);
    fckfft::DynamicResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f));
    bank.process(x.data(), x.size());
    CHECK(bank.magnitude(2) > 0.0f);
    bank.reset();
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK(bank.value(k).re == 0.0f && bank.value(k).im == 0.0f);
    bank.process(x.data(), x.size());
    const fckfft::Complex before = bank.value(1);
    CHECK(bank.setAlpha(1, 0.02f));
    CHECK_NEAR(bank.alpha(1), 0.02f, 0.0);
    CHECK_NEAR(bank.value(1).re, before.re, 1e-6 * std::fabs(before.re));
    CHECK_NEAR(bank.timeConstant(1), fckfft::timeConstantFromAlpha(0.02f, 48000.0f), 1e-9);
    CHECK(bank.setBeta(1, 0.5f));
    CHECK_NEAR(bank.beta(1), 0.5f, 0.0);
    CHECK(bank.setFrequency(4, 1000.0f));
    CHECK_NEAR(bank.frequency(4), 1000.0, 0.0);
    const std::vector<float> y = signals::tone(48000, 48000.0, 1000.0);
    bank.reset();
    bank.process(y.data(), y.size());
    CHECK_NEAR(bank.magnitude(4), 0.5, 0.01);
}

TEST_CASE(bank_alpha_change_keeps_resonance_continuous)
{
    const float f[1] = {500.0f};
    const std::vector<float> x = signals::tone(9600, 48000.0, 500.0);
    fckfft::DynamicResonatorBank a, b;
    CHECK(a.init(f, 1, 48000.0f));
    CHECK(b.init(f, 1, 48000.0f));
    a.process(x.data(), 4800);
    b.process(x.data(), 4800);
    const float newAlpha = 0.5f * a.alpha(0);
    CHECK(a.setAlpha(0, newAlpha));
    a.process(x.data() + 4800, 1);
    b.process(x.data() + 4800, 1);
    CHECK_NEAR(a.magnitude(0), b.magnitude(0), 2e-3 * b.magnitude(0));
}

TEST_CASE(bank_long_run_stability)
{
    const std::vector<float> f = semitoneBins(100.0f, 16);
    fckfft::DynamicResonatorBank aged, fresh;
    CHECK(aged.init(f.data(), f.size(), 48000.0f));
    CHECK(fresh.init(f.data(), f.size(), 48000.0f));
    const std::vector<float> n = signals::noise(48000, 5u);
    for (int second = 0; second < 30; ++second)
        aged.process(n.data(), n.size());
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK(std::isfinite(aged.magnitude(k)) && aged.magnitude(k) < 1.0f);
    const std::vector<float> silence(96000, 0.0f);
    aged.process(silence.data(), silence.size());
    const std::vector<float> t = signals::tone(48000, 48000.0, 400.0);
    aged.process(t.data(), t.size());
    fresh.process(t.data(), t.size());
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK_NEAR(aged.magnitude(k), fresh.magnitude(k), 1e-4);
}

TEST_CASE(bank_silence_decays_to_zero)
{
    const std::vector<float> f = semitoneBins(1000.0f, 8);
    fckfft::DynamicResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f));
    const std::vector<float> n = signals::noise(4800, 2u);
    bank.process(n.data(), n.size());
    const std::vector<float> silence(48000, 0.0f);
    for (int i = 0; i < 8; ++i)
        bank.process(silence.data(), silence.size());
    for (std::size_t k = 0; k < f.size(); ++k)
        CHECK(bank.power(k) == 0.0f);
}

TEST_CASE(bank_static_and_dynamic_agree)
{
    const std::vector<float> f = semitoneBins(60.0f, 30);
    const std::vector<float> x = signals::noise(2000, 4u);
    fckfft::StaticResonatorBank<32> s;
    fckfft::DynamicResonatorBank d;
    std::vector<float> storage(fckfft::ResonatorBank::storageSize(f.size()));
    fckfft::ResonatorBank e;
    CHECK(s.init(f.data(), f.size(), 48000.0f));
    CHECK(d.init(f.data(), f.size(), 48000.0f));
    CHECK(e.init(storage.data(), storage.size(), f.data(), f.size(), 48000.0f));
    s.process(x.data(), x.size());
    d.process(x.data(), x.size());
    e.process(x.data(), x.size());
    for (std::size_t k = 0; k < f.size(); ++k)
    {
        CHECK_NEAR(s.value(k).re, d.value(k).re, 0.0);
        CHECK_NEAR(e.value(k).im, d.value(k).im, 0.0);
    }
}

TEST_CASE(bank_analyze_frames)
{
    const std::vector<float> f = semitoneBins(300.0f, 4);
    const std::vector<float> x = signals::noise(1000, 8u);
    fckfft::DynamicResonatorBank a, b;
    CHECK(a.init(f.data(), f.size(), 48000.0f));
    CHECK(b.init(f.data(), f.size(), 48000.0f));
    std::vector<fckfft::Complex> frames(3 * f.size());
    CHECK(fckfft::analyze(a, x.data(), x.size(), 300, frames.data()) == 3);
    CHECK(fckfft::analyze(a, x.data(), x.size(), 0, frames.data()) == 0);
    for (std::size_t frame = 0; frame < 3; ++frame)
    {
        b.process(x.data() + frame * 300, 300);
        for (std::size_t k = 0; k < f.size(); ++k)
            CHECK_NEAR(frames[frame * f.size() + k].re, b.value(k).re, 1e-6 * std::fabs(b.value(k).re));
    }
}

TEST_CASE(bank_frequency_selectivity)
{
    const std::vector<float> f = semitoneBins(220.0f, 25);
    fckfft::DynamicResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f));
    for (std::size_t target = 0; target < f.size(); target += 6)
    {
        bank.reset();
        const std::vector<float> x = signals::tone(48000, 48000.0, f[target]);
        bank.process(x.data(), x.size());
        std::size_t best = 0;
        for (std::size_t k = 1; k < f.size(); ++k)
            if (bank.power(k) > bank.power(best))
                best = k;
        CHECK(best == target);
    }
}

TEST_CASE(bank_recovers_from_non_finite_input)
{
    const std::vector<float> f = semitoneBins(200.0f, 19);
    const float bad[2] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    for (float poison : bad)
    {
        fckfft::DynamicResonatorBank bank, clean;
        CHECK(bank.init(f.data(), f.size(), 48000.0f));
        CHECK(clean.init(f.data(), f.size(), 48000.0f));
        std::vector<float> x = signals::tone(9600, 48000.0, 440.0);
        x[1000] = poison;
        bank.process(x.data(), 1000 + fckfft::ResonatorBank::kRecoveryInterval);
        for (std::size_t k = 0; k < f.size(); ++k)
            CHECK(std::isfinite(bank.power(k)));
        bank.reset();
        bank.process(x.data() + 2000, 7600);
        clean.process(x.data() + 2000, 7600);
        for (std::size_t k = 0; k < f.size(); ++k)
            CHECK_NEAR(bank.value(k).re, clean.value(k).re, 0.0);
    }
}

TEST_CASE(bank_move_and_contract)
{
    static_assert(std::is_nothrow_move_constructible<fckfft::DynamicResonatorBank>::value, "dynamic banks move without throwing");
    static_assert(std::is_nothrow_move_assignable<fckfft::DynamicResonatorBank>::value, "dynamic banks move-assign without throwing");
    static_assert(!std::is_copy_constructible<fckfft::DynamicResonatorBank>::value, "banks are not copyable");
    static_assert(!std::is_move_constructible<fckfft::StaticResonatorBank<4>>::value, "static banks own inline storage");
    static_assert(noexcept(std::declval<fckfft::DynamicResonatorBank&>().process(nullptr, 0)), "process does not throw");
    const float f[4] = {220.0f, 330.0f, 440.0f, 660.0f};
    const std::vector<float> x = signals::tone(4800, 48000.0, 440.0);
    fckfft::DynamicResonatorBank reference;
    CHECK(reference.init(f, 4, 48000.0f));
    reference.process(x.data(), x.size());
    std::vector<fckfft::DynamicResonatorBank> channels;
    for (int c = 0; c < 3; ++c)
    {
        fckfft::DynamicResonatorBank bank;
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
    fckfft::DynamicResonatorBank assigned;
    assigned = std::move(channels[0]);
    CHECK(assigned.isInitialized() && !channels[0].isInitialized());
    CHECK_NEAR(assigned.power(2), reference.power(2), 0.0);
}

TEST_CASE(bank_matches_double_model_on_random_configurations)
{
    std::uint32_t seed = 12345u;
    const auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    const double sr = 48000.0;
    double worst = 0.0, worstComplex = 0.0;
    for (int trial = 0; trial < 120; ++trial)
    {
        const std::size_t n = 1 + static_cast<std::size_t>(next() * 40.0);
        std::vector<float> f(n), a(n), b(n);
        for (std::size_t k = 0; k < n; ++k)
        {
            const double r = next();
            f[k] = r < 0.05 ? 0.0f : (r > 0.95 ? static_cast<float>(sr / 2) : static_cast<float>(next() * sr / 2));
            a[k] = static_cast<float>(std::pow(10.0, -3.0 * next()));
            b[k] = static_cast<float>(std::pow(10.0, -3.0 * next()));
        }
        const std::size_t length = 2000 + static_cast<std::size_t>(next() * 2000.0);
        const std::size_t stride = 1 + static_cast<std::size_t>(next() * 3.0);
        std::vector<float> x(length * stride);
        for (float& v : x)
            v = static_cast<float>(2.0 * next() - 1.0);
        fckfft::DynamicResonatorBank bank;
        CHECK(bank.init(f.data(), a.data(), b.data(), n, static_cast<float>(sr)));
        for (std::size_t i = 0; i < length;)
        {
            const std::size_t block = std::min<std::size_t>(length - i, 1 + static_cast<std::size_t>(next() * 300.0));
            bank.process(x.data() + i * stride, block, stride);
            i += block;
        }
        for (std::size_t k = 0; k < n; ++k)
        {
            const double omega = -2.0 * signals::kPi * static_cast<double>(f[k]) / sr;
            double rr = 0.0, ri = 0.0, sr2 = 0.0, si = 0.0;
            for (std::size_t i = 0; i < length; ++i)
            {
                const double s = x[i * stride];
                const double phase = omega * static_cast<double>(i);
                rr += a[k] * (s * std::cos(phase) - rr);
                ri += a[k] * (s * std::sin(phase) - ri);
                sr2 += b[k] * (rr - sr2);
                si += b[k] * (ri - si);
            }
            const double error = std::fabs(static_cast<double>(bank.magnitude(k)) - std::hypot(sr2, si));
            worst = std::max(worst, error);
            const fckfft::Complex v = bank.value(k);
            worstComplex = std::max(worstComplex, std::max(std::fabs(v.re - sr2), std::fabs(v.im - si)));
        }
    }
    CHECK_NEAR(worst, 0.0, 1e-5);
    CHECK_NEAR(worstComplex, 0.0, 1e-5);
}
