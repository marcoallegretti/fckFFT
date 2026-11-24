#include <fckfft/fckfft.h>

#include <vector>

float exerciseApi(const float* x, std::size_t n);

float exerciseApi(const float* x, std::size_t n)
{
    constexpr std::size_t kBins = 40;
    constexpr float kSampleRate = 48000.0f;
    float f[kBins], out[kBins];
    fckfft::Complex values[kBins];
    fckfft::logSpacedFrequencies(30.0f, 8000.0f, kBins, f);

    fckfft::StaticResonatorBank<kBins> fixed;
    fckfft::DynamicResonatorBank dynamic;
    fixed.init(f, kBins, kSampleRate);
    dynamic.init(f, kBins, kSampleRate);
    fixed.process(x, n);
    dynamic.process(x, n / 2, 2);
    fixed.setFrequency(1, 100.0f);
    fixed.setAlpha(1, 0.01f);
    fixed.setBeta(1, 0.01f);
    fixed.values(values);
    fixed.powers(out);
    fixed.magnitudes(out);
    fixed.phases(out);

    fckfft::StaticTrackingResonatorBank<kBins> tracking;
    fckfft::DynamicTrackingResonatorBank untracked;
    tracking.init(f, kBins, kSampleRate, fckfft::TrackingRule::normalizedChord);
    untracked.init(f, kBins, kSampleRate, fckfft::TrackingRule::none);
    tracking.setRule(fckfft::TrackingRule::tangent);
    tracking.setThresholdDb(-50.0f);
    tracking.setPowerTimeConstant(0.0f);
    tracking.setPowerTimeConstant(0.1f);
    tracking.setGamma(0, 0.001f);
    tracking.setNaturalFrequency(0, 50.0f);
    tracking.setAlpha(0, 0.01f);
    tracking.setBeta(0, 0.01f);
    tracking.process(x, n);
    untracked.process(x, n);
    tracking.values(values);
    tracking.powers(out);
    tracking.magnitudes(out);
    tracking.phases(out);
    tracking.resonantFrequencies(out);
    tracking.instantaneousFrequencies(out);

    std::vector<float> storage(fckfft::ResonatorBank::storageSize(kBins));
    fckfft::ResonatorBank borrowed;
    borrowed.init(storage.data(), storage.size(), f, kBins, kSampleRate);
    std::vector<fckfft::Complex> frames(kBins * (n / 64 + 1));
    fckfft::analyze(borrowed, x, n, 64, frames.data());

    float m[kBins];
    fckfft::melFrequencies(30.0f, 8000.0f, kBins, m, true);
    fckfft::linearSpacedFrequencies(0.0f, 100.0f, kBins, m);
    fckfft::logUniformFrequencies(27.5f, 12.0f, kBins, m);
    fckfft::heuristicAlphas(f, kBins, kSampleRate, m);
    fckfft::heuristicGammas(f, kBins, kSampleRate, m);

    return out[0] + values[0].re + m[0] + fckfft::alphaForQ(440.0f, 10.0f, kSampleRate)
           + fckfft::alphaForBandwidth(10.0f, kSampleRate) + fckfft::alphaFromTimeConstant(0.0f, kSampleRate)
           + fckfft::timeConstantFromAlpha(0.01f, kSampleRate) + fckfft::midiToHz(60.0f) + fckfft::hzToMidi(440.0f)
           + fckfft::melToHz(fckfft::hzToMel(300.0f)) + fixed.phase(0) + fixed.magnitude(0) + fixed.timeConstant(0)
           + tracking.deltaPhase(0) + tracking.thresholdDb() + borrowed.power(0)
           + static_cast<float>(fckfft::supportedIsa() == fckfft::isaLimit());
}
