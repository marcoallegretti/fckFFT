#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <fckfft/fckfft.h>

namespace {

class Reader
{
public:
    Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

    std::uint8_t byte()
    {
        return pos_ < size_ ? data_[pos_++] : 0;
    }

    float value()
    {
        std::uint32_t bits = 0;
        for (int i = 0; i < 4; ++i)
            bits = (bits << 8) | byte();
        float v;
        std::memcpy(&v, &bits, sizeof v);
        return v;
    }

    float unit() { return static_cast<float>(byte()) / 255.0f; }
    bool empty() const { return pos_ >= size_; }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    Reader in(data, size);
    const std::size_t n = 1 + in.byte() % 48;
    const bool raw = in.byte() & 1;
    const float sampleRate = raw ? in.value() : 8000.0f + 1000.0f * static_cast<float>(in.byte());
    std::vector<float> f(n), a(n), b(n), g(n);
    for (std::size_t k = 0; k < n; ++k)
    {
        f[k] = raw ? in.value() : in.unit() * 0.5f * sampleRate;
        a[k] = raw ? in.value() : 1e-4f + in.unit();
        b[k] = raw ? in.value() : 1e-4f + in.unit();
        g[k] = raw ? in.value() : in.unit();
    }
    const auto rule = static_cast<fckfft::TrackingRule>(in.byte() % 4);
    const float thresholdDb = raw ? in.value() : -100.0f + in.unit() * 100.0f;

    fckfft::DynamicResonatorBank fixed;
    fckfft::DynamicTrackingResonatorBank tracking;
    const bool fixedReady = fixed.init(f.data(), a.data(), b.data(), n, sampleRate);
    const bool trackingReady = tracking.init(f.data(), a.data(), b.data(), g.data(), n, sampleRate, rule, thresholdDb);
    if (in.byte() & 1)
        tracking.setPowerTimeConstant(in.value());

    std::vector<float> samples;
    while (!in.empty() && samples.size() < 4096)
        samples.push_back(in.value());
    const std::size_t stride = 1 + samples.size() % 3;
    const std::size_t frames = samples.size() / stride;
    std::size_t offset = 0;
    std::size_t step = 1;
    while (offset < frames)
    {
        const std::size_t count = frames - offset < step ? frames - offset : step;
        fixed.process(samples.data() + offset * stride, count, stride);
        tracking.process(samples.data() + offset * stride, count, stride);
        offset += count;
        step = step * 3 + 1;
    }

    float sink = 0.0f;
    std::vector<float> out(n);
    std::vector<fckfft::Complex> values(n);
    if (fixedReady)
    {
        fixed.powers(out.data());
        fixed.phases(out.data());
        fixed.values(values.data());
        for (std::size_t k = 0; k < n; ++k)
            sink += fixed.magnitude(k) + fixed.timeConstant(k);
        fixed.setAlpha(n - 1, samples.empty() ? 0.5f : samples[0]);
        fixed.setBeta(0, samples.empty() ? 0.5f : samples.back());
        fixed.setFrequency(0, samples.empty() ? 0.0f : samples[samples.size() / 2]);
        fixed.process(samples.data(), frames, stride);
    }
    if (trackingReady)
    {
        tracking.instantaneousFrequencies(out.data());
        tracking.resonantFrequencies(out.data());
        tracking.magnitudes(out.data());
        tracking.values(values.data());
        for (std::size_t k = 0; k < n; ++k)
            sink += tracking.deltaPhase(k) + tracking.phase(k);
        tracking.setGamma(0, samples.empty() ? 0.5f : samples[0]);
        tracking.setThresholdDb(samples.empty() ? -60.0f : samples.back());
        tracking.setRule(static_cast<fckfft::TrackingRule>((static_cast<unsigned>(rule) + 1) % 4));
        tracking.process(samples.data(), frames, stride);
    }
    fckfft::DynamicResonatorBank moved(std::move(fixed));
    moved.process(samples.data(), frames, stride);
    fixed.process(samples.data(), frames, stride);
    volatile float keep = sink;
    (void)keep;
    return 0;
}
