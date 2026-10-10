#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fckfft/fckfft.h>

namespace {

constexpr float kSampleRate = 48000.0f;

std::vector<float> testSignal(std::size_t count)
{
    std::vector<float> x(count);
    std::uint32_t seed = 1u;
    double phase = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const float noise = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
        x[i] = 0.5f * static_cast<float>(std::sin(phase)) + 0.05f * noise;
        phase += 2.0 * 3.14159265358979323846 * (220.0 + 660.0 * static_cast<double>(i) / count) / kSampleRate;
    }
    return x;
}

template <class Bank>
double measure(Bank& bank, const std::vector<float>& x, std::size_t block, int repetitions)
{
    std::vector<double> runs;
    for (int r = 0; r < repetitions; ++r)
    {
        bank.reset();
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i + block <= x.size(); i += block)
            bank.process(x.data() + i, block);
        const auto t1 = std::chrono::steady_clock::now();
        runs.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count()
                       / static_cast<double>(x.size() / block * block));
    }
    std::sort(runs.begin(), runs.end());
    return runs[runs.size() / 2];
}

void report(const char* kernel, std::size_t bins, std::size_t block, double ns)
{
    const double budget = 1e9 / kSampleRate;
    std::printf("%-10s %6zu %6zu %12.1f %10.3f %10.2f %9.1f%%\n", kernel, bins, block, ns,
                ns / static_cast<double>(bins), 1e3 / ns, 100.0 * ns / budget);
}

}

int main(int argc, char** argv)
{
    std::vector<std::size_t> binCounts = {88, 264, 440, 880};
    std::vector<std::size_t> blocks = {48, 256};
    int repetitions = 15;
    bool tracking = true;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--bins") == 0 && i + 1 < argc)
        {
            binCounts.clear();
            for (char* p = argv[++i]; *p;)
            {
                binCounts.push_back(std::strtoul(p, &p, 10));
                if (*p == ',')
                    ++p;
            }
        }
        else if (std::strcmp(argv[i], "--blocks") == 0 && i + 1 < argc)
        {
            blocks.clear();
            for (char* p = argv[++i]; *p;)
            {
                blocks.push_back(std::strtoul(p, &p, 10));
                if (*p == ',')
                    ++p;
            }
        }
        else if (std::strcmp(argv[i], "--repetitions") == 0 && i + 1 < argc)
            repetitions = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--no-tracking") == 0)
            tracking = false;
        else
        {
            std::printf("usage: %s [--bins 88,264] [--blocks 48,256] [--repetitions 15] [--no-tracking]\n", argv[0]);
            return 1;
        }
    }

    const std::vector<float> x = testSignal(static_cast<std::size_t>(kSampleRate / 4));
    std::printf("fixed lanes %zu, tracking lanes %zu, sample rate %.0f Hz, median of %d runs\n",
                fckfft::ResonatorBank::kLanes, fckfft::TrackingResonatorBank::kLanes, kSampleRate, repetitions);
    std::printf("%-10s %6s %6s %12s %10s %10s %10s\n", "kernel", "bins", "block", "ns/sample", "ns/bin", "Msamples/s",
                "realtime");
    for (std::size_t bins : binCounts)
    {
        if (bins == 0)
            continue;
        std::vector<float> f(bins);
        fckfft::logSpacedFrequencies(27.5f, 4186.009f, bins, f.data());
        for (std::size_t block : blocks)
        {
            if (block == 0)
                continue;
            fckfft::DynamicResonatorBank fixed;
            fixed.init(f.data(), bins, kSampleRate);
            report("fixed", bins, block, measure(fixed, x, block, repetitions));
            if (!tracking)
                continue;
            const char* names[4] = {"none", "ewma", "chord", "tangent"};
            for (int rule = 0; rule < 4; ++rule)
            {
                fckfft::DynamicTrackingResonatorBank bank;
                bank.init(f.data(), bins, kSampleRate, static_cast<fckfft::TrackingRule>(rule));
                report(names[rule], bins, block, measure(bank, x, block, repetitions));
            }
        }
    }
    return 0;
}
