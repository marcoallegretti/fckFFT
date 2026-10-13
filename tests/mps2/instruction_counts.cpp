#include <cmath>
#include <cstdint>
#include <cstdio>

#include <fckfft/fckfft.h>

namespace {

volatile std::uint32_t* const kSysTickControl = reinterpret_cast<volatile std::uint32_t*>(0xE000E010u);
volatile std::uint32_t* const kSysTickReload = reinterpret_cast<volatile std::uint32_t*>(0xE000E014u);
volatile std::uint32_t* const kSysTickCurrent = reinterpret_cast<volatile std::uint32_t*>(0xE000E018u);

constexpr std::size_t kBlock = 48;
constexpr std::size_t kBlocks = 16;
constexpr std::size_t kWarmup = 4800;
constexpr std::size_t kLength = kWarmup + kBlock * kBlocks;
constexpr float kSampleRate = 48000.0f;
constexpr int kSignals = 4;
constexpr std::uint32_t kToleranceTicks = 2;

const char* const kSignalNames[kSignals] = {"silence", "noise", "tone", "decay"};

float input[kLength];
float frequencies[88];
fckfft::StaticResonatorBank<88> fixedBank;
fckfft::StaticTrackingResonatorBank<49> trackingBank;

void render(int signal)
{
    std::uint32_t seed = 12345u;
    for (std::size_t i = 0; i < kLength; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const float t = static_cast<float>(i) / kSampleRate;
        switch (signal)
        {
        case 1:
            input[i] = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
            break;
        case 2:
            input[i] = std::sin(6.2831853f * 440.0f * t);
            break;
        case 3:
            input[i] = i == 0 ? 1.0f : 0.0f;
            break;
        default:
            input[i] = 0.0f;
            break;
        }
    }
}

template <class Bank>
std::uint32_t ticks(Bank& bank)
{
    bank.process(input, kWarmup);
    const std::uint32_t start = *kSysTickCurrent;
    for (std::size_t b = 0; b < kBlocks; ++b)
        bank.process(input + kWarmup + b * kBlock, kBlock);
    const std::uint32_t end = *kSysTickCurrent;
    return (start - end) & 0xFFFFFFu;
}

template <class Init>
bool check(const char* name, std::uint32_t budget, Init init)
{
    std::uint32_t counts[kSignals];
    std::uint32_t lowest = 0xFFFFFFFFu, highest = 0;
    for (int signal = 0; signal < kSignals; ++signal)
    {
        render(signal);
        counts[signal] = init();
        lowest = counts[signal] < lowest ? counts[signal] : lowest;
        highest = counts[signal] > highest ? counts[signal] : highest;
    }
    const bool ok = highest - lowest <= kToleranceTicks && highest <= budget + budget / 50;
    std::printf("%s %-8s", ok ? "PASS" : "FAIL", name);
    for (int signal = 0; signal < kSignals; ++signal)
        std::printf("  %s %lu", kSignalNames[signal], static_cast<unsigned long>(counts[signal]));
    std::printf("  ticks, budget %lu\n", static_cast<unsigned long>(budget));
    return ok;
}

}

int main()
{
    *kSysTickReload = 0xFFFFFFu;
    *kSysTickCurrent = 0;
    *kSysTickControl = 5u;
    fckfft::logSpacedFrequencies(27.5f, 4186.0f, 88, frequencies);
    bool ok = check("fixed", 24198, [] {
        fixedBank.init(frequencies, 88, kSampleRate);
        return ticks(fixedBank);
    });
    const char* const names[4] = {"none", "ewma", "chord", "tangent"};
    const std::uint32_t budgets[4] = {39203, 139493, 103099, 101678};
    for (int rule = 0; rule < 4; ++rule)
        ok = check(names[rule], budgets[rule], [rule] {
                 trackingBank.init(frequencies, 49, kSampleRate, static_cast<fckfft::TrackingRule>(rule));
                 return ticks(trackingBank);
             }) && ok;
    return ok ? 0 : 1;
}
