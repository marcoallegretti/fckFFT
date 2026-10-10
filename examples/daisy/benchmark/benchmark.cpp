#include "daisy_seed.h"

#include <fckfft/fckfft.h>

#include <cmath>

using namespace daisy;

namespace {

constexpr std::size_t kBlock = 48;
constexpr std::size_t kInputLength = 64 * kBlock;
constexpr float kSampleRate = 48000.0f;
constexpr int kRuns = 64;

DaisySeed hw;
DTCM_MEM_SECTION fckfft::StaticResonatorBank<256> fixedBank;
DTCM_MEM_SECTION fckfft::StaticTrackingResonatorBank<64> trackingBank;
fckfft::StaticResonatorBank<256> sramBank;
DTCM_MEM_SECTION float input[kInputLength];
float frequencies[256];

void enableCycleCounter()
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

template <class Bank>
void measure(Bank& bank, std::uint32_t& best, std::uint32_t& average)
{
    bank.reset();
    bank.process(input, kBlock);
    best = 0xFFFFFFFFu;
    std::uint64_t total = 0;
    for (int run = 0; run < kRuns; ++run)
    {
        const float* block = input + (static_cast<std::size_t>(run) % (kInputLength / kBlock)) * kBlock;
        const std::uint32_t start = DWT->CYCCNT;
        bank.process(block, kBlock);
        const std::uint32_t elapsed = DWT->CYCCNT - start;
        best = elapsed < best ? elapsed : best;
        total += elapsed;
    }
    average = static_cast<std::uint32_t>(total / kRuns);
}

void report(const char* kernel, std::size_t bins, std::uint32_t best, std::uint32_t average)
{
    const unsigned long perBin = static_cast<unsigned long>(
        std::lround(100.0 * static_cast<double>(best) / static_cast<double>(kBlock * bins)));
    const unsigned long load = static_cast<unsigned long>(std::lround(
        1000.0 * static_cast<double>(average) * (kSampleRate / kBlock) / static_cast<double>(SystemCoreClock)));
    hw.PrintLine("%-8s %4u bins  best %7lu  avg %7lu cycles/block  %lu.%02lu cycles/bin-sample  %lu.%lu%% CPU",
                 kernel, static_cast<unsigned>(bins), static_cast<unsigned long>(best),
                 static_cast<unsigned long>(average), perBin / 100, perBin % 100, load / 10, load % 10);
}

}

int main()
{
    hw.Init(true);
    hw.StartLog(true);
    enableCycleCounter();

    std::uint32_t seed = 1u;
    for (std::size_t i = 0; i < kInputLength; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const float noise = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
        input[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / kSampleRate) + 0.05f * noise;
    }

    hw.PrintLine("fckFFT benchmark: core clock %lu Hz, block %u, sample rate %u Hz",
                 static_cast<unsigned long>(SystemCoreClock), static_cast<unsigned>(kBlock),
                 static_cast<unsigned>(kSampleRate));

    const std::size_t fixedSizes[] = {16, 32, 64, 88, 128, 256};
    for (std::size_t bins : fixedSizes)
    {
        fckfft::logSpacedFrequencies(27.5f, 4186.0f, bins, frequencies);
        fixedBank.init(frequencies, bins, kSampleRate);
        std::uint32_t best, average;
        measure(fixedBank, best, average);
        report("fixed", bins, best, average);
        sramBank.init(frequencies, bins, kSampleRate);
        measure(sramBank, best, average);
        report("sram", bins, best, average);
    }

    const char* names[4] = {"none", "ewma", "chord", "tangent"};
    const std::size_t trackingSizes[] = {16, 32, 64};
    for (std::size_t bins : trackingSizes)
    {
        fckfft::logSpacedFrequencies(27.5f, 4186.0f, bins, frequencies);
        for (int rule = 0; rule < 4; ++rule)
        {
            trackingBank.init(frequencies, bins, kSampleRate, static_cast<fckfft::TrackingRule>(rule));
            std::uint32_t best, average;
            measure(trackingBank, best, average);
            report(names[rule], bins, best, average);
        }
    }

    hw.PrintLine("done");
    for (;;)
    {
        hw.SetLed(true);
        System::Delay(250);
        hw.SetLed(false);
        System::Delay(250);
    }
}
