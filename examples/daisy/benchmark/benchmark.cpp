#if defined(FCKFFT_PATCH_SM)
#include "daisy_patch_sm.h"
#else
#include "daisy_seed.h"
#endif

#include <fckfft/fckfft.h>

#include <cmath>

using namespace daisy;

namespace {

#if defined(FCKFFT_PATCH_SM)
using Board = patch_sm::DaisyPatchSM;
constexpr const char* kBoardName = "Daisy Patch SM";
#else
using Board = DaisySeed;
constexpr const char* kBoardName = "Daisy Seed";
#endif

constexpr std::size_t kBlock = 48;
constexpr std::size_t kInputLength = 64 * kBlock;
constexpr float kSampleRate = 48000.0f;
constexpr int kRuns = 256;

Board hw;
DTCM_MEM_SECTION fckfft::StaticResonatorBank<256> fixedBank;
DTCM_MEM_SECTION fckfft::StaticTrackingResonatorBank<64> trackingBank;
fckfft::StaticResonatorBank<256> sramBank;
DTCM_MEM_SECTION float input[kInputLength];
float frequencies[256];
float scratch[256];
fckfft::Complex values[256];

struct Timing
{
    std::uint32_t best = 0xFFFFFFFFu;
    std::uint32_t worst = 0;
    std::uint64_t total = 0;
    std::uint32_t runs = 0;

    void add(std::uint32_t cycles)
    {
        best = cycles < best ? cycles : best;
        worst = cycles > worst ? cycles : worst;
        total += cycles;
        ++runs;
    }

    std::uint32_t average() const { return static_cast<std::uint32_t>(total / runs); }
};

void initBoard()
{
#if defined(FCKFFT_PATCH_SM)
    hw.Init();
#else
    hw.Init(true);
#endif
}

void enableCycleCounter()
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->LAR = 0xC5ACCE55u;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

Timing measure(int runs, void (*f)(int))
{
    Timing timing;
    for (int run = 0; run < runs; ++run)
    {
        __disable_irq();
        const std::uint32_t start = DWT->CYCCNT;
        f(run);
        const std::uint32_t elapsed = DWT->CYCCNT - start;
        __enable_irq();
        timing.add(elapsed);
    }
    return timing;
}

const float* block(int run)
{
    return input + (static_cast<std::size_t>(run) % (kInputLength / kBlock)) * kBlock;
}

unsigned long hundredths(double value)
{
    return static_cast<unsigned long>(std::lround(100.0 * value));
}

unsigned long load(std::uint32_t cyclesPerBlock)
{
    return static_cast<unsigned long>(std::lround(1000.0 * static_cast<double>(cyclesPerBlock) * (kSampleRate / kBlock)
                                                  / static_cast<double>(SystemCoreClock)));
}

void reportProcess(const char* kernel, std::size_t bins, const Timing& t)
{
    const unsigned long perBin = hundredths(static_cast<double>(t.best) / static_cast<double>(kBlock * bins));
    const unsigned long average = load(t.average()), worst = load(t.worst);
    hw.PrintLine("%-8s %4u bins  block best %7lu avg %7lu worst %7lu  %lu.%02lu cycles/bin-sample  CPU avg %lu.%lu%% "
                 "worst %lu.%lu%%",
                 kernel, static_cast<unsigned>(bins), static_cast<unsigned long>(t.best),
                 static_cast<unsigned long>(t.average()), static_cast<unsigned long>(t.worst), perBin / 100,
                 perBin % 100, average / 10, average % 10, worst / 10, worst % 10);
}

void reportCall(const char* name, std::size_t bins, const Timing& t)
{
    const unsigned long perBin = hundredths(static_cast<double>(t.best) / static_cast<double>(bins));
    hw.PrintLine("%-24s %4u bins  best %7lu worst %7lu cycles  %lu.%02lu cycles/bin", name,
                 static_cast<unsigned>(bins), static_cast<unsigned long>(t.best), static_cast<unsigned long>(t.worst),
                 perBin / 100, perBin % 100);
}

template <class Bank>
Bank* measured = nullptr;

template <class Bank>
void processBlock(int run)
{
    measured<Bank>->process(block(run), kBlock);
}

template <class Bank>
Timing measureProcess(Bank& bank)
{
    bank.reset();
    measured<Bank> = &bank;
    return measure(kRuns, &processBlock<Bank>);
}

void measureFixed()
{
    const std::size_t sizes[] = {16, 32, 64, 88, 128, 256};
    for (std::size_t bins : sizes)
    {
        fckfft::logSpacedFrequencies(27.5f, 4186.0f, bins, frequencies);
        fixedBank.init(frequencies, bins, kSampleRate);
        reportProcess("fixed", bins, measureProcess(fixedBank));
        sramBank.init(frequencies, bins, kSampleRate);
        reportProcess("sram", bins, measureProcess(sramBank));
    }
}

void measureTracking()
{
    const char* names[4] = {"none", "ewma", "chord", "tangent"};
    const std::size_t sizes[] = {16, 32, 49, 64};
    for (std::size_t bins : sizes)
    {
        fckfft::logSpacedFrequencies(27.5f, 4186.0f, bins, frequencies);
        for (int rule = 0; rule < 4; ++rule)
        {
            trackingBank.init(frequencies, bins, kSampleRate, static_cast<fckfft::TrackingRule>(rule));
            reportProcess(names[rule], bins, measureProcess(trackingBank));
        }
    }
}

void measureReadouts()
{
    const std::size_t sizes[] = {88, 256};
    for (std::size_t bins : sizes)
    {
        fckfft::logSpacedFrequencies(27.5f, 4186.0f, bins, frequencies);
        fixedBank.init(frequencies, bins, kSampleRate);
        fixedBank.process(input, kInputLength);
        reportCall("fixed values", bins, measure(kRuns, [](int) { fixedBank.values(values); }));
        reportCall("fixed phases", bins, measure(kRuns, [](int) { fixedBank.phases(scratch); }));
        reportCall("fixed magnitudes", bins, measure(kRuns, [](int) { fixedBank.magnitudes(scratch); }));
        reportCall("fixed powers", bins, measure(kRuns, [](int) { fixedBank.powers(scratch); }));
    }
    fckfft::logSpacedFrequencies(27.5f, 4186.0f, 64, frequencies);
    trackingBank.init(frequencies, 64, kSampleRate, fckfft::TrackingRule::normalizedChord);
    trackingBank.process(input, kInputLength);
    reportCall("tracking magnitudes", 64, measure(kRuns, [](int) { trackingBank.magnitudes(scratch); }));
    reportCall("tracking resonant", 64, measure(kRuns, [](int) { trackingBank.resonantFrequencies(scratch); }));
    reportCall("tracking instantaneous", 64,
               measure(kRuns, [](int) { trackingBank.instantaneousFrequencies(scratch); }));
}

void measureSetters()
{
    fckfft::logSpacedFrequencies(27.5f, 4186.0f, 88, frequencies);
    reportCall("fixed init", 88, measure(16, [](int) { fixedBank.init(frequencies, 88, kSampleRate); }));
    fixedBank.process(input, kInputLength);
    reportCall("fixed setFrequency", 1, measure(kRuns, [](int run) {
                   fixedBank.setFrequency(40, (run & 1) != 0 ? 440.0f : 442.0f);
               }));
    reportCall("fixed setAlpha", 1, measure(kRuns, [](int run) {
                   fixedBank.setAlpha(40, (run & 1) != 0 ? 0.003f : 0.004f);
               }));
    reportCall("fixed setBeta", 1, measure(kRuns, [](int run) {
                   fixedBank.setBeta(40, (run & 1) != 0 ? 0.003f : 0.004f);
               }));
    fckfft::logSpacedFrequencies(27.5f, 4186.0f, 64, frequencies);
    reportCall("tracking init", 64, measure(16, [](int) {
                   trackingBank.init(frequencies, 64, kSampleRate, fckfft::TrackingRule::normalizedChord);
               }));
}

}

int main()
{
    initBoard();
    hw.StartLog(true);
    enableCycleCounter();

    std::uint32_t seed = 1u;
    for (std::size_t i = 0; i < kInputLength; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const float noise = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
        input[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(i) / kSampleRate) + 0.05f * noise;
    }

    hw.PrintLine("fckFFT benchmark on %s: core clock %lu Hz, block %u, sample rate %u Hz, %d runs", kBoardName,
                 static_cast<unsigned long>(SystemCoreClock), static_cast<unsigned>(kBlock),
                 static_cast<unsigned>(kSampleRate), kRuns);
    reportCall("empty call", 1, measure(kRuns, [](int) {}));
    measureFixed();
    measureTracking();
    measureReadouts();
    measureSetters();
    hw.PrintLine("done");

    for (;;)
    {
        hw.SetLed(true);
        System::Delay(250);
        hw.SetLed(false);
        System::Delay(250);
    }
}
