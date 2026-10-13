#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

#include <fckfft/fckfft.h>

#include "check.h"
#include "signals.h"

namespace {

long allocationCount = 0;

}

void* operator new(std::size_t size)
{
    ++allocationCount;
    void* p = std::malloc(size != 0 ? size : 1);
    if (p == nullptr)
        std::abort();
    return p;
}

void operator delete(void* p) noexcept
{
    std::free(p);
}

void operator delete(void* p, std::size_t) noexcept
{
    std::free(p);
}

namespace {

#if defined(FCKFFT_X86_MXCSR)
constexpr std::uint64_t kControlMask = 0xFFC0u;
std::uint64_t controlRegister()
{
    return _mm_getcsr();
}
void setControlRegister(std::uint64_t value)
{
    _mm_setcsr(static_cast<unsigned int>(value));
}
constexpr std::uint64_t kFlushBits = 0x8040u;
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
constexpr std::uint64_t kControlMask = 0x03C00000u;
std::uint64_t controlRegister()
{
    std::uint64_t value;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(value));
    return value;
}
void setControlRegister(std::uint64_t value)
{
    __asm__ __volatile__("msr fpcr, %0" : : "r"(value));
}
constexpr std::uint64_t kFlushBits = 1u << 24;
#elif defined(__arm__) && defined(__ARM_FP) && (defined(__GNUC__) || defined(__clang__))
constexpr std::uint64_t kControlMask = 0x03C00000u;
std::uint64_t controlRegister()
{
    std::uint32_t value;
    __asm__ __volatile__("vmrs %0, fpscr" : "=r"(value));
    return value;
}
void setControlRegister(std::uint64_t value)
{
    const std::uint32_t v = static_cast<std::uint32_t>(value);
    __asm__ __volatile__("vmsr fpscr, %0" : : "r"(v));
}
constexpr std::uint64_t kFlushBits = 1u << 24;
#else
constexpr std::uint64_t kControlMask = 0;
std::uint64_t controlRegister()
{
    return 0;
}
void setControlRegister(std::uint64_t) {}
constexpr std::uint64_t kFlushBits = 0;
#endif

std::vector<float> testFrequencies(std::size_t count)
{
    std::vector<float> f(count);
    fckfft::logSpacedFrequencies(55.0f, 7040.0f, count, f.data());
    return f;
}

}

TEST_CASE(banks_do_not_allocate_after_init)
{
    const std::vector<float> f = testFrequencies(37);
    const std::vector<float> x = signals::noise(4800, 3u, 0.5f);
    std::vector<fckfft::Complex> values(f.size());
    std::vector<float> out(f.size());

    fckfft::DynamicResonatorBank fixed;
    CHECK(fixed.init(f.data(), f.size(), 48000.0f));
    fckfft::DynamicTrackingResonatorBank tracking;
    CHECK(tracking.init(f.data(), f.size(), 48000.0f));

    const long before = allocationCount;
    fixed.process(x.data(), x.size());
    fixed.process(x[0]);
    fixed.values(values.data());
    fixed.phases(out.data());
    fixed.magnitudes(out.data());
    fixed.powers(out.data());
    CHECK(fixed.setFrequency(3, 120.0f));
    CHECK(fixed.setAlpha(3, 0.01f));
    CHECK(fixed.setBeta(3, 0.02f));
    fixed.reset();
    for (int rule = 0; rule < 4; ++rule)
    {
        tracking.setRule(static_cast<fckfft::TrackingRule>(rule));
        tracking.process(x.data(), x.size());
        tracking.process(x[0]);
    }
    tracking.values(values.data());
    tracking.phases(out.data());
    tracking.magnitudes(out.data());
    tracking.powers(out.data());
    tracking.resonantFrequencies(out.data());
    tracking.instantaneousFrequencies(out.data());
    CHECK(tracking.setGamma(3, 0.002f));
    tracking.setThresholdDb(-50.0f);
    tracking.setPowerTimeConstant(0.1f);
    tracking.reset();
    CHECK(allocationCount == before);
}

TEST_CASE(process_restores_floating_point_control)
{
    const std::vector<float> f = testFrequencies(9);
    std::vector<float> x(2400, 0.0f);
    x[0] = 1e-30f;
    fckfft::DynamicResonatorBank fixed;
    CHECK(fixed.init(f.data(), f.size(), 48000.0f));
    fckfft::DynamicTrackingResonatorBank tracking;
    CHECK(tracking.init(f.data(), f.size(), 48000.0f));
    const std::uint64_t original = controlRegister();
    for (std::uint64_t host : {original & ~kFlushBits, original | kFlushBits})
    {
        setControlRegister(host);
        fixed.process(x.data(), x.size());
        CHECK((controlRegister() & kControlMask) == (host & kControlMask));
        tracking.process(x.data(), x.size());
        CHECK((controlRegister() & kControlMask) == (host & kControlMask));
    }
    setControlRegister(original);
}

TEST_CASE(process_flushes_denormal_state)
{
    const std::vector<float> f = testFrequencies(9);
    std::vector<float> x(48, 0.0f);
    fckfft::DynamicTrackingResonatorBank bank;
    CHECK(bank.init(f.data(), f.size(), 48000.0f, fckfft::TrackingRule::none));
    x[0] = 1e-30f;
    bank.process(x.data(), x.size());
    x[0] = 0.0f;
    std::vector<fckfft::Complex> state(f.size());
    bool denormal = false;
    for (int block = 0; block < 2000 && !denormal; ++block)
    {
        bank.process(x.data(), x.size());
        bank.values(state.data());
        for (const fckfft::Complex& v : state)
            for (float c : {v.re, v.im})
                denormal = denormal || (c != 0.0f && std::fabs(c) < 1.17549435e-38f);
    }
    CHECK(!denormal);
}
