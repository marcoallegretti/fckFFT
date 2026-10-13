#ifndef FCKFFT_SIMD_H
#define FCKFFT_SIMD_H

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(_MSC_VER) && !defined(__clang__)
#define FCKFFT_INLINE __forceinline
#else
#define FCKFFT_INLINE inline __attribute__((always_inline))
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define FCKFFT_VFP_FLAGS "cc", "vfpcc"
#else
#define FCKFFT_VFP_FLAGS "cc"
#endif

#define FCKFFT_BACKEND_SCALAR 1
#define FCKFFT_BACKEND_SSE2 2
#define FCKFFT_BACKEND_AVX 3
#define FCKFFT_BACKEND_AVX2 4
#define FCKFFT_BACKEND_AVX512 5
#define FCKFFT_BACKEND_NEON 6

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FCKFFT_X86 1
#endif

#if defined(FCKFFT_NO_SIMD)
#define FCKFFT_BASELINE FCKFFT_BACKEND_SCALAR
#elif defined(__AVX512F__)
#define FCKFFT_BASELINE FCKFFT_BACKEND_AVX512
#elif defined(__AVX__) && (defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__)))
#define FCKFFT_BASELINE FCKFFT_BACKEND_AVX2
#elif defined(__AVX__)
#define FCKFFT_BASELINE FCKFFT_BACKEND_AVX
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FCKFFT_BASELINE FCKFFT_BACKEND_SSE2
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define FCKFFT_BASELINE FCKFFT_BACKEND_NEON
#else
#define FCKFFT_BASELINE FCKFFT_BACKEND_SCALAR
#endif

#if defined(FCKFFT_X86) && FCKFFT_BASELINE != FCKFFT_BACKEND_SCALAR && !defined(FCKFFT_NO_DISPATCH)                  \
    && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#define FCKFFT_DISPATCH 1
#endif

#if defined(FCKFFT_X86) && FCKFFT_BASELINE != FCKFFT_BACKEND_SCALAR
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#elif defined(FCKFFT_DISPATCH)
#include <cpuid.h>
#endif
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_NEON
#include <arm_neon.h>
#if defined(_MSC_VER) && defined(_M_ARM64)
#include <intrin.h>
#endif
#endif

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define FCKFFT_X86_MXCSR 1
#include <xmmintrin.h>
#endif

#if FCKFFT_BASELINE == FCKFFT_BACKEND_SCALAR
#define FCKFFT_ABI_BASE scalar
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_SSE2
#define FCKFFT_ABI_BASE sse2
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX
#define FCKFFT_ABI_BASE avx
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX2
#define FCKFFT_ABI_BASE avx2
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX512
#define FCKFFT_ABI_BASE avx512
#else
#define FCKFFT_ABI_BASE neon
#endif

#define FCKFFT_ABI_JOIN2(a, b) a##b
#define FCKFFT_ABI_JOIN(a, b) FCKFFT_ABI_JOIN2(a, b)
#if defined(FCKFFT_DISPATCH)
#define FCKFFT_ABI FCKFFT_ABI_JOIN(FCKFFT_ABI_JOIN(abi_, FCKFFT_ABI_BASE), _dispatch)
#else
#define FCKFFT_ABI FCKFFT_ABI_JOIN(abi_, FCKFFT_ABI_BASE)
#endif

namespace fckfft {
inline namespace FCKFFT_ABI {

enum class Isa
{
    scalar,
    sse2,
    avx,
    avx2,
    avx512,
    neon
};

struct Complex
{
    float re;
    float im;
};

enum class TrackingRule
{
    none,
    ewma,
    normalizedChord,
    tangent
};

namespace simd {

class ScopedFlushDenormals
{
public:
#if defined(FCKFFT_X86_MXCSR)
    ScopedFlushDenormals() : saved_(_mm_getcsr()) { _mm_setcsr(saved_ | 0x8040u); }
    ~ScopedFlushDenormals() { _mm_setcsr(saved_); }
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    ScopedFlushDenormals()
    {
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(saved_));
        __asm__ __volatile__("msr fpcr, %0" : : "r"(saved_ | (1ull << 24)));
    }
    ~ScopedFlushDenormals() { __asm__ __volatile__("msr fpcr, %0" : : "r"(saved_)); }
#elif defined(_MSC_VER) && defined(_M_ARM64)
    ScopedFlushDenormals() : saved_(_ReadStatusReg(ARM64_FPCR)) { _WriteStatusReg(ARM64_FPCR, saved_ | (1ll << 24)); }
    ~ScopedFlushDenormals() { _WriteStatusReg(ARM64_FPCR, saved_); }
#elif defined(__arm__) && defined(__ARM_FP) && (defined(__GNUC__) || defined(__clang__))
    ScopedFlushDenormals()
    {
        __asm__ __volatile__("vmrs %0, fpscr" : "=r"(saved_));
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(saved_ | (1u << 24)));
    }
    ~ScopedFlushDenormals() { __asm__ __volatile__("vmsr fpscr, %0" : : "r"(saved_)); }
#else
    ScopedFlushDenormals() {}
    ~ScopedFlushDenormals() {}
#endif
    ScopedFlushDenormals(const ScopedFlushDenormals&) = delete;
    ScopedFlushDenormals& operator=(const ScopedFlushDenormals&) = delete;

private:
#if defined(FCKFFT_X86_MXCSR)
    unsigned int saved_;
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    unsigned long long saved_ = 0;
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __int64 saved_;
#elif defined(__arm__) && defined(__ARM_FP) && (defined(__GNUC__) || defined(__clang__))
    unsigned int saved_ = 0;
#endif
};

}

namespace detail {

struct BankLayout
{
    enum Field
    {
        kCr,
        kCi,
        kDr,
        kDi,
        kVr,
        kVi,
        kSr,
        kSi,
        kFields
    };
    static constexpr std::size_t kPacks = 3;
};

struct TrackingLayout
{
    enum Field
    {
        kOa,
        kOb,
        kAb,
        kGm,
        kOg,
        kNwr,
        kNwi,
        kZr,
        kZi,
        kUr,
        kUi,
        kSr,
        kSi,
        kPr,
        kPi,
        kWr,
        kWi,
        kQr,
        kQi,
        kPw,
        kFields
    };
    static constexpr std::size_t kPacks = 2;
};

struct BankBackend
{
    Isa isa;
    std::size_t lanes;
    void (*process)(float*, std::size_t, const float*, std::size_t, std::size_t, bool);
};

struct TrackingBackend
{
    Isa isa;
    std::size_t lanes;
    void (*process[4])(float*, std::size_t, const float*, std::size_t, std::size_t, bool, float, float, float&);
    void (*sanitize)(float*, std::size_t);
};

namespace baseline {
#define FCKFFT_BACKEND FCKFFT_BASELINE
#if FCKFFT_BASELINE == FCKFFT_BACKEND_SCALAR
#define FCKFFT_BACKEND_ISA Isa::scalar
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_SSE2
#define FCKFFT_BACKEND_ISA Isa::sse2
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX
#define FCKFFT_BACKEND_ISA Isa::avx
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX2
#define FCKFFT_BACKEND_ISA Isa::avx2
#elif FCKFFT_BASELINE == FCKFFT_BACKEND_AVX512
#define FCKFFT_BACKEND_ISA Isa::avx512
#else
#define FCKFFT_BACKEND_ISA Isa::neon
#endif
#include "backend.inc"
#undef FCKFFT_BACKEND_ISA
#undef FCKFFT_BACKEND
}

#if defined(FCKFFT_DISPATCH) && FCKFFT_BASELINE < FCKFFT_BACKEND_AVX2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx,avx2,fma")
#endif
namespace avx2 {
#define FCKFFT_BACKEND FCKFFT_BACKEND_AVX2
#define FCKFFT_BACKEND_ISA Isa::avx2
#include "backend.inc"
#undef FCKFFT_BACKEND_ISA
#undef FCKFFT_BACKEND
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#define FCKFFT_HAS_AVX2_BACKEND 1
#endif

#if defined(FCKFFT_DISPATCH) && FCKFFT_BASELINE < FCKFFT_BACKEND_AVX512
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx2,fma,avx512f"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx,avx2,fma,avx512f")
#endif
namespace avx512 {
#define FCKFFT_BACKEND FCKFFT_BACKEND_AVX512
#define FCKFFT_BACKEND_ISA Isa::avx512
#include "backend.inc"
#undef FCKFFT_BACKEND_ISA
#undef FCKFFT_BACKEND
}
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#define FCKFFT_HAS_AVX512_BACKEND 1
#endif

constexpr std::size_t kAlignmentFloats = 16;

inline float* alignStorage(float* storage) noexcept
{
    const std::size_t misalignment = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(storage) % 64u);
    return storage + (64u - misalignment) % 64u / sizeof(float);
}

constexpr std::size_t maxLanes(std::size_t a, std::size_t b)
{
    return a > b ? a : b;
}

constexpr std::size_t kMaxBankLanes = maxLanes(baseline::kBankBackend.lanes,
#if defined(FCKFFT_HAS_AVX512_BACKEND)
                                               avx512::kBankBackend.lanes
#elif defined(FCKFFT_HAS_AVX2_BACKEND)
                                               avx2::kBankBackend.lanes
#else
                                               baseline::kBankBackend.lanes
#endif
);

constexpr std::size_t kMaxTrackingLanes = maxLanes(baseline::kTrackingBackend.lanes,
#if defined(FCKFFT_HAS_AVX512_BACKEND)
                                                   avx512::kTrackingBackend.lanes
#elif defined(FCKFFT_HAS_AVX2_BACKEND)
                                                   avx2::kTrackingBackend.lanes
#else
                                                   baseline::kTrackingBackend.lanes
#endif
);

#if defined(FCKFFT_DISPATCH)
#if defined(_MSC_VER) && !defined(__clang__)
inline void cpuid(unsigned leaf, unsigned sub, unsigned out[4])
{
    int r[4];
    __cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i)
        out[i] = static_cast<unsigned>(r[i]);
}

inline unsigned long long xcr0()
{
    return _xgetbv(0);
}
#else
inline void cpuid(unsigned leaf, unsigned sub, unsigned out[4])
{
    __cpuid_count(leaf, sub, out[0], out[1], out[2], out[3]);
}

inline unsigned long long xcr0()
{
    unsigned lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

inline Isa detectIsa()
{
    unsigned r[4];
    cpuid(0, 0, r);
    const unsigned maxLeaf = r[0];
    cpuid(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1u, avx = (r[2] >> 28) & 1u, fma = (r[2] >> 12) & 1u;
    if (!osxsave || !avx || maxLeaf < 7)
        return Isa::sse2;
    const unsigned long long state = xcr0();
    if ((state & 0x6u) != 0x6u)
        return Isa::sse2;
    cpuid(7, 0, r);
    const bool avx2 = (r[1] >> 5) & 1u, avx512f = (r[1] >> 16) & 1u;
    if (avx2 && fma && avx512f && (state & 0xE6u) == 0xE6u)
        return Isa::avx512;
    return avx2 && fma ? Isa::avx2 : Isa::avx;
}
#endif

inline std::atomic<int>& isaLimitStorage()
{
    static std::atomic<int> limit{-1};
    return limit;
}

}

inline Isa baselineIsa() noexcept
{
    return detail::baseline::kBankBackend.isa;
}

inline Isa supportedIsa() noexcept
{
#if defined(FCKFFT_DISPATCH)
    const Isa cpu = detail::detectIsa();
    return static_cast<int>(cpu) > static_cast<int>(baselineIsa()) ? cpu : baselineIsa();
#else
    return baselineIsa();
#endif
}

inline Isa isaLimit() noexcept
{
    const int limit = detail::isaLimitStorage().load(std::memory_order_relaxed);
    if (limit >= 0)
        return static_cast<Isa>(limit);
    return static_cast<int>(baselineIsa()) > static_cast<int>(Isa::avx2) ? baselineIsa() : Isa::avx2;
}

inline void setIsaLimit(Isa limit) noexcept
{
    detail::isaLimitStorage().store(static_cast<int>(limit), std::memory_order_relaxed);
}

namespace detail {

inline Isa selectedIsa() noexcept
{
#if defined(FCKFFT_DISPATCH)
    const int cpu = static_cast<int>(supportedIsa()), limit = static_cast<int>(isaLimit());
#if defined(FCKFFT_HAS_AVX512_BACKEND)
    if (cpu >= static_cast<int>(Isa::avx512) && limit >= static_cast<int>(Isa::avx512))
        return Isa::avx512;
#endif
#if defined(FCKFFT_HAS_AVX2_BACKEND)
    if (cpu >= static_cast<int>(Isa::avx2) && limit >= static_cast<int>(Isa::avx2))
        return Isa::avx2;
#endif
    (void)cpu;
    (void)limit;
#endif
    return baselineIsa();
}

inline const BankBackend& selectBankBackend() noexcept
{
    const Isa isa = selectedIsa();
#if defined(FCKFFT_HAS_AVX512_BACKEND)
    if (isa == Isa::avx512)
        return avx512::kBankBackend;
#endif
#if defined(FCKFFT_HAS_AVX2_BACKEND)
    if (isa == Isa::avx2)
        return avx2::kBankBackend;
#endif
    (void)isa;
    return baseline::kBankBackend;
}

inline const TrackingBackend& selectTrackingBackend() noexcept
{
    const Isa isa = selectedIsa();
#if defined(FCKFFT_HAS_AVX512_BACKEND)
    if (isa == Isa::avx512)
        return avx512::kTrackingBackend;
#endif
#if defined(FCKFFT_HAS_AVX2_BACKEND)
    if (isa == Isa::avx2)
        return avx2::kTrackingBackend;
#endif
    (void)isa;
    return baseline::kTrackingBackend;
}

}

}
}

#endif
