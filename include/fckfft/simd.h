#ifndef FCKFFT_SIMD_H
#define FCKFFT_SIMD_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && !defined(__clang__)
#define FCKFFT_INLINE __forceinline
#else
#define FCKFFT_INLINE inline __attribute__((always_inline))
#endif

#if !defined(FCKFFT_NO_SIMD)
#if defined(__AVX512F__)
#define FCKFFT_SIMD_AVX512 1
#include <immintrin.h>
#elif defined(__AVX__)
#define FCKFFT_SIMD_AVX 1
#include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FCKFFT_SIMD_SSE 1
#include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
#define FCKFFT_SIMD_NEON 1
#include <arm_neon.h>
#endif
#endif

#if defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#define FCKFFT_SIMD_X86_FMA 1
#endif

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define FCKFFT_X86_MXCSR 1
#include <xmmintrin.h>
#endif

namespace fckfft {
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
#elif defined(__arm__) && defined(__ARM_FP) && (defined(__GNUC__) || defined(__clang__))
    unsigned int saved_ = 0;
#endif
};

struct ScalarOps
{
    using V = float;
    static constexpr std::size_t width = 1;
    static FCKFFT_INLINE V load(const float* p) { return *p; }
    static FCKFFT_INLINE void store(float* p, V v) { *p = v; }
    static FCKFFT_INLINE V set(float x) { return x; }
    static FCKFFT_INLINE V sub(V a, V b) { return a - b; }
    static FCKFFT_INLINE V mul(V a, V b) { return a * b; }
    static FCKFFT_INLINE V madd(V a, V b, V c) { return a * b + c; }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return c - a * b; }
    static FCKFFT_INLINE V sqrt(V a)
    {
#if defined(__ARM_FP) && (__ARM_FP & 4) && !defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
        V r;
        __asm__("vsqrt.f32 %0, %1" : "=t"(r) : "t"(a));
        return r;
#else
        return std::sqrt(a);
#endif
    }
    static FCKFFT_INLINE V recip(V a) { return 1.0f / a; }
    static FCKFFT_INLINE V rsqrt(V a) { return 1.0f / sqrt(a); }
    static FCKFFT_INLINE V abs(V a) { return std::fabs(a); }
#if defined(__ARM_FP) && (__ARM_FP & 4) && !defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
    static FCKFFT_INLINE V min(V a, V b) { return std::fmin(a, b); }
    static FCKFFT_INLINE V max(V a, V b) { return std::fmax(a, b); }
#if defined(__ARM_ARCH) && __ARM_ARCH >= 8
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f)
    {
        V r;
        __asm__("vcmp.f32 %1, %2\n\t"
                "vmrs APSR_nzcv, fpscr\n\t"
                "vselge.f32 %0, %3, %4"
                : "=t"(r)
                : "t"(a), "t"(b), "t"(t), "t"(f)
                : "cc");
        return r;
    }
#else
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f)
    {
        V r = f;
        __asm__("vcmp.f32 %1, %2\n\t"
                "vmrs APSR_nzcv, fpscr\n\t"
                "it ge\n\t"
                "vmovge.f32 %0, %3"
                : "+t"(r)
                : "t"(a), "t"(b), "t"(t)
                : "cc");
        return r;
    }
#endif
#else
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f)
    {
        const std::uint32_t mask = 0u - static_cast<std::uint32_t>(a >= b);
        std::uint32_t tb, fb;
        std::memcpy(&tb, &t, sizeof tb);
        std::memcpy(&fb, &f, sizeof fb);
        const std::uint32_t rb = (tb & mask) | (fb & ~mask);
        V r;
        std::memcpy(&r, &rb, sizeof r);
        return r;
    }
    static FCKFFT_INLINE V min(V a, V b) { return selectGe(a, b, b, a); }
    static FCKFFT_INLINE V max(V a, V b) { return selectGe(a, b, a, b); }
#endif
    static FCKFFT_INLINE float reduceMax(V a) { return a; }
};

#if defined(FCKFFT_SIMD_AVX512)
struct NativeOps
{
    using V = __m512;
    static constexpr std::size_t width = 16;
    static FCKFFT_INLINE V load(const float* p) { return _mm512_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm512_set1_ps(x); }
    static FCKFFT_INLINE V sub(V a, V b) { return _mm512_sub_ps(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return _mm512_mul_ps(a, b); }
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm512_fnmadd_ps(a, b, c); }
    static FCKFFT_INLINE V recip(V a)
    {
        const V r = _mm512_rcp14_ps(a);
        return _mm512_mul_ps(r, _mm512_fnmadd_ps(a, r, _mm512_set1_ps(2.0f)));
    }
    static FCKFFT_INLINE V rsqrt(V a)
    {
        const V r = _mm512_rsqrt14_ps(a);
        const V h = _mm512_mul_ps(_mm512_set1_ps(0.5f), a);
        return _mm512_mul_ps(r, _mm512_fnmadd_ps(h, _mm512_mul_ps(r, r), _mm512_set1_ps(1.5f)));
    }
    static FCKFFT_INLINE V abs(V a) { return _mm512_abs_ps(a); }
    static FCKFFT_INLINE V min(V a, V b) { return _mm512_min_ps(a, b); }
    static FCKFFT_INLINE V max(V a, V b) { return _mm512_max_ps(a, b); }
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f) { return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(a, b, _CMP_GE_OQ), f, t); }
    static FCKFFT_INLINE float reduceMax(V a) { return _mm512_reduce_max_ps(a); }
};
#elif defined(FCKFFT_SIMD_AVX)
struct NativeOps
{
    using V = __m256;
    static constexpr std::size_t width = 8;
    static FCKFFT_INLINE V load(const float* p) { return _mm256_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm256_set1_ps(x); }
    static FCKFFT_INLINE V sub(V a, V b) { return _mm256_sub_ps(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return _mm256_mul_ps(a, b); }
#if defined(FCKFFT_SIMD_X86_FMA)
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm256_fnmadd_ps(a, b, c); }
#else
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
#endif
    static FCKFFT_INLINE V recip(V a)
    {
        const V r = _mm256_rcp_ps(a);
        return _mm256_mul_ps(r, nmadd(a, r, _mm256_set1_ps(2.0f)));
    }
    static FCKFFT_INLINE V rsqrt(V a)
    {
        const V r = _mm256_rsqrt_ps(a);
        const V h = _mm256_mul_ps(_mm256_set1_ps(0.5f), a);
        return _mm256_mul_ps(r, nmadd(h, _mm256_mul_ps(r, r), _mm256_set1_ps(1.5f)));
    }
    static FCKFFT_INLINE V abs(V a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a); }
    static FCKFFT_INLINE V min(V a, V b) { return _mm256_min_ps(a, b); }
    static FCKFFT_INLINE V max(V a, V b) { return _mm256_max_ps(a, b); }
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f) { return _mm256_blendv_ps(f, t, _mm256_cmp_ps(a, b, _CMP_GE_OQ)); }
    static FCKFFT_INLINE float reduceMax(V a)
    {
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        return _mm_cvtss_f32(m);
    }
};
#elif defined(FCKFFT_SIMD_SSE)
struct NativeOps
{
    using V = __m128;
    static constexpr std::size_t width = 4;
    static FCKFFT_INLINE V load(const float* p) { return _mm_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm_set1_ps(x); }
    static FCKFFT_INLINE V sub(V a, V b) { return _mm_sub_ps(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return _mm_mul_ps(a, b); }
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm_sub_ps(c, _mm_mul_ps(a, b)); }
    static FCKFFT_INLINE V recip(V a)
    {
        const V r = _mm_rcp_ps(a);
        return _mm_mul_ps(r, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(a, r)));
    }
    static FCKFFT_INLINE V rsqrt(V a)
    {
        const V r = _mm_rsqrt_ps(a);
        const V h = _mm_mul_ps(_mm_set1_ps(0.5f), a);
        return _mm_mul_ps(r, _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(h, _mm_mul_ps(r, r))));
    }
    static FCKFFT_INLINE V abs(V a) { return _mm_andnot_ps(_mm_set1_ps(-0.0f), a); }
    static FCKFFT_INLINE V min(V a, V b) { return _mm_min_ps(a, b); }
    static FCKFFT_INLINE V max(V a, V b) { return _mm_max_ps(a, b); }
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f)
    {
        const V m = _mm_cmpge_ps(a, b);
        return _mm_or_ps(_mm_and_ps(m, t), _mm_andnot_ps(m, f));
    }
    static FCKFFT_INLINE float reduceMax(V a)
    {
        V m = _mm_max_ps(a, _mm_movehl_ps(a, a));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        return _mm_cvtss_f32(m);
    }
};
#elif defined(FCKFFT_SIMD_NEON)
struct NativeOps
{
    using V = float32x4_t;
    static constexpr std::size_t width = 4;
    static FCKFFT_INLINE V load(const float* p) { return vld1q_f32(p); }
    static FCKFFT_INLINE void store(float* p, V v) { vst1q_f32(p, v); }
    static FCKFFT_INLINE V set(float x) { return vdupq_n_f32(x); }
    static FCKFFT_INLINE V sub(V a, V b) { return vsubq_f32(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return vmulq_f32(a, b); }
#if defined(__aarch64__) || defined(_M_ARM64)
    static FCKFFT_INLINE V madd(V a, V b, V c) { return vfmaq_f32(c, a, b); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return vfmsq_f32(c, a, b); }
    static FCKFFT_INLINE float reduceMax(V a) { return vmaxvq_f32(a); }
#else
    static FCKFFT_INLINE V madd(V a, V b, V c) { return vmlaq_f32(c, a, b); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return vmlsq_f32(c, a, b); }
    static FCKFFT_INLINE float reduceMax(V a)
    {
        float32x2_t m = vpmax_f32(vget_low_f32(a), vget_high_f32(a));
        m = vpmax_f32(m, m);
        return vget_lane_f32(m, 0);
    }
#endif
    static FCKFFT_INLINE V recip(V a)
    {
        V r = vrecpeq_f32(a);
        r = vmulq_f32(r, vrecpsq_f32(a, r));
        return vmulq_f32(r, vrecpsq_f32(a, r));
    }
    static FCKFFT_INLINE V rsqrt(V a)
    {
        V r = vrsqrteq_f32(a);
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(a, r), r));
        return vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(a, r), r));
    }
    static FCKFFT_INLINE V abs(V a) { return vabsq_f32(a); }
    static FCKFFT_INLINE V min(V a, V b) { return vminq_f32(a, b); }
    static FCKFFT_INLINE V max(V a, V b) { return vmaxq_f32(a, b); }
    static FCKFFT_INLINE V selectGe(V a, V b, V t, V f) { return vbslq_f32(vcgeq_f32(a, b), t, f); }
};
#else
using NativeOps = ScalarOps;
#endif

template <class Ops>
FCKFFT_INLINE typename Ops::V atan2(typename Ops::V y, typename Ops::V x)
{
    using V = typename Ops::V;
    const V zero = Ops::set(0.0f);
    const V ax = Ops::abs(x);
    const V ay = Ops::abs(y);
    const V hi = Ops::max(ax, ay);
    const V lo = Ops::min(ax, ay);
    const V t = Ops::mul(lo, Ops::recip(Ops::max(hi, Ops::set(1e-30f))));
    const V t2 = Ops::mul(t, t);
    const V t4 = Ops::mul(t2, t2);
    const V t8 = Ops::mul(t4, t4);
    const V q0 = Ops::madd(t2, Ops::set(-0.333173662f), Ops::set(0.999996126f));
    const V q1 = Ops::madd(t2, Ops::set(-0.132333264f), Ops::set(0.198078111f));
    const V q2 = Ops::madd(t2, Ops::set(-0.033603959f), Ops::set(0.079623379f));
    const V q3 = Ops::madd(t4, Ops::set(0.006811703f), q2);
    const V p = Ops::madd(t8, q3, Ops::madd(t4, q1, q0));
    V r = Ops::mul(p, t);
    r = Ops::selectGe(ax, ay, r, Ops::sub(Ops::set(1.57079632679f), r));
    r = Ops::selectGe(x, zero, r, Ops::sub(Ops::set(3.14159265359f), r));
    return Ops::selectGe(y, zero, r, Ops::sub(zero, r));
}

template <class Ops>
FCKFFT_INLINE void sincos(typename Ops::V theta, typename Ops::V& s, typename Ops::V& c)
{
    using V = typename Ops::V;
    const V h = Ops::mul(theta, Ops::set(0.5f));
    const V h2 = Ops::mul(h, h);
    V ps = Ops::set(2.5904894e-06f);
    ps = Ops::madd(ps, h2, Ops::set(-1.9800899e-04f));
    ps = Ops::madd(ps, h2, Ops::set(8.3328998e-03f));
    ps = Ops::madd(ps, h2, Ops::set(-1.6666648e-01f));
    ps = Ops::madd(ps, h2, Ops::set(1.0f));
    const V sh = Ops::mul(ps, h);
    V pc = Ops::set(2.3153952e-05f);
    pc = Ops::madd(pc, h2, Ops::set(-1.3853705e-03f));
    pc = Ops::madd(pc, h2, Ops::set(4.1663583e-02f));
    pc = Ops::madd(pc, h2, Ops::set(-4.9999905e-01f));
    pc = Ops::madd(pc, h2, Ops::set(0.99999994f));
    s = Ops::mul(Ops::set(2.0f), Ops::mul(sh, pc));
    c = Ops::nmadd(Ops::set(2.0f), Ops::mul(sh, sh), Ops::set(1.0f));
}

}
}

#endif
