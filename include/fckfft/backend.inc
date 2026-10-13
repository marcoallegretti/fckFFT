#if FCKFFT_BACKEND == FCKFFT_BACKEND_SCALAR
struct Ops
{
    using V = float;
    static constexpr std::size_t width = 1;
    static FCKFFT_INLINE V load(const float* p) { return *p; }
    static FCKFFT_INLINE void store(float* p, V v) { *p = v; }
    static FCKFFT_INLINE V set(float x) { return x; }
    static FCKFFT_INLINE V add(V a, V b) { return a + b; }
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

#elif FCKFFT_BACKEND == FCKFFT_BACKEND_SSE2
struct Ops
{
    using V = __m128;
    static constexpr std::size_t width = 4;
    static FCKFFT_INLINE V load(const float* p) { return _mm_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm_set1_ps(x); }
    static FCKFFT_INLINE V add(V a, V b) { return _mm_add_ps(a, b); }
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
#elif FCKFFT_BACKEND == FCKFFT_BACKEND_AVX
struct Ops
{
    using V = __m256;
    static constexpr std::size_t width = 8;
    static FCKFFT_INLINE V load(const float* p) { return _mm256_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm256_set1_ps(x); }
    static FCKFFT_INLINE V add(V a, V b) { return _mm256_add_ps(a, b); }
    static FCKFFT_INLINE V sub(V a, V b) { return _mm256_sub_ps(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return _mm256_mul_ps(a, b); }
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
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
#elif FCKFFT_BACKEND == FCKFFT_BACKEND_AVX2
struct Ops
{
    using V = __m256;
    static constexpr std::size_t width = 8;
    static FCKFFT_INLINE V load(const float* p) { return _mm256_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm256_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm256_set1_ps(x); }
    static FCKFFT_INLINE V add(V a, V b) { return _mm256_add_ps(a, b); }
    static FCKFFT_INLINE V sub(V a, V b) { return _mm256_sub_ps(a, b); }
    static FCKFFT_INLINE V mul(V a, V b) { return _mm256_mul_ps(a, b); }
    static FCKFFT_INLINE V madd(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
    static FCKFFT_INLINE V nmadd(V a, V b, V c) { return _mm256_fnmadd_ps(a, b, c); }
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
#elif FCKFFT_BACKEND == FCKFFT_BACKEND_AVX512
struct Ops
{
    using V = __m512;
    static constexpr std::size_t width = 16;
    static FCKFFT_INLINE V load(const float* p) { return _mm512_loadu_ps(p); }
    static FCKFFT_INLINE void store(float* p, V v) { _mm512_storeu_ps(p, v); }
    static FCKFFT_INLINE V set(float x) { return _mm512_set1_ps(x); }
    static FCKFFT_INLINE V add(V a, V b) { return _mm512_add_ps(a, b); }
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
#elif FCKFFT_BACKEND == FCKFFT_BACKEND_NEON
struct Ops
{
    using V = float32x4_t;
    static constexpr std::size_t width = 4;
    static FCKFFT_INLINE V load(const float* p) { return vld1q_f32(p); }
    static FCKFFT_INLINE void store(float* p, V v) { vst1q_f32(p, v); }
    static FCKFFT_INLINE V set(float x) { return vdupq_n_f32(x); }
    static FCKFFT_INLINE V add(V a, V b) { return vaddq_f32(a, b); }
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
#endif

FCKFFT_INLINE Ops::V atan2(Ops::V y, Ops::V x)
{
    using V = Ops::V;
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

FCKFFT_INLINE void sincos(Ops::V theta, Ops::V& s, Ops::V& c)
{
    using V = Ops::V;
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

struct BankKernel : BankLayout
{
    using V = Ops::V;
    static constexpr std::size_t kWidth = Ops::width;
    static constexpr std::size_t kLanes = kPacks * kWidth;
    static constexpr std::size_t kGroupFloats = kFields * kLanes;

    struct Pack
    {
        V cr[kPacks], ci[kPacks], dr[kPacks], di[kPacks], vr[kPacks], vi[kPacks], sr[kPacks], si[kPacks];
    };

    static FCKFFT_INLINE void loadOne(Pack& p, std::size_t j, const float* g)
    {
        p.cr[j] = Ops::load(g + kCr * kLanes);
        p.ci[j] = Ops::load(g + kCi * kLanes);
        p.dr[j] = Ops::load(g + kDr * kLanes);
        p.di[j] = Ops::load(g + kDi * kLanes);
        p.vr[j] = Ops::load(g + kVr * kLanes);
        p.vi[j] = Ops::load(g + kVi * kLanes);
        p.sr[j] = Ops::load(g + kSr * kLanes);
        p.si[j] = Ops::load(g + kSi * kLanes);
    }

    static FCKFFT_INLINE void storeOne(const Pack& p, std::size_t j, float* g)
    {
        Ops::store(g + kVr * kLanes, p.vr[j]);
        Ops::store(g + kVi * kLanes, p.vi[j]);
        Ops::store(g + kSr * kLanes, p.sr[j]);
        Ops::store(g + kSi * kLanes, p.si[j]);
    }

    static FCKFFT_INLINE void load(Pack& p, const float* g)
    {
        loadOne(p, 0, g);
        loadOne(p, 1, g + kWidth);
        loadOne(p, 2, g + 2 * kWidth);
    }

    static FCKFFT_INLINE void store(const Pack& p, float* g)
    {
        storeOne(p, 0, g);
        storeOne(p, 1, g + kWidth);
        storeOne(p, 2, g + 2 * kWidth);
    }

    static FCKFFT_INLINE void stepOne(Pack& p, std::size_t j, V x, std::false_type)
    {
        const V t = Ops::nmadd(p.ci[j], p.vi[j], x);
        p.vi[j] = Ops::madd(p.cr[j], p.vi[j], Ops::mul(p.ci[j], p.vr[j]));
        p.vr[j] = Ops::madd(p.cr[j], p.vr[j], t);
        const V u = Ops::nmadd(p.di[j], p.si[j], p.vr[j]);
        p.si[j] = Ops::madd(p.dr[j], p.si[j], Ops::madd(p.di[j], p.sr[j], p.vi[j]));
        p.sr[j] = Ops::madd(p.dr[j], p.sr[j], u);
    }

    static FCKFFT_INLINE void stepOne(Pack& p, std::size_t j, V x, std::true_type)
    {
        const V vi = Ops::madd(p.cr[j], p.vi[j], Ops::mul(p.ci[j], p.vr[j]));
        const V vr = Ops::sub(Ops::mul(p.cr[j], p.vr[j]), Ops::mul(p.ci[j], p.vi[j]));
        p.vr[j] = Ops::add(vr, x);
        p.vi[j] = vi;
        const V si = Ops::madd(p.dr[j], p.si[j], Ops::mul(p.di[j], p.sr[j]));
        const V sr = Ops::sub(Ops::mul(p.dr[j], p.sr[j]), Ops::mul(p.di[j], p.si[j]));
        p.sr[j] = Ops::add(sr, p.vr[j]);
        p.si[j] = Ops::add(si, p.vi[j]);
    }

    static FCKFFT_INLINE void step(Pack& p, V x)
    {
        const std::integral_constant<bool, kWidth == 1> scalar{};
        stepOne(p, 0, x, scalar);
        stepOne(p, 1, x, scalar);
        stepOne(p, 2, x, scalar);
    }

    static FCKFFT_INLINE void sanitizeOne(Pack& p, std::size_t j)
    {
        const V zero = Ops::set(0.0f);
        V t = Ops::mul(p.vr[j], zero);
        t = Ops::madd(p.vi[j], zero, t);
        t = Ops::madd(p.sr[j], zero, t);
        t = Ops::madd(p.si[j], zero, t);
        p.vr[j] = Ops::selectGe(t, zero, p.vr[j], zero);
        p.vi[j] = Ops::selectGe(t, zero, p.vi[j], zero);
        p.sr[j] = Ops::selectGe(t, zero, p.sr[j], zero);
        p.si[j] = Ops::selectGe(t, zero, p.si[j], zero);
    }

    static FCKFFT_INLINE void sanitize(Pack& p)
    {
        sanitizeOne(p, 0);
        sanitizeOne(p, 1);
        sanitizeOne(p, 2);
    }

    static void process(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool sanitizeAtEnd)
    {
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            float* base = groups + g * kGroupFloats;
            Pack a;
            load(a, base);
            const float* in = x;
            std::size_t i = 0;
            for (; i + 2 <= n; i += 2)
            {
                step(a, Ops::set(in[0]));
                step(a, Ops::set(in[stride]));
                in += 2 * stride;
            }
            if (i < n)
                step(a, Ops::set(in[0]));
            if (sanitizeAtEnd)
                sanitize(a);
            store(a, base);
        }
    }
};

struct TrackingKernel : TrackingLayout
{
    using V = Ops::V;
    static constexpr std::size_t kWidth = Ops::width;
    static constexpr std::size_t kLanes = kPacks * kWidth;
    static constexpr std::size_t kGroupFloats = kFields * kLanes;

    static FCKFFT_INLINE V ld(const float* g, int f) { return Ops::load(g + static_cast<std::size_t>(f) * kLanes); }
    static FCKFFT_INLINE void st(float* g, int f, V v) { Ops::store(g + static_cast<std::size_t>(f) * kLanes, v); }

    static FCKFFT_INLINE V accumulate(float* g, V s, V zr, V zi)
    {
        const V oa = ld(g, kOa), ob = ld(g, kOb), ab = ld(g, kAb);
        const V ur = Ops::madd(oa, ld(g, kUr), Ops::mul(s, zr));
        const V ui = Ops::madd(oa, ld(g, kUi), Ops::mul(s, zi));
        const V sr = Ops::madd(ob, ld(g, kSr), Ops::mul(ab, ur));
        const V si = Ops::madd(ob, ld(g, kSi), Ops::mul(ab, ui));
        const V pw = Ops::madd(sr, sr, Ops::mul(si, si));
        st(g, kUr, ur);
        st(g, kUi, ui);
        st(g, kSr, sr);
        st(g, kSi, si);
        st(g, kPw, pw);
        return pw;
    }

    static FCKFFT_INLINE V accumulate(float* g, V s) { return accumulate(g, s, ld(g, kZr), ld(g, kZi)); }

    static FCKFFT_INLINE void renormalize(V& re, V& im)
    {
        const V m = Ops::madd(re, re, Ops::mul(im, im));
        const V k = Ops::nmadd(Ops::set(0.5f), m, Ops::set(1.5f));
        re = Ops::mul(re, k);
        im = Ops::mul(im, k);
    }

    template <TrackingRule Rule>
    static FCKFFT_INLINE void track(float* g, V threshold, bool renorm, V& zr, V& zi)
    {
        const V sr = ld(g, kSr), si = ld(g, kSi);
        const V pr = ld(g, kPr), pi = ld(g, kPi);
        const V dr = Ops::madd(sr, pr, Ops::mul(si, pi));
        const V di = Ops::nmadd(si, pr, Ops::mul(sr, pi));
        const V gm = ld(g, kGm), og = ld(g, kOg);
        st(g, kPr, sr);
        st(g, kPi, si);
        st(g, kQr, Ops::madd(og, ld(g, kQr), Ops::mul(gm, dr)));
        st(g, kQi, Ops::madd(og, ld(g, kQi), Ops::mul(gm, di)));
        V wr = ld(g, kWr), wi = ld(g, kWi);
        if (Rule != TrackingRule::none)
        {
            V cr, ci;
            if (Rule == TrackingRule::ewma)
            {
                sincos(Ops::mul(gm, atan2(di, dr)), ci, cr);
            }
            else
            {
                const V dd = Ops::madd(dr, dr, Ops::mul(di, di));
                const V valid = Ops::set(1e-20f);
                const V gi = Ops::mul(gm, Ops::rsqrt(Ops::max(dd, valid)));
                if (Rule == TrackingRule::normalizedChord)
                    cr = Ops::madd(gi, dr, og);
                else
                    cr = Ops::set(1.0f);
                ci = Ops::mul(gi, di);
                cr = Ops::selectGe(dd, valid, cr, Ops::set(1.0f));
                ci = Ops::selectGe(dd, valid, ci, Ops::set(0.0f));
            }
            V tr = Ops::nmadd(wi, ci, Ops::mul(wr, cr));
            V ti = Ops::madd(wr, ci, Ops::mul(wi, cr));
            if (Rule != TrackingRule::ewma)
                renormalize(tr, ti);
            const V pw = ld(g, kPw);
            wr = Ops::selectGe(pw, threshold, tr, ld(g, kNwr));
            wi = Ops::selectGe(pw, threshold, ti, ld(g, kNwi));
        }
        const V z0 = ld(g, kZr), z1 = ld(g, kZi);
        zr = Ops::nmadd(z1, wi, Ops::mul(z0, wr));
        zi = Ops::madd(z0, wi, Ops::mul(z1, wr));
        if (renorm)
        {
            renormalize(zr, zi);
            renormalize(wr, wi);
        }
        st(g, kZr, zr);
        st(g, kZi, zi);
        st(g, kWr, wr);
        st(g, kWi, wi);
    }

    static void sanitize(float* groups, std::size_t numGroups)
    {
        const V zero = Ops::set(0.0f);
        for (std::size_t g = 0; g < numGroups; ++g)
            for (std::size_t half = 0; half < 2; ++half)
            {
                float* base = groups + g * kGroupFloats + half * kWidth;
                V t = zero;
                for (int f = kZr; f < kFields; ++f)
                    t = Ops::madd(ld(base, f), zero, t);
                if (Ops::reduceMax(Ops::selectGe(t, zero, zero, Ops::set(1.0f))) == 0.0f)
                    continue;
                for (int f = kZr; f < kFields; ++f)
                {
                    V reset = zero;
                    if (f == kZr)
                        reset = Ops::set(1.0f);
                    else if (f == kWr)
                        reset = ld(base, kNwr);
                    else if (f == kWi)
                        reset = ld(base, kNwi);
                    st(base, f, Ops::selectGe(t, zero, ld(base, f), reset));
                }
            }
    }

    template <TrackingRule Rule>
    static void process(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool renormalizeAtEnd, float sigma, float ratio, float& accPower)
    {
        const float omSigma = 1.0f - sigma;
        V peak = Ops::set(0.0f);
        {
            const V s = Ops::set(x[0]);
            for (std::size_t g = 0; g < numGroups; ++g)
            {
                float* base = groups + g * kGroupFloats;
                const V pa = accumulate(base, s);
                const V pb = accumulate(base + kWidth, s);
                peak = Ops::max(peak, Ops::max(pa, pb));
            }
        }
        for (std::size_t i = 1; i < n; ++i)
        {
            accPower = omSigma * accPower + sigma * Ops::reduceMax(peak);
            const V threshold = Ops::set(accPower * ratio);
            const V s = Ops::set(x[i * stride]);
            peak = Ops::set(0.0f);
            for (std::size_t g = 0; g < numGroups; ++g)
            {
                float* base = groups + g * kGroupFloats;
                V zra, zia, zrb, zib;
                track<Rule>(base, threshold, false, zra, zia);
                track<Rule>(base + kWidth, threshold, false, zrb, zib);
                const V pa = accumulate(base, s, zra, zia);
                const V pb = accumulate(base + kWidth, s, zrb, zib);
                peak = Ops::max(peak, Ops::max(pa, pb));
            }
        }
        accPower = omSigma * accPower + sigma * Ops::reduceMax(peak);
        const V threshold = Ops::set(accPower * ratio);
        for (std::size_t g = 0; g < numGroups; ++g)
        {
            float* base = groups + g * kGroupFloats;
            V zr, zi;
            track<Rule>(base, threshold, renormalizeAtEnd, zr, zi);
            track<Rule>(base + kWidth, threshold, renormalizeAtEnd, zr, zi);
        }
    }
};


inline void processBank(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                        bool sanitize)
{
    BankKernel::process(groups, numGroups, x, n, stride, sanitize);
}

template <TrackingRule Rule>
void processTracking(float* groups, std::size_t numGroups, const float* x, std::size_t n, std::size_t stride,
                     bool renormalize, float sigma, float ratio, float& accPower)
{
    TrackingKernel::process<Rule>(groups, numGroups, x, n, stride, renormalize, sigma, ratio, accPower);
}

inline void sanitizeTracking(float* groups, std::size_t numGroups)
{
    TrackingKernel::sanitize(groups, numGroups);
}

constexpr Backend kBackend = {FCKFFT_BACKEND_ISA,
                              BankKernel::kLanes,
                              TrackingKernel::kLanes,
                              &processBank,
                              {&processTracking<TrackingRule::none>, &processTracking<TrackingRule::ewma>,
                               &processTracking<TrackingRule::normalizedChord>, &processTracking<TrackingRule::tangent>},
                              &sanitizeTracking};
