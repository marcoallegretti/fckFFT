#ifndef FCKFFT_TESTS_CHECK_H
#define FCKFFT_TESTS_CHECK_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace check {

struct Case
{
    const char* name;
    void (*fn)();
};

inline std::vector<Case>& cases()
{
    static std::vector<Case> all;
    return all;
}

inline int& failures()
{
    static int count = 0;
    return count;
}

struct Register
{
    Register(const char* name, void (*fn)()) { cases().push_back(Case{name, fn}); }
};

inline void fail(const char* file, int line, const char* expr)
{
    ++failures();
    std::printf("  FAIL %s:%d: %s\n", file, line, expr);
}

inline void failNear(const char* file, int line, const char* expr, double a, double b, double tol)
{
    ++failures();
    std::printf("  FAIL %s:%d: %s (%.9g vs %.9g, tol %.3g)\n", file, line, expr, a, b, tol);
}

inline double sameTolerance(double reference)
{
#if defined(__FAST_MATH__)
    return 1e-5 * std::fabs(reference) + 1e-7;
#else
    (void)reference;
    return 0.0;
#endif
}

inline float fromBits(std::uint32_t u)
{
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

inline bool finite(float x)
{
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return (u & 0x7F800000u) != 0x7F800000u;
}

inline int run()
{
    int failedCases = 0;
    failures() = 0;
    for (const Case& c : cases())
    {
        const int before = failures();
        c.fn();
        const bool ok = failures() == before;
        if (!ok)
            ++failedCases;
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
    }
    std::printf("%d/%d passed\n", static_cast<int>(cases().size()) - failedCases, static_cast<int>(cases().size()));
    return failedCases == 0 ? 0 : 1;
}

}

#define CHECK_CONCAT_INNER(a, b) a##b
#define CHECK_CONCAT(a, b) CHECK_CONCAT_INNER(a, b)

#define TEST_CASE(name)                                                                                                \
    static void CHECK_CONCAT(test_, name)();                                                                           \
    static const check::Register CHECK_CONCAT(reg_, name)(#name, &CHECK_CONCAT(test_, name));                          \
    static void CHECK_CONCAT(test_, name)()

#define CHECK(expr)                                                                                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
            check::fail(__FILE__, __LINE__, #expr);                                                                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        const double check_a = static_cast<double>(a), check_b = static_cast<double>(b);                               \
        if (!(std::fabs(check_a - check_b) <= (tol)))                                                                  \
            check::failNear(__FILE__, __LINE__, #a " ~ " #b, check_a, check_b, (tol));                                 \
    } while (0)

#define CHECK_SAME(a, b) CHECK_NEAR(a, b, check::sameTolerance(static_cast<double>(b)))

#endif
