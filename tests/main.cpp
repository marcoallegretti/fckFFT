#include <cstdio>

#include <fckfft/simd.h>

#include "check.h"

int main()
{
    static const char* const names[] = {"scalar", "sse2", "avx", "avx2", "avx512", "neon"};
    const fckfft::Isa isas[] = {fckfft::Isa::scalar, fckfft::Isa::sse2,   fckfft::Isa::avx,
                                fckfft::Isa::avx2,   fckfft::Isa::avx512, fckfft::Isa::neon};
    int status = 0;
    for (fckfft::Isa isa : isas)
    {
        if (isa != fckfft::baselineIsa()
            && (static_cast<int>(isa) <= static_cast<int>(fckfft::baselineIsa())
                || static_cast<int>(isa) > static_cast<int>(fckfft::supportedIsa()) || isa == fckfft::Isa::avx))
            continue;
        fckfft::setIsaLimit(isa);
        std::printf("== %s\n", names[static_cast<int>(isa)]);
        status |= check::run();
    }
    return status;
}
