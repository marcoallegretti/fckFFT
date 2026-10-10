#include <fckfft/fckfft.h>

#include <cmath>
#include <cstdio>

static_assert(FCKFFT_VERSION_MAJOR == PACKAGE_VERSION_MAJOR && FCKFFT_VERSION_MINOR == PACKAGE_VERSION_MINOR &&
                  FCKFFT_VERSION_PATCH == PACKAGE_VERSION_PATCH,
              "installed headers and package version differ");

int main()
{
    constexpr int kBins = 25;
    constexpr float kSampleRate = 48000.0f;
    float frequencies[kBins];
    fckfft::logSpacedFrequencies(110.0f, 1760.0f, kBins, frequencies);

    fckfft::DynamicResonatorBank bank;
    if (!bank.init(frequencies, kBins, kSampleRate))
        return 1;

    const int target = 12;
    float block[480];
    for (int n = 0; n < 48000; n += 480)
    {
        for (int i = 0; i < 480; ++i)
            block[i] = std::sin(6.2831853f * frequencies[target] * static_cast<float>(n + i) / kSampleRate);
        bank.process(block, 480);
    }

    float magnitudes[kBins];
    bank.magnitudes(magnitudes);
    int peak = 0;
    for (int k = 1; k < kBins; ++k)
        if (magnitudes[k] > magnitudes[peak])
            peak = k;
    std::printf("peak bin %d, magnitude %.4f\n", peak, magnitudes[peak]);
    return peak == target && std::fabs(magnitudes[peak] - 0.5f) < 0.01f ? 0 : 1;
}
