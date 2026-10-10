# fckFFT

A portable, header-only C++14 implementation of Alexandre R. J. François's
[Resonate](https://alexandrefrancois.org/Resonate/) algorithm: low-latency spectral
analysis with a bank of resonators, updated per sample, no FFT, no windowing, no
buffering.

The reference implementation, [noFFT](https://github.com/alexandrefrancois/noFFT), and the
[Oscillators](https://github.com/alexandrefrancois/Oscillators) package depend on Apple's
Accelerate framework. fckFFT has no platform dependency. It runs on desktop operating
systems, on bare-metal microcontrollers such as the Electro-Smith Daisy, and anywhere
else a C++14 compiler is available.

- **ResonatorBank**: a bank of fixed-frequency resonators with complex, power,
  magnitude and phase outputs.
- **TrackingResonatorBank**: a bank of self-tuning resonators that follow the
  frequency components of the input, with the EWMA, normalized-chord and tangent
  tracking rules. The `none` rule keeps the frequencies fixed and estimates each
  bin's instantaneous frequency (super-resolution).
- Frequency layouts (log, linear, MIDI, mel with Slaney or HTK scaling) and
  time-constant helpers.
- SIMD kernels for SSE2, AVX, AVX2/FMA, AVX-512, NEON (AArch64 and ARMv7) and a
  scalar kernel tuned for in-order cores such as the Cortex-M7.
- No exceptions, no RTTI and no allocation in the audio path. Storage is owned
  by the caller, so banks can live in static memory, in DTCM or in a `std::vector`.
- Output does not depend on how the input is split into blocks: processing
  sample by sample or in blocks of any size gives bit-identical results.

## Quick start

```cpp
#include <fckfft/fckfft.h>

float frequencies[88];
fckfft::logSpacedFrequencies(27.5f, 4186.0f, 88, frequencies);

fckfft::StaticResonatorBank<88> bank;
bank.init(frequencies, 88, 48000.0f);

bank.process(input, numSamples);
float magnitudes[88];
bank.magnitudes(magnitudes);
```

Without `alphas`, each resonator uses the per-frequency heuristic from the Resonate
paper (`fckfft::heuristicAlpha`). Pass explicit `alphas` and `betas` arrays to
choose the time constants; `fckfft::alphaFromTimeConstant` converts from seconds.
A full-scale sine at a resonator's frequency produces a magnitude of 0.5.

Banks come in three storage flavours with the same interface:

| Type | Storage |
|---|---|
| `StaticResonatorBank<N>` / `StaticTrackingResonatorBank<N>` | inline array sized for `N` resonators |
| `DynamicResonatorBank` / `DynamicTrackingResonatorBank` | `std::vector`, allocated in `init` |
| `ResonatorBank` / `TrackingResonatorBank` | caller-provided buffer of `storageSize(n)` floats |

`init` returns `false` for invalid parameters: a frequency outside `[0, sampleRate/2]`,
`alpha` or `beta` outside `(0, 1]`, or storage that is too small.

### Frequency tracking

```cpp
fckfft::DynamicTrackingResonatorBank bank;
bank.init(frequencies, 88, 48000.0f, fckfft::TrackingRule::normalizedChord, -60.0f);
bank.process(input, numSamples);
float tuned = bank.resonantFrequency(k);
float estimate = bank.instantaneousFrequency(k);
```

Resonators retune only while their power is within `thresholdDb` of the strongest
resonator in the bank, and return to their natural frequency below it.
`setPowerTimeConstant` smooths the reference power, and `gamma` (by default
`alpha / 2`) sets the tracking rate.

## Integration

### CMake

```cmake
add_subdirectory(fckFFT)
target_link_libraries(my_target PRIVATE fckfft::fckfft)
```

`cmake --install` also exports a `fckfft` package for `find_package(fckfft)`.

### Electro-Smith Daisy (libDaisy)

The library is header-only. Add the include path to the project Makefile:

```make
FCKFFT_DIR ?= ../fckFFT
C_INCLUDES += -I$(FCKFFT_DIR)/include
```

It builds with libDaisy's defaults (`-O2 -std=gnu++14 -fno-exceptions -fno-rtti`).
Place banks in DTCM for deterministic, zero-wait-state access:

```cpp
DTCM_MEM_SECTION fckfft::StaticResonatorBank<128> bank;
```

`examples/daisy/benchmark` measures every kernel on the device with the DWT cycle
counter and prints the results over USB serial. `examples/daisy/tuner` is a
chromatic tuner that uses the tracking bank's instantaneous frequency estimate.
Build either one with `make LIBDAISY_DIR=/path/to/libDaisy`.

### JUCE

`juce/fckfft` is a JUCE module. It includes the headers from this repository, so add
it by path rather than copying it:

```cmake
juce_add_module(path/to/fckFFT/juce/fckfft)
target_link_libraries(MyPlugin PRIVATE fckfft)
```

In Projucer, add the module from `path/to/fckFFT/juce` without enabling "copy
module". `examples/juce/spectrogram` is a VST3/AU/Standalone plugin that draws a
scrolling spectrogram and reports its DSP load. Build it with
`cmake -S examples/juce/spectrogram -B build -DJUCE_DIR=/path/to/JUCE`.

## SIMD selection

The kernel is chosen at compile time from the target flags: AVX-512 (`-mavx512f`),
AVX (`-mavx`, with FMA when `-mfma` or MSVC `/arch:AVX2` is set), SSE2 (the x86-64
default), NEON, or scalar. Define `FCKFFT_NO_SIMD` to force the scalar kernel.
Plugins built with default x86-64 flags use SSE2; enabling AVX2 and FMA roughly
halves the cost.

`process` scopes flush-to-zero on x86 (MXCSR), AArch64 (FPCR) and 32-bit ARM VFP
(FPSCR), so decaying resonators never hit denormal slow paths, whatever the host's
floating-point state. Without it, x86 slowed down about 100× after one second of
silence.

## Accuracy

- **noFFT parity.** Compared with noFFT's output (Apple Accelerate) on an 88-bin
  piano-range chirp, the maximum magnitude difference is 2.9e-6. The maximum complex
  difference is 5.8e-4. That comes from phase drift, since both float implementations
  round the per-sample phasor increment. Against an extended-precision evaluation,
  noFFT deviates by 5.7e-4 (complex). fckFFT deviates by 2.0e-4 (complex) and 5.6e-7
  (magnitude), because it computes the phasor increment in double precision before
  rounding.
- **Tracking parity.** The tracking bank matches a double-precision model of the
  reference tracking algorithm within 0.03 Hz and 2.1e-4 magnitude on tones, vibrato
  and chirps.
- **Instantaneous frequency.** With the `none` rule, the estimate is accurate to
  0.01–0.02 Hz on steady tones. The tuner example reads to within 0.21 cents from
  E2 to E6, harmonics included, and settles within 1 cent in 14–190 ms.

## Performance

### Desktop

Intel Xeon @ 2.1 GHz, 48 kHz, 48-sample blocks, `bench/bench.cpp`, median of 15
runs, ns per input sample:

| Kernel, bins | SSE2 | AVX2 + FMA | AVX-512 |
|---|---|---|---|
| fixed, 88 | 77 | 32 | 21 |
| fixed, 264 | 150 | 86 | 59 |
| fixed, 440 | 258 | 136 | 92 |
| fixed, 880 | 491 | 294 | 183 |
| tracking `none`, 88 | 183 | 92 | |
| tracking chord, 88 | 312 | 219 | |
| tracking tangent, 88 | 301 | 142 | |
| tracking EWMA, 88 | 580 | 313 | |

On the same machine and instruction set (AVX2 + FMA), the Rust
[resonators](https://github.com/jhartquist/resonators) crate takes 68.6, 182, 276 and
499 ns per sample for 88, 264, 440 and 880 bins. Its README reports it 1.6× faster
than noFFT on an Apple M2.

The fixed bank keeps each group of resonators in registers for a whole block,
instead of streaming the full state through memory for every sample. It also stores
`r / alpha` instead of `r`, which removes one multiply per resonator per sample.

### Cortex-M7 (Daisy Seed)

Cycle counts from the LLVM Cortex-M7 pipeline model (`llvm-mca`), applied to the hot
loops that GCC generates with libDaisy's flags. These are model estimates; run
`examples/daisy/benchmark` to measure on hardware.

| Kernel | Cycles per resonator per sample | CPU at 480 MHz, 48 kHz |
|---|---|---|
| fixed | 17.5 | 88 bins: 15%, 256 bins: 45% |
| tracking `none` | ~86 | 49 bins: 42% |
| tracking tangent / chord | ~157 / ~160 | 32 bins: ~51% |
| tracking EWMA | ~216 | 32 bins: ~69% |

On this model, the reference loop order (every sample streams every resonator's
state through memory) costs 77 cycles per resonator per sample, about 4.4× the fixed
kernel. The test suite runs on a Cortex-M7 under QEMU (`mps2-an500`).

Memory per resonator: 56 bytes for the fixed bank and 96 bytes for the tracking bank
on the scalar kernel, plus padding to the kernel's lane count.

## Differences from the reference implementations

The algorithm follows noFFT and the Oscillators C++/Swift code. These changes come
from measurements against those implementations and against a double-precision
model:

- **Tracking from silence.** The normalized-chord and tangent rules of the
  vectorised reference produce NaN when the delta phase is zero, as it is when
  processing starts from silence. fckFFT skips the update in that case, as the
  reference's scalar tracking resonator does.
- **Phasor-step magnitude.** The chord and tangent updates change the magnitude of
  the phasor step. Left unnormalised (as in the reference), this caused magnitude
  errors up to 0.22 and frequency errors up to 255 Hz in the chord rule. fckFFT
  renormalises the step after each update.
- **Power smoothing.** The reference C++ tracking bank leaves its power-smoothing
  coefficient uninitialised. It defaults to 1 (no smoothing), as in the Swift version.
- **Vector edge cases.** The reference NEON threshold merge writes its scalar tail
  to the wrong array and reads out of bounds for fewer than four resonators. fckFFT
  has no separate tail path.
- **Renormalisation timing.** Phasors are renormalised every 128 samples (32 for
  tracking banks), counted from the last reset, instead of once per `update` call.
  Long blocks therefore cannot drift, and results do not depend on block size.
- **Instantaneous frequency.** The readout uses the delta phase smoothed with
  `gamma`, as in the reference scalar resonator. Tracking itself uses the raw delta
  phase, as in the reference bank.

## Tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

The suite covers noFFT parity, tracking parity with a double-precision model,
block-size independence, stride handling, padding lanes, long-run stability,
silence, parameter changes and the frequency utilities, the latter against
librosa's mel tables. It passes on SSE2, AVX, AVX2/FMA, AVX-512 and scalar builds
(GCC and Clang), AArch64 NEON, ARMv7 NEON and a bare-metal Cortex-M7.

To run the suite on the QEMU Cortex-M7 machine:

```sh
cmake -S . -B build-m7 -DCMAKE_TOOLCHAIN_FILE=cmake/arm-none-eabi-cortex-m7.cmake \
      -DFCKFFT_TEST_MPS2_AN500=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-m7
ctest --test-dir build-m7
```

## Credits

The Resonate algorithm is by Alexandre R. J. François
([project page](https://alexandrefrancois.org/Resonate/),
*Resonate: Efficient Low Latency Spectral Analysis of Audio Signals*, ICMC 2025).
The reference test fixture `tests/data/chirp_88.npz` was generated with noFFT by
John Hartquist for [resonators](https://github.com/jhartquist/resonators).

## License

MIT. See [LICENSE](LICENSE).
