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

## Guarantees

**Real time.** `process()` performs no heap allocation, takes no locks and makes no
system calls. On every SIMD backend and on the GCC-built scalar kernel, the number of
executed instructions per block is identical for silence, noise, full-scale tones and
decaying signals (verified with Callgrind; Clang's scalar build varies by less than
0.1%). On ARM VFP targets the scalar kernel selects
values with `vcmp`/`vmrs`/conditional moves instead of branches. The only
data-dependent work is the recovery path described below, which adds 0.02% when it
triggers.

**Floating-point environment.** `process()` scopes flush-to-zero on x86 (MXCSR),
AArch64 (FPCR) and 32-bit ARM VFP (FPSCR), so decaying resonators never hit denormal
slow paths, whatever the host's floating-point state. Without it, x86 slowed down
about 100× after one second of silence.

**Invalid input.** A NaN or infinite input sample corrupts the resonators for at most
128 samples: at the next renormalisation boundary, every resonator whose state is
not finite is reset, and the rest are untouched. Recovery relies on IEEE
semantics, so do not compile with `-ffinite-math-only` or `-ffast-math`.

**Errors.** No function throws, except the `init` of the `Dynamic` banks, which can
throw `std::bad_alloc` from `std::vector`. Every other member is `noexcept`. `init`
returns `false` for invalid parameters, and an uninitialised bank ignores `process`.
Out-of-range indices passed to the read-out functions are not checked.

**Threads.** A bank has no internal synchronisation. `process`, the setters and
the read-outs of one bank must not run concurrently; separate banks are independent.
To display results on another thread, copy them out after `process` (the JUCE
example uses a lock-free FIFO).

**Ownership.** Banks are not copyable. `Dynamic` banks and banks over caller-owned
storage are movable (for example into a `std::vector`, one per channel); a moved-from
bank is uninitialised. `Static` banks are neither copyable nor movable because their
storage is inline.

**Precision limits** (float32 state, measured):

| Quantity | Limit |
|---|---|
| Resonator frequency | relative error ≤ 8e-8 (phase drifts by at most 2π·f·8e-8 rad/s against an exact clock) |
| Time constant | relative error ≈ 3e-8 / α: 1.7e-4 at α = 1e-4, 1.8e-3 at α = 1e-5; α below 1e-5 (τ > 2 s at 48 kHz) is not recommended |
| Input amplitude | exact up to ±1e18; larger values overflow the power |
| Long-run stability | after one hour of input, the response matches a freshly reset bank within 6e-8 |

## Performance

### Desktop

Intel Xeon @ 2.1 GHz (shared virtual machine), 48 kHz, 48-sample blocks,
`bench/bench.cpp`, best of three medians of 11 runs, ns per input sample:

| Kernel, bins | SSE2 | AVX2 + FMA | AVX-512 |
|---|---|---|---|
| fixed, 88 | 50 | 28 | 15 |
| fixed, 264 | 151 | 76 | 43 |
| fixed, 440 | 245 | 132 | 69 |
| fixed, 880 | 455 | 241 | 128 |
| tracking `none`, 88 | 107 | 84 | 65 |
| tracking chord, 88 | 295 | 158 | 106 |
| tracking tangent, 88 | 307 | 126 | 102 |
| tracking EWMA, 88 | 572 | 273 | 160 |

On the same machine and instruction set (AVX2 + FMA), the Rust
[resonators](https://github.com/jhartquist/resonators) crate takes 68.6, 182, 276 and
499 ns per sample for 88, 264, 440 and 880 bins, so the AVX2 fixed bank is 2.1–2.4×
faster. Its README reports it 1.6× faster than noFFT on an Apple M2.

The fixed bank keeps each group of resonators in registers for a whole block,
instead of streaming the full state through memory for every sample. It also stores
`r / alpha` instead of `r`, which removes one multiply per resonator per sample.

Exact tracking advances every resonator in lockstep, one sample at a time, because
each sample's threshold depends on the strongest resonator in the bank, so its state
(96 bytes per resonator) streams through the cache on every sample. At 880 bins this
state (84 KB) exceeds the L1 data cache of every x86 machine measured here (32–48 KB),
yet the cost per bin at 880 bins is 0.8–1.0× the cost per bin at 88 bins, for the
chord and EWMA rules, on this machine and on the four x86 runners below.

### Continuous integration runners

Single runs (median of 9) on shared GitHub-hosted machines, so differences of ±30%
between runs of the same job are common. ns per input sample, 48-sample blocks:

| Machine, build | fixed 88 | fixed 880 | chord 88 | chord 880 | EWMA 88 | EWMA 880 |
|---|---|---|---|---|---|---|
| AMD EPYC 9V45, GCC, AVX-512 | 17.2 | 158 | 90 | 793 | 170 | 1399 |
| AMD EPYC 7763, Clang, AVX2 | 30.8 | 276 | 150 | 1311 | 236 | 2112 |
| AMD EPYC 9V74, MSVC, AVX2 | 21.9 | 202 | 130 | 1164 | 209 | 1893 |
| Intel Xeon 8370C, MSVC, SSE2 | 58.7 | 591 | 340 | 3392 | 624 | 6243 |
| Apple M1 (virtual), Clang, NEON | 31.2–34.2 | 318–540 | 158–179 | 1559–2820 | 282–305 | 2814–4487 |

The Apple row gives the range over two runs. The resonators README reports 41.9 ns
(88 bins) and 357 ns (880 bins) for the Rust crate on an Apple M2 Max, and 66.8 and
578 ns for noFFT with Accelerate. On the slower M1, fckFFT is faster at 88 bins in
both runs; at 880 bins the runs straddle the Rust figure.

### Cortex-M7 (Daisy Seed)

Cycle counts from the LLVM Cortex-M7 pipeline model (`llvm-mca`), applied to the hot
loops that GCC generates with libDaisy's flags. These are model estimates; run
`examples/daisy/benchmark` to measure on hardware.

| Kernel | Cycles per resonator per sample | CPU at 480 MHz, 48 kHz |
|---|---|---|
| fixed | 17.5 | 88 bins: 15%, 256 bins: 45% |
| tracking `none` | ~86 | 49 bins: 42% |
| tracking tangent / chord | ~159 / ~164 | 32 bins: ~52% |
| tracking EWMA | ~233 | 32 bins: ~75% |

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

## Versioning

Releases follow semantic versioning. `FCKFFT_VERSION_MAJOR`, `FCKFFT_VERSION_MINOR`,
`FCKFFT_VERSION_PATCH` and `FCKFFT_VERSION_PRERELEASE` (empty for final releases) in
`fckfft.h` identify the headers in use. The CMake package takes its version from
these macros, and configuration fails if the JUCE module declares a different one.
Before 1.0, minor versions may break the API, so `find_package(fckfft 0.1)` accepts
0.1.x only.

## Tests

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

The suite covers noFFT parity, tracking parity with a double-precision model,
block-size independence, stride handling, padding lanes, long-run stability,
silence, invalid input, move semantics, parameter changes and the frequency
utilities, the latter against librosa's mel tables. A randomised test compares 120
configurations (1–40 bins from DC to Nyquist, α and β from 1e-3 to 1, random block
sizes and strides) with a double-precision model; the worst error is 1.5e-6.

Continuous integration runs the suite with GCC, Clang and MSVC on SSE2, AVX,
AVX2/FMA, AVX-512 and scalar builds, under AddressSanitizer and
UndefinedBehaviorSanitizer, on Apple Silicon, on AArch64 and ARMv7 NEON, and on a
bare-metal Cortex-M7. It also builds both Daisy examples against libDaisy and the
JUCE example plugin, builds a project against the installed CMake package
(`tests/package`), and fuzzes both banks for two minutes.

To fuzz locally (Clang):

```sh
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DFCKFFT_BUILD_FUZZERS=ON
cmake --build build-fuzz --target fckfft_fuzz_banks
./build-fuzz/fckfft_fuzz_banks -max_total_time=600
```

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
