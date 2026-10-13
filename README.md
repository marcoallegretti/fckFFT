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
choose the time constants; `fckfft::alphaFromTimeConstant` converts from seconds,
`fckfft::alphaForBandwidth` from a −3 dB bandwidth in Hz and `fckfft::alphaForQ`
from a quality factor. A full-scale sine at a resonator's frequency produces a
magnitude of 0.5.

Banks come in three storage flavours with the same interface:

| Type | Storage |
|---|---|
| `StaticResonatorBank<N>` / `StaticTrackingResonatorBank<N>` | inline array sized for `N` resonators |
| `DynamicResonatorBank` / `DynamicTrackingResonatorBank` | `std::vector`, allocated in `init` |
| `ResonatorBank` / `TrackingResonatorBank` | caller-provided buffer of `storageSize(n)` floats |

`init` returns `false` for invalid parameters: a frequency outside `[0, sampleRate/2]`,
`alpha` or `beta` outside `(0, 1]` or too small for single precision (the coefficient
actually stored must realise the requested value within 25 %, which holds for every
`alpha` above about 2e-7), or storage that is too small. A rejected `init` or setter
leaves the bank exactly as it was.

`setFrequency`, `setAlpha` and `setBeta` retune one resonator while it runs. The
resonator keeps its current value, and its phase continues from where it was at the
new frequency.

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
`alpha / 2`) sets the tracking rate. Keep `gamma` at or below `alpha`: up to there,
every rule follows a double-precision model of the same algorithm, while at 1.5–2 ×
`alpha` the tracking loop amplifies rounding differences into frequency errors of
hundreds of Hz.

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

On x86, one binary carries SSE2, AVX2 + FMA and AVX-512 kernels and each bank picks
one in `init` from the processor it runs on, so a plugin built with default flags
runs AVX2 on every machine that has it. The default limit is AVX2: AVX-512 is used
only when the build targets it (`-mavx512f`) or after `fckfft::setIsaLimit(Isa::avx512)`,
because on some Xeon generations 512-bit arithmetic lowers the clock of the whole
core, other plugins included. `setIsaLimit(Isa::sse2)` gives identical results on
every machine; results otherwise differ between kernels in the last bits, since
AVX2 fuses multiply-adds. `bank.isa()` reports the kernel in use.

On ARM the kernel is chosen at compile time: NEON, or scalar on Cortex-M. Define
`FCKFFT_NO_SIMD` to force the scalar kernel, or `FCKFFT_NO_DISPATCH` to compile only
the kernel the build flags target.

Every translation unit that uses fckFFT must be compiled with the same instruction
set flags. The library's symbols carry those flags, so passing a bank between units
built with different flags is a link error rather than silent corruption.

## Accuracy

- **noFFT parity.** Compared with noFFT's output (Apple Accelerate) on an 88-bin
  piano-range chirp, the maximum magnitude difference is 6.2e-6. Against a
  double-precision evaluation of the same recurrence, fckFFT deviates by 7.0e-6 in
  magnitude and 7.0e-6 in complex value; noFFT deviates by 2.4e-6 in magnitude and
  7.6e-4 in complex value, because its per-sample phasor drifts in float.
- **Tracking parity.** The tracking bank matches a double-precision model of the
  reference tracking algorithm within 0.03 Hz and 2.1e-4 magnitude on tones, vibrato
  and chirps.
- **Instantaneous frequency.** With the `none` rule, the estimate is accurate to
  0.01–0.02 Hz on steady tones. The tuner example reads to within 0.21 cents from
  E2 to E6, harmonics included, and settles within 1 cent in 14–190 ms.

## Frequency response

Each resonator (with `beta = alpha`) is a pair of identical complex one-pole filters,
so its magnitude response falls by 12 dB per octave of offset and its −3 dB bandwidth
is `0.205·alpha·sampleRate` Hz for small `alpha`. The table gives the response of the
Resonate heuristic `alpha` at 48 kHz, computed from the transfer function and
verified on the bank with tones (within 0.03 dB at 55 and 440 Hz, and 2e-7 in
magnitude against a double-precision recurrence at 4186 Hz):

| Frequency | `alpha` | Q | −3 dB width | Noise bandwidth | −1 / +1 semitone | −2 / +2 semitones | 10–90 % rise | Time to 90 % |
|---|---|---|---|---|---|---|---|---|
| 27.5 Hz | 0.00039 | 7.1 | 3.9 Hz | 4.7 Hz | −2.0 / −2.2 dB | −6.0 / −7.1 dB | 178 ms | 206 ms |
| 110 Hz | 0.00112 | 10.0 | 11.0 Hz | 13.4 Hz | −3.6 / −4.0 dB | −9.4 / −10.8 dB | 62 ms | 72 ms |
| 440 Hz | 0.00346 | 12.9 | 34.1 Hz | 41.6 Hz | −5.4 / −5.9 dB | −12.6 / −14.2 dB | 20 ms | 23 ms |
| 1760 Hz | 0.01123 | 15.8 | 111 Hz | 136 Hz | −7.3 / −7.9 dB | −15.5 / −17.2 dB | 6.2 ms | 7.2 ms |
| 4186 Hz | 0.02379 | 17.7 | 237 Hz | 289 Hz | −8.4 / −9.0 dB | −17.1 / −18.9 dB | 2.9 ms | 3.4 ms |

The heuristic therefore favours speed over selectivity: with semitone-spaced bins, a
tone also appears in its neighbours at −2 to −9 dB. A Hann-windowed STFT bin with
the same −3 dB width has about the same latency (half its window: 65 ms at 110 Hz,
21 ms at 440 Hz, 3.1 ms at 4186 Hz, against the times to 90 % above) and rejects
the neighbouring semitone slightly better (−3.8 to −13.7 dB), but rejects the second
semitone far better (−17 to −47 dB against −9 to −19 dB), because the resonator's
skirts fall at only 12 dB per octave. To trade latency for selectivity, choose
`alpha` with `alphaForQ`: at Q = 17 the neighbouring semitone is at least 8 dB down
everywhere, and the time to 90 % becomes 490 ms at 27.5 Hz, 31 ms at 440 Hz and
3.2 ms at 4186 Hz. A steeper resonator would not help here: at the same −3 dB
width, four cascaded poles instead of two improve the neighbouring semitone by less
than 1 dB and the second semitone by 2–5 dB, for twice the arithmetic and 16 % more
time to 90 %.

## Guarantees

**Real time.** `process()` performs no heap allocation, takes no locks and makes no
system calls. On every SIMD backend and on the GCC-built scalar kernel, the number of
executed instructions per block is identical for silence, noise, full-scale tones and
decaying signals (verified with Callgrind; Clang's scalar build varies by less than
0.1%). On ARM VFP targets the scalar kernel selects
values with `vcmp`/`vmrs`/conditional moves instead of branches. The only
data-dependent work is the recovery path described below, which adds 0.02% when it
triggers. The read-outs take the same time at any uptime. On x86 (2.1 GHz Xeon),
`powers` and `magnitudes` cost 1–2 ns per resonator, `values` 4–5 ns and
`phases` 16–21 ns; on the Cortex-M7 they execute 22, 29, 82 and 190 instructions
per resonator.

**Floating-point environment.** `process()` scopes flush-to-zero on x86 (MXCSR),
AArch64 (FPCR) and 32-bit ARM VFP (FPSCR), so decaying resonators never hit denormal
slow paths, whatever the host's floating-point state. Without it, x86 slowed down
about 100× after one second of silence.

**Invalid input.** A NaN or infinite input sample corrupts the resonators for at most
128 samples: at the next 128-sample check, every resonator whose state is
not finite is reset, and the rest are untouched. Recovery relies on IEEE
semantics, so do not compile with `-ffinite-math-only` or `-ffast-math`.

**Errors.** No function throws, except the `init` of the `Dynamic` banks, which can
throw `std::bad_alloc` from `std::vector`. Every other member is `noexcept`. `init`
and the setters return `false` for invalid parameters and then change nothing, so a
bank that fails to re-initialise keeps its previous configuration. An uninitialised
bank ignores `process`.
Out-of-range indices passed to the read-out functions, including any index on an
uninitialised bank, are not checked.

**Threads.** A bank has no internal synchronisation. `process`, the setters and
the read-outs of one bank must not run concurrently; separate banks are independent.
To display results on another thread, copy them out after `process` (the JUCE
example uses a lock-free FIFO).

**Ownership.** Banks are not copyable. `Dynamic` banks and banks over caller-owned
storage are movable (for example into a `std::vector`, one per channel); a moved-from
bank is uninitialised. `Static` banks are neither copyable nor movable because their
storage is inline. The owning banks do not convert to `ResonatorBank` or
`TrackingResonatorBank`, so their storage cannot be moved out from under them.

**Precision limits** (float32 state, measured):

| Quantity | Limit |
|---|---|
| Phase | within 2e-6 rad of the double-precision recurrence at the heuristic time constants |
| Phase reference | advances by a correctly rounded 64-bit fraction of a turn per sample, so it drifts by at most 1.7e-19 rad per sample: measured, below 6e-8 rad after 30 days and 2e-7–4.4e-7 rad after one year at 44.1–96 kHz |
| Time constant | relative error ≈ 3e-8 / α: 1.7e-4 at α = 1e-4, 1.8e-3 at α = 1e-5; α below 1e-5 (τ > 2 s at 48 kHz) is not recommended |
| Input amplitude | exact up to ±1e18; larger values overflow the power |
| Long-run stability | after one hour of input, magnitude and phase match the double-precision recurrence as closely as after one second (3.4e-7, 2e-7 rad) |

## Performance

### Desktop

Intel Xeon @ 2.1 GHz (shared virtual machine), 48 kHz, 48-sample blocks,
`bench/bench.cpp`, best median over repeated runs, ns per input sample:

| Kernel, bins | SSE2 | AVX2 + FMA | AVX-512 |
|---|---|---|---|
| fixed, 88 | 45 | 18 | 10 |
| fixed, 264 | 128 | 50 | 31 |
| fixed, 440 | 204 | 85 | 53 |
| fixed, 880 | 424 | 170 | 96 |
| tracking `none`, 88 | 97 | 57 | 24 |
| tracking chord, 88 | 279 | 134 | 102 |
| tracking tangent, 88 | 341 | 118 | 98 |
| tracking EWMA, 88 | 572 | 273 | 160 |

On the same machine and instruction set (AVX2 + FMA), the Rust
[resonators](https://github.com/jhartquist/resonators) crate takes 68.6, 182, 276 and
499 ns per sample for 88, 264, 440 and 880 bins, so the AVX2 fixed bank is 2.9–3.7×
faster. Its README reports it 1.6× faster than noFFT on an Apple M2.

The fixed bank runs each resonator as a complex one-pole filter at its own frequency,
`v = (1 − α)·e^(−jω)·v + x`, followed by a second complex pole for the β smoothing,
instead of a baseband EWMA behind a rotating phasor. The two are the same filter:
the state differs only by the rotation `e^(−jωn)`, which `values` and `phases` apply
at read-out. This removes the phasor, its renormalisation and four multiplies per
resonator per sample, so the SIMD kernels run on 8 fused multiply-adds per resonator
per sample and 32 bytes of state. Each group of resonators stays in registers for
a whole block. On the same machine the previous kernel measured 50, 28 and 15 ns per
sample at 88 bins on SSE2, AVX2 and AVX-512.

Exact tracking advances every resonator in lockstep, one sample at a time, because
each sample's threshold depends on the strongest resonator in the bank, so its state
(96 bytes per resonator) streams through the cache on every sample. At 880 bins this
state (84 KB) exceeds the L1 data cache of every x86 machine measured here (32–48 KB),
yet the cost per bin at 880 bins is 0.8–1.0× the cost per bin at 88 bins, for the
chord and EWMA rules, on this machine and on the four x86 runners below. Without a
tracking rule the threshold has no effect on the state, so the `none` rule runs each
group of resonators through up to 32 samples in registers and applies the per-sample
peaks to the reference power afterwards, with bit-identical results.

### Continuous integration runners

Single runs (median of 9) on shared GitHub-hosted machines, so differences of ±30%
between runs of the same job are common and the machine behind a job changes from
run to run. ns per input sample, 48-sample blocks:

| Machine, build | fixed 88 | fixed 880 | chord 88 | chord 880 | EWMA 88 | EWMA 880 |
|---|---|---|---|---|---|---|
| AMD EPYC 9V74, GCC, AVX2 + FMA | 13.7 | 128 | 129 | 1139 | 256 | 1938 |
| AMD EPYC 7763, Clang, AVX2 + FMA | 19.1 | 176 | 153 | 1314 | 240 | 2200 |
| Intel Xeon 8370C, MSVC, AVX2 + FMA | 16.9 | 320 | 150 | 1299 | 244 | 2180 |
| AMD EPYC 7763, MSVC, SSE2 | 45.6 | 427 | 341 | 3243 | 588 | 6193 |
| Apple M1 (virtual), Clang, NEON | 24.8 | 220 | 167 | 1670 | 308 | 3019 |

On the same Apple M1 runner, the previous phasor kernel measured 31–34 ns at 88 bins
and 318–540 ns at 880 bins over two runs. The resonators README reports 41.9 ns
(88 bins) and 357 ns (880 bins) for the Rust crate on an Apple M2 Max, and 66.8 and
578 ns for noFFT with Accelerate; on the slower M1, fckFFT takes 24.8 and 220 ns.

### Cortex-M7 (Daisy Seed)

Cycle counts from the LLVM Cortex-M7 pipeline model (`llvm-mca`), applied to the hot
loops that GCC generates with libDaisy's flags. These are model estimates; run
`examples/daisy/benchmark` to measure on hardware.

| Kernel | Cycles per resonator per sample | CPU at 480 MHz, 48 kHz |
|---|---|---|
| fixed | 15.8 | 88 bins: 14%, 256 bins: 40% |
| tracking `none` | ~50 | 49 bins: 24% |
| tracking tangent / chord | ~211 / ~215 | 32 bins: ~68% |
| tracking EWMA | ~233 | 32 bins: ~75% |

On this model, the reference loop order (every sample streams every resonator's
state through memory) costs 77 cycles per resonator per sample, about 4.9× the fixed
kernel. The VFP's fused multiply-add overwrites its accumulator, so the scalar kernel
uses a copy-free formulation with 11 operations per resonator per sample instead of
the 8 fused operations of the SIMD kernels; the previous phasor kernel measured 17.5
cycles on this model. The test suite runs on a Cortex-M7 under QEMU (`mps2-an500`).

Memory per resonator: 64 bytes for the fixed bank (32 of state and coefficients) and
96 bytes for the tracking bank on the scalar kernel, plus padding to the kernel's lane
count.

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
  normalises the step exactly after each update; a single Newton step is not enough
  at wide bandwidths, where it left magnitude errors of 0.44 at `alpha` = 0.5.
- **Power smoothing.** The reference C++ tracking bank leaves its power-smoothing
  coefficient uninitialised. It defaults to 1 (no smoothing), as in the Swift version.
- **Vector edge cases.** The reference NEON threshold merge writes its scalar tail
  to the wrong array and reads out of bounds for fewer than four resonators. fckFFT
  has no separate tail path.
- **Resonator form.** The fixed bank has no phasor: each resonator is a complex
  one-pole filter at its own frequency, and the read-out rotates the state back to
  the baseband from a 64-bit integer phase. Magnitudes match the reference recurrence
  within rounding, the phase drifts by at most 1.7e-19 rad per sample, and results do
  not depend on block size.
  Tracking banks keep the phasor form and renormalise it every 32 samples, counted
  from the last reset, instead of once per `update` call.
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
