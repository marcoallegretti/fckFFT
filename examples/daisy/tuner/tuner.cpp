#include "daisy_seed.h"

#include <fckfft/fckfft.h>

#include <cmath>

using namespace daisy;

namespace {

constexpr std::size_t kBins = 49;
constexpr float kLowestNote = 40.0f;
constexpr float kMinimumMagnitude = 0.005f;
constexpr float kInTuneCents = 5.0f;

const char* const kNoteNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};

DaisySeed hw;
DTCM_MEM_SECTION fckfft::StaticTrackingResonatorBank<kBins> bank;
volatile float detectedFrequency = 0.0f;
volatile float detectedMagnitude = 0.0f;

void AudioCallback(AudioHandle::InputBuffer in, AudioHandle::OutputBuffer out, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        out[0][i] = in[0][i];
        out[1][i] = in[1][i];
    }
    bank.process(in[0], size);
    std::size_t strongest = 0;
    float strongestPower = bank.power(0);
    for (std::size_t k = 1; k < kBins; ++k)
    {
        const float p = bank.power(k);
        if (p > strongestPower)
        {
            strongestPower = p;
            strongest = k;
        }
    }
    detectedMagnitude = std::sqrt(strongestPower);
    detectedFrequency = bank.instantaneousFrequency(strongest);
}

void printReading(int note, float cents, float frequency)
{
    const long centTenths = std::lround(cents * 10.0f);
    const long absTenths = centTenths < 0 ? -centTenths : centTenths;
    const long hundredthsHz = std::lround(frequency * 100.0f);
    hw.PrintLine("%-2s%d  %c%ld.%ld cents  %ld.%02ld Hz", kNoteNames[note % 12], note / 12 - 1,
                 centTenths < 0 ? '-' : '+', absTenths / 10, absTenths % 10, hundredthsHz / 100, hundredthsHz % 100);
}

}

int main()
{
    hw.Init(true);
    hw.SetAudioSampleRate(SaiHandle::Config::SampleRate::SAI_48KHZ);
    hw.SetAudioBlockSize(48);

    float frequencies[kBins];
    for (std::size_t k = 0; k < kBins; ++k)
        frequencies[k] = fckfft::midiToHz(kLowestNote + static_cast<float>(k));
    bank.init(frequencies, kBins, hw.AudioSampleRate(), fckfft::TrackingRule::none);

    hw.StartLog(false);
    hw.StartAudio(AudioCallback);

    for (;;)
    {
        System::Delay(100);
        const float magnitude = detectedMagnitude;
        const float frequency = detectedFrequency;
        if (magnitude < kMinimumMagnitude || !(frequency > 0.0f))
        {
            hw.SetLed(false);
            continue;
        }
        const float midi = fckfft::hzToMidi(frequency);
        const int note = static_cast<int>(std::lround(midi));
        if (note < 0)
            continue;
        const float cents = 100.0f * (midi - static_cast<float>(note));
        hw.SetLed(std::fabs(cents) < kInTuneCents);
        printReading(note, cents, frequency);
    }
}
