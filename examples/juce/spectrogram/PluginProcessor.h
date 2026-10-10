#pragma once

#include <fckfft/fckfft.h>
#include <juce_audio_processors/juce_audio_processors.h>

#include <atomic>
#include <vector>

class SpectrogramProcessor final : public juce::AudioProcessor
{
public:
    static constexpr int kBins = 160;
    static constexpr int kFifoFrames = 512;
    static constexpr double kFramesPerSecond = 100.0;
    static constexpr float kMinFrequency = 30.0f;
    static constexpr float kMaxFrequency = 15000.0f;

    SpectrogramProcessor();

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    int popFrames(float* destination, int maxFrames);
    float topFrequency() const { return topFrequency_.load(); }
    double dspLoad() const { return loadMeasurer_.getLoadAsProportion(); }

    juce::AudioProcessorValueTreeState parameters;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void pushFrame();

    fckfft::DynamicResonatorBank bank_;
    std::vector<float> mono_;
    std::vector<float> frames_;
    juce::AbstractFifo fifo_{kFifoFrames};
    std::atomic<float> topFrequency_{kMaxFrequency};
    int samplesPerFrame_ = 480;
    int samplesUntilFrame_ = 480;
    juce::AudioProcessLoadMeasurer loadMeasurer_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramProcessor)
};
