#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <algorithm>
#include <cmath>

SpectrogramProcessor::SpectrogramProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, "state", createLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout SpectrogramProcessor::createLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;
    layout.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{"floor", 1}, "Floor",
                                                           juce::NormalisableRange<float>(-140.0f, -30.0f, 1.0f),
                                                           -100.0f, juce::AudioParameterFloatAttributes().withLabel("dB")));
    return layout;
}

void SpectrogramProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    const float top = std::min(kMaxFrequency, 0.45f * static_cast<float>(sampleRate));
    std::vector<float> frequencies(kBins);
    fckfft::logSpacedFrequencies(kMinFrequency, top, kBins, frequencies.data());
    bank_.init(frequencies.data(), kBins, static_cast<float>(sampleRate));
    topFrequency_.store(top);
    mono_.assign(static_cast<size_t>(std::max(samplesPerBlock, 1)), 0.0f);
    frames_.assign(static_cast<size_t>(kFifoFrames * kBins), 0.0f);
    fifo_.reset();
    samplesPerFrame_ = std::max(1, static_cast<int>(std::lround(sampleRate / kFramesPerSecond)));
    samplesUntilFrame_ = samplesPerFrame_;
    loadMeasurer_.reset(sampleRate, samplesPerBlock);
}

void SpectrogramProcessor::releaseResources()
{
}

bool SpectrogramProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return layouts.getMainInputChannelSet() == out;
}

void SpectrogramProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const juce::AudioProcessLoadMeasurer::ScopedTimer timer(loadMeasurer_, buffer.getNumSamples());
    const int numSamples = buffer.getNumSamples();
    const int inputs = getTotalNumInputChannels();
    for (int ch = inputs; ch < getTotalNumOutputChannels(); ++ch)
        buffer.clear(ch, 0, numSamples);
    if (inputs == 0 || !bank_.isInitialized() || mono_.empty())
        return;

    const float gain = 1.0f / static_cast<float>(inputs);
    const int capacity = static_cast<int>(mono_.size());
    for (int start = 0; start < numSamples; start += capacity)
    {
        const int count = std::min(capacity, numSamples - start);
        juce::FloatVectorOperations::copyWithMultiply(mono_.data(), buffer.getReadPointer(0, start), gain, count);
        for (int ch = 1; ch < inputs; ++ch)
            juce::FloatVectorOperations::addWithMultiply(mono_.data(), buffer.getReadPointer(ch, start), gain, count);

        int offset = 0;
        while (offset < count)
        {
            const int step = std::min(count - offset, samplesUntilFrame_);
            bank_.process(mono_.data() + offset, static_cast<std::size_t>(step));
            offset += step;
            samplesUntilFrame_ -= step;
            if (samplesUntilFrame_ == 0)
            {
                pushFrame();
                samplesUntilFrame_ = samplesPerFrame_;
            }
        }
    }
}

void SpectrogramProcessor::pushFrame()
{
    const auto scope = fifo_.write(1);
    const int index = scope.blockSize1 > 0 ? scope.startIndex1 : (scope.blockSize2 > 0 ? scope.startIndex2 : -1);
    if (index < 0)
        return;
    float* frame = frames_.data() + static_cast<size_t>(index) * kBins;
    for (int k = 0; k < kBins; ++k)
        frame[k] = 20.0f * std::log10(std::max(2.0f * bank_.magnitude(static_cast<std::size_t>(k)), 1e-9f));
}

int SpectrogramProcessor::popFrames(float* destination, int maxFrames)
{
    const auto scope = fifo_.read(std::min(maxFrames, fifo_.getNumReady()));
    const auto copy = [this, &destination](int start, int size) {
        std::copy_n(frames_.data() + static_cast<size_t>(start) * kBins, static_cast<size_t>(size) * kBins, destination);
        destination += static_cast<size_t>(size) * kBins;
    };
    if (scope.blockSize1 > 0)
        copy(scope.startIndex1, scope.blockSize1);
    if (scope.blockSize2 > 0)
        copy(scope.startIndex2, scope.blockSize2);
    return scope.blockSize1 + scope.blockSize2;
}

juce::AudioProcessorEditor* SpectrogramProcessor::createEditor()
{
    return new SpectrogramEditor(*this);
}

void SpectrogramProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto xml = parameters.copyState().createXml())
        copyXmlToBinary(*xml, destData);
}

void SpectrogramProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto xml = getXmlFromBinary(data, sizeInBytes))
        if (xml->hasTagName(parameters.state.getType()))
            parameters.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SpectrogramProcessor();
}
