#pragma once

#include "PluginProcessor.h"

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_basics/juce_gui_basics.h>

#include <array>
#include <vector>

class SpectrogramEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit SpectrogramEditor(SpectrogramProcessor& owner);
    ~SpectrogramEditor() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    static constexpr int kHistory = 600;

    void timerCallback() override;
    void appendColumn(const float* frame, float floorDb);
    juce::Rectangle<int> plotArea() const;

    SpectrogramProcessor& processor_;
    juce::Image image_{juce::Image::RGB, kHistory, SpectrogramProcessor::kBins, true};
    std::array<juce::Colour, 256> palette_;
    std::vector<float> frames_;
    juce::Slider floor_;
    juce::Label floorLabel_;
    juce::Label loadLabel_;
    juce::AudioProcessorValueTreeState::SliderAttachment floorAttachment_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramEditor)
};
