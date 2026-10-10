#include "PluginEditor.h"

#include <algorithm>
#include <cmath>

SpectrogramEditor::SpectrogramEditor(SpectrogramProcessor& owner)
    : AudioProcessorEditor(owner),
      processor_(owner),
      frames_(static_cast<size_t>(SpectrogramProcessor::kFifoFrames * SpectrogramProcessor::kBins)),
      floorAttachment_(owner.parameters, "floor", floor_)
{
    juce::ColourGradient gradient(juce::Colours::black, 0.0f, 0.0f, juce::Colours::white, 1.0f, 0.0f, false);
    gradient.addColour(0.25, juce::Colour(0xff1b0c41));
    gradient.addColour(0.5, juce::Colour(0xff9c179e));
    gradient.addColour(0.75, juce::Colour(0xffed7953));
    gradient.addColour(0.92, juce::Colour(0xfffbd524));
    for (size_t i = 0; i < palette_.size(); ++i)
        palette_[i] = gradient.getColourAtPosition(static_cast<double>(i) / static_cast<double>(palette_.size() - 1));

    floor_.setSliderStyle(juce::Slider::LinearHorizontal);
    floor_.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 20);
    floor_.setTextValueSuffix(" dB");
    floorLabel_.setText("Floor", juce::dontSendNotification);
    floorLabel_.attachToComponent(&floor_, true);
    loadLabel_.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(floor_);
    addAndMakeVisible(loadLabel_);

    setResizable(true, true);
    setResizeLimits(400, 260, 2400, 1600);
    setSize(800, 480);
    startTimerHz(60);
}

SpectrogramEditor::~SpectrogramEditor()
{
    stopTimer();
}

juce::Rectangle<int> SpectrogramEditor::plotArea() const
{
    return getLocalBounds().reduced(8).withTrimmedTop(32).withTrimmedLeft(48);
}

void SpectrogramEditor::resized()
{
    auto top = getLocalBounds().reduced(8).removeFromTop(24);
    loadLabel_.setBounds(top.removeFromRight(180));
    floor_.setBounds(top.withTrimmedLeft(48).removeFromLeft(320));
}

void SpectrogramEditor::timerCallback()
{
    const int count = processor_.popFrames(frames_.data(), SpectrogramProcessor::kFifoFrames);
    const float floorDb = processor_.parameters.getRawParameterValue("floor")->load();
    for (int f = 0; f < count; ++f)
        appendColumn(frames_.data() + static_cast<size_t>(f) * SpectrogramProcessor::kBins, floorDb);
    loadLabel_.setText("DSP load " + juce::String(100.0 * processor_.dspLoad(), 1) + " %", juce::dontSendNotification);
    if (count > 0)
        repaint(plotArea());
}

void SpectrogramEditor::appendColumn(const float* frame, float floorDb)
{
    image_.moveImageSection(0, 0, 1, 0, kHistory - 1, SpectrogramProcessor::kBins);
    juce::Image::BitmapData pixels(image_, kHistory - 1, 0, 1, SpectrogramProcessor::kBins,
                                   juce::Image::BitmapData::writeOnly);
    const float scale = static_cast<float>(palette_.size() - 1) / -floorDb;
    for (int k = 0; k < SpectrogramProcessor::kBins; ++k)
    {
        const float level = juce::jlimit(0.0f, static_cast<float>(palette_.size() - 1), (frame[k] - floorDb) * scale);
        pixels.setPixelColour(0, SpectrogramProcessor::kBins - 1 - k, palette_[static_cast<size_t>(level)]);
    }
}

void SpectrogramEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    const auto area = plotArea();
    g.drawImage(image_, area.toFloat(), juce::RectanglePlacement::stretchToFit);

    g.setColour(juce::Colours::white.withAlpha(0.7f));
    g.setFont(juce::FontOptions(12.0f));
    const float low = SpectrogramProcessor::kMinFrequency;
    const float high = processor_.topFrequency();
    const float span = std::log(high / low);
    for (float hz = 55.0f; hz < high; hz *= 2.0f)
    {
        if (hz < low)
            continue;
        const float y = static_cast<float>(area.getBottom()) - std::log(hz / low) / span * static_cast<float>(area.getHeight());
        g.drawHorizontalLine(static_cast<int>(y), static_cast<float>(area.getX()), static_cast<float>(area.getX() + 6));
        const juce::String text = hz >= 1000.0f ? juce::String(hz / 1000.0f, 2) + "k" : juce::String(static_cast<int>(hz));
        g.drawText(text, area.getX() - 46, static_cast<int>(y) - 7, 42, 14, juce::Justification::centredRight);
    }
}
