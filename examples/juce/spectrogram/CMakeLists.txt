cmake_minimum_required(VERSION 3.22)

project(FckfftSpectrogram VERSION 1.0.0 LANGUAGES C CXX)

set(JUCE_DIR "" CACHE PATH "Path to a JUCE source tree; leave empty to use an installed JUCE package")

if(JUCE_DIR)
    add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/JUCE)
else()
    find_package(JUCE CONFIG REQUIRED)
endif()

juce_add_module(${CMAKE_CURRENT_SOURCE_DIR}/../../../juce/fckfft)

juce_add_plugin(FckfftSpectrogram
    COMPANY_NAME fckFFT
    PLUGIN_MANUFACTURER_CODE Fckf
    PLUGIN_CODE Fspg
    FORMATS AU VST3 Standalone
    PRODUCT_NAME "fckFFT Spectrogram"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    NEEDS_WEB_BROWSER FALSE
    NEEDS_CURL FALSE)

target_sources(FckfftSpectrogram
    PRIVATE
        PluginProcessor.cpp
        PluginEditor.cpp)

target_compile_definitions(FckfftSpectrogram
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(FckfftSpectrogram
    PRIVATE
        fckfft
        juce::juce_audio_utils
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
