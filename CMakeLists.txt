cmake_minimum_required(VERSION 3.15)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/include/fckfft/fckfft.h FCKFFT_VERSION_DEFINES REGEX "^#define FCKFFT_VERSION_(MAJOR|MINOR|PATCH) ")
string(REGEX REPLACE "[^;]*MAJOR ([0-9]+);[^;]*MINOR ([0-9]+);[^;]*PATCH ([0-9]+)" "\\1.\\2.\\3"
    FCKFFT_HEADER_VERSION "${FCKFFT_VERSION_DEFINES}")

project(fckFFT VERSION ${FCKFFT_HEADER_VERSION} LANGUAGES CXX)

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/include/fckfft/fckfft.h FCKFFT_VERSION_PRERELEASE REGEX "^#define FCKFFT_VERSION_PRERELEASE ")
string(REGEX REPLACE "^.*\"(.*)\".*$" "\\1" FCKFFT_VERSION_PRERELEASE "${FCKFFT_VERSION_PRERELEASE}")
set(FCKFFT_VERSION ${PROJECT_VERSION})
if(FCKFFT_VERSION_PRERELEASE)
    string(APPEND FCKFFT_VERSION "-${FCKFFT_VERSION_PRERELEASE}")
endif()

file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/juce/fckfft/fckfft.h FCKFFT_JUCE_VERSION REGEX "^ *version: ")
string(REGEX REPLACE "^ *version: *" "" FCKFFT_JUCE_VERSION "${FCKFFT_JUCE_VERSION}")
if(NOT FCKFFT_JUCE_VERSION STREQUAL FCKFFT_VERSION)
    message(FATAL_ERROR "JUCE module version ${FCKFFT_JUCE_VERSION} differs from ${FCKFFT_VERSION}")
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(FCKFFT_TOP_LEVEL ON)
else()
    set(FCKFFT_TOP_LEVEL OFF)
endif()

option(FCKFFT_BUILD_TESTS "Build the test suite" ${FCKFFT_TOP_LEVEL})
option(FCKFFT_BUILD_BENCHMARKS "Build the benchmark" ${FCKFFT_TOP_LEVEL})
option(FCKFFT_NO_SIMD "Use the portable scalar kernels" OFF)
option(FCKFFT_NO_DISPATCH "Compile only the kernel selected by the build flags" OFF)
option(FCKFFT_TEST_MPS2_AN500 "Link the tests for the QEMU mps2-an500 Cortex-M7 machine" OFF)
option(FCKFFT_BUILD_FUZZERS "Build the libFuzzer targets (Clang)" OFF)

add_library(fckfft INTERFACE)
add_library(fckfft::fckfft ALIAS fckfft)
target_include_directories(fckfft INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(fckfft INTERFACE cxx_std_14)
if(FCKFFT_NO_SIMD)
    target_compile_definitions(fckfft INTERFACE FCKFFT_NO_SIMD)
endif()
if(FCKFFT_NO_DISPATCH)
    target_compile_definitions(fckfft INTERFACE FCKFFT_NO_DISPATCH)
endif()

if(FCKFFT_TOP_LEVEL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(DIRECTORY include/fckfft DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS fckfft EXPORT fckfftTargets)
    install(EXPORT fckfftTargets NAMESPACE fckfft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fckfft)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfig.cmake
        "include(\${CMAKE_CURRENT_LIST_DIR}/fckfftTargets.cmake)\n")
    if(PROJECT_VERSION_MAJOR EQUAL 0)
        set(FCKFFT_COMPATIBILITY SameMinorVersion)
    else()
        set(FCKFFT_COMPATIBILITY SameMajorVersion)
    endif()
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/fckfftConfigVersion.cmake
        COMPATIBILITY ${FCKFFT_COMPATIBILITY} ARCH_INDEPENDENT)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fckfft)
endif()

if(FCKFFT_BUILD_TESTS)
    enable_testing()
    add_executable(fckfft_tests
        tests/main.cpp
        tests/test_utilities.cpp
        tests/test_resonator_bank.cpp
        tests/test_tracking_resonator_bank.cpp
        tests/test_realtime.cpp)
    target_link_libraries(fckfft_tests PRIVATE fckfft)
    target_compile_definitions(fckfft_tests PRIVATE FCKFFT_TEST_DATA_DIR="${PROJECT_SOURCE_DIR}/tests/data")
    if(MSVC)
        target_compile_options(fckfft_tests PRIVATE /W4)
    else()
        target_compile_options(fckfft_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
    endif()
    if(FCKFFT_TEST_MPS2_AN500)
        target_sources(fckfft_tests PRIVATE tests/mps2/startup.cpp)
        target_link_options(fckfft_tests PRIVATE --specs=rdimon.specs
            -T${PROJECT_SOURCE_DIR}/tests/mps2/mps2_an500.ld)
        set_target_properties(fckfft_tests PROPERTIES CROSSCOMPILING_EMULATOR
            "qemu-system-arm;-M;mps2-an500;-nographic;-monitor;none;-semihosting-config;enable=on,target=native;-kernel")
    endif()
    add_test(NAME fckfft_tests COMMAND fckfft_tests)
    set_tests_properties(fckfft_tests PROPERTIES TIMEOUT 600)
    if(FCKFFT_TEST_MPS2_AN500)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        add_test(NAME fckfft_kernel_code
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/mps2/check_kernels.py ${CMAKE_OBJDUMP}
                    $<TARGET_FILE:fckfft_tests>)
    endif()
endif()

if(FCKFFT_BUILD_BENCHMARKS AND NOT FCKFFT_TEST_MPS2_AN500)
    add_executable(fckfft_bench bench/bench.cpp)
    target_link_libraries(fckfft_bench PRIVATE fckfft)
endif()

if(FCKFFT_BUILD_FUZZERS)
    add_executable(fckfft_fuzz_banks tests/fuzz/fuzz_banks.cpp)
    target_link_libraries(fckfft_fuzz_banks PRIVATE fckfft)
    target_compile_options(fckfft_fuzz_banks PRIVATE -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all)
    target_link_options(fckfft_fuzz_banks PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
