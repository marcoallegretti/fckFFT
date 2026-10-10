cmake_minimum_required(VERSION 3.15)

project(fckFFT VERSION 1.0.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(FCKFFT_TOP_LEVEL ON)
else()
    set(FCKFFT_TOP_LEVEL OFF)
endif()

option(FCKFFT_BUILD_TESTS "Build the test suite" ${FCKFFT_TOP_LEVEL})
option(FCKFFT_BUILD_BENCHMARKS "Build the benchmark" ${FCKFFT_TOP_LEVEL})
option(FCKFFT_NO_SIMD "Use the portable scalar kernels" OFF)
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

if(FCKFFT_TOP_LEVEL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(DIRECTORY include/fckfft DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(TARGETS fckfft EXPORT fckfftTargets)
    install(EXPORT fckfftTargets NAMESPACE fckfft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fckfft)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfig.cmake
        "include(\${CMAKE_CURRENT_LIST_DIR}/fckfftTargets.cmake)\n")
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/fckfftConfigVersion.cmake
        COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/fckfftConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fckfft)
endif()

if(FCKFFT_BUILD_TESTS)
    enable_testing()
    add_executable(fckfft_tests
        tests/main.cpp
        tests/test_utilities.cpp
        tests/test_resonator_bank.cpp
        tests/test_tracking_resonator_bank.cpp)
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
