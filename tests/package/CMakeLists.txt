cmake_minimum_required(VERSION 3.15)

project(fckfft_package_test LANGUAGES CXX)

find_package(fckfft 0.2 REQUIRED)

add_executable(fckfft_package_test main.cpp)
target_link_libraries(fckfft_package_test PRIVATE fckfft::fckfft)
target_compile_definitions(fckfft_package_test PRIVATE
    PACKAGE_VERSION_MAJOR=${fckfft_VERSION_MAJOR}
    PACKAGE_VERSION_MINOR=${fckfft_VERSION_MINOR}
    PACKAGE_VERSION_PATCH=${fckfft_VERSION_PATCH})
