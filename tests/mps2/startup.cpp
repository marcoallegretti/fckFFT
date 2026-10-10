#include <cstdint>

extern "C" {
extern std::uint32_t __stack_top;
void _start();
void Reset_Handler();
void Default_Handler();
}

extern "C" void Reset_Handler()
{
    *reinterpret_cast<volatile std::uint32_t*>(0xE000ED88u) |= 0xFu << 20;
    __asm__ __volatile__("dsb\n\tisb");
    _start();
    for (;;)
    {
    }
}

extern "C" void Default_Handler()
{
    for (;;)
    {
    }
}

using Handler = void (*)();

__attribute__((section(".vectors"), used)) const Handler kVectors[16] = {
    reinterpret_cast<Handler>(&__stack_top), Reset_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, Default_Handler, nullptr,
    nullptr, nullptr, nullptr, Default_Handler,
    Default_Handler, nullptr, Default_Handler, Default_Handler};
