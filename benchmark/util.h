#ifndef UTIL_H
#define UTIL_H

#ifdef __ARM_ARCH_ISA_A64
#include <stdint.h>

inline uint64_t rdtsc() {
    uint64_t val;
    __asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

#else
#include <stdint.h>
#include <x86intrin.h>

inline uint64_t rdtsc() { return __rdtsc(); }
#endif

#endif // UTIL_H
