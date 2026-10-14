// cpu.h - which SIMD instructions this CPU has, checked at run time.
//
// On x86 the fast kernels need AVX2 + FMA (Intel since 2013, AMD since
// 2015). Rather than compiling the whole program for AVX2 (the binary
// would crash on older CPUs), only the kernels are compiled for it, with
// MRT2_AVX2 in front of the function, and the program asks the CPU at
// start-up whether to use them. One binary then runs on every x86 PC.
//
// On ARM nothing is checked: NEON is always there on 64-bit ARM.
#pragma once
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#define MRT2_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define MRT2_AVX2 // MSVC compiles AVX2 intrinsics anywhere; no attribute needed
#else
#include <immintrin.h>
#define MRT2_AVX2 __attribute__((target("avx2,fma")))
#endif
#endif

namespace mrt2 {

// True if the AVX2 + FMA kernels can run here (always false off x86).
// Set MRT2_NO_AVX2=1 in the environment to force the plain C++ kernels
// (for tests: both paths must give the same results).
bool cpu_avx2();

// The processor's name ("AMD EPYC 7763 64-Core Processor", "Apple M4"),
// for benchmark reports; "" if unknown.
std::string cpu_name();

} // namespace mrt2
