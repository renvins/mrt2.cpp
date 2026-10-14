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
#define MRT2_AVX2 __attribute__((target("avx2,fma,f16c")))
#endif
#endif

namespace mrt2 {

// True if the AVX2 + FMA (+ F16C, for half-precision weights) kernels can
// run here (always false off x86). Every AVX2 CPU has F16C; checked anyway.
// Set MRT2_NO_AVX2=1 in the environment to force the plain C++ kernels
// (for tests: both paths must give the same results).
bool cpu_avx2();

// True if this ARM CPU has the int8 dot-product instruction (sdot: every
// Apple chip, Raspberry Pi 5, recent phones and servers; not the oldest
// 64-bit ARM cores). GCC's default ARM target predates it, so the kernel
// that uses it is compiled with a per-function attribute and chosen here.
// MRT2_NO_DOTPROD=1 forces the plain kernel. Always false off ARM.
bool cpu_dotprod();

// Whether the fast mode should store the codec's first layers as fp16
// (make_fp16 in codec.h). It halves the bytes those layers read, which
// pays where they wait on memory: measured faster on x86 (CI: -2.5 ms on
// Linux, -3.8 ms on Windows) and Apple chips (-1 ms on M4); slower on the
// Linux ARM server (+1.2 ms: plenty of bandwidth, so only the conversion
// cost remains) and neutral on Windows ARM. Fixed per platform, so a seed
// always gives the same music on a given machine.
inline bool cpu_prefers_fp16_weights() {
#if defined(MRT2_X86) || defined(__APPLE__)
    return true;
#else
    return false;
#endif
}

// The processor's name ("AMD EPYC 7763 64-Core Processor", "Apple M4"),
// for benchmark reports; "" if unknown.
std::string cpu_name();

} // namespace mrt2
