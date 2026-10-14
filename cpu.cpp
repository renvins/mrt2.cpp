#include "cpu.h"
#include <cstdlib>
#include <cstring>
#if defined(MRT2_X86) && !(defined(_MSC_VER) && !defined(__clang__))
#include <cpuid.h>
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mrt2 {

#if defined(MRT2_X86)
static bool detect_avx2() {
#if defined(_MSC_VER) && !defined(__clang__)
    // Ask the CPU (cpuid), and check that the OS saves the 256-bit
    // registers on thread switches (xgetbv): without that, AVX2 code would
    // silently corrupt other threads' data.
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool fma = r[2] & (1 << 12), osxsave = r[2] & (1 << 27), avx = r[2] & (1 << 28);
    if (!fma || !osxsave || !avx) return false;
    if ((_xgetbv(0) & 6) != 6) return false; // XMM and YMM state enabled by the OS
    __cpuidex(r, 7, 0);
    return r[1] & (1 << 5); // AVX2
#else
    // The compiler's built-in does the same checks, OS support included.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}
#endif

bool cpu_avx2() {
#if defined(MRT2_X86)
    static const bool yes = !std::getenv("MRT2_NO_AVX2") && detect_avx2(); // checked once
    return yes;
#else
    return false;
#endif
}

std::string cpu_name() {
#if defined(MRT2_X86)
    // cpuid leaves 0x80000002..4 hold the 48-character brand string.
    unsigned r[12] = {};
    for (unsigned i = 0; i < 3; i++) {
#if defined(_MSC_VER) && !defined(__clang__)
        int v[4];
        __cpuid(v, (int)(0x80000002u + i));
        std::memcpy(r + 4 * i, v, 16);
#else
        __get_cpuid(0x80000002u + i, &r[4 * i], &r[4 * i + 1], &r[4 * i + 2], &r[4 * i + 3]);
#endif
    }
    char s[49] = {};
    std::memcpy(s, r, 48);
    std::string name(s);
    name.erase(0, name.find_first_not_of(' '));
    return name;
#elif defined(__APPLE__)
    char s[128] = {};
    size_t n = sizeof s - 1;
    return sysctlbyname("machdep.cpu.brand_string", s, &n, nullptr, 0) == 0 ? std::string(s) : std::string();
#else
    return "";
#endif
}

} // namespace mrt2
