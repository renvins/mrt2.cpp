// test.h - a tiny test harness: two counters and a few macros, no framework.
//
//   TEST(gelu) { CHECK(x > 0); CHECK_NEAR(y, 0.8411920, 1e-6); }
//   int main() { RUN(gelu); return test::summary("test_ops"); }
//
// A failed check prints the file, line, both values and their difference,
// then the test continues, so one run shows every failure at once.
#pragma once
#include <cmath>
#include <cstdio>

namespace test {

inline int checks = 0;
inline int failures = 0;

inline void check(bool ok, const char *expr, const char *file, int line) {
    checks++;
    if (!ok) {
        failures++;
        std::printf("    FAIL %s:%d: %s\n", file, line, expr);
    }
}

inline void check_near(double a, double b, double tol, const char *ea, const char *eb,
                       const char *file, int line) {
    checks++;
    // Written as !(diff <= tol) so that a NaN counts as a failure.
    if (!(std::fabs(a - b) <= tol)) {
        failures++;
        std::printf("    FAIL %s:%d: %s = %.9g, %s = %.9g, diff %.3g > %.3g\n",
                    file, line, ea, a, eb, b, std::fabs(a - b), tol);
    }
}

// Final line and exit code: 0 if every check passed.
inline int summary(const char *suite) {
    std::printf("%s: %d checks, %d failed -> %s\n", suite, checks, failures,
                failures ? "SOME TESTS FAILED" : "all tests passed");
    return failures ? 1 : 0;
}

} // namespace test

#define CHECK(cond) test::check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) test::check_near((a), (b), (tol), #a, #b, __FILE__, __LINE__)
#define TEST(name) static void name()
#define RUN(name)                                                              \
    do {                                                                       \
        int f0_ = test::failures, c0_ = test::checks;                          \
        name();                                                                \
        std::printf("%s  %-34s (%d checks)\n",                                 \
                    test::failures == f0_ ? "ok  " : "FAIL", #name,            \
                    test::checks - c0_);                                       \
    } while (0)
