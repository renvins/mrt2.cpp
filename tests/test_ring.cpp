// test_ring.cpp - the lock-free queue between the generator and the audio
// callback: nothing lost, nothing duplicated, order kept, under real
// concurrency (also run it under ThreadSanitizer).
//
// Usage: test_ring
#include "../ring.h"
#include "test.h"
#include <cstdio>
#include <thread>
#include <vector>

using namespace mrt2;

TEST(basic_rules) {
    SpscRing r(1000);
    CHECK(r.capacity() == 1024); // rounded up to a power of two
    std::vector<float> a(1500), b(1500);
    for (int i = 0; i < 1500; i++) a[i] = (float)i;
    CHECK(r.write(a.data(), 1500) == 1024); // only what fits
    CHECK(r.size() == 1024);
    CHECK(r.read(b.data(), 100) == 100);
    CHECK(b[0] == 0.0f && b[99] == 99.0f);
    CHECK(r.write(a.data() + 1024, 476) == 100); // 100 slots free again, wrapping around
    CHECK(r.read(b.data(), 2000) == 1024);       // only what is there
    CHECK(b[0] == 100.0f && b[923] == 1023.0f && b[924] == 1024.0f && b[1023] == 1123.0f);
    CHECK(r.read(b.data(), 10) == 0);            // empty: returns at once
}

TEST(two_threads_lose_nothing) {
    // The writer sends 0, 1, 2, ... in chunks of varying size; the reader
    // takes chunks of other sizes and checks every value arrives once, in order.
    const size_t total = 10'000'000;
    SpscRing r(4096);
    bool in_order = true;
    std::thread reader([&] {
        std::vector<float> buf(777);
        size_t expect = 0, k = 0;
        while (expect < total) {
            const size_t n = r.read(buf.data(), 1 + (k++ * 7919) % 777);
            for (size_t i = 0; i < n; i++, expect++)
                if (buf[i] != (float)(expect % 16777216)) in_order = false; // floats are exact below 2^24
        }
    });
    std::vector<float> chunk(1000);
    size_t sent = 0, k = 0;
    while (sent < total) {
        const size_t want = std::min<size_t>(total - sent, 1 + (k++ * 104729) % 1000);
        for (size_t i = 0; i < want; i++) chunk[i] = (float)((sent + i) % 16777216);
        sent += r.write(chunk.data(), want);
    }
    reader.join();
    std::printf("      10M floats through a 4096-float ring: %s\n", in_order ? "all in order" : "LOST OR REORDERED");
    CHECK(in_order);
    CHECK(r.size() == 0);
}

int main() {
    RUN(basic_rules);
    RUN(two_threads_lose_nothing);
    return test::summary("test_ring");
}
