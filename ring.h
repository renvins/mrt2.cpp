// ring.h - a lock-free queue of floats between exactly two threads: one
// writes (the generator), one reads (the audio callback).
//
// The audio callback runs on the OS's real-time thread: if it ever waits
// (a lock, a memory allocation, a system call), the sound drops out. So
// neither side here ever waits: write() takes what fits, read() takes
// what is there, and both return how many floats they moved.
//
// How it works: two counters that only grow, `head` (floats written so
// far, changed only by the writer) and `tail` (floats read so far, changed
// only by the reader). head - tail is what is waiting to be read. Each
// side reads the other's counter with "acquire" and publishes its own with
// "release": once the reader sees the new head, it is guaranteed to also
// see the floats written before it (and the same the other way round).
#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>

namespace mrt2 {

class SpscRing {
public:
    // Room for at least `capacity` floats (rounded up to a power of two, so
    // a position is `counter & mask` instead of a division).
    explicit SpscRing(size_t capacity) {
        size_t n = 1;
        while (n < capacity) n *= 2;
        buf_.resize(n);
        mask_ = n - 1;
    }

    size_t capacity() const { return buf_.size(); }

    // Floats waiting to be read. Exact for the reader; for the writer, a
    // lower bound on the free space is capacity() - size().
    size_t size() const { return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire); }

    // Writer only. Copies min(n, free space) floats; returns that count.
    size_t write(const float *x, size_t n) {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_acquire);
        n = std::min(n, buf_.size() - (h - t));
        for (size_t i = 0; i < n; i++) buf_[(h + i) & mask_] = x[i];
        head_.store(h + n, std::memory_order_release);
        return n;
    }

    // Reader only. Copies min(n, waiting) floats; returns that count.
    size_t read(float *x, size_t n) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_acquire);
        n = std::min(n, h - t);
        for (size_t i = 0; i < n; i++) x[i] = buf_[(t + i) & mask_];
        tail_.store(t + n, std::memory_order_release);
        return n;
    }

private:
    std::vector<float> buf_;
    size_t mask_ = 0;
    // On separate cache lines: each is written by a different thread, and
    // sharing a line would make the two cores fight over it.
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

} // namespace mrt2
