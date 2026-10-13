// threads.h - a persistent thread pool for splitting a frame's big
// operations (convs, matrix products) across the CPU's cores.
//
//   set_threads(10);                         // once, at startup
//   pool().run(chunks, [&](int i) { ... });  // calls fn(0..chunks-1)
//
// Workers are created once and wait for work: creating threads per
// operation would cost more than many operations themselves. Chunks are
// handed out one at a time from a shared counter, so fast cores (Apple's
// performance cores) simply take more chunks than slow ones. The caller
// works on chunks too, and run() returns when every chunk is done.
//
// run() never allocates: the work is passed as a function pointer plus a
// context pointer, not a std::function (which may allocate on every call,
// and a frame makes ~300 calls).
//
// Waiting workers spin (with yield) so they react within microseconds;
// they use a core while waiting. Fine for generating audio; an idle-friendly
// mode can come with the real-time engine.
#pragma once
#include <atomic>
#include <thread>
#include <type_traits>
#include <vector>

namespace mrt2 {

class ThreadPool {
public:
    explicit ThreadPool(int threads); // total threads, including the caller
    ~ThreadPool();
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return n_; }

    // Call fn(i) for every i in [0, chunks), spread over all threads.
    template <class F> void run(int chunks, F &&fn) {
        using Fn = std::remove_reference_t<F>;
        run_impl(chunks, [](void *ctx, int i) { (*static_cast<Fn *>(ctx))(i); }, (void *)&fn);
    }

private:
    void run_impl(int chunks, void (*call)(void *, int), void *ctx);
    void work();       // take chunks until none are left
    void worker_main();

    int n_;
    std::vector<std::thread> workers_;
    std::atomic<int> generation_{0}; // bumped once per run(): wakes the workers
    std::atomic<int> next_{0};       // next chunk to hand out
    std::atomic<int> done_{0};       // workers finished with this run
    std::atomic<bool> quit_{false};
    void (*call_)(void *, int) = nullptr;
    void *ctx_ = nullptr;
    int chunks_ = 0;
};

// The pool used by ops, conv and the codec. Default: 1 thread (no workers).
void set_threads(int threads);
ThreadPool &pool();

// Ask the OS to run the calling thread on performance cores (Apple:
// QoS "user interactive"; elsewhere nothing to do).
void prefer_performance_cores();

} // namespace mrt2
