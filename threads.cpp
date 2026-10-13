#include "threads.h"
#include <memory>
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

namespace mrt2 {

void prefer_performance_cores() {
#if defined(__APPLE__)
    // Apple chips mix performance and efficiency cores; macOS may otherwise
    // put a thread on an efficiency core, about 3x slower for our work.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

ThreadPool::ThreadPool(int threads) : n_(threads < 1 ? 1 : threads) {
    for (int i = 1; i < n_; i++) workers_.emplace_back([this] { worker_main(); });
}

ThreadPool::~ThreadPool() {
    quit_.store(true);
    generation_.fetch_add(1, std::memory_order_release); // wake everyone up to exit
    for (std::thread &t : workers_) t.join();
}

void ThreadPool::worker_main() {
    prefer_performance_cores();
    int seen = 0;
    for (;;) {
        // Wait for the next run(): spin on the generation counter.
        while (generation_.load(std::memory_order_acquire) == seen) std::this_thread::yield();
        seen = generation_.load(std::memory_order_acquire);
        if (quit_.load()) return;
        work();
        done_.fetch_add(1, std::memory_order_release);
    }
}

// True while this thread is running chunks of some run(): a run() started
// from inside a chunk then executes inline instead of waiting for workers
// that are busy running the outer job (which would deadlock).
static thread_local bool t_in_pool = false;

void ThreadPool::work() {
    // Take chunk numbers from the shared counter until they run out.
    t_in_pool = true;
    for (int i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < chunks_;) call_(ctx_, i);
    t_in_pool = false;
}

void ThreadPool::run_impl(int chunks, void (*call)(void *, int), void *ctx) {
    if (n_ == 1 || chunks <= 1 || t_in_pool) { // nothing to share, or nested: run here
        for (int i = 0; i < chunks; i++) call(ctx, i);
        return;
    }
    // Publish the job before waking the workers: the release on
    // generation_ makes these writes visible to every worker that sees it.
    call_ = call;
    ctx_ = ctx;
    chunks_ = chunks;
    next_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    work(); // the caller helps
    while (done_.load(std::memory_order_acquire) != n_ - 1) std::this_thread::yield();
}

static std::unique_ptr<ThreadPool> &global_pool() {
    static std::unique_ptr<ThreadPool> p = std::make_unique<ThreadPool>(1);
    return p;
}

void set_threads(int threads) { global_pool() = std::make_unique<ThreadPool>(threads); }

ThreadPool &pool() { return *global_pool(); }

} // namespace mrt2
