// profile.h - a tiny profiler for the benchmark (tools/bench).
//
// Model code marks each operation with its cost:
//
//   MRT2_STAGE("temporal");                       // what part of the model
//   MRT2_PROF("attn.qkv", flops, bytes);          // until the end of the scope
//
// The benchmark then knows, per (stage, operation): time, calls, FLOPs and
// bytes moved, so it can tell what limits each operation (arithmetic or
// memory) and how close it is to that limit.
//
// Only built with -DMRT2_PROFILE (the benchmark build). In every other build
// both macros expand to nothing, so the engine pays no cost at all.
//
// Scopes must not nest: each one is a leaf operation, so their times add up.
#pragma once
#include <cstdint>
#include <vector>

namespace mrt2::prof {

struct Entry {
    const char *stage, *op;
    double seconds = 0, flops = 0, bytes = 0;
    long calls = 0;
};

#ifdef MRT2_PROFILE

void record(const char *op, double seconds, double flops, double bytes);
const char *&current_stage();
std::vector<Entry> entries(); // everything recorded so far
void reset();
double now(); // seconds, monotonic

struct Scope {
    const char *op;
    double flops, bytes, t0;
    Scope(const char *o, double f, double b) : op(o), flops(f), bytes(b), t0(now()) {}
    ~Scope() { record(op, now() - t0, flops, bytes); }
};

struct StageGuard {
    const char *saved;
    explicit StageGuard(const char *s) : saved(current_stage()) { current_stage() = s; }
    ~StageGuard() { current_stage() = saved; }
};

#define MRT2_CAT2(a, b) a##b
#define MRT2_CAT(a, b) MRT2_CAT2(a, b)
#define MRT2_PROF(op, flops, bytes) ::mrt2::prof::Scope MRT2_CAT(prof_, __LINE__)(op, flops, bytes)
#define MRT2_STAGE(name) ::mrt2::prof::StageGuard MRT2_CAT(stage_, __LINE__)(name)

#else

// sizeof() never evaluates its operand: the arguments count as "used" (no
// unused-variable warnings) but no code is generated.
#define MRT2_PROF(op, flops, bytes) ((void)sizeof(op), (void)sizeof(flops), (void)sizeof(bytes))
#define MRT2_STAGE(name) ((void)sizeof(name))

#endif

} // namespace mrt2::prof
