#include "profile.h"

#ifdef MRT2_PROFILE
#include <chrono>
#include <cstring>

namespace mrt2::prof {

// A small fixed table, searched linearly: there are only ~40 distinct
// (stage, operation) pairs. Single-threaded, like the engine for now.
static std::vector<Entry> g_entries;

const char *&current_stage() {
    static const char *stage = "other";
    return stage;
}

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void record(const char *op, double seconds, double flops, double bytes) {
    const char *stage = current_stage();
    for (Entry &e : g_entries)
        if (std::strcmp(e.op, op) == 0 && std::strcmp(e.stage, stage) == 0) {
            e.seconds += seconds;
            e.flops += flops;
            e.bytes += bytes;
            e.calls++;
            return;
        }
    g_entries.push_back({stage, op, seconds, flops, bytes, 1});
}

std::vector<Entry> entries() { return g_entries; }
void reset() { g_entries.clear(); }

} // namespace mrt2::prof
#endif
