// mrt2info.cpp - list the tensors in a .safetensors file.
//
// Only the header is read, so this is fast even on multi-gigabyte files.
#include "safetensors.h"
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.safetensors [name-substring]\n", argv[0]);
        return 1;
    }
    // Optional second argument filters tensor names by substring.
    const char *filter = argc >= 3 ? argv[2] : nullptr;

    mrt2::SafeTensorsHeader st;
    std::string err;
    if (!mrt2::safetensors_read_header(argv[1], st, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    uint64_t total_params = 0;
    uint64_t total_bytes = 0;
    size_t shown = 0;
    for (const auto &kv : st.tensors) {
        const mrt2::TensorInfo &t = kv.second;
        total_params += t.numel();
        total_bytes += t.nbytes;
        if (filter && t.name.find(filter) == std::string::npos) continue;
        shown++;

        std::printf("%-6s [", t.dtype.c_str());
        for (size_t i = 0; i < t.shape.size(); i++)
            std::printf("%s%lld", i ? ", " : "", (long long)t.shape[i]);
        std::printf("]  %s\n", t.name.c_str());
    }

    std::printf("tensors=%zu  params=%llu  bytes=%llu  data_start=%llu\n",
                st.tensors.size(),
                (unsigned long long)total_params,
                (unsigned long long)total_bytes,
                (unsigned long long)st.data_start);
    if (filter) std::printf("shown=%zu for \"%s\"\n", shown, filter);
    return 0;
}
