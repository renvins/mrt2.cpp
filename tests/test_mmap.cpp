// test_mmap.cpp - map a file and print the safetensors header length
// read straight from the mapping
#include "../mmap.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 1;
    }

    mrt2::MappedFile mf;
    std::string err;
    if (!mf.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (mf.size() < 8) {
        std::fprintf(stderr, "file too small\n");
        return 1;
    }

    //memcpy, not *(uint64_t*)ptr: the compiler turns it into a single
    // load, and it stays legal even if the address is unaligned
    uint64_t hlen = 0;
    std::memcpy(&hlen, mf.data(), sizeof(hlen));

    std::printf("mapped %zu bytes, header length %llu\n",
                mf.size(), (unsigned long long)hlen);
    return 0;
}