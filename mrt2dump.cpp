// mrt2dump.cpp - print the header JSON of a SafeTensors file.
//
// SafeTensors layout:
//   [8-bytes little-endian header length N][N-byte JSON header][tensor data]
//
// The header maps each tensor name to its dtype, shape, and the byte
// range holding its data (offsets are relative to the data section).
#include "mmap.h"
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

    // 8-byte little-endian length of the JSON header.
    uint64_t hlen = 0;
    if (mf.size() >= 8) std::memcpy(&hlen, mf.data(), sizeof(hlen));
    // Check before trusting it: in a file that is not safetensors these
    // 8 bytes are arbitrary, and could claim an exabyte-long header.
    if (mf.size() < 8 || hlen > mf.size() - 8) {
        std::fprintf(stderr, "%s: not a safetensors file\n", argv[1]);
        return 1;
    }

    std::printf("%s: header is %llu bytes\n", argv[1], (unsigned long long)hlen);
    // fwrite prints exactly hlen bytes straight from the mapping:
    // no copy, and no reliance on a terminating '\0'.
    std::fwrite(mf.data() + 8, 1, hlen, stdout);
    std::printf("\n");
    return 0;
}
