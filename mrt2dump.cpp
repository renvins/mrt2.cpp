// mrt2dump.cpp - print the header JSON of a SafeTensors file.
//
// SafeTensors layout:
//   [8-bytes little-endian header length N][N-byte JSON header][tensor data]
//
// The header maps each tensor name to its dtype, shape, and the byte
// range holding its data (offsets are relative to the data section).
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s model.safetensors\n", argv[0]);
        return 1;
    }

    std::ifstream fp(argv[1], std::ios::binary);
    if (!fp) {
        std::perror(argv[1]);
        return 1;
    }

    // 8-byte little-endian length of the JSON header.
    uint64_t hlen = 0;
    fp.read(reinterpret_cast<char *>(&hlen), sizeof(hlen));
    if (!fp) {
        std::fprintf(stderr, "short read: header length\n");
        return 1;
    }

    std::string hdr(hlen, '\0');
    fp.read(hdr.data(), static_cast<std::streamsize>(hlen));
    if (!fp) {
        std::fprintf(stderr, "short read: header\n");
        return 1;
    }

    std::printf("%s: header is %llu bytes\n", argv[1],
                (unsigned long long)hlen);
    std::printf("%s\n", hdr.c_str());
    return 0;
}
