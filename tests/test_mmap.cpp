// test_mmap.cpp - map a file, read the safetensors header length straight
// from the mapping, and check that moving a MappedFile hands over
// ownership correctly.
#include "../mmap.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>

// Copying must stay impossible; moving must work.
static_assert(!std::is_copy_constructible_v<mrt2::MappedFile>);
static_assert(std::is_nothrow_move_constructible_v<mrt2::MappedFile>);
static_assert(std::is_nothrow_move_assignable_v<mrt2::MappedFile>);

static int failures = 0;

static void check(bool ok, const char *what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// First 8 bytes of a safetensors file: the header length.
static uint64_t header_len(const mrt2::MappedFile &mf) {
    // memcpy, not *(uint64_t*)ptr: the compiler turns it into a single
    // load, and it stays legal even if the address is unaligned.
    uint64_t hlen = 0;
    std::memcpy(&hlen, mf.data(), sizeof(hlen));
    return hlen;
}

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

    const uint64_t hlen = header_len(mf);
    std::printf("mapped %zu bytes, header length %llu\n",
                mf.size(), (unsigned long long)hlen);

    // Move construction: b takes over the mapping, mf is left empty.
    const uint8_t *addr = mf.data();
    const size_t size = mf.size();
    mrt2::MappedFile b(std::move(mf));
    check(b.data() == addr && b.size() == size, "move constructor hands over address and size");
    check(mf.data() == nullptr && mf.size() == 0, "moved-from object is empty");
    check(header_len(b) == hlen, "data is still readable after the move");

    // Move assignment into an object that already owns a mapping: its old
    // mapping is closed first, then it takes the new one.
    mrt2::MappedFile c;
    if (!c.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    c = std::move(b);
    check(c.data() == addr && c.size() == size, "move assignment hands over the mapping");
    check(b.data() == nullptr, "move-assigned-from object is empty");

    // Self-move must not destroy the object. (`alias` avoids the
    // compiler's -Wself-move warning on a literal `c = std::move(c)`.)
    mrt2::MappedFile &alias = c;
    c = std::move(alias);
    check(c.data() == addr && header_len(c) == hlen, "self-move keeps the mapping");

    // When main returns, only `c` unmaps; `mf` and `b` are empty, so their
    // destructors do nothing (the sanitizer build checks this).
    std::printf("%s\n", failures ? "SOME TESTS FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
