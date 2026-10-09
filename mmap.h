// mmap.h - map a whole file read-only into memory.
//
// The OS pages the file in lazily on first touch, so opening a
// multi-gb model is instant and costs no extra RAM: the
// mapping *is* the page cache.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace mrt2 {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    // Not copyable: two copies would both unmap the same memory.
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Map `path` read-only. Returns false and fills `err` on failure.
    bool open(const std::string &path, std::string &err);
    void close();

    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t *data_ = nullptr;
    size_t size_ = 0;
}; 

} // namespace mrt2