// mmap.h - map a whole file read-only into memory.
//
// The OS pages the file in lazily on first touch, so opening a
// multi-gb model is instant and costs no extra RAM: the
// mapping *is* the page cache.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace mrt2 {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    // Not copyable: two copies would both unmap the same memory.
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;

    // Movable: ownership of the mapping is handed over, and the source is
    // left empty, so only the new owner will ever unmap it. The mapped
    // address itself does not change, so pointers into it stay valid.
    MappedFile(MappedFile &&o) noexcept
        : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}

    MappedFile &operator=(MappedFile &&o) noexcept {
        if (this != &o) {
            close(); // drop whatever this object mapped before
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }

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