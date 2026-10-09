#include "mmap.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mrt2 {

MappedFile::~MappedFile() { close(); }

#ifndef _WIN32

bool MappedFile::open(const std::string &path, std::string &err) {
    close(); // reopening drops the previous mapping

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        err = "cannot stat " + path + ": " + std::strerror(errno);
        ::close(fd);
        return false;
    }
    // mmap() rejects a zero length, and an empty file is not a model
    if (st.st_size == 0) {
        err = "empty file: " + path;
        ::close(fd);
        return false;
    }

    void *p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    // The mapping holds its own reference to the file, so the
    // descriptor is no longer needed whether or not mmap succeeded.
    ::close(fd);
    if (p == MAP_FAILED) {
        err = "mmap failed for " + path + ": " + std::strerror(errno);
        return false;
    }

    data_ = static_cast<const uint8_t *>(p);
    size_ = (size_t)st.st_size;
    return true;
}

void MappedFile::close() {
    if (data_) munmap(const_cast<uint8_t *>(data_), size_);
    data_ = nullptr;
    size_ = 0;
}

#else // _WIN32

bool MappedFile::open(const std::string &path, std::string &err) {
    close();

    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "cannot open " + path; return false; }

    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart == 0) {
        err = "cannot size (or empty) " + path;
        CloseHandle(f);
        return false;
    }

    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    void *p = m ? MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0) : nullptr;
    // Like POSIX: the view keeps the mapping and file alive by itself,
    // so both handles can be closed right away.
    if (m) CloseHandle(m);
    CloseHandle(f);
    if (!p) { err = "mmap failed for " + path; return false; }

    data_ = static_cast<const uint8_t *>(p);
    size_ = (size_t)sz.QuadPart;
    return true;
}

void MappedFile::close() {
    if (data_) UnmapViewOfFile(data_);
    data_ = nullptr;
    size_ = 0;
}

#endif

} // namespace mrt2