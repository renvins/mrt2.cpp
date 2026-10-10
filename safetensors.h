// safetensors.h - read the header of a .safetensors file.
//
// Layout of a .safetensors file:
//
//   [ 8 bytes: little-endian u64 header length N ]
//   [ N bytes: JSON header (UTF-8)               ]
//   [ rest: raw tensor bytes                     ]
//
// The JSON header is one object keyed by tensor name.
// Each value is:
//
//   { "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24] }
//
// data_offsets[0] and [1] are byte positions *inside the data section*:
// relative to the first byte after the header, NOT absolute file offsets.
// So a tensor's bytes sit at file position (8 + N + data_offsets[0]) .. .
#pragma once
#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace mrt2 {

struct TensorInfo {
    std::string name;
    std::string dtype; // "F32", "BF16", ... as written in the header
    std::vector<int64_t> shape; // row-major dimensions
    uint64_t offset = 0; // byte offset inside the data section
    uint64_t nbytes = 0; // size of this tensor's data, in bytes

    // Number of elements = product of all dimensions.
    // A scalar (empty shape) has one element, which is why the accumulator
    // starts at 1, not 0.
    uint64_t numel() const;
};

struct SafeTensorsHeader {
    // std::map keeps tensors sorted by name, so our
    // output is reproducible run to run. There are
    // only hundreds of tensors, so the log factor is moot.
    std::map<std::string, TensorInfo> tensors;

    uint64_t data_start = 0; // absolute file offset where tensor bytes begin
    uint64_t file_size = 0; // total file size in bytes
};

// Parse only the header; the (huge) tensor bytes are never read.
// Returns false and fills `err` on failure.
bool safetensors_read_header(const std::string &path, SafeTensorsHeader &out,
                             std::string &err);

// Parse a safetensors header from an in-memory image of the whole file
// (e.g. a MappedFile). Every tensor's dtype, shape and byte range is
// checked, so later code can trust them. Returns false and fills `err`.
bool safetensors_parse(const uint8_t *buf, size_t size, SafeTensorsHeader &out,
                       std::string &err);

} // namespace mrt2
