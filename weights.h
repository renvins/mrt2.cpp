// weights.h - a model's tensors, mapped from a .safetensors file.
//
// open() maps the file and parses its header. After that a tensor is
// just a pointer into the mapping: nothing is copied or converted.
//
// Lifetime rule: every pointer returned by f32() points into the
// mapping, so the Weights object must outlive everything that holds
// one of those pointers.
#pragma once
#include "mmap.h"
#include "safetensors.h"
#include <cstdint>
#include <initializer_list>
#include <string>

namespace mrt2 {

class Weights {
public:
    // Map `path` and parse its header. Returns false and fills `err`.
    bool open(const std::string &path, std::string &err);

    // Header entry for `name`, or nullptr if the file has no such tensor.
    const TensorInfo *find(const std::string &name) const;

    // Data of an F32 tensor, after checking that it exists, is F32 and
    // has exactly `shape`. Throws std::runtime_error otherwise.
    // Alignment is not checked here: safetensors_parse already
    // guarantees every tensor starts on a multiple of its element size.
    const float *f32(const std::string &name,
                     std::initializer_list<int64_t> shape) const;

    // Same as f32(), for I32 tensors (e.g. token ids in reference files).
    const int32_t *i32(const std::string &name,
                       std::initializer_list<int64_t> shape) const;

    // First byte of a tensor's data, for code that walks every tensor
    // (e.g. mrt2info --verify). `t` must come from this object's header.
    const uint8_t *raw(const TensorInfo &t) const {
        return file_.data() + hdr_.data_start + t.offset;
    }

    const SafeTensorsHeader &header() const { return hdr_; }

private:
    // The tensor `name`, after checking it exists and has this dtype and
    // shape; throws std::runtime_error otherwise.
    const TensorInfo &checked(const std::string &name, const char *dtype,
                              std::initializer_list<int64_t> shape) const;

    MappedFile file_;
    SafeTensorsHeader hdr_;
};

} // namespace mrt2
