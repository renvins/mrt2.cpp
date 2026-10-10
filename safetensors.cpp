#include "safetensors.h"
#include "json.h"
#include "mmap.h"
#include <cstring>

namespace mrt2 {

uint64_t TensorInfo::numel() const {
    uint64_t n = 1;
    for (int64_t d : shape) n *= (uint64_t)d;
    return n;
}

// Bytes per element for each dtype safetensors defines; 0 if unknown.
static uint64_t dtype_size(const std::string &dt) {
    if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
    if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
    if (dt == "F16" || dt == "BF16" || dt == "I16" || dt == "U16") return 2;
    if (dt == "I8" || dt == "U8" || dt == "BOOL") return 1;
    return 0;
}

// True if `v` can be used as a size or offset: a whole, non-negative
// number. JSON numbers are doubles, and 2^53 is the largest integer a
// double holds exactly, so anything above it is rejected.
static bool is_count(const Json &v) {
    return v.type == Json::Num && v.num >= 0 && v.num <= 9007199254740992.0 &&
           v.num == (double)(uint64_t)v.num;
}

bool safetensors_parse(const uint8_t *buf, size_t size, SafeTensorsHeader &out,
                       std::string &err) {

    out = SafeTensorsHeader{};
    if (size < 8) { 
        err = "file too small for a safetensors header";
        return false;
    }

    // The 8-byte length is little-endian; memcpy is a legal unaligned load.
    uint64_t hlen = 0;
    std::memcpy(&hlen, buf, sizeof(hlen));
    if (hlen > size - 8) { 
        err = "header length exceeds file size";
        return false;
    }

    Json root;
    if (!json_parse(reinterpret_cast<const char *>(buf + 8), hlen, root, err)) return false;
    // The top level must be an object, else this isn't a safetensors header.
    if (root.type != Json::Obj) { err = "header is not a JSON object"; return false; }

    // Tensor data begins right after the length field and the header.
    out.file_size = size;
    out.data_start = 8 + hlen;
    const uint64_t data_size = size - out.data_start;

    for (const auto &kv : root.obj) {
        // "__metadata__" is an optional free-form field, not a tensor.
        if (kv.first == "__metadata__") continue;

        const Json &m = kv.second;
        const Json *dt = m.get("dtype");
        const Json *sh = m.get("shape");
        const Json *of = m.get("data_offsets");
        // Check every type before reading a value, so a malformed entry is
        // an error instead of a thrown exception or a crash.
        if (!dt || dt->type != Json::Str ||
            !sh || sh->type != Json::Arr ||
            !of || of->type != Json::Arr || of->as_arr().size() != 2 ||
            !is_count(of->as_arr()[0]) || !is_count(of->as_arr()[1])) {
            err = "bad tensor entry: " + kv.first;
            return false;
        }

        TensorInfo t;
        t.name = kv.first;
        t.dtype = dt->as_str();
        for (const Json &d : sh->as_arr()) {
            if (!is_count(d)) { err = "bad shape: " + t.name; return false; }
            t.shape.push_back((int64_t)d.as_num());
        }

        // data_offsets is the half-open range [begin, end) in the data
        // section. Checking it here means no later read can leave the file.
        uint64_t begin = (uint64_t)of->as_arr()[0].as_num();
        uint64_t end = (uint64_t)of->as_arr()[1].as_num();
        if (begin > end || end > data_size) {
            err = "data_offsets out of bounds: " + t.name;
            return false;
        }
        t.offset = begin;
        t.nbytes = end - begin;

        // The range must hold exactly numel elements of the dtype.
        uint64_t es = dtype_size(t.dtype);
        if (es == 0) { err = "unknown dtype " + t.dtype + ": " + t.name; return false; }
        if (t.numel() * es != t.nbytes) {
            err = "byte size does not match shape: " + t.name;
            return false;
        }

        out.tensors.emplace(t.name, std::move(t));
    }
    return true;
}

bool safetensors_read_header(const std::string &path, SafeTensorsHeader &out,
                             std::string &err) {
    // Mapping is lazy, so only the header pages are actually read.
    MappedFile f;
    if (!f.open(path, err)) return false;
    return safetensors_parse(f.data(), f.size(), out, err);
}

} // namespace mrt2
