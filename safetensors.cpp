#include "safetensors.h"
#include "json.h"
#include <fstream>

namespace mrt2 {

uint64_t TensorInfo::numel() const {
    uint64_t n = 1;
    for (int64_t d : shape) n *= (uint64_t)d;
    return n;
}

bool safetensors_read_header(const std::string &path, SafeTensorsHeader &out,
                             std::string &err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }

    // Size up front so we can reject a bogus header length before allocating.
    f.seekg(0, std::ios::end);
    out.file_size = (uint64_t)f.tellg();
    f.seekg(0, std::ios::beg);

    // The 8-byte length is little-endian; reading straight into a uint64_t
    // is correct on x86 and ARM (both little-endian).
    uint64_t hlen = 0;
    f.read(reinterpret_cast<char *>(&hlen), sizeof(hlen));
    if (!f) { err = "short read: header length"; return false; }

    // The file is at least 8 bytes here, so file_size - 8 cannot underflow.
    if (hlen > out.file_size - 8) { err = "header length exceeds file size"; return false; }

    std::string hdr(hlen, '\0');
    f.read(hdr.data(), (std::streamsize)hlen);
    if (!f) { err = "short read: header"; return false; }

    Json root;
    if (!json_parse(hdr.data(), hdr.size(), root, err)) return false;
    // The top level must be an object, else this isn't a safetensors header.
    if (root.type != Json::Obj) { err = "header is not a JSON object"; return false; }

    // Tensor data begins right after the length field and the header.
    out.data_start = 8 + hlen;

    for (const auto &kv : root.obj) {
        // "__metadata__" is an optional free-form field, not a tensor.
        if (kv.first == "__metadata__") continue;

        const Json &m = kv.second;
        const Json *dt = m.get("dtype");
        const Json *sh = m.get("shape");
        const Json *of = m.get("data_offsets");
        // Validate before casting below, so a malformed entry is an error
        // instead of a crash.
        if (!dt || dt->type != Json::Str ||
            !sh || sh->type != Json::Arr ||
            !of || of->type != Json::Arr || of->as_arr().size() != 2) {
            err = "bad tensor entry: " + kv.first;
            return false;
        }

        TensorInfo t;
        t.name = kv.first;
        t.dtype = dt->as_str();
        for (const Json &d : sh->as_arr()) t.shape.push_back((int64_t)d.as_num());

        // data_offsets is the half-open range [begin, end) in the data section.
        t.offset = (uint64_t)of->as_arr()[0].as_num();
        t.nbytes = (uint64_t)of->as_arr()[1].as_num() - t.offset;

        out.tensors.emplace(t.name, std::move(t));
    }
    return true;
}

} // namespace mrt2
