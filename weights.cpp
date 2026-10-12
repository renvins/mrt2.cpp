#include "weights.h"
#include <algorithm>
#include <stdexcept>

namespace mrt2 {

bool Weights::open(const std::string &path, std::string &err) {
    if (!file_.open(path, err)) return false;
    if (!safetensors_parse(file_.data(), file_.size(), hdr_, err)) {
        file_.close();
        return false;
    }
    return true;
}

const TensorInfo *Weights::find(const std::string &name) const {
    auto it = hdr_.tensors.find(name);
    return it == hdr_.tensors.end() ? nullptr : &it->second;
}

// "[12294, 1024]", for error messages.
static std::string shape_str(const int64_t *b, const int64_t *e) {
    std::string s = "[";
    for (const int64_t *p = b; p != e; p++) {
        if (p != b) s += ", ";
        s += std::to_string(*p);
    }
    return s + "]";
}

const TensorInfo &Weights::checked(const std::string &name, const char *dtype,
                                   std::initializer_list<int64_t> shape) const {
    const TensorInfo *t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    if (t->dtype != dtype)
        throw std::runtime_error(name + ": expected " + dtype + ", got " + t->dtype);

    // The 4-iterator std::equal also fails if the ranks differ.
    if (!std::equal(t->shape.begin(), t->shape.end(), shape.begin(), shape.end()))
        throw std::runtime_error(name + ": expected shape " +
                                 shape_str(shape.begin(), shape.end()) + ", got " +
                                 shape_str(t->shape.data(), t->shape.data() + t->shape.size()));
    return *t;
}

// For both accessors: the parser proved [offset, offset + nbytes) lies
// inside the file and that the offset is aligned to the element size, so
// the returned pointer is safe to use.
const float *Weights::f32(const std::string &name,
                          std::initializer_list<int64_t> shape) const {
    return reinterpret_cast<const float *>(raw(checked(name, "F32", shape)));
}

const int32_t *Weights::i32(const std::string &name,
                            std::initializer_list<int64_t> shape) const {
    return reinterpret_cast<const int32_t *>(raw(checked(name, "I32", shape)));
}

} // namespace mrt2
