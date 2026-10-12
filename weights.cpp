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

const float *Weights::f32(const std::string &name,
                          std::initializer_list<int64_t> shape) const {
    const TensorInfo *t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    if (t->dtype != "F32")
        throw std::runtime_error(name + ": expected F32, got " + t->dtype);

    // The 4-iterator std::equal also fails if the ranks differ.
    if (!std::equal(t->shape.begin(), t->shape.end(), shape.begin(), shape.end()))
        throw std::runtime_error(name + ": expected shape " +
                                 shape_str(shape.begin(), shape.end()) + ", got " +
                                 shape_str(t->shape.data(), t->shape.data() + t->shape.size()));

    // The parser proved [offset, offset + nbytes) lies inside the file and
    // that the offset is 4-byte aligned, so this pointer is safe to use.
    return reinterpret_cast<const float *>(raw(*t));
}

} // namespace mrt2
