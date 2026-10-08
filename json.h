// json.h - a small JSON parser, just enough for model
// metadata files.
#pragma once
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mrt2 {

struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj };
    Type type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj; // preserves order

    const Json *get(const std::string &key) const; // object lookup or nullptr
    double as_num() const; // throws if not Num
    const std::string &as_str() const; // throws if not Str
    const std::vector<Json> &as_arr() const; // throws if not Arr
};

// Parse a whole document; false on error, with `err` set.
bool json_parse(const char *s, size_t len, Json &out, std::string &err);

// Reset a value to Null, releasing whatever it held.
void json_reset(Json &v);

} // namespace mrt2
