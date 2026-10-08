#include "json.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace mrt2 {

// Reset a value to Null, releasing whatever it held.
void json_reset(Json &v) {
    v = Json{};
}

const Json *Json::get(const std::string &key) const {
    if (type != Obj) return nullptr;
    for (const auto &kv : obj) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

double Json::as_num() const {
    if (type != Num) throw std::runtime_error("json: expected number");
    return num;
}

const std::string &Json::as_str() const {
    if (type != Str) throw std::runtime_error("json: expected string");
    return str;
}

const std::vector<Json> &Json::as_arr() const {
    if (type != Arr) throw std::runtime_error("json: expected array");
    return arr;
}

namespace {

// A recursive-descent parser over the byte range [p, end).
struct Parser {
    const char *p;
    const char *end;
    std::string err;

    // Record the first error and report failure.
    bool fail(const char *msg) {
        if (err.empty()) err = msg;
        return false;
    }

    // Skip spaces, tabs and newlines.
    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'))
            p++;
    }

    bool value(Json &out);
    bool string(std::string &out);
    bool literal(const char *word);
    bool number(double &out);
};

// Parse a string; called with *p == '"'.
bool Parser::string(std::string &out) {
    p++; // opening quote
    out.clear();
    while (p < end && *p != '"') {
        char c = *p++;
        if (c != '\\') {
            out.push_back(c);
            continue;
        }
        if (p >= end) return fail("json: bad escape");
        char e = *p++;
        switch (e) {
        case '"':  out.push_back('"');  break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/');  break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'u': {
            // \uXXXX -> UTF-8 (BMP only).
            if (end - p < 4) return fail("json: bad \\u");
            unsigned cp = 0;
            for (int i = 0; i < 4; i++) {
                char h = *p++;
                cp <<= 4;
                if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                else return fail("json: bad hex");
            }
            if (cp < 0x80) {
                out.push_back((char)cp);
            } else if (cp < 0x800) {
                out.push_back((char)(0xC0 | (cp >> 6)));
                out.push_back((char)(0x80 | (cp & 0x3F)));
            } else {
                out.push_back((char)(0xE0 | (cp >> 12)));
                out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back((char)(0x80 | (cp & 0x3F)));
            }
            break;
        }
        default: return fail("json: bad escape");
        }
    }
    if (p >= end) return fail("json: unterminated string");
    p++; // closing quote
    return true;
}

// Match and consume an exact keyword such as "true".
bool Parser::literal(const char *word) {
    size_t n = std::strlen(word);
    if ((size_t)(end - p) < n || std::memcmp(p, word, n) != 0)
        return fail("json: bad literal");
    p += n;
    return true;
}

// Parse a number with strtod; handles signs, fractions and exponents.
bool Parser::number(double &out) {
    char *stop = nullptr;
    out = std::strtod(p, &stop);
    if (stop == p) return fail("json: bad number");
    p = stop;
    return true;
}

bool Parser::value(Json &out) {
    ws();
    if (p >= end) return fail("json: unexpected end");
    char c = *p;

    if (c == '{') {
        p++;
        out.type = Json::Obj;
        ws();
        if (p < end && *p == '}') { p++; return true; }
        while (true) {
            ws();
            if (p >= end || *p != '"') return fail("json: expected key");
            std::string key;
            if (!string(key)) return false;
            ws();
            if (p >= end || *p != ':') return fail("json: expected ':'");
            p++;
            Json child;
            if (!value(child)) return false;
            out.obj.emplace_back(std::move(key), std::move(child));
            ws();
            if (p < end && *p == ',') { p++; continue; }
            if (p < end && *p == '}') { p++; return true; }
            return fail("json: expected ',' or '}'");
        }
    }

    if (c == '[') {
        p++;
        out.type = Json::Arr;
        ws();
        if (p < end && *p == ']') { p++; return true; }
        while (true) {
            Json child;
            if (!value(child)) return false;
            out.arr.push_back(std::move(child));
            ws();
            if (p < end && *p == ',') { p++; continue; }
            if (p < end && *p == ']') { p++; return true; }
            return fail("json: expected ',' or ']'");
        }
    }

    if (c == '"') { out.type = Json::Str; return string(out.str); }
    if (c == 't') { out.type = Json::Bool; out.b = true;  return literal("true"); }
    if (c == 'f') { out.type = Json::Bool; out.b = false; return literal("false"); }
    if (c == 'n') { out.type = Json::Null; return literal("null"); }

    out.type = Json::Num;
    return number(out.num);
}

} // namespace

bool json_parse(const char *s, size_t len, Json &out, std::string &err) {
    Parser parser{s, s + len, {}};
    json_reset(out);
    bool ok = parser.value(out);
    if (ok) {
        parser.ws();
        if (parser.p != parser.end) ok = parser.fail("json: trailing data");
    }
    err = parser.err;
    return ok;
}

} // namespace mrt2
