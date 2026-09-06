// Compact JSON reader for the files this repository generates (layout.json,
// dump_plan.json). It parses the subset those files use -- objects, arrays,
// strings, numbers, true/false/null -- into a value tree that is read by
// dotted path ("programs.decode.addr"). Every failure throws with the byte
// offset or the path that was missing.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace qjson {

class Value {
 public:
  enum class Kind { Null, Bool, Num, Str, Arr, Obj };

  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0.0;
  int64_t integer = 0;
  std::string text;
  std::vector<Value> items;                        // Arr
  std::vector<std::pair<std::string, Value>> keys; // Obj, in file order

  bool is_null() const { return kind == Kind::Null; }
  bool is_num() const { return kind == Kind::Num; }
  bool is_str() const { return kind == Kind::Str; }
  bool is_arr() const { return kind == Kind::Arr; }
  bool is_obj() const { return kind == Kind::Obj; }

  size_t size() const { return kind == Kind::Arr ? items.size() : keys.size(); }

  const Value* find(const std::string& key) const {
    for (const auto& kv : keys) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

  bool has(const std::string& key) const { return find(key) != nullptr; }

  const Value& operator[](const std::string& key) const {
    const Value* v = find(key);
    if (v == nullptr) throw std::runtime_error("json: no key \"" + key + "\"");
    return *v;
  }

  const Value& operator[](size_t i) const {
    if (i >= items.size()) throw std::runtime_error("json: array index out of range");
    return items[i];
  }

  // Dotted path lookup: at("programs.decode.addr").
  const Value& at(const std::string& path) const {
    const Value* v = this;
    size_t pos = 0;
    while (pos <= path.size()) {
      size_t dot = path.find('.', pos);
      std::string part = path.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
      const Value* nxt = v->find(part);
      if (nxt == nullptr) throw std::runtime_error("json: no path \"" + path + "\"");
      v = nxt;
      if (dot == std::string::npos) break;
      pos = dot + 1;
    }
    return *v;
  }

  const Value* find_path(const std::string& path) const {
    const Value* v = this;
    size_t pos = 0;
    while (pos <= path.size()) {
      size_t dot = path.find('.', pos);
      std::string part = path.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
      v = v->find(part);
      if (v == nullptr) return nullptr;
      if (dot == std::string::npos) break;
      pos = dot + 1;
    }
    return v;
  }

  int64_t i64() const {
    if (kind != Kind::Num) throw std::runtime_error("json: value is not a number");
    return integer;
  }
  double num() const {
    if (kind != Kind::Num) throw std::runtime_error("json: value is not a number");
    return number;
  }
  const std::string& str() const {
    if (kind != Kind::Str) throw std::runtime_error("json: value is not a string");
    return text;
  }
  bool as_bool() const { return kind == Kind::Bool ? boolean : false; }

  int64_t i64_or(const std::string& path, int64_t fallback) const {
    const Value* v = find_path(path);
    return (v != nullptr && v->is_num()) ? v->integer : fallback;
  }
};

class Parser {
 public:
  explicit Parser(std::string src) : s_(std::move(src)) {}

  Value parse() {
    skip();
    Value v = value();
    skip();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  std::string s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("json: " + what + " at byte " + std::to_string(i_));
  }
  void skip() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) i_++;
  }
  char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
  void expect(char c) {
    if (peek() != c) fail(std::string("expected '") + c + "'");
    i_++;
  }

  Value value() {
    switch (peek()) {
      case '{': return object();
      case '[': return array();
      case '"': {
        Value v;
        v.kind = Value::Kind::Str;
        v.text = string();
        return v;
      }
      case 't': case 'f': return boolean();
      case 'n': {
        literal("null");
        return Value();
      }
      default: return number();
    }
  }

  void literal(const char* lit) {
    for (const char* p = lit; *p != '\0'; ++p) {
      if (peek() != *p) fail(std::string("expected ") + lit);
      i_++;
    }
  }

  Value boolean() {
    Value v;
    v.kind = Value::Kind::Bool;
    if (peek() == 't') { literal("true"); v.boolean = true; }
    else { literal("false"); v.boolean = false; }
    return v;
  }

  Value object() {
    Value v;
    v.kind = Value::Kind::Obj;
    expect('{');
    skip();
    if (peek() == '}') { i_++; return v; }
    for (;;) {
      skip();
      std::string k = string();
      skip();
      expect(':');
      skip();
      v.keys.emplace_back(std::move(k), value());
      skip();
      if (peek() == ',') { i_++; continue; }
      expect('}');
      return v;
    }
  }

  Value array() {
    Value v;
    v.kind = Value::Kind::Arr;
    expect('[');
    skip();
    if (peek() == ']') { i_++; return v; }
    for (;;) {
      skip();
      v.items.push_back(value());
      skip();
      if (peek() == ',') { i_++; continue; }
      expect(']');
      return v;
    }
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size()) {
      char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') { out.push_back(c); continue; }
      if (i_ >= s_.size()) fail("unterminated escape");
      char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          if (i_ + 4 > s_.size()) fail("short \\u escape");
          unsigned cp = static_cast<unsigned>(strtoul(s_.substr(i_, 4).c_str(), nullptr, 16));
          i_ += 4;
          if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
          } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
          }
          break;
        }
        default: fail("unknown escape");
      }
    }
    fail("unterminated string");
  }

  Value number() {
    size_t start = i_;
    if (peek() == '-' || peek() == '+') i_++;
    bool is_int = true;
    while (i_ < s_.size()) {
      char c = s_[i_];
      if (c >= '0' && c <= '9') { i_++; continue; }
      if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') { is_int = false; i_++; continue; }
      break;
    }
    if (i_ == start) fail("expected a value");
    std::string t = s_.substr(start, i_ - start);
    Value v;
    v.kind = Value::Kind::Num;
    v.number = strtod(t.c_str(), nullptr);
    v.integer = is_int ? static_cast<int64_t>(strtoll(t.c_str(), nullptr, 10))
                       : static_cast<int64_t>(v.number);
    return v;
  }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

inline Value parse_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("json: cannot open " + path);
  std::string data;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
  fclose(f);
  try {
    return parse(data);
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

}  // namespace qjson
