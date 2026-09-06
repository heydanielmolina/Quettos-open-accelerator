// Token printing: the vocabulary written by the compiler as tokens.bin (u32
// count, then u16 length and the raw bytes of every id) and a streaming writer
// that holds an incomplete UTF-8 sequence until the token that finishes it
// arrives, so a multi-byte character is never split across two writes.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace qcore {

class Vocabulary {
 public:
  void load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open " + path);
    uint32_t count = 0;
    if (fread(&count, 4, 1, f) != 1) { fclose(f); throw std::runtime_error(path + ": short header"); }
    tokens_.resize(count);
    for (uint32_t i = 0; i < count; i++) {
      uint16_t n = 0;
      if (fread(&n, 2, 1, f) != 1) { fclose(f); throw std::runtime_error(path + ": truncated"); }
      tokens_[i].resize(n);
      if (n != 0 && fread(&tokens_[i][0], 1, n, f) != n) {
        fclose(f);
        throw std::runtime_error(path + ": truncated token " + std::to_string(i));
      }
    }
    fclose(f);
  }

  bool loaded() const { return !tokens_.empty(); }
  size_t size() const { return tokens_.size(); }

  const std::string& bytes(uint32_t id) const {
    static const std::string empty;
    return id < tokens_.size() ? tokens_[id] : empty;
  }

 private:
  std::vector<std::string> tokens_;
};

// Appends token bytes and writes out every complete UTF-8 sequence.
class TokenStream {
 public:
  TokenStream(const Vocabulary* vocab, FILE* out) : vocab_(vocab), out_(out) {}

  void push(uint32_t id) {
    if (vocab_ != nullptr && vocab_->loaded()) pending_ += vocab_->bytes(id);
    else pending_ += "<" + std::to_string(id) + ">";
    flush_complete();
  }

  // Writes the bytes of every character that is complete; keeps the rest.
  void flush_complete() {
    size_t cut = pending_.size();
    while (cut > 0) {
      size_t start = cut - 1;
      while (start > 0 && (static_cast<uint8_t>(pending_[start]) & 0xC0) == 0x80) start--;
      uint8_t lead = static_cast<uint8_t>(pending_[start]);
      size_t need = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 1;
      if (start + need <= pending_.size()) break;  // the tail is a complete character
      cut = start;                                 // hold back the incomplete one
    }
    if (cut != 0) {
      fwrite(pending_.data(), 1, cut, out_);
      fflush(out_);
      pending_.erase(0, cut);
    }
  }

  // Writes whatever is left, replacing an unfinished sequence.
  void finish() {
    if (!pending_.empty()) {
      fwrite("\xEF\xBF\xBD", 1, 3, out_);
      pending_.clear();
    }
    fflush(out_);
  }

 private:
  const Vocabulary* vocab_;
  FILE* out_;
  std::string pending_;
};

}  // namespace qcore
