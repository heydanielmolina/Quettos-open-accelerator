// The prefix file: a saved KV region with the record of what it belongs to.
//
// A prefix is the state a run leaves behind after it has consumed some
// positions of a sequence -- the KV region of docs/MEMORY_MAP.md, byte for byte
// at its image layout -- so a later run can restore it and prefill only the
// positions that come after. That is only exact if the file and the run agree
// on every term the KV bytes were computed under, so the header carries them:
// the ISA version, the port width, MAX_CTX, the model name, the SHA-256 of the
// image the bytes were produced from, where the region sits and how large it
// is, and the token id of every position the file covers. `verify_prefix` holds
// all of them to the run that is about to restore the file and names the first
// one that differs; a mismatched file is refused there rather than restored
// into a machine it does not belong to.
//
// Layout, little-endian, header then payload:
//
//   0   8  magic "QKVPRFX1"
//   8   4  format version
//  12   4  header bytes (this header including the ids, so the payload starts here)
//  16   4  ISA version
//  20   4  WB
//  24   4  MAX_CTX
//  28   4  positions: how many leading positions of the sequence the file covers
//  32   8  KV base address in the image
//  40   8  KV region size, which is also the payload size
//  48  64  model name, NUL padded
// 112  64  SHA-256 of image.bin, as the 64 hex characters layout.json records
// 176  4*positions  the token id fed in at each covered position, in order
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfg.hpp"
#include "csr_defs.hpp"

namespace qcore {
namespace {

constexpr char PREFIX_MAGIC[8] = {'Q', 'K', 'V', 'P', 'R', 'F', 'X', '1'};
constexpr uint32_t PREFIX_VERSION = 1u;
constexpr uint32_t PREFIX_NAME_CHARS = 64u;  // model name and image SHA-256 alike
constexpr uint32_t PREFIX_FIXED_BYTES = 176u;

// What the file says it belongs to. `ids[p]` is the token that was fed in at
// position p, so a run restoring the file can hold its own prompt to it.
struct PrefixHeader {
  uint32_t version = PREFIX_VERSION;
  uint32_t isa_version = 0;
  uint32_t wb = 0;
  uint32_t max_ctx = 0;
  uint64_t kv_base = 0;
  uint64_t kv_size = 0;
  std::string model;
  std::string image_sha256;
  std::vector<uint32_t> ids;

  uint32_t positions() const { return static_cast<uint32_t>(ids.size()); }
  uint32_t header_bytes() const { return PREFIX_FIXED_BYTES + 4u * positions(); }
};

inline void put_u32(std::vector<uint8_t>& out, size_t at, uint32_t v) {
  for (int i = 0; i < 4; i++) out[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

inline void put_u64(std::vector<uint8_t>& out, size_t at, uint64_t v) {
  for (int i = 0; i < 8; i++) out[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

inline uint32_t get_u32(const std::vector<uint8_t>& in, size_t at) {
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) v |= static_cast<uint32_t>(in[at + i]) << (8 * i);
  return v;
}

inline uint64_t get_u64(const std::vector<uint8_t>& in, size_t at) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= static_cast<uint64_t>(in[at + i]) << (8 * i);
  return v;
}

// A NUL-padded field, written as far as it fits and refused when it does not:
// a name this cannot carry whole would compare equal to a different one.
inline void put_name(std::vector<uint8_t>& out, size_t at, const std::string& s,
                     const char* what) {
  if (s.size() > PREFIX_NAME_CHARS) {
    throw std::runtime_error(std::string("prefix: ") + what + " is longer than " +
                             std::to_string(PREFIX_NAME_CHARS) + " characters: " + s);
  }
  memcpy(out.data() + at, s.data(), s.size());
}

inline std::string get_name(const std::vector<uint8_t>& in, size_t at) {
  size_t n = 0;
  while (n < PREFIX_NAME_CHARS && in[at + n] != 0) n++;
  return std::string(reinterpret_cast<const char*>(in.data() + at), n);
}

// Header then payload, in one write. `kv` is the whole KV region.
inline void write_prefix(const std::string& path, const PrefixHeader& h, const uint8_t* kv) {
  std::vector<uint8_t> head(h.header_bytes(), 0);
  memcpy(head.data(), PREFIX_MAGIC, sizeof(PREFIX_MAGIC));
  put_u32(head, 8, h.version);
  put_u32(head, 12, h.header_bytes());
  put_u32(head, 16, h.isa_version);
  put_u32(head, 20, h.wb);
  put_u32(head, 24, h.max_ctx);
  put_u32(head, 28, h.positions());
  put_u64(head, 32, h.kv_base);
  put_u64(head, 40, h.kv_size);
  put_name(head, 48, h.model, "the model name");
  put_name(head, 112, h.image_sha256, "the image SHA-256");
  for (uint32_t i = 0; i < h.positions(); i++) put_u32(head, PREFIX_FIXED_BYTES + 4 * i, h.ids[i]);
  FILE* f = fopen(path.c_str(), "wb");
  if (f == nullptr) throw std::runtime_error("cannot write " + path);
  bool ok = fwrite(head.data(), 1, head.size(), f) == head.size() &&
            fwrite(kv, 1, static_cast<size_t>(h.kv_size), f) == h.kv_size;
  fclose(f);
  if (!ok) throw std::runtime_error("short write to " + path + " (out of disk?)");
}

// Parses the file and reads the payload into `kv`. Everything here is a
// property of the file alone; `verify` is what holds it to a run.
inline PrefixHeader read_prefix(const std::string& path, std::vector<uint8_t>* kv) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> head(PREFIX_FIXED_BYTES, 0);
  size_t got = fread(head.data(), 1, head.size(), f);
  if (got != head.size()) {
    fclose(f);
    throw std::runtime_error(path + " is " + std::to_string(got) +
                             " B, shorter than the " + std::to_string(PREFIX_FIXED_BYTES) +
                             " B header of a prefix file");
  }
  if (memcmp(head.data(), PREFIX_MAGIC, sizeof(PREFIX_MAGIC)) != 0) {
    fclose(f);
    throw std::runtime_error(path + " does not start with the prefix magic (--kv-save writes it)");
  }
  PrefixHeader h;
  h.version = get_u32(head, 8);
  if (h.version != PREFIX_VERSION) {
    fclose(f);
    throw std::runtime_error(path + " is prefix format version " + std::to_string(h.version) +
                             "; this harness writes and reads version " +
                             std::to_string(PREFIX_VERSION));
  }
  uint32_t header_bytes = get_u32(head, 12);
  h.isa_version = get_u32(head, 16);
  h.wb = get_u32(head, 20);
  h.max_ctx = get_u32(head, 24);
  uint32_t positions = get_u32(head, 28);
  h.kv_base = get_u64(head, 32);
  h.kv_size = get_u64(head, 40);
  h.model = get_name(head, 48);
  h.image_sha256 = get_name(head, 112);
  if (header_bytes != PREFIX_FIXED_BYTES + 4u * positions) {
    fclose(f);
    throw std::runtime_error(path + " says its header is " + std::to_string(header_bytes) +
                             " B, which is not the " +
                             std::to_string(PREFIX_FIXED_BYTES + 4u * positions) + " B of " +
                             std::to_string(positions) + " positions");
  }
  std::vector<uint8_t> ids(4u * positions, 0);
  if (fread(ids.data(), 1, ids.size(), f) != ids.size()) {
    fclose(f);
    throw std::runtime_error(path + " ends inside the token ids of its header");
  }
  h.ids.resize(positions);
  for (uint32_t i = 0; i < positions; i++) h.ids[i] = get_u32(ids, 4 * i);
  kv->assign(static_cast<size_t>(h.kv_size), 0);
  size_t payload = fread(kv->data(), 1, kv->size(), f);
  uint8_t extra = 0;
  bool trailing = fread(&extra, 1, 1, f) == 1;
  fclose(f);
  if (payload != kv->size() || trailing) {
    throw std::runtime_error(path + " carries " + std::to_string(payload) + (trailing ? "+" : "") +
                             " B after its header, not the " + std::to_string(h.kv_size) +
                             " B KV region it describes");
  }
  return h;
}

// Every term the restored bytes were computed under, held to the run that is
// about to restore them. The first one that differs stops the run: a file that
// belongs to another image, another port width or another prompt would be
// restored into a machine whose KV addresses, tile shape or token history are
// not the ones it was written from, and the generation after it would be wrong
// without saying so.
inline void verify_prefix(const PrefixHeader& h, const Layout& layout,
                          const std::vector<int64_t>& prompt, const std::string& path) {
  auto refuse = [&](const std::string& why) {
    throw std::runtime_error(path + " does not belong to this run: " + why);
  };
  if (h.isa_version != ISA_VERSION) {
    refuse("ISA version " + std::to_string(h.isa_version) + ", this harness " +
           std::to_string(ISA_VERSION));
  }
  if (h.wb != static_cast<uint32_t>(Build::wb)) {
    refuse("saved at WB=" + std::to_string(h.wb) + ", this model is WB=" +
           std::to_string(Build::wb));
  }
  if (h.max_ctx != static_cast<uint32_t>(layout.max_ctx)) {
    refuse("saved at MAX_CTX=" + std::to_string(h.max_ctx) + ", this image is MAX_CTX=" +
           std::to_string(layout.max_ctx));
  }
  if (h.model != layout.model_name) refuse("model " + h.model + ", this image " + layout.model_name);
  if (h.image_sha256 != layout.image_sha256) {
    // The first 16 characters name a file among any set a person has; the whole
    // digest is in the header and in layout.json for anyone comparing files.
    refuse("image sha256 " + h.image_sha256.substr(0, 16) + ", this image " +
           layout.image_sha256.substr(0, 16));
  }
  if (h.kv_base != layout.kv_base || h.kv_size != layout.kv_size) {
    refuse("KV region " + std::to_string(h.kv_size) + " B at " + std::to_string(h.kv_base) +
           ", this image " + std::to_string(layout.kv_size) + " B at " +
           std::to_string(layout.kv_base));
  }
  if (h.positions() == 0) refuse("it covers no position, so there is nothing to restore");
  if (h.positions() + 1 > prompt.size()) {
    refuse("it covers " + std::to_string(h.positions()) + " positions of a " +
           std::to_string(prompt.size()) + "-id prompt, which leaves no position for this run");
  }
  for (uint32_t i = 0; i < h.positions(); i++) {
    if (h.ids[i] != static_cast<uint32_t>(prompt[i])) {
      refuse("position " + std::to_string(i) + " was token " + std::to_string(h.ids[i]) +
             ", this prompt has " + std::to_string(prompt[i]));
    }
  }
}

}  // namespace
}  // namespace qcore
