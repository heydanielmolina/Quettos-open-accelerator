// The sixteen PERF counters as the host reads them (docs/ISA.md, PERF
// indices): a snapshot, the difference between two snapshots, the exclusivity
// check on the six busy buckets, and the perf.json the run writes.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "csr_defs.hpp"

namespace qcore {

inline const char* perf_name(uint32_t i) {
  static const char* names[PERF_COUNT] = {
      "CYCLES", "BUSY",     "MAC_ACTIVE", "STALL_MEM", "STALL_VPU",   "STALL_KV",
      "STALL_SEQ", "STALL_DRAIN", "RD_BEATS", "RD_BYTES", "WT_BYTES", "WR_BEATS",
      "WR_BYTES", "MACS",   "DESCRIPTORS", "FETCH_BEATS"};
  return i < PERF_COUNT ? names[i] : "?";
}

struct PerfSnapshot {
  uint64_t v[PERF_COUNT] = {};

  uint64_t operator[](uint32_t i) const { return v[i]; }
  uint64_t& operator[](uint32_t i) { return v[i]; }

  PerfSnapshot plus(const PerfSnapshot& other) const {
    PerfSnapshot t;
    for (uint32_t i = 0; i < PERF_COUNT; i++) t.v[i] = v[i] + other.v[i];
    return t;
  }

  uint64_t bucket_sum() const {
    uint64_t s = 0;
    for (uint32_t i = PERF_MAC_ACTIVE; i <= PERF_STALL_DRAIN; i++) s += v[i];
    return s;
  }

  // BUSY is the sum of the six exclusive buckets, cycle by cycle.
  bool buckets_exclusive() const { return bucket_sum() == v[PERF_BUSY]; }

  double mac_utilization() const {
    return v[PERF_BUSY] == 0 ? 0.0 : static_cast<double>(v[PERF_MAC_ACTIVE]) / static_cast<double>(v[PERF_BUSY]);
  }
  double bytes_per_cycle() const {
    return v[PERF_CYCLES] == 0 ? 0.0 : static_cast<double>(v[PERF_RD_BYTES]) / static_cast<double>(v[PERF_CYCLES]);
  }
};

// What one token cost, printed per token and kept for perf.json.
struct TokenRecord {
  int index = 0;
  uint32_t pos = 0;
  uint32_t in_id = 0;
  int64_t out_id = -1;
  PerfSnapshot delta;
};

// A small writer for the harness's own output file.
class JsonOut {
 public:
  explicit JsonOut(FILE* f) : f_(f) {}

  void open(char c) { sep(); fputc(c, f_); stack_.push_back(true); }
  void close(char c) { fputc(c, f_); if (!stack_.empty()) stack_.pop_back(); }
  void key(const std::string& k) {
    sep();
    fprintf(f_, "\"%s\": ", k.c_str());
    pending_key_ = true;
  }
  void str(const std::string& v) { sep(); fprintf(f_, "\"%s\"", v.c_str()); }
  void num(uint64_t v) { sep(); fprintf(f_, "%llu", static_cast<unsigned long long>(v)); }
  void snum(int64_t v) { sep(); fprintf(f_, "%lld", static_cast<long long>(v)); }
  void real(double v) { sep(); fprintf(f_, "%.6f", v); }
  void boolean(bool v) { sep(); fputs(v ? "true" : "false", f_); }

  void kv(const std::string& k, uint64_t v) { key(k); num(v); }
  void kv_i(const std::string& k, int64_t v) { key(k); snum(v); }
  void kv_s(const std::string& k, const std::string& v) { key(k); str(v); }
  void kv_d(const std::string& k, double v) { key(k); real(v); }
  void kv_b(const std::string& k, bool v) { key(k); boolean(v); }

 private:
  void sep() {
    if (pending_key_) {
      pending_key_ = false;
      return;
    }
    if (stack_.empty()) return;
    if (!stack_.back()) fputs(", ", f_);
    stack_.back() = false;
  }

  FILE* f_;
  std::vector<bool> stack_;
  bool pending_key_ = false;
};

// The sixteen counters as a JSON object keyed by their names.
inline void write_json_perf(JsonOut& j, const PerfSnapshot& p) {
  j.open('{');
  for (uint32_t i = 0; i < PERF_COUNT; i++) j.kv(perf_name(i), p[i]);
  j.close('}');
}

}  // namespace qcore
