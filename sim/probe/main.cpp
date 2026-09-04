// Quettos Core -- speed probe harness: a simulation cost model. The accelerator harness lives in sim/verilator/.
//
// Drives probe_top through a synthetic Qwen2.5-0.5B-shaped decode token:
//   per layer (x24): GEMV N=1152 K=896 | VPU | GEMV N=896 K=896 | VPU |
//                    GEMV N=9728 K=896 | VPU | GEMV N=896 K=4864 | VPU
//   then LM head GEMV N=151936 K=896 in ARGMAX mode.
// Weight/meta beats come from a 64 MB xorshift-filled buffer through a fixed-
// latency (default 32 cycles), one-beat-per-cycle memory model. Wall-clock is
// measured with std::chrono over a fixed number of simulated cycles.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "Vprobe_top.h"
#include "verilated.h"

#ifndef PROBE_WB
#define PROBE_WB 64
#endif
static constexpr int WB = PROBE_WB;
static constexpr int RD_WORDS = WB * 8 / 32;      // 32-bit words per beat
static constexpr int META_BEATS_PER_TILE = 8;     // WB channels x 8 B / (WB B per beat)

struct XorShift64 {
  uint64_t s;
  explicit XorShift64(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

struct Beat { uint64_t arrive; uint8_t tag; };

// Fixed-latency, 1 beat/cycle external memory model. Beats of one GEMV are
// issued in stream order (K weight beats then 8 meta beats per tile) into a
// delay line; a beat is presented to the RTL when it has arrived and rd_ready.
struct MemModel {
  std::vector<uint32_t> buf;
  uint64_t buf_words = 0;
  uint64_t beat_idx = 0;        // global beat counter -> address
  uint32_t lat = 32;
  uint32_t max_inflight = 64;
  std::deque<Beat> inflight;
  // current stream
  uint32_t tiles_left = 0;
  uint32_t k = 0;
  uint32_t tile_pos = 0;        // 0..k+META_BEATS_PER_TILE-1
  uint64_t issued_total = 0, delivered_total = 0;

  void init(size_t bytes, uint64_t seed) {
    buf_words = bytes / 4;
    buf.resize(buf_words);
    XorShift64 x(seed);
    for (size_t i = 0; i < buf_words; i += 2) {
      uint64_t v = x.next();
      buf[i] = (uint32_t)v;
      if (i + 1 < buf_words) buf[i + 1] = (uint32_t)(v >> 32);
    }
  }
  void start_gemv(uint32_t n_tiles, uint32_t k_len) { tiles_left = n_tiles; k = k_len; tile_pos = 0; }
  bool stream_active() const { return tiles_left != 0; }
  void issue(uint64_t now) {
    if (!stream_active() || inflight.size() >= max_inflight) return;
    uint8_t tag = (tile_pos < k) ? 0 : 1;
    inflight.push_back({now + lat, tag});
    issued_total++;
    if (++tile_pos == k + META_BEATS_PER_TILE) { tile_pos = 0; tiles_left--; }
  }
  // returns true if a beat is presented this cycle
  bool deliver(uint64_t now, Vprobe_top* top) {
    if (inflight.empty() || inflight.front().arrive > now) return false;
    uint64_t base = (beat_idx * RD_WORDS) % buf_words;
    uint8_t tag = inflight.front().tag;
    if (tag == 0) {
      for (int i = 0; i < RD_WORDS; i++) top->rd_data[i] = buf[(base + i) % buf_words];
    } else {
      // per-channel meta {i32 bias, u16 Sw_m in [2^15,2^16), i8 Sw_e in [-8,-1], u8 pad=0},
      // shaped like compiler output so the requant exercises its non-saturating path
      for (int i = 0; i < RD_WORDS; i += 2) {
        uint32_t r0 = buf[(base + i) % buf_words], r1 = buf[(base + i + 1) % buf_words];
        int32_t bias = (int32_t)(r0 << 12) >> 12;                 // 20-bit signed
        uint32_t swm = 0x8000u | (r1 & 0x7FFFu);
        uint32_t swe = (uint32_t)(int8_t)(-1 - (int)((r1 >> 16) & 7)) & 0xFFu;
        top->rd_data[i] = (uint32_t)bias;
        top->rd_data[i + 1] = swm | (swe << 16);
      }
    }
    top->rd_tag = tag;
    inflight.pop_front();
    beat_idx++;
    delivered_total++;
    return true;
  }
};

struct Op { bool vpu; uint32_t n; uint32_t k; uint32_t src; uint32_t dst; bool argmax; };

static std::vector<Op> build_token(uint32_t vpu_words) {
  // vsram word map (elements / 8), from the Qwen VSRAM element map in docs/MEMORY_MAP.md
  const uint32_t X_W = 0, A_W = 224, QKV_W = 336, CTXQ_W = 592, GU_W = 704, HQ_W = 1920;
  std::vector<Op> ops;
  for (int layer = 0; layer < 24; layer++) {
    ops.push_back({false, 1152, 896, A_W, QKV_W, false});
    ops.push_back({true, 0, vpu_words, 0, A_W, false});
    ops.push_back({false, 896, 896, CTXQ_W, X_W, false});
    ops.push_back({true, 0, vpu_words, 0, CTXQ_W, false});
    ops.push_back({false, 9728, 896, A_W, GU_W, false});
    ops.push_back({true, 0, vpu_words, 0, 1628, false});
    ops.push_back({false, 896, 4864, HQ_W, X_W, false});
    ops.push_back({true, 0, vpu_words, 0, X_W, false});
  }
  ops.push_back({false, 151936, 896, A_W, X_W, true});
  return ops;
}

int main(int argc, char** argv) {
  int tokens = 2;
  uint32_t lat = 32;
  uint32_t vpu_words = 900;
  uint64_t seed = 0x243F6A8885A308D3ull;
  bool verbose = true;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](void) -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
    if (a == "--tokens") tokens = atoi(next());
    else if (a == "--lat") lat = (uint32_t)atoi(next());
    else if (a == "--vpu-words") vpu_words = (uint32_t)atoi(next());
    else if (a == "--seed") seed = strtoull(next(), nullptr, 0);
    else if (a == "--quiet") verbose = false;
    else { fprintf(stderr, "usage: probe [--tokens N] [--lat N] [--vpu-words N] [--seed X] [--quiet]\n"); return 2; }
  }

  VerilatedContext ctx;
  ctx.commandArgs(argc, argv);
  Vprobe_top top(&ctx);

  MemModel mem;
  mem.lat = lat;
  mem.init(64u << 20, seed);

  auto ops = build_token(vpu_words);
  uint64_t beats_per_token = 0;
  for (auto& op : ops) if (!op.vpu) beats_per_token += (uint64_t)(op.n / WB) * (op.k + META_BEATS_PER_TILE);

  uint64_t cycles = 0;
  auto tick = [&](void) {
    top.rd_valid = 0;
    if (top.rd_ready && mem.deliver(cycles, &top)) top.rd_valid = 1;
    mem.issue(cycles);
    top.clk = 1; top.eval();
    top.clk = 0; top.eval();
    cycles++;
  };

  // reset
  top.clk = 0; top.rst = 1; top.start = 0; top.rd_valid = 0; top.rd_tag = 0;
  top.sx_m = 0xC0DE; top.sx_e = (uint8_t)(-10); top.sbias = 4; top.vsh1 = 14; top.vsh2 = 18;
  for (int i = 0; i < 4; i++) tick();
  top.rst = 0;
  tick();

  printf("probe_top WB=%d  lat=%u  vpu_words=%u  tokens=%d  beats/token=%llu\n",
         WB, lat, vpu_words, tokens, (unsigned long long)beats_per_token);
  fflush(stdout);

  auto t0 = std::chrono::steady_clock::now();
  uint64_t cycles0 = cycles;
  for (int t = 0; t < tokens; t++) {
    uint64_t tok_start = cycles;
    for (size_t oi = 0; oi < ops.size(); oi++) {
      const Op& op = ops[oi];
      top.start = 1;
      top.cmd_vpu = op.vpu;
      top.cmd_argmax = op.argmax;
      top.n_tiles = op.vpu ? 0 : (op.n / WB);
      top.k_len = op.k;
      top.vs_src = op.src;
      top.vs_dst = op.dst;
      if (!op.vpu) mem.start_gemv(op.n / WB, op.k);
      tick();
      top.start = 0;
      uint64_t guard = cycles + 200000000ull;
      while (!top.done) {
        tick();
        if (cycles > guard) { fprintf(stderr, "TIMEOUT waiting for done at op %zu\n", oi); return 1; }
      }
    }
    if (verbose) {
      printf("  token %d: %llu cycles  argmax id=%u val=%d  checksum=%08x\n", t,
             (unsigned long long)(cycles - tok_start), top.argmax_idx, (int32_t)top.argmax_val, top.checksum);
      fflush(stdout);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  uint64_t sim_cycles = cycles - cycles0;

  printf("cycles=%llu seconds=%.3f Mcycles/s=%.3f\n", (unsigned long long)sim_cycles, secs, sim_cycles / secs / 1e6);
  printf("checksum=%08x absmax_req=%u absmax_vpu=%u argmax_idx=%u argmax_val=%d sat=%u err=%u\n",
         top.checksum, top.absmax_req, top.absmax_vpu, top.argmax_idx, (int32_t)top.argmax_val, top.sat_count, top.err_count);
  printf("perf: cycles=%llu busy=%llu beats=%llu mac_active=%llu  (mem model: issued=%llu delivered=%llu)\n",
         (unsigned long long)top.perf_cycles, (unsigned long long)top.perf_busy,
         (unsigned long long)top.perf_beats, (unsigned long long)top.perf_mac,
         (unsigned long long)mem.issued_total, (unsigned long long)mem.delivered_total);
  printf("RESULT WB=%d cycles=%llu seconds=%.3f mcps=%.3f checksum=%08x\n", WB,
         (unsigned long long)sim_cycles, secs, sim_cycles / secs / 1e6, top.checksum);
  top.final();
  return 0;
}
