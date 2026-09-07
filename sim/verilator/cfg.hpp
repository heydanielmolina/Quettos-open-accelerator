// Configuration of one harness run: the widths the model was built with, the
// command line, and the typed view of the compiled model's layout.json. The
// build-time widths come from -D flags set by the Makefile and are checked
// against layout.json before the first cycle.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr_defs.hpp"
#include "json.hpp"

#ifndef QCORE_WB
#define QCORE_WB 64
#endif
#ifndef QCORE_B_MAX
#define QCORE_B_MAX 1
#endif
#ifndef QCORE_VSRAM_WORDS
#define QCORE_VSRAM_WORDS 4096
#endif
#ifndef QCORE_VL
#define QCORE_VL 4
#endif
#ifndef QCORE_FIFO_BEATS
#define QCORE_FIFO_BEATS 128
#endif
#ifndef QCORE_ACC_W
#define QCORE_ACC_W 40
#endif
#ifndef QCORE_MAX_BURST
#define QCORE_MAX_BURST 64
#endif
#ifndef QCORE_THREADS
#define QCORE_THREADS 1
#endif
#ifndef QCORE_TOP_NAME
#define QCORE_TOP_NAME qcore_top
#endif
#define QCORE_STR_(x) #x
#define QCORE_STR(x) QCORE_STR_(x)

namespace qcore {

// Widths of the model this binary links against.
struct Build {
  static constexpr int wb = QCORE_WB;
  static constexpr int b_max = QCORE_B_MAX;
  static constexpr int vsram_words = QCORE_VSRAM_WORDS;
  static constexpr int vl = QCORE_VL;
  static constexpr int fifo_beats = QCORE_FIFO_BEATS;
  static constexpr int acc_w = QCORE_ACC_W;
  static constexpr int max_burst = QCORE_MAX_BURST;
  static constexpr int threads = QCORE_THREADS;
  static constexpr const char* top = QCORE_STR(QCORE_TOP_NAME);
#ifdef QCORE_TRACE
  static constexpr bool trace = true;
#else
  static constexpr bool trace = false;
#endif
};

static_assert(Build::wb >= 16, "WB below 16 is outside the v1 configuration set");
static_assert(Build::wb % 16 == 0, "WB is a multiple of 16");

const char* const USAGE =
    "usage: qcore_sim --image DIR [options]\n"
    "\n"
    "  --image DIR        compiled model directory (image.bin, layout.json, ...)\n"
    "  --lat N            QMEM read latency and write-ack latency in cycles (32)\n"
    "  --bw-div N         one returned beat every N cycles (1)\n"
    "  --threads N        Verilator thread count; must match the built model\n"
    "  --max-new N        tokens to generate (16)\n"
    "  --step             run every descriptor through CTRL.STEP\n"
    "  --dump-ops         with --step, write per-descriptor state from dump_plan.json\n"
    "  --allow-sat        report the saturation and error counters instead of failing\n"
    "  --trace FILE       write a VCD (the model must be built with TRACE=1)\n"
    "  --traffic          rewrite the VROPE and VSOFTMAX descriptors of the program\n"
    "                     regions to NOP before the run: a traffic and cycle\n"
    "                     measurement over the real image, no value check\n"
    "  --program FILE     run this raw descriptor blob instead of the token loop\n"
    "  --program-addr N   where --program is loaded and PC starts (required with it)\n"
    "  --sreg B:I=WORD    load SREG[B][I] with WORD before the run (repeatable)\n"
    "  --tok N            TOK for a --program run (0)\n"
    "  --pos N            POS for a --program run (0)\n"
    "  --row-en N         ROW_EN for a --program run (1)\n"
    "  --dump-vsram B:S:C record VSRAM[B][S .. S+C-1] per descriptor (repeatable)\n"
    "  --dump-mem A:S     record the S bytes at A as int32 after the run (repeatable)\n"
    "  --bringup-json P   where a --program run writes its records\n"
    "  --prompt FILE      token ids to prefill, one per line (DIR/prompt.tokens)\n"
    "  --prompt-ids LIST  comma-separated ids, instead of --prompt\n"
    "  --eos LIST         comma-separated end-of-sequence ids (default: the\n"
    "                     model.eos_ids of layout.json)\n"
    "  --perf-json PATH   where to write the counters (build/perf/perf.json)\n"
    "  --dump-dir DIR     where --dump-ops writes (build/perf/steps)\n"
    "  --kv-save FILE     write the KV region to FILE after the run\n"
    "  --kv-load FILE     load the KV region from FILE before the run\n"
    "  --max-cycles N     stop and fail after N cycles (0: no limit)\n"
    "  --quiet            counters and the final line only\n";

// One SREG word the host loads before a run, and one VSRAM range it records.
struct SregLoad {
  int bank = 0;
  uint32_t index = 0;
  uint32_t word = 0;
};

struct VsramRange {
  int bank = 0;
  uint32_t start = 0;
  uint32_t count = 0;
};

struct MemRange {
  uint64_t addr = 0;
  uint32_t size = 0;
};

struct Options {
  std::string image;
  std::string prompt;
  std::string program;
  std::string bringup_json = "build/perf/bringup.json";
  std::vector<SregLoad> sreg_loads;
  std::vector<VsramRange> dump_vsram;
  std::vector<MemRange> dump_mem;
  uint64_t program_addr = 0;
  uint32_t tok = 0;
  uint32_t pos = 0;
  uint32_t row_en = 1;
  std::string perf_json = "build/perf/perf.json";
  std::string dump_dir = "build/perf/steps";
  std::string trace_file;
  std::string kv_save;
  std::string kv_load;
  std::vector<int64_t> prompt_ids;
  std::vector<int64_t> eos_ids;
  uint32_t lat = 32;
  uint32_t bw_div = 1;
  int threads = 1;
  int max_new = 16;
  uint64_t max_cycles = 0;
  bool step = false;
  bool dump_ops = false;
  bool allow_sat = false;
  bool trace = false;
  bool traffic = false;
  bool quiet = false;
};

inline std::vector<int64_t> parse_ids(const std::string& list) {
  std::vector<int64_t> out;
  size_t pos = 0;
  while (pos < list.size()) {
    size_t comma = list.find(',', pos);
    std::string part = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!part.empty()) out.push_back(strtoll(part.c_str(), nullptr, 0));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

// "bank:index=word", every part accepting 0x.
inline SregLoad parse_sreg(const std::string& spec) {
  SregLoad l;
  size_t colon = spec.find(':');
  size_t eq = spec.find('=');
  if (colon == std::string::npos || eq == std::string::npos || eq < colon) {
    throw std::runtime_error("--sreg wants BANK:INDEX=WORD, got " + spec);
  }
  l.bank = static_cast<int>(strtol(spec.substr(0, colon).c_str(), nullptr, 0));
  l.index = static_cast<uint32_t>(strtoul(spec.substr(colon + 1, eq - colon - 1).c_str(), nullptr, 0));
  l.word = static_cast<uint32_t>(strtoul(spec.substr(eq + 1).c_str(), nullptr, 0));
  if (l.bank < 0 || l.bank >= Build::b_max) throw std::runtime_error("--sreg bank outside B_MAX: " + spec);
  if (l.index >= SREG_COUNT) throw std::runtime_error("--sreg index outside the bank: " + spec);
  return l;
}

// "bank:start:count" in VSRAM elements.
inline VsramRange parse_range(const std::string& spec) {
  VsramRange r;
  size_t a = spec.find(':');
  size_t b = a == std::string::npos ? a : spec.find(':', a + 1);
  if (a == std::string::npos || b == std::string::npos) {
    throw std::runtime_error("--dump-vsram wants BANK:START:COUNT, got " + spec);
  }
  r.bank = static_cast<int>(strtol(spec.substr(0, a).c_str(), nullptr, 0));
  r.start = static_cast<uint32_t>(strtoul(spec.substr(a + 1, b - a - 1).c_str(), nullptr, 0));
  r.count = static_cast<uint32_t>(strtoul(spec.substr(b + 1).c_str(), nullptr, 0));
  if (r.bank < 0 || r.bank >= Build::b_max) throw std::runtime_error("--dump-vsram bank outside B_MAX: " + spec);
  uint64_t end = static_cast<uint64_t>(r.start) + r.count;
  if (end > static_cast<uint64_t>(Build::vsram_words) * VSRAM_WORD_ELEMS) {
    throw std::runtime_error("--dump-vsram range past the end of the VSRAM: " + spec);
  }
  return r;
}

// "addr:size" in bytes; the region is read back as int32 words.
inline MemRange parse_mem(const std::string& spec) {
  MemRange r;
  size_t colon = spec.find(':');
  if (colon == std::string::npos) {
    throw std::runtime_error("--dump-mem wants ADDR:SIZE, got " + spec);
  }
  r.addr = strtoull(spec.substr(0, colon).c_str(), nullptr, 0);
  r.size = static_cast<uint32_t>(strtoul(spec.substr(colon + 1).c_str(), nullptr, 0));
  if (r.size == 0 || (r.size % 4) != 0) {
    throw std::runtime_error("--dump-mem size is a positive multiple of 4: " + spec);
  }
  return r;
}

// Parses the command line; returns false when --help was asked for.
inline bool parse_args(int argc, char** argv, Options* o) {
  auto next = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
    return argv[++i];
  };
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") return false;
    else if (a == "--image") o->image = next(i, "--image");
    else if (a == "--lat") o->lat = static_cast<uint32_t>(strtoul(next(i, "--lat").c_str(), nullptr, 0));
    else if (a == "--bw-div") o->bw_div = static_cast<uint32_t>(strtoul(next(i, "--bw-div").c_str(), nullptr, 0));
    else if (a == "--threads") o->threads = static_cast<int>(strtol(next(i, "--threads").c_str(), nullptr, 0));
    else if (a == "--max-new") o->max_new = static_cast<int>(strtol(next(i, "--max-new").c_str(), nullptr, 0));
    else if (a == "--max-cycles") o->max_cycles = strtoull(next(i, "--max-cycles").c_str(), nullptr, 0);
    else if (a == "--step") o->step = true;
    else if (a == "--dump-ops") { o->dump_ops = true; o->step = true; }
    else if (a == "--allow-sat") o->allow_sat = true;
    else if (a == "--trace") { o->trace = true; o->trace_file = next(i, "--trace"); }
    else if (a == "--traffic") o->traffic = true;
    else if (a == "--quiet") o->quiet = true;
    else if (a == "--prompt") o->prompt = next(i, "--prompt");
    else if (a == "--prompt-ids") o->prompt_ids = parse_ids(next(i, "--prompt-ids"));
    else if (a == "--eos") o->eos_ids = parse_ids(next(i, "--eos"));
    else if (a == "--perf-json") o->perf_json = next(i, "--perf-json");
    else if (a == "--dump-dir") o->dump_dir = next(i, "--dump-dir");
    else if (a == "--kv-save") o->kv_save = next(i, "--kv-save");
    else if (a == "--kv-load") o->kv_load = next(i, "--kv-load");
    else if (a == "--program") o->program = next(i, "--program");
    else if (a == "--program-addr") o->program_addr = strtoull(next(i, "--program-addr").c_str(), nullptr, 0);
    else if (a == "--bringup-json") o->bringup_json = next(i, "--bringup-json");
    else if (a == "--sreg") o->sreg_loads.push_back(parse_sreg(next(i, "--sreg")));
    else if (a == "--dump-vsram") o->dump_vsram.push_back(parse_range(next(i, "--dump-vsram")));
    else if (a == "--dump-mem") o->dump_mem.push_back(parse_mem(next(i, "--dump-mem")));
    else if (a == "--tok") o->tok = static_cast<uint32_t>(strtoul(next(i, "--tok").c_str(), nullptr, 0));
    else if (a == "--pos") o->pos = static_cast<uint32_t>(strtoul(next(i, "--pos").c_str(), nullptr, 0));
    else if (a == "--row-en") o->row_en = static_cast<uint32_t>(strtoul(next(i, "--row-en").c_str(), nullptr, 0));
    else throw std::runtime_error("unknown option " + a);
  }
  if (o->image.empty()) throw std::runtime_error("--image is required");
  if (o->bw_div < 1) throw std::runtime_error("--bw-div is at least 1");
  if (o->threads != Build::threads) {
    throw std::runtime_error("this model was built with --threads " + std::to_string(Build::threads) +
                             "; rebuild with THREADS=" + std::to_string(o->threads));
  }
  if (o->trace && !Build::trace) {
    throw std::runtime_error("this model was built without tracing; rebuild with TRACE=1");
  }
  if (!o->program.empty() && o->program_addr == 0) {
    throw std::runtime_error("--program needs --program-addr");
  }
  if (o->program.empty() &&
      (!o->sreg_loads.empty() || !o->dump_vsram.empty() || !o->dump_mem.empty())) {
    throw std::runtime_error("--sreg, --dump-vsram and --dump-mem belong to a --program run");
  }
  if (o->prompt.empty() && o->prompt_ids.empty()) o->prompt = o->image + "/prompt.tokens";
  return true;
}

// One program of the compiled image.
struct Program {
  uint32_t addr = 0;
  uint32_t size = 0;
  uint32_t descriptors = 0;
};

// The parts of layout.json the harness needs, checked against the build.
struct Layout {
  qjson::Value root;
  std::string dir;
  std::string image_file;
  uint64_t image_size = 0;
  int wb = 0;
  int isa_version = 0;
  int max_ctx = 0;
  std::string model_name;
  int64_t vocab = 0;
  std::vector<int64_t> eos_ids;
  Program decode;
  Program prefill;
  uint64_t kv_base = 0;
  uint64_t kv_size = 0;
  bool has_tokens_bin = false;
  // The compiler's own traffic model, which the PERF counters must reproduce
  // (docs/ISA.md, WT_BYTES and MACS).
  uint64_t decode_wt_bytes = 0, decode_macs = 0;
  uint64_t prefill_wt_bytes = 0, prefill_macs = 0;
  uint64_t head_layers = 0, scores_macs_per_tile = 0, pv_macs_per_token = 0;

  uint64_t expected_macs(bool decode, uint32_t pos) const {
    uint64_t base = decode ? decode_macs : prefill_macs;
    uint64_t tiles = (pos + 1 + Build::wb - 1) / Build::wb;
    return base + head_layers * (tiles * scores_macs_per_tile + (pos + 1) * pv_macs_per_token);
  }
  uint64_t expected_wt_bytes(bool decode) const {
    return decode ? decode_wt_bytes : prefill_wt_bytes;
  }

  static Program program_of(const qjson::Value& v) {
    Program p;
    p.addr = static_cast<uint32_t>(v["addr"].i64());
    p.size = static_cast<uint32_t>(v["size"].i64());
    p.descriptors = static_cast<uint32_t>(v["descriptors"].i64());
    return p;
  }

  static Layout load(const std::string& dir) {
    Layout l;
    l.dir = dir;
    l.root = qjson::parse_file(dir + "/layout.json");
    const qjson::Value& r = l.root;
    if (r.at("format").str() != "quettos-layout") throw std::runtime_error("layout.json: wrong format");
    l.isa_version = static_cast<int>(r.at("isa_version").i64());
    if (l.isa_version != static_cast<int>(ISA_VERSION)) {
      throw std::runtime_error("layout.json ISA_VERSION " + std::to_string(l.isa_version) +
                               " against the harness table " + std::to_string(ISA_VERSION));
    }
    l.wb = static_cast<int>(r.at("wb").i64());
    if (l.wb != Build::wb) {
      throw std::runtime_error("layout.json was compiled for WB=" + std::to_string(l.wb) +
                               "; this model is WB=" + std::to_string(Build::wb));
    }
    l.max_ctx = static_cast<int>(r.at("max_ctx").i64());
    l.model_name = r.at("model.name").str();
    l.vocab = r.at("model.vocab").i64();
    const qjson::Value* eos = r.find_path("model.eos_ids");
    if (eos != nullptr && eos->is_arr()) {
      for (size_t i = 0; i < eos->size(); i++) l.eos_ids.push_back((*eos)[i].i64());
    }
    l.image_file = r.at("image.file").str();
    l.image_size = static_cast<uint64_t>(r.at("image.size").i64());
    l.decode = program_of(r.at("programs.decode"));
    l.prefill = program_of(r.at("programs.prefill"));
    l.kv_base = static_cast<uint64_t>(r.at("bases.kv").i64());
    l.kv_size = l.image_size - l.kv_base;
    l.decode_wt_bytes = static_cast<uint64_t>(r.at("traffic.decode.wt_bytes").i64());
    l.decode_macs = static_cast<uint64_t>(r.at("traffic.decode.macs").i64());
    l.prefill_wt_bytes = static_cast<uint64_t>(r.at("traffic.prefill.wt_bytes").i64());
    l.prefill_macs = static_cast<uint64_t>(r.at("traffic.prefill.macs").i64());
    l.head_layers = static_cast<uint64_t>(r.at("traffic.attention.head_layers").i64());
    l.scores_macs_per_tile = static_cast<uint64_t>(r.at("traffic.attention.scores_macs_per_tile").i64());
    l.pv_macs_per_token = static_cast<uint64_t>(r.at("traffic.attention.pv_macs_per_token").i64());
    const qjson::Value* tb = r.find("tokens_bin");
    l.has_tokens_bin = tb != nullptr && tb->is_obj();
    return l;
  }

  std::string image_path() const { return dir + "/" + image_file; }
  std::string tokens_path() const { return dir + "/tokens.bin"; }
};

}  // namespace qcore
