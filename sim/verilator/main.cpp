// Quettos Core harness: the host side of the machine. It maps the compiled
// image.bin behind the QMEM bus model, drives qcore_top over the CSR window,
// and runs one of two workloads: the prefill/decode loop of
// sw/quettos/isa_sim.py, or a descriptor blob given with --program, whose
// per-descriptor state is written out for sw/quettos/compare.py to check
// against the simulator.
//
// The model, the bus model and the clock are device.hpp; this file is the run:
// the options, the loops, the checks and the output.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cfg.hpp"
#include "csr.hpp"
#include "detok.hpp"
#include "bringup.hpp"
#include "device.hpp"
#include "json.hpp"
#include "perf.hpp"

namespace qcore {
namespace {

// A path as the filesystem names it, so a record of the file a run read is
// read back the same way from any directory. POSIX.1-2008 realpath allocates
// the buffer, which keeps the path length out of this file.
std::string real_path(const std::string& path) {
  char* resolved = realpath(path.c_str(), nullptr);
  if (resolved == nullptr) return path;
  std::string out(resolved);
  free(resolved);
  return out;
}

std::vector<int64_t> read_prompt(const std::string& path) {
  std::vector<int64_t> ids;
  FILE* f = fopen(path.c_str(), "r");
  if (f == nullptr) return ids;
  long long v = 0;
  while (fscanf(f, "%lld", &v) == 1) ids.push_back(v);
  fclose(f);
  return ids;
}

// What makes a run's numbers something other than a clean end-to-end
// measurement, recorded in perf.json next to the counters: the stop that ended
// the run.
struct RunMarks {
  bool stopped = false;
  std::string reason;
  uint32_t fault = 0;
  uint32_t fault_op = 0;
  uint32_t pc = 0;

  const char* status() const { return stopped ? "stopped" : "ok"; }

  void set_stop(const std::string& why, Status s, uint32_t at) {
    stopped = true;
    reason = why;
    fault = s.fault();
    fault_op = s.fault_op();
    pc = at;
  }
};

// ---------------------------------------------------------------- token run

struct Run {
  Machine* m;
  const Layout* layout;
  const Options* opt;
  Vocabulary vocab;
  qjson::Value dump_plan;
  std::vector<TokenRecord> tokens;
  uint64_t descriptors_dumped = 0;
  bool ok = true;
  // What the run was given, recorded beside its counters: the prompt ids the
  // token loop prefilled and where they came from, and the image file the
  // hardware executed with the bytes the memory model mapped of it.
  std::vector<int64_t> prompt;
  std::string prompt_from;
  std::string image_path;
  uint64_t image_bytes = 0;
  // What the counters read before this token's first descriptor. START clears
  // them, so the base is zero on the token path; STEP clears none of them
  // (docs/ISA.md, CTRL), so a stepped token's cost is measured from here.
  PerfSnapshot perf_base;
  Events event_base;

  PerfSnapshot raw_perf() {
    PerfSnapshot s;
    for (uint32_t i = 0; i < PERF_COUNT; i++) s[i] = m->perf(i);
    return s;
  }

  // This token's counters, whichever way it ran.
  PerfSnapshot snapshot() { return raw_perf().minus(perf_base); }

  Events events() {
    Events now = read_events(*m);
    Events e;
    e.sat_req = now.sat_req - event_base.sat_req;
    e.sat_vpu = now.sat_vpu - event_base.sat_vpu;
    e.err_shift = now.err_shift - event_base.err_shift;
    e.err_bounds = now.err_bounds - event_base.err_bounds;
    return e;
  }

  // One descriptor at a time, dumping the state its plan entry names.
  bool step_program(const std::string& which, uint32_t pos) {
    const qjson::Value* plan = dump_plan.is_obj() ? dump_plan.find(which) : nullptr;
    std::string path = opt->dump_dir + "/" + which + "_pos" + std::to_string(pos) + ".json";
    FILE* out = nullptr;
    if (opt->dump_ops) {
      make_dir(opt->dump_dir);
      out = fopen(path.c_str(), "w");
      if (out == nullptr) { fprintf(stderr, "cannot write %s\n", path.c_str()); return false; }
      fputs("[", out);
    }
    uint32_t index = 0;
    uint32_t dumped = 0;
    bool first = true;
    for (;;) {
      m->write(CSR_CTRL, 1u << CTRL_STEP);
      if (!m->run_until_halt()) { ok = false; break; }
      Status s = m->status();
      if (s.err()) { ok = false; break; }
      if (out != nullptr && plan != nullptr && index < plan->size()) {
        if (!first) fputs(",\n", out);
        first = false;
        dump_entry(out, (*plan)[index], index);
        dumped++;
      }
      index++;
      m->write(CSR_STATUS, s.raw);
      if (s.done()) break;
    }
    if (out != nullptr) {
      fputs("]\n", out);
      fclose(out);
      descriptors_dumped += dumped;
    }
    return ok;
  }

  void dump_entry(FILE* out, const qjson::Value& e, uint32_t index) {
    JsonOut j(out);
    j.open('{');
    j.kv("index", index);
    j.kv_s("name", e.has("name") && e["name"].is_str() ? e["name"].str() : "");
    j.kv_s("op", e.has("op") && e["op"].is_str() ? e["op"].str() : "");
    j.kv("pc", m->read(CSR_PC));
    const qjson::Value* vs = e.find("vsram");
    j.key("vsram");
    if (vs != nullptr && vs->is_obj()) {
      uint32_t start = static_cast<uint32_t>((*vs)["start"].i64());
      uint32_t count = static_cast<uint32_t>((*vs)["count"].i64());
      int bank = 0;  // v1 programs run row 0 (docs/ISA.md, rows)
      j.open('{');
      j.kv("start", start);
      j.kv("count", count);
      j.key("values");
      j.open('[');
      for (uint32_t i = 0; i < count; i++) j.snum(static_cast<int32_t>(m->vsram(bank, start + i)));
      j.close(']');
      j.close('}');
    } else {
      j.str("");
    }
    j.key("sreg");
    j.open('{');
    const qjson::Value* sr = e.find("sreg");
    if (sr != nullptr && sr->is_arr()) {
      for (size_t i = 0; i < sr->size(); i++) {
        uint32_t idx = static_cast<uint32_t>((*sr)[i].i64());
        j.kv(std::to_string(idx), m->read_sreg(0, idx));
      }
    }
    j.close('}');
    j.key("mem");
    j.open('{');
    const qjson::Value* mem = e.find("mem");
    if (mem != nullptr && mem->is_arr()) {
      for (size_t i = 0; i < mem->size(); i++) {
        const qjson::Value& r = (*mem)[i];
        uint64_t addr = static_cast<uint64_t>(r["addr"].i64());
        uint64_t size = static_cast<uint64_t>(r["size"].i64());
        j.key(r["name"].str());
        j.open('{');
        j.kv("addr", addr);
        j.kv("size", size);
        // FNV-1a over the region: a KV region is 128 KB and the comparison
        // side hashes isa_sim's bytes the same way. --dump-bytes N carries the
        // bytes themselves for a region of at most N, so a difference is
        // reported at the byte it is in rather than as a hash.
        j.key("fnv1a64");
        std::vector<uint8_t> buf(static_cast<size_t>(size));
        m->bytes()->read(addr, static_cast<uint32_t>(size), buf.data());
        uint64_t h = 14695981039346656037ull;
        for (uint8_t b : buf) {
          h ^= b;
          h *= 1099511628211ull;
        }
        j.num(h);
        if (size <= opt->dump_bytes) {
          static const char* kHex = "0123456789abcdef";
          std::string hex(2 * buf.size(), '0');
          for (size_t b = 0; b < buf.size(); b++) {
            hex[2 * b] = kHex[buf[b] >> 4];
            hex[2 * b + 1] = kHex[buf[b] & 0xF];
          }
          j.kv_s("hex", hex);
        }
        j.close('}');
      }
    }
    j.close('}');
    j.key("csr");
    j.open('{');
    const qjson::Value* cs = e.find("csr");
    if (cs != nullptr && cs->is_arr()) {
      for (size_t i = 0; i < cs->size(); i++) {
        const std::string& name = (*cs)[i].str();
        // The plan names a register of the CSR window; a name the table does
        // not carry stops the dump, so a comparison is never made against the
        // wrong register.
        uint32_t word = 0;
        if (!csr_word_of(name, &word)) {
          throw std::runtime_error("dump_plan.json names \"" + name +
                                   "\", which is no register of the CSR window (" +
                                   csr_name_list() + ")");
        }
        j.kv(name, m->read(word));
      }
    }
    j.close('}');
    // The three counters the ISA simulator models and the four event counters,
    // both since this token's first descriptor.
    PerfSnapshot p = snapshot();
    j.key("perf");
    j.open('{');
    j.kv("DESCRIPTORS", p[PERF_DESCRIPTORS]);
    j.kv("MACS", p[PERF_MACS]);
    j.kv("WT_BYTES", p[PERF_WT_BYTES]);
    j.close('}');
    Events ev = events();
    j.key("events");
    j.open('{');
    j.kv("SAT_REQ", ev.sat_req);
    j.kv("SAT_VPU", ev.sat_vpu);
    j.kv("ERR_SHIFT", ev.err_shift);
    j.kv("ERR_BOUNDS", ev.err_bounds);
    j.close('}');
    j.close('}');
  }

  // TOK / POS / ROW_EN, then START (or one STEP per descriptor), as
  // isa_sim.run_token sequences it.
  int64_t run_token(const Program& prog, const std::string& which, uint32_t tok, uint32_t pos) {
    m->write(CSR_STATUS, m->read(CSR_STATUS));  // clear what the last run left
    m->write(CSR_PC, prog.addr);
    m->write(CSR_TOK, tok);
    m->write(CSR_POS, pos);
    m->write(CSR_ROW_EN, 1);
    perf_base = opt->step ? raw_perf() : PerfSnapshot{};
    event_base = opt->step ? read_events(*m) : Events{};
    if (opt->step) {
      if (!step_program(which, pos)) return -1;
    } else {
      m->write(CSR_CTRL, 1u << CTRL_START);
      if (!m->run_until_halt()) { ok = false; return -1; }
    }
    Status s = m->status();
    if (s.err()) {
      ok = false;
      return -1;
    }
    return static_cast<int64_t>(m->read(CSR_ARGMAX_TOK));
  }
};

// WT_BYTES and MACS against the compiler's traffic model (docs/ISA.md).
int check_token(const Layout& layout, const PerfSnapshot& p, bool decode, uint32_t pos,
                uint32_t descriptors) {
  int rc = 0;
  if (p[PERF_DESCRIPTORS] != descriptors) {
    // The traffic model is the whole program's, so a token that retired
    // something other than the program's descriptor count is reported here
    // rather than compared against it.
    printf("FAIL: token at pos %u retired %llu descriptors, not the program's %u\n", pos,
           (unsigned long long)p[PERF_DESCRIPTORS], descriptors);
    rc = 1;
  } else {
    uint64_t wt = layout.expected_wt_bytes(decode);
    uint64_t macs = layout.expected_macs(decode, pos);
    if (p[PERF_WT_BYTES] != wt) {
      printf("FAIL: WT_BYTES=%llu against layout.json %llu\n", (unsigned long long)p[PERF_WT_BYTES],
             (unsigned long long)wt);
      rc = 1;
    }
    if (p[PERF_MACS] != macs) {
      printf("FAIL: MACS=%llu against layout.json %llu\n", (unsigned long long)p[PERF_MACS],
             (unsigned long long)macs);
      rc = 1;
    }
  }
  if (!p.buckets_exclusive()) {
    printf("FAIL: token at pos %u: BUSY=%llu is not the sum of the six buckets (%llu)\n", pos,
           (unsigned long long)p[PERF_BUSY], (unsigned long long)p.bucket_sum());
    rc = 1;
  }
  return rc;
}

void print_counters(const PerfSnapshot& p, const char* label) {
  printf("%s cycles=%llu busy=%llu mac=%llu mem=%llu vpu=%llu kv=%llu seq=%llu drain=%llu\n", label,
         (unsigned long long)p[PERF_CYCLES], (unsigned long long)p[PERF_BUSY],
         (unsigned long long)p[PERF_MAC_ACTIVE], (unsigned long long)p[PERF_STALL_MEM],
         (unsigned long long)p[PERF_STALL_VPU], (unsigned long long)p[PERF_STALL_KV],
         (unsigned long long)p[PERF_STALL_SEQ], (unsigned long long)p[PERF_STALL_DRAIN]);
  printf("%s rd_beats=%llu rd_bytes=%llu wr_beats=%llu wr_bytes=%llu wt_bytes=%llu macs=%llu "
         "descriptors=%llu fetch_beats=%llu\n",
         label, (unsigned long long)p[PERF_RD_BEATS], (unsigned long long)p[PERF_RD_BYTES],
         (unsigned long long)p[PERF_WR_BEATS], (unsigned long long)p[PERF_WR_BYTES],
         (unsigned long long)p[PERF_WT_BYTES], (unsigned long long)p[PERF_MACS],
         (unsigned long long)p[PERF_DESCRIPTORS], (unsigned long long)p[PERF_FETCH_BEATS]);
}

void write_perf_json(const Options& o, const Layout& layout, const Machine& m, const Run& run,
                     const Events& ev, const PerfSnapshot& total, double seconds, const char* mode,
                     const RunMarks& marks, uint64_t sim_cycles) {
  size_t slash = o.perf_json.find_last_of('/');
  if (slash != std::string::npos) make_dir(o.perf_json.substr(0, slash));
  FILE* f = fopen(o.perf_json.c_str(), "w");
  if (f == nullptr) {
    fprintf(stderr, "cannot write %s\n", o.perf_json.c_str());
    return;
  }
  JsonOut j(f);
  j.open('{');
  j.kv_s("format", "quettos-perf");
  j.kv("isa_version", ISA_VERSION);
  j.kv_s("model", layout.model_name);
  j.kv_s("mode", mode);
  j.key("build");
  j.open('{');
  j.kv_s("top", Build::top);
  j.kv("wb", Build::wb);
  j.kv("b_max", Build::b_max);
  j.kv("vl", Build::vl);
  j.kv("vsram_words", Build::vsram_words);
  j.kv("fifo_beats", Build::fifo_beats);
  j.kv("acc_w", Build::acc_w);
  j.kv("threads", Build::threads);
  j.close('}');
  j.key("run");
  j.open('{');
  j.kv("lat", o.lat);
  j.kv("bw_div", o.bw_div);
  j.kv("max_new", static_cast<uint64_t>(o.max_new));
  j.kv_b("step", o.step);
  j.kv_s("status", marks.status());
  if (marks.stopped) {
    j.kv_s("stop_reason", marks.reason);
    j.kv("fault", marks.fault);
    j.kv("fault_op", marks.fault_op);
    j.kv("fault_pc", marks.pc);
  }
  j.kv("clock_cycles", sim_cycles);
  j.kv_d("wall_seconds", seconds);
  j.kv_d("mcycles_per_s", seconds > 0 ? sim_cycles / seconds / 1e6 : 0.0);
  // The image the hardware executed and the prompt the loop was given: a run's
  // ids belong to one prompt, and a reader holds them to the record of that
  // one (sw/quettos/cli.py, demo-report).
  j.kv_s("image", run.image_path);
  j.kv("image_bytes", run.image_bytes);
  j.key("prompt");
  j.open('{');
  j.kv_s("from", run.prompt_from);
  j.kv("count", static_cast<uint64_t>(run.prompt.size()));
  j.key("ids");
  j.open('[');
  for (int64_t id : run.prompt) j.snum(id);
  j.close(']');
  j.close('}');
  j.close('}');
  j.key("counters");
  write_json_perf(j, total);
  j.key("events");
  j.open('{');
  j.kv("SAT_REQ", ev.sat_req);
  j.kv("SAT_VPU", ev.sat_vpu);
  j.kv("ERR_SHIFT", ev.err_shift);
  j.kv("ERR_BOUNDS", ev.err_bounds);
  j.close('}');
  j.key("memory");
  j.open('{');
  j.kv("rd_requests", m.qmem().rd_requests);
  j.kv("rd_beats", m.qmem().rd_beats);
  j.kv("rd_bytes", m.qmem().rd_bytes);
  j.kv("wr_beats", m.qmem().wr_beats);
  j.kv("wr_bytes", m.qmem().wr_bytes);
  j.close('}');
  j.key("tokens");
  j.open('[');
  for (const TokenRecord& t : run.tokens) {
    j.open('{');
    j.kv("index", static_cast<uint64_t>(t.index));
    // Which program ran, rather than what came out of it: a token that faulted
    // has no id, and it still belongs to the program it faulted in.
    j.kv_s("pass", t.decode ? "decode" : "prefill");
    j.kv("pos", t.pos);
    j.kv("in", t.in_id);
    j.kv_i("out", t.out_id);
    j.kv("cycles", t.delta[PERF_CYCLES]);
    j.kv("mac_active", t.delta[PERF_MAC_ACTIVE]);
    j.kv("rd_bytes", t.delta[PERF_RD_BYTES]);
    j.kv("descriptors", t.delta[PERF_DESCRIPTORS]);
    j.close('}');
  }
  j.close(']');
  j.close('}');
  fputc('\n', f);
  fclose(f);
}

int main_impl(int argc, char** argv) {
  Options o;
  if (!parse_args(argc, argv, &o)) {
    fputs(USAGE, stdout);
    return 0;
  }
  Layout layout = Layout::load(o.image);
  // The compiled model names its own end-of-sequence ids; --eos overrides them.
  if (o.eos_ids.empty()) o.eos_ids = layout.eos_ids;
  MemBytes bytes;
  bytes.open_image(layout.image_path());
  // The image the hardware is about to execute is the one layout.json
  // describes. The size costs nothing and is checked here; the SHA-256 is
  // 0.179 s on the 513,950,464 B Qwen image, so it is checked where the demo
  // checks its other invariants (`quettos demo-report`), once at the end of a
  // run rather than at the start of every one.
  if (bytes.size() != layout.image_size) {
    throw std::runtime_error(layout.image_path() + " is " + std::to_string(bytes.size()) +
                             " B; layout.json describes " + std::to_string(layout.image_size));
  }
  if (!o.kv_load.empty()) {
    FILE* f = fopen(o.kv_load.c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open " + o.kv_load);
    std::vector<uint8_t> buf(static_cast<size_t>(layout.kv_size));
    size_t n = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    bytes.write(layout.kv_base, static_cast<uint32_t>(n), buf.data());
    printf("kv: loaded %llu bytes from %s\n", (unsigned long long)n, o.kv_load.c_str());
  }

  VerilatedContext ctx;
  ctx.commandArgs(argc, argv);
  Dut dut(&ctx);
  Machine m(&dut, &bytes, o);
#ifdef QCORE_TRACE
  if (o.trace) m.open_trace(o.trace_file, &ctx, o.trace_cycles);
#endif
  m.reset();

  if (!o.quiet) {
    printf("qcore_sim: top=%s WB=%d B_MAX=%d VSRAM_WORDS=%d threads=%d\n", Build::top, Build::wb,
           Build::b_max, Build::vsram_words, Build::threads);
    if (o.trace) {
      // A cycle is about 12 KB of VCD here, so the window is what keeps the
      // file to a size a disk holds (--trace-cycles).
      if (o.trace_cycles == 0) {
        printf("trace: %s, every cycle of the run\n", o.trace_file.c_str());
      } else {
        printf("trace: %s, the first %llu cycles\n", o.trace_file.c_str(),
               (unsigned long long)o.trace_cycles);
      }
    }
    printf("image: %s (%llu B) model=%s isa_version=%d\n", layout.image_path().c_str(),
           (unsigned long long)layout.image_size, layout.model_name.c_str(), layout.isa_version);
    printf("qmem: lat=%u bw_div=%u  programs: decode@0x%08x (%u) prefill@0x%08x (%u)\n", o.lat,
           o.bw_div, layout.decode.addr, layout.decode.descriptors, layout.prefill.addr,
           layout.prefill.descriptors);
  }
  if (m.read(CSR_ISA_VERSION) != ISA_VERSION) {
    throw std::runtime_error("the model reports a different ISA_VERSION than the harness table");
  }

  // --- a descriptor blob of our own: the bring-up path
  if (!o.program.empty()) {
    Bringup b(&m, &o);
    int rc = b.run(layout);
    if (b.records().empty()) {
      printf("stopped: %s\n", m.stop_reason().empty() ? "no descriptor retired"
                                                      : m.stop_reason().c_str());
      printf("%s PC=0x%08x\n", m.status().text().c_str(), m.read(CSR_PC));
      m.close_trace();
      return 2;
    }
    const StepRecord& last = b.records().back();
    Status s{last.status};
    printf("bringup: records=%zu %s PC=0x%08x ARGMAX_TOK=%u ARGMAX_VAL=%d\n", b.records().size(),
           s.text().c_str(), last.pc, last.argmax_tok, static_cast<int32_t>(last.argmax_val));
    print_counters(last.perf, "perf:");
    printf("events: SAT_REQ=%u SAT_VPU=%u ERR_SHIFT=%u ERR_BOUNDS=%u\n", last.events.sat_req,
           last.events.sat_vpu, last.events.err_shift, last.events.err_bounds);
    if (!m.stop_reason().empty()) printf("stopped: %s\n", m.stop_reason().c_str());
    if (last.events.total() != 0 && !o.allow_sat) {
      printf("FAIL: %u saturation and range events (--allow-sat reports them instead)\n",
             last.events.total());
      if (rc == 0) rc = 1;
    }
    if (!last.perf.buckets_exclusive()) {
      printf("FAIL: BUSY=%llu is not the sum of the six buckets (%llu)\n",
             (unsigned long long)last.perf[PERF_BUSY],
             (unsigned long long)last.perf.bucket_sum());
      if (rc == 0) rc = 1;
    }
    printf("RESULT top=%s wb=%d clock_cycles=%llu seconds=%.3f mcps=%.3f records=%zu\n", Build::top,
           Build::wb, (unsigned long long)b.cycles(), b.seconds(),
           b.seconds() > 0 ? b.cycles() / b.seconds() / 1e6 : 0.0, b.records().size());
    m.close_trace();
    return rc;
  }

  // --- the compiled programs
  RunMarks marks;

  Run run{&m, &layout, &o, {}, {}, {}, 0, true};
  if (layout.has_tokens_bin) run.vocab.load(layout.tokens_path());
  if (o.step || o.dump_ops) run.dump_plan = qjson::parse_file(o.image + "/dump_plan.json");

  std::vector<int64_t> prompt = o.prompt_ids;
  std::string prompt_from = "--prompt-ids";
  if (prompt.empty()) {
    prompt = read_prompt(o.prompt);
    prompt_from = real_path(o.prompt);
  }
  if (prompt.empty()) {
    prompt.push_back(0);
    prompt_from = "id 0, the harness default";
  }
  run.prompt = prompt;
  run.prompt_from = prompt_from;
  run.image_path = real_path(layout.image_path());
  run.image_bytes = bytes.size();
  if (!o.quiet) {
    printf("prompt: %zu tokens from %s, max_new=%d\n", prompt.size(), prompt_from.c_str(),
           o.max_new);
  }

  TokenStream out(&run.vocab, stdout);
  auto t0 = std::chrono::steady_clock::now();
  uint64_t cycles0 = m.cycles();

  PerfSnapshot total;
  Events ev_total;
  int rc_counters = 0;
  int index = 0;
  for (size_t i = 0; i + 1 < prompt.size() && run.ok; i++) {
    run.run_token(layout.prefill, "prefill", static_cast<uint32_t>(prompt[i]),
                  static_cast<uint32_t>(i));
    // START clears the counters, so a token's snapshot is that token's cost.
    PerfSnapshot now = run.snapshot();
    TokenRecord rec;
    rec.index = index++;
    rec.pos = static_cast<uint32_t>(i);
    rec.in_id = static_cast<uint32_t>(prompt[i]);
    rec.delta = now;
    rec.decode = false;
    total = total.plus(now);
    ev_total = ev_total.plus(run.events());
    run.tokens.push_back(rec);
    rc_counters |= check_token(layout, now, false, rec.pos, layout.prefill.descriptors);
    if (!o.quiet) {
      printf("prefill %zu: pos=%zu cycles=%llu mac=%.1f%% descriptors=%llu\n", i, i,
             (unsigned long long)rec.delta[PERF_CYCLES], 100.0 * rec.delta.mac_utilization(),
             (unsigned long long)rec.delta[PERF_DESCRIPTORS]);
    }
  }

  uint32_t tok = static_cast<uint32_t>(prompt.back());
  uint32_t pos = static_cast<uint32_t>(prompt.size() - 1);
  for (int jj = 0; jj < o.max_new && run.ok; jj++) {
    int64_t nxt = run.run_token(layout.decode, "decode", tok, pos + static_cast<uint32_t>(jj));
    PerfSnapshot now = run.snapshot();
    TokenRecord rec;
    rec.index = index++;
    rec.pos = pos + static_cast<uint32_t>(jj);
    rec.in_id = tok;
    rec.out_id = nxt;
    rec.delta = now;
    rec.decode = true;
    total = total.plus(now);
    ev_total = ev_total.plus(run.events());
    run.tokens.push_back(rec);
    // Every token that ran is held to the traffic model and to the bucket sum,
    // the one that faulted included: it retired fewer descriptors than the
    // program has, which is what check_token reports.
    rc_counters |= check_token(layout, now, true, rec.pos, layout.decode.descriptors);
    if (nxt < 0) break;
    if (!o.quiet) {
      printf("[token %d pos=%u id=%lld cycles=%llu mac=%.1f%%] ", jj, rec.pos,
             (long long)nxt, (unsigned long long)rec.delta[PERF_CYCLES],
             100.0 * rec.delta.mac_utilization());
    }
    out.push(static_cast<uint32_t>(nxt));
    bool stop = false;
    for (int64_t e : o.eos_ids) stop = stop || e == nxt;
    if (stop) break;
    tok = static_cast<uint32_t>(nxt);
  }
  out.finish();
  auto t1 = std::chrono::steady_clock::now();
  double seconds = std::chrono::duration<double>(t1 - t0).count();
  uint64_t sim_cycles = m.cycles() - cycles0;
  printf("\n");

  Events ev = ev_total;  // START clears them per token, so the run's events are the sum
  const char* mode = o.step ? "step" : "token";
  print_counters(total, "perf:");
  printf("events: SAT_REQ=%u SAT_VPU=%u ERR_SHIFT=%u ERR_BOUNDS=%u\n", ev.sat_req, ev.sat_vpu,
         ev.err_shift, ev.err_bounds);
  printf("memory: rd_requests=%llu rd_beats=%llu rd_bytes=%llu wr_beats=%llu wr_bytes=%llu\n",
         (unsigned long long)m.qmem().rd_requests, (unsigned long long)m.qmem().rd_beats,
         (unsigned long long)m.qmem().rd_bytes, (unsigned long long)m.qmem().wr_beats,
         (unsigned long long)m.qmem().wr_bytes);

  int rc = rc_counters;
  if (!total.buckets_exclusive()) {
    printf("FAIL: BUSY=%llu is not the sum of the six buckets (%llu)\n",
           (unsigned long long)total[PERF_BUSY], (unsigned long long)total.bucket_sum());
    rc = 1;
  }
  // A read burst still in flight when a program halts lands after the PERF
  // snapshot that HALT takes, so RD_BEATS trails the bus model by at most the
  // fetch unit's two outstanding bursts per token (docs/RTL.md 3.4).
  uint64_t slack = run.tokens.size() * 2 * (WB >= static_cast<int>(DESC_BYTES) ? 1 : DESC_BYTES / WB);
  uint64_t model = m.qmem().rd_beats;
  uint64_t counted = total[PERF_RD_BEATS];
  if (counted > model || model - counted > slack) {
    printf("FAIL: PERF counted %llu read beats, the memory model %llu (at most %llu may be in "
           "flight at a HALT)\n",
           (unsigned long long)counted, (unsigned long long)model, (unsigned long long)slack);
    rc = 1;
  }
  if (total[PERF_RD_BYTES] != counted * static_cast<uint64_t>(WB)) {
    printf("FAIL: RD_BYTES=%llu is not RD_BEATS * WB\n", (unsigned long long)total[PERF_RD_BYTES]);
    rc = 1;
  }
  if (total[PERF_WR_BEATS] != m.qmem().wr_beats || total[PERF_WR_BYTES] != m.qmem().wr_bytes) {
    printf("FAIL: PERF counted %llu write beats / %llu bytes, the memory model %llu/%llu\n",
           (unsigned long long)total[PERF_WR_BEATS], (unsigned long long)total[PERF_WR_BYTES],
           (unsigned long long)m.qmem().wr_beats, (unsigned long long)m.qmem().wr_bytes);
    rc = 1;
  }
  if (ev.total() != 0 && !o.allow_sat) {
    printf("FAIL: %u saturation and range events (--allow-sat reports them instead)\n", ev.total());
    rc = 1;
  }
  if (!m.stop_reason().empty()) {
    Status s = m.status();
    marks.set_stop(m.stop_reason(), s, m.read(CSR_PC));
    printf("stopped: %s\n", marks.reason.c_str());
    printf("%s PC=0x%08x\n", s.text().c_str(), marks.pc);
    rc = 2;
  } else if (!run.ok) {
    Status s = m.status();
    marks.set_stop("a descriptor faulted", s, m.read(CSR_PC));
    printf("stopped: the descriptor at PC=0x%08x faulted; STATUS carries the fault code and the "
           "opcode byte\n",
           marks.pc);
    printf("%s PC=0x%08x\n", s.text().c_str(), marks.pc);
    rc = 2;
  }
  if (o.dump_ops) printf("dumps: %llu descriptors into %s\n",
                         (unsigned long long)run.descriptors_dumped, o.dump_dir.c_str());

  if (!o.kv_save.empty()) {
    FILE* f = fopen(o.kv_save.c_str(), "wb");
    if (f != nullptr) {
      std::vector<uint8_t> buf(static_cast<size_t>(layout.kv_size));
      bytes.read(layout.kv_base, static_cast<uint32_t>(buf.size()), buf.data());
      fwrite(buf.data(), 1, buf.size(), f);
      fclose(f);
      printf("kv: wrote %llu bytes to %s\n", (unsigned long long)buf.size(), o.kv_save.c_str());
    }
  }

  write_perf_json(o, layout, m, run, ev, total, seconds, mode, marks, sim_cycles);
  printf("RESULT top=%s wb=%d clock_cycles=%llu seconds=%.3f mcps=%.3f mac=%.1f%% "
         "rd_bytes_per_cycle=%.2f\n",
         Build::top, Build::wb, (unsigned long long)sim_cycles, seconds,
         seconds > 0 ? sim_cycles / seconds / 1e6 : 0.0, 100.0 * total.mac_utilization(),
         total.bytes_per_cycle());
  m.close_trace();
  return rc;
}

}  // namespace
}  // namespace qcore

int main(int argc, char** argv) {
  try {
    return qcore::main_impl(argc, argv);
  } catch (const std::exception& e) {
    fprintf(stderr, "qcore_sim: %s\n", e.what());
    return 2;
  }
}
