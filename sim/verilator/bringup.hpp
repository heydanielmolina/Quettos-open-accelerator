// The bring-up run: a descriptor blob of the host's own, loaded into the
// copy-on-write image and stepped one descriptor at a time, with the VSRAM
// ranges, SREG banks, CSRs and PERF counters each descriptor leaves written out
// for sw/quettos/compare.py to check against sw/quettos/isa_sim.py.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cfg.hpp"
#include "csr.hpp"
#include "device.hpp"
#include "perf.hpp"

namespace qcore {
namespace {

std::vector<uint8_t> read_file(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> out;
  uint8_t buf[4096];
  size_t n = 0;
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.insert(out.end(), buf, buf + n);
  fclose(f);
  return out;
}

// One descriptor's worth of state, as sw/quettos/compare.py reads it back.
struct StepRecord {
  uint32_t index = 0;
  uint32_t pc = 0;
  uint32_t status = 0;
  uint32_t argmax_tok = 0, argmax_val = 0;
  Events events;
  PerfSnapshot perf;
  std::vector<std::vector<int32_t>> vsram;
  std::vector<std::vector<uint32_t>> sreg;
};

class Bringup {
 public:
  Bringup(Machine* m, const Options* o) : m_(m), o_(o) {}

  int run(const Layout& layout) {
    std::vector<uint8_t> blob = read_file(o_->program);
    if (blob.empty() || blob.size() % DESC_BYTES != 0) {
      throw std::runtime_error("--program is not a whole number of 32-byte descriptors");
    }
    if ((o_->program_addr % DESC_BYTES) != 0) {
      throw std::runtime_error("--program-addr is not a multiple of 32");
    }
    m_->bytes()->write(o_->program_addr, static_cast<uint32_t>(blob.size()), blob.data());
    for (const SregLoad& l : o_->sreg_loads) m_->load_sreg(l.bank, l.index, l.word);

    if (!o_->quiet) {
      printf("bringup: %zu descriptors at 0x%08llx, TOK=%u POS=%u ROW_EN=0x%x, %zu SREG word(s)\n",
             blob.size() / DESC_BYTES, (unsigned long long)o_->program_addr, o_->tok, o_->pos,
             o_->row_en, o_->sreg_loads.size());
      for (const SregLoad& l : o_->sreg_loads) {
        printf("bringup: SREG[%d][%u] = 0x%08x (host load: no unit in this build writes a scale)\n",
               l.bank, l.index, l.word);
      }
    }

    m_->write(CSR_STATUS, m_->read(CSR_STATUS));
    m_->write(CSR_PC, static_cast<uint32_t>(o_->program_addr));
    m_->write(CSR_TOK, o_->tok);
    m_->write(CSR_POS, o_->pos);
    m_->write(CSR_ROW_EN, o_->row_en);
    if (m_->read(CSR_PC) != static_cast<uint32_t>(o_->program_addr)) {
      throw std::runtime_error("PC did not take the host write");
    }

    auto t0 = std::chrono::steady_clock::now();
    uint64_t c0 = m_->cycles();
    int rc = o_->step ? step_all(blob.size() / DESC_BYTES) : free_run();
    seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    cycles_ = m_->cycles() - c0;
    write_records(layout);
    return rc;
  }

  const std::vector<StepRecord>& records() const { return records_; }
  uint64_t cycles() const { return cycles_; }
  double seconds() const { return seconds_; }

 private:
  // One CTRL.STEP per descriptor; PERF is snapshotted at every retire and no
  // counter is cleared, so a descriptor's contribution is the difference
  // between two consecutive records (docs/RTL.md 3.5, step mode).
  int step_all(size_t descriptors) {
    for (uint32_t i = 0; i < descriptors + 1; i++) {
      m_->write(CSR_CTRL, 1u << CTRL_STEP);
      if (!m_->run_until_halt()) return 2;
      Status s{m_->read(CSR_STATUS)};
      capture(i, s);
      if (s.err()) return 2;
      m_->write(CSR_STATUS, s.raw);
      if (s.done()) return 0;
    }
    m_->stop("the program did not reach HALT within its descriptor count");
    return 2;
  }

  int free_run() {
    m_->write(CSR_CTRL, 1u << CTRL_START);
    if (!m_->run_until_halt()) return 2;
    Status s{m_->read(CSR_STATUS)};
    capture(0, s);
    return s.err() ? 2 : 0;
  }

  void capture(uint32_t index, Status s) {
    StepRecord r;
    r.index = index;
    r.status = s.raw;
    r.pc = m_->read(CSR_PC);
    r.argmax_tok = m_->read(CSR_ARGMAX_TOK);
    r.argmax_val = m_->read(CSR_ARGMAX_VAL);
    r.events = read_events(*m_);
    for (uint32_t i = 0; i < PERF_COUNT; i++) r.perf[i] = m_->perf(i);
    for (const VsramRange& v : o_->dump_vsram) {
      std::vector<int32_t> vals(v.count);
      for (uint32_t e = 0; e < v.count; e++) {
        vals[e] = static_cast<int32_t>(m_->vsram(v.bank, v.start + e));
      }
      r.vsram.push_back(std::move(vals));
    }
    for (int b = 0; b < Build::b_max; b++) {
      std::vector<uint32_t> bank(SREG_COUNT);
      for (uint32_t i = 0; i < SREG_COUNT; i++) bank[i] = m_->read_sreg(b, i);
      r.sreg.push_back(std::move(bank));
    }
    records_.push_back(std::move(r));
  }

  void write_records(const Layout& layout) {
    size_t slash = o_->bringup_json.find_last_of('/');
    if (slash != std::string::npos) make_dir(o_->bringup_json.substr(0, slash));
    FILE* f = fopen(o_->bringup_json.c_str(), "w");
    if (f == nullptr) throw std::runtime_error("cannot write " + o_->bringup_json);
    JsonOut j(f);
    j.open('{');
    j.kv_s("format", "quettos-bringup");
    j.kv("isa_version", ISA_VERSION);
    j.kv_s("model", layout.model_name);
    j.key("build");
    j.open('{');
    j.kv_s("top", Build::top);
    j.kv("wb", Build::wb);
    j.kv("b_max", Build::b_max);
    j.kv("vsram_words", Build::vsram_words);
    j.kv("threads", Build::threads);
    j.close('}');
    j.key("run");
    j.open('{');
    j.kv("lat", o_->lat);
    j.kv("bw_div", o_->bw_div);
    j.kv_b("step", o_->step);
    j.kv("addr", o_->program_addr);
    j.kv("tok", o_->tok);
    j.kv("pos", o_->pos);
    j.kv("row_en", o_->row_en);
    j.kv("clock_cycles", cycles_);
    j.kv_d("wall_seconds", seconds_);
    j.kv_d("mcycles_per_s", seconds_ > 0 ? cycles_ / seconds_ / 1e6 : 0.0);
    j.close('}');
    // The regions a DUMP wrote, read once the run has ended: HALT auto-fences,
    // so every write the program issued has been acknowledged by then
    // (docs/RTL.md 4, auto-fence).
    j.key("mem");
    j.open('[');
    for (const MemRange& r : o_->dump_mem) {
      std::vector<uint8_t> buf(r.size);
      m_->bytes()->read(r.addr, r.size, buf.data());
      j.open('{');
      j.kv("addr", r.addr);
      j.kv("size", r.size);
      j.key("values");
      j.open('[');
      for (uint32_t i = 0; i + 4 <= r.size; i += 4) {
        int32_t v = 0;
        memcpy(&v, buf.data() + i, 4);
        j.snum(v);
      }
      j.close(']');
      j.close('}');
    }
    j.close(']');
    j.key("vsram_ranges");
    j.open('[');
    for (const VsramRange& v : o_->dump_vsram) {
      j.open('{');
      j.kv("bank", static_cast<uint64_t>(v.bank));
      j.kv("start", v.start);
      j.kv("count", v.count);
      j.close('}');
    }
    j.close(']');
    j.key("memory");
    j.open('{');
    j.kv("rd_requests", m_->qmem().rd_requests);
    j.kv("rd_beats", m_->qmem().rd_beats);
    j.kv("rd_bytes", m_->qmem().rd_bytes);
    j.kv("wr_beats", m_->qmem().wr_beats);
    j.kv("wr_bytes", m_->qmem().wr_bytes);
    j.close('}');
    j.key("records");
    j.open('[');
    for (const StepRecord& r : records_) {
      j.open('{');
      j.kv("index", r.index);
      j.kv("pc", r.pc);
      j.kv("status", r.status);
      j.kv("fault", Status{r.status}.fault());
      j.kv("fault_op", Status{r.status}.fault_op());
      j.kv("argmax_tok", r.argmax_tok);
      j.kv_i("argmax_val", static_cast<int32_t>(r.argmax_val));
      j.key("events");
      j.open('{');
      j.kv("SAT_REQ", r.events.sat_req);
      j.kv("SAT_VPU", r.events.sat_vpu);
      j.kv("ERR_SHIFT", r.events.err_shift);
      j.kv("ERR_BOUNDS", r.events.err_bounds);
      j.close('}');
      j.key("perf");
      write_json_perf(j, r.perf);
      j.key("vsram");
      j.open('[');
      for (const std::vector<int32_t>& vals : r.vsram) {
        j.open('[');
        for (int32_t v : vals) j.snum(v);
        j.close(']');
      }
      j.close(']');
      j.key("sreg");
      j.open('[');
      for (const std::vector<uint32_t>& bank : r.sreg) {
        j.open('[');
        for (uint32_t v : bank) j.num(v);
        j.close(']');
      }
      j.close(']');
      j.close('}');
    }
    j.close(']');
    j.close('}');
    fputc('\n', f);
    fclose(f);
  }

  Machine* m_;
  const Options* o_;
  std::vector<StepRecord> records_;
  uint64_t cycles_ = 0;
  double seconds_ = 0;
};

}  // namespace
}  // namespace qcore
