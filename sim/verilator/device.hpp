// The device the harness drives: rtl/qcore_top.sv over its two ports -- the
// QMEM bus (docs/RTL.md 2.1) with the bus model of mem_model.hpp on it, and
// the host CSR window (3.3) reached through CsrBus. Everything the host used
// to stand in for -- descriptor fetch, dispatch, the register file and the
// PERF counters -- is inside the model now, so this file is the clock, the two
// ports, and the two backdoors a bring-up run needs: the VSRAM words a
// descriptor wrote and the SREG scale a GEMV reads.
// Machine::tick is the clock: one rising edge, then the inputs the next edge
// samples.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "Vqcore_top.h"
#include "Vqcore_top___024root.h"
#include "verilated.h"
#ifdef QCORE_TRACE
#include "verilated_vcd_c.h"
#endif

#include "cfg.hpp"
#include "csr.hpp"
#include "mem_model.hpp"
#include "perf.hpp"

namespace qcore {
namespace {

using Dut = Vqcore_top;

constexpr int WB = Build::wb;
constexpr int B_MAX = Build::b_max;

// The VSRAM and the SREG bank of row r, by their flattened names. qcore_top
// names the generate block g_row and the instances u_vsram and u_row.
#define QCORE_VSRAM_MEM(r) \
  (dut->rootp->qcore_top__DOT__g_row__BRA__##r##__KET____DOT__u_vsram__DOT__mem)
#define QCORE_SREG_MEM(r) \
  (dut->rootp->qcore_top__DOT__g_row__BRA__##r##__KET____DOT__u_row__DOT__sreg)

uint32_t vsram_elem(Dut* dut, int bank, uint32_t elem) {
  uint32_t w = elem / VSRAM_WORD_ELEMS;
  uint32_t slot = elem % VSRAM_WORD_ELEMS;
  if (w >= static_cast<uint32_t>(Build::vsram_words)) return 0;
  switch (bank) {
    case 0: return QCORE_VSRAM_MEM(0)[w][slot];
#if QCORE_B_MAX > 1
    case 1: return QCORE_VSRAM_MEM(1)[w][slot];
#endif
    default: return 0;
  }
}

uint32_t sreg_word(Dut* dut, int bank, uint32_t index) {
  if (index >= SREG_COUNT) return 0;
  switch (bank) {
    case 0: return QCORE_SREG_MEM(0)[index];
#if QCORE_B_MAX > 1
    case 1: return QCORE_SREG_MEM(1)[index];
#endif
    default: return 0;
  }
}

void set_sreg_word(Dut* dut, int bank, uint32_t index, uint32_t value) {
  if (index >= SREG_COUNT) return;
  switch (bank) {
    case 0: QCORE_SREG_MEM(0)[index] = value; break;
#if QCORE_B_MAX > 1
    case 1: QCORE_SREG_MEM(1)[index] = value; break;
#endif
    default: break;
  }
}

void make_dir(const std::string& path) {
  std::string acc;
  size_t pos = 0;
  while (pos <= path.size()) {
    size_t slash = path.find('/', pos);
    acc = path.substr(0, slash == std::string::npos ? path.size() : slash);
    if (!acc.empty()) mkdir(acc.c_str(), 0777);
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
}

// The machine: the model, the memory behind it, and the host port.
class Machine : public CsrAccess {
 public:
  Machine(Dut* dut, MemBytes* bytes, const Options& opt)
      : dut_(dut),
        bytes_(bytes),
        opt_(opt),
        qmem_(dut, bytes, opt.lat, opt.bw_div),
        bus_(dut, [this]() { tick(); }) {}

  // --- host side (CsrAccess): one clock per operation, as the port defines it
  uint32_t read(uint32_t word) override { return bus_.read(word); }
  void write(uint32_t word, uint32_t value) override { bus_.write(word, value); }

  uint64_t cycles() const { return cycle_; }
  const Qmem<Dut, WB>& qmem() const { return qmem_; }
  const std::string& stop_reason() const { return stop_reason_; }
  void stop(const std::string& why) { stop_reason_ = why; }

  void reset() {
    dut_->rst = 1;
    dut_->clk = 0;
    bus_.idle();
    qmem_.reset_ports();
    dut_->eval();
    for (int i = 0; i < 4; i++) tick();
    dut_->rst = 0;
    tick();
  }

  // One clock: the rising edge, then the inputs the next edge will sample.
  // Traces carry two samples per cycle so the clock is visible in the VCD.
  void tick() {
    dut_->clk = 1;
    dut_->eval();
#ifdef QCORE_TRACE
    if (trace_ != nullptr) trace_->dump(2 * cycle_);
#endif
    dut_->clk = 0;
    dut_->eval();
    cycle_++;
    qmem_.drive(cycle_);
    dut_->eval();
    qmem_.sample();
#ifdef QCORE_TRACE
    if (trace_ != nullptr) {
      trace_->dump(2 * cycle_ - 1);
      // The trace is bounded in cycles, so a run of any length writes a file of
      // a known size: the window closes here and the run carries on.
      if (trace_limit_ != 0 && cycle_ >= trace_limit_) close_trace();
    }
#endif
  }

#ifdef QCORE_TRACE
  // ``limit`` cycles of VCD, then the file is closed and the run goes on; 0 is
  // the whole run.
  void open_trace(const std::string& path, VerilatedContext* ctx, uint64_t limit) {
    ctx->traceEverOn(true);
    trace_ = new VerilatedVcdC();
    dut_->trace(trace_, 8);
    trace_->open(path.c_str());
    trace_limit_ = limit;
    traced_ = 0;
  }
  void close_trace() {
    if (trace_ != nullptr) {
      trace_->close();
      delete trace_;
      trace_ = nullptr;
      traced_ = cycle_;
    }
  }
#else
  void close_trace() {}
#endif

  // How many cycles the VCD carries: the whole run, or the bound it stopped at.
  uint64_t traced_cycles() const { return traced_; }

  // Polls STATUS until the core halts, as a host does; false on a stop.
  bool run_until_halt() {
    for (;;) {
      Status s{read(CSR_STATUS)};
      if (s.halted() && !s.busy()) return true;
      if (opt_.max_cycles != 0 && cycle_ > opt_.max_cycles) {
        stop_reason_ = "cycle budget spent";
        return false;
      }
    }
  }

  // Zero-cycle views of the row memories: the VSRAM words a descriptor wrote
  // (docs/RTL.md 3.11) and the SREG bank (2.7).
  uint32_t vsram(int bank, uint32_t elem) { return vsram_elem(dut_, bank, elem); }
  uint32_t read_sreg(int bank, uint32_t index) { return sreg_word(dut_, bank, index); }
  void load_sreg(int bank, uint32_t index, uint32_t value) {
    set_sreg_word(dut_, bank, index, value);
  }

  MemBytes* bytes() { return bytes_; }

 private:
  Dut* dut_;
  MemBytes* bytes_;
  const Options& opt_;
  Qmem<Dut, WB> qmem_;
  CsrBus<Dut> bus_;
  uint64_t cycle_ = 0;
  uint64_t traced_ = 0;
  std::string stop_reason_;
#ifdef QCORE_TRACE
  VerilatedVcdC* trace_ = nullptr;
  uint64_t trace_limit_ = 0;
#endif
};

}  // namespace
}  // namespace qcore
