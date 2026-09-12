// The host's view of the machine: the 64-word CSR window, the STATUS decode
// with its fault fields, the 32-byte descriptor, and two implementations of the
// CsrAccess calls the rest of the harness makes -- CsrBus, which drives a DUT's
// csr_* port with the one-cycle read latency of docs/RTL.md 3.3, and CsrFile,
// the same register semantics in C++. device.hpp drives qcore_top through
// CsrBus; csr_check.cpp runs both against rtl/qcore_csr.sv and compares every
// read, which is what keeps them one specification. Every name and bit position
// comes from the generated csr_defs.hpp.
#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "csr_defs.hpp"

namespace qcore {

inline const char* opcode_name(uint32_t op) {
  switch (op) {
    case OP_NOP: return "NOP";
    case OP_HALT: return "HALT";
    case OP_GEMV: return "GEMV";
    case OP_EMBED: return "EMBED";
    case OP_VRMSNORM: return "VRMSNORM";
    case OP_VQUANT: return "VQUANT";
    case OP_VROPE: return "VROPE";
    case OP_VSILUMUL: return "VSILUMUL";
    case OP_VSOFTMAX: return "VSOFTMAX";
    case OP_VSUBC: return "VSUBC";
    case OP_KVWRITE: return "KVWRITE";
    case OP_FENCE: return "FENCE";
    default: return "?";
  }
}

inline bool is_opcode(uint32_t op) { return opcode_name(op)[0] != '?'; }

inline const char* fault_name(uint32_t code) {
  switch (code) {
    case FAULT_NONE: return "NONE";
    case FAULT_OPCODE: return "OPCODE";
    case FAULT_ROW: return "ROW";
    case FAULT_PC_ALIGN: return "PC_ALIGN";
    default: return "?";
  }
}

// The named registers of the CSR window, in word order (docs/RTL.md 3.3). The
// PERF halves are reached by index instead, through CsrAccess::perf.
struct CsrName {
  const char* name;
  uint32_t word;
};

inline const CsrName* csr_names(size_t* count) {
  static const CsrName kNames[] = {
      {"CTRL", CSR_CTRL},           {"STATUS", CSR_STATUS},
      {"PC", CSR_PC},               {"ROW_EN", CSR_ROW_EN},
      {"TOK", CSR_TOK},             {"POS", CSR_POS},
      {"ARGMAX_TOK", CSR_ARGMAX_TOK}, {"ARGMAX_VAL", CSR_ARGMAX_VAL},
      {"SAT_REQ", CSR_SAT_REQ},     {"SAT_VPU", CSR_SAT_VPU},
      {"ERR_SHIFT", CSR_ERR_SHIFT}, {"ERR_BOUNDS", CSR_ERR_BOUNDS},
      {"ISA_VERSION", CSR_ISA_VERSION}};
  *count = sizeof(kNames) / sizeof(kNames[0]);
  return kNames;
}

// The word a name addresses, or false when it is none of them. A caller that
// takes a register name from a file uses this and fails on false: a name the
// table does not carry is a file the harness cannot honour, not a register to
// guess at.
inline bool csr_word_of(const std::string& name, uint32_t* word) {
  size_t count = 0;
  const CsrName* names = csr_names(&count);
  for (size_t i = 0; i < count; i++) {
    if (name == names[i].name) {
      *word = names[i].word;
      return true;
    }
  }
  return false;
}

// Every name the table carries, for an error message that says what is allowed.
inline std::string csr_name_list() {
  size_t count = 0;
  const CsrName* names = csr_names(&count);
  std::string out;
  for (size_t i = 0; i < count; i++) {
    if (i != 0) out += ", ";
    out += names[i].name;
  }
  return out;
}

inline const char* csr_name(uint32_t word) {
  switch (word) {
    case CSR_CTRL: return "CTRL";
    case CSR_STATUS: return "STATUS";
    case CSR_PC: return "PC";
    case CSR_ROW_EN: return "ROW_EN";
    case CSR_TOK: return "TOK";
    case CSR_POS: return "POS";
    case CSR_ARGMAX_TOK: return "ARGMAX_TOK";
    case CSR_ARGMAX_VAL: return "ARGMAX_VAL";
    case CSR_SAT_REQ: return "SAT_REQ";
    case CSR_SAT_VPU: return "SAT_VPU";
    case CSR_ERR_SHIFT: return "ERR_SHIFT";
    case CSR_ERR_BOUNDS: return "ERR_BOUNDS";
    case CSR_ISA_VERSION: return "ISA_VERSION";
    default: return word >= PERF_BASE && word < PERF_BASE + 2 * PERF_COUNT ? "PERF" : "";
  }
}

// STATUS as the host reads it.
struct Status {
  uint32_t raw = 0;

  bool done() const { return (raw >> STATUS_DONE) & 1u; }
  bool busy() const { return (raw >> STATUS_BUSY) & 1u; }
  bool step_halted() const { return (raw >> STATUS_STEP_HALTED) & 1u; }
  bool err() const { return (raw >> STATUS_ERR) & 1u; }
  uint32_t fault() const { return (raw >> STATUS_FAULT_LSB) & ((1u << STATUS_FAULT_W) - 1u); }
  uint32_t fault_op() const { return (raw >> STATUS_FAULT_OP_LSB) & ((1u << STATUS_FAULT_OP_W) - 1u); }
  bool halted() const { return done() || step_halted() || err(); }

  std::string text() const {
    char buf[128];
    snprintf(buf, sizeof(buf), "STATUS=0x%08x [%s%s%s%s] FAULT=%s FAULT_OP=0x%02x", raw,
             done() ? "DONE " : "", busy() ? "BUSY " : "", step_halted() ? "STEP_HALTED " : "",
             err() ? "ERR" : "", fault_name(fault()), fault_op());
    return std::string(buf);
  }
};

// One 32-byte descriptor, read through the generated field positions.
struct Descriptor {
  uint8_t b[DESC_BYTES] = {};

  uint32_t field(uint32_t lsb, uint32_t w) const {
    uint64_t acc = 0;
    for (int i = 7; i >= 0; i--) {
      uint32_t byte = (lsb / 8) + static_cast<uint32_t>(i);
      acc = (acc << 8) | (byte < DESC_BYTES ? b[byte] : 0u);
    }
    acc >>= (lsb % 8);
    return w >= 32 ? static_cast<uint32_t>(acc) : static_cast<uint32_t>(acc & ((1ull << w) - 1));
  }

  uint32_t opcode() const { return field(DESC_OPCODE_LSB, DESC_OPCODE_W); }
  uint32_t flags() const { return field(DESC_FLAGS_LSB, DESC_FLAGS_W); }
  uint32_t row_mask() const { return field(DESC_ROW_MASK_LSB, DESC_ROW_MASK_W); }
  bool accumulate() const { return field(DESC_ACCUMULATE_LSB, DESC_ACCUMULATE_W) != 0; }
  bool unit_meta() const { return field(DESC_UNIT_META_LSB, DESC_UNIT_META_W) != 0; }
  bool n_from_pos() const { return field(DESC_N_FROM_POS_LSB, DESC_N_FROM_POS_W) != 0; }
  bool k_from_pos() const { return field(DESC_K_FROM_POS_LSB, DESC_K_FROM_POS_W) != 0; }
  uint32_t out_mode() const { return field(DESC_OUT_MODE_LSB, DESC_OUT_MODE_W); }
  bool len_from_pos() const { return field(DESC_LEN_FROM_POS_LSB, DESC_LEN_FROM_POS_W) != 0; }
  bool track_absmax() const { return field(DESC_TRACK_ABSMAX_LSB, DESC_TRACK_ABSMAX_W) != 0; }
  uint32_t addr_a() const { return field(DESC_ADDR_A_LSB, DESC_ADDR_A_W); }
  uint32_t addr_m() const { return field(DESC_ADDR_M_LSB, DESC_ADDR_M_W); }
  uint32_t n() const { return field(DESC_N_LSB, DESC_N_W); }
  uint32_t k() const { return field(DESC_K_LSB, DESC_K_W); }
  uint32_t vs_src() const { return field(DESC_VS_SRC_LSB, DESC_VS_SRC_W); }
  uint32_t vs_dst() const { return field(DESC_VS_DST_LSB, DESC_VS_DST_W); }
  uint32_t vs_aux() const { return field(DESC_VS_AUX_LSB, DESC_VS_AUX_W); }
  uint32_t sreg_src() const { return field(DESC_SREG_SRC_LSB, DESC_SREG_SRC_W); }
  uint32_t sreg_dst() const { return field(DESC_SREG_DST_LSB, DESC_SREG_DST_W); }
  uint32_t src_row() const { return field(DESC_SRC_ROW_LSB, DESC_SRC_ROW_W); }
  uint32_t dst_row() const { return field(DESC_DST_ROW_LSB, DESC_DST_ROW_W); }
  uint32_t sh0() const { return field(DESC_SH0_LSB, DESC_SH0_W); }
  int32_t sh1() const { return static_cast<int8_t>(field(DESC_SH1_LSB, DESC_SH1_W)); }
  uint32_t imm32() const { return field(DESC_IMM32_LSB, DESC_IMM32_W); }

  bool reads_qmem() const {
    switch (opcode()) {
      case OP_GEMV: case OP_EMBED: case OP_VRMSNORM:
      case OP_VROPE: case OP_VSOFTMAX: case OP_VSUBC:
        return true;
      default:
        return false;
    }
  }
};

// What the harness does to the machine, however the registers are reached.
class CsrAccess {
 public:
  virtual ~CsrAccess() = default;
  virtual uint32_t read(uint32_t word) = 0;
  virtual void write(uint32_t word, uint32_t value) = 0;

  Status status() { return Status{read(CSR_STATUS)}; }
  uint64_t perf(uint32_t index) {
    uint64_t lo = read(PERF_BASE + 2 * index);
    uint64_t hi = read(PERF_BASE + 2 * index + 1);
    return lo | (hi << 32);
  }
};

// The four event counters as one read.
struct Events {
  uint32_t sat_req = 0, sat_vpu = 0, err_shift = 0, err_bounds = 0;
  uint32_t total() const { return sat_req + sat_vpu + err_shift + err_bounds; }

  Events plus(const Events& other) const {
    return Events{sat_req + other.sat_req, sat_vpu + other.sat_vpu,
                  err_shift + other.err_shift, err_bounds + other.err_bounds};
  }
};

inline Events read_events(CsrAccess& m) {
  Events e;
  e.sat_req = m.read(CSR_SAT_REQ);
  e.sat_vpu = m.read(CSR_SAT_VPU);
  e.err_shift = m.read(CSR_ERR_SHIFT);
  e.err_bounds = m.read(CSR_ERR_BOUNDS);
  return e;
}

// The register file of docs/RTL.md 3.3 in C++: the access classes, the sticky
// status bits with their fault fields, the four wrapping event counters and the
// PERF halves. rtl/qcore_csr.sv implements the same table in hardware.
class CsrFile : public CsrAccess {
 public:
  uint32_t read(uint32_t word) override {
    if (word >= PERF_BASE && word < PERF_BASE + 2 * PERF_COUNT) {
      uint32_t i = word - PERF_BASE;
      return static_cast<uint32_t>(perf_snap[i / 2] >> (32 * (i % 2)));
    }
    switch (word) {
      case CSR_CTRL: return 0;
      case CSR_STATUS: return status_word();
      case CSR_PC: return pc;
      case CSR_ROW_EN: return row_en;
      case CSR_TOK: return tok;
      case CSR_POS: return pos;
      case CSR_ARGMAX_TOK: return argmax_tok;
      case CSR_ARGMAX_VAL: return argmax_val;
      case CSR_SAT_REQ: return sat_req;
      case CSR_SAT_VPU: return sat_vpu;
      case CSR_ERR_SHIFT: return err_shift;
      case CSR_ERR_BOUNDS: return err_bounds;
      case CSR_ISA_VERSION: return ISA_VERSION;
      default: return 0;
    }
  }

  void write(uint32_t word, uint32_t value) override {
    if (word == CSR_CTRL) {
      if (busy && ((value >> CTRL_ABORT) & 1u)) req_abort = true;
      if (!busy && ((value >> CTRL_START) & 1u)) req_start = true;
      else if (!busy && ((value >> CTRL_STEP) & 1u)) req_step = true;
      return;
    }
    if (word == CSR_STATUS) {
      if ((value >> STATUS_DONE) & 1u) done = false;
      if ((value >> STATUS_STEP_HALTED) & 1u) step_halted = false;
      if ((value >> STATUS_ERR) & 1u) {
        err = false;
        fault = FAULT_NONE;
        fault_op = 0;
      }
      return;
    }
    if (busy) return;  // the rw registers belong to the core while it runs
    switch (word) {
      case CSR_PC: pc = value; break;
      case CSR_ROW_EN: row_en = value; break;
      case CSR_TOK: tok = value; break;
      case CSR_POS: pos = value; break;
      default: break;
    }
  }

  uint32_t status_word() const {
    uint32_t v = 0;
    v |= static_cast<uint32_t>(done) << STATUS_DONE;
    v |= static_cast<uint32_t>(busy) << STATUS_BUSY;
    v |= static_cast<uint32_t>(step_halted) << STATUS_STEP_HALTED;
    v |= static_cast<uint32_t>(err) << STATUS_ERR;
    v |= (fault & ((1u << STATUS_FAULT_W) - 1u)) << STATUS_FAULT_LSB;
    v |= (fault_op & ((1u << STATUS_FAULT_OP_W) - 1u)) << STATUS_FAULT_OP_LSB;
    return v;
  }

  // --- the core's side
  bool take_start() { bool v = req_start; req_start = false; return v; }
  bool take_step() { bool v = req_step; req_step = false; return v; }
  bool take_abort() { bool v = req_abort; req_abort = false; return v; }

  void clear_status() {
    done = step_halted = err = false;
    fault = FAULT_NONE;
    fault_op = 0;
  }
  void clear_counters() { sat_req = sat_vpu = err_shift = err_bounds = 0; }
  void set_fault(uint32_t code, uint32_t op) {
    err = true;
    fault = code;
    fault_op = op;
  }
  void add_events(uint32_t sat_r, uint32_t sat_v, uint32_t shift, uint32_t bounds) {
    sat_req += sat_r;
    sat_vpu += sat_v;
    err_shift += shift;
    err_bounds += bounds;
  }
  void reset() {
    *this = CsrFile();
  }

  uint32_t pc = 0, row_en = 0, tok = 0, pos = 0;
  uint32_t argmax_tok = 0, argmax_val = 0;
  uint32_t sat_req = 0, sat_vpu = 0, err_shift = 0, err_bounds = 0;
  bool busy = false, done = false, step_halted = false, err = false;
  uint32_t fault = FAULT_NONE, fault_op = 0;
  uint64_t perf_snap[PERF_COUNT] = {};

 private:
  bool req_start = false, req_step = false, req_abort = false;
};

// The host port itself: one write per cycle, a read whose data arrives the
// cycle after csr_re. `tick` advances the model by one clock.
template <class Dut>
class CsrBus : public CsrAccess {
 public:
  CsrBus(Dut* dut, std::function<void()> tick) : dut_(dut), tick_(std::move(tick)) {}

  void idle() {
    dut_->csr_we = 0;
    dut_->csr_re = 0;
    dut_->csr_addr = 0;
    dut_->csr_wdata = 0;
  }

  void write(uint32_t word, uint32_t value) override {
    dut_->csr_we = 1;
    dut_->csr_addr = word & (CSR_WORDS - 1);
    dut_->csr_wdata = value;
    tick_();
    dut_->csr_we = 0;
    dut_->csr_wdata = 0;
  }

  uint32_t read(uint32_t word) override {
    dut_->csr_re = 1;
    dut_->csr_addr = word & (CSR_WORDS - 1);
    tick_();  // csr_rdata carries the addressed word one cycle after csr_re
    dut_->csr_re = 0;
    return static_cast<uint32_t>(dut_->csr_rdata);
  }

 private:
  Dut* dut_;
  std::function<void()> tick_;
};

}  // namespace qcore
