// The harness's CSR driver against the register file it drives. CsrBus performs
// every host operation on rtl/qcore_csr.sv and CsrFile performs the same
// operation on the C++ register table; every read is compared against both, so
// the RTL and the model of it stay one specification of STATUS, PC, the event
// counters and the PERF halves. Run it with `make harness-csr` (or
// `make -C sim/verilator csr-check`); it prints the check and failure counts
// and exits non-zero on any mismatch.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "Vqcore_csr.h"
#include "csr.hpp"
#include "verilated.h"

namespace {

using qcore::CsrBus;
using qcore::CsrFile;
using qcore::Status;

int failures = 0;
int checks = 0;

void expect(bool cond, const std::string& what) {
  checks++;
  if (!cond) {
    failures++;
    printf("FAIL %s\n", what.c_str());
  }
}

void expect_eq(uint32_t got, uint32_t want, const std::string& what) {
  checks++;
  if (got != want) {
    failures++;
    printf("FAIL %s: got 0x%08x, expected 0x%08x\n", what.c_str(), got, want);
  }
}

class Bench {
 public:
  explicit Bench(Vqcore_csr* dut) : dut_(dut), bus_(dut, [this] { tick(); }) {}

  void tick() {
    dut_->clk = 1;
    dut_->eval();
    dut_->clk = 0;
    dut_->eval();
    cycles_++;
  }

  void reset() {
    dut_->rst = 1;
    dut_->clk = 0;
    dut_->csr_we = 0;
    dut_->csr_re = 0;
    dut_->csr_addr = 0;
    dut_->csr_wdata = 0;
    dut_->pc_set = 0;
    dut_->pc_set_val = 0;
    dut_->busy_i = 0;
    dut_->done_set = 0;
    dut_->step_halted_set = 0;
    dut_->err_set = 0;
    dut_->fault_code = 0;
    dut_->fault_op = 0;
    dut_->argmax_we = 0;
    dut_->argmax_tok = 0;
    dut_->argmax_val = 0;
    dut_->sat_req_inc = 0;
    dut_->sat_vpu_inc = 0;
    dut_->err_shift_inc = 0;
    dut_->err_bounds_inc = 0;
    memset(&dut_->perf_snap, 0, sizeof(dut_->perf_snap));
    dut_->eval();
    for (int i = 0; i < 3; i++) tick();
    dut_->rst = 0;
    tick();
    file_.reset();
  }

  // Every host operation goes to the hardware and to the C++ register file.
  void write(uint32_t word, uint32_t value) {
    bus_.write(word, value);
    file_.write(word, value);
  }

  uint32_t read(uint32_t word, const std::string& what) {
    uint32_t hw = bus_.read(word);
    uint32_t sw = file_.read(word);
    expect_eq(hw, sw, what + ": " + qcore::csr_name(word) + " hardware against the C++ file");
    return hw;
  }

  void set_busy(bool v) {
    dut_->busy_i = v ? 1 : 0;
    file_.busy = v;
    dut_->eval();
  }

  Vqcore_csr* dut() { return dut_; }
  CsrFile& file() { return file_; }
  uint64_t cycles() const { return cycles_; }

 private:
  Vqcore_csr* dut_;
  CsrBus<Vqcore_csr> bus_;
  CsrFile file_;
  uint64_t cycles_ = 0;
};

void test_reset_and_rw(Bench& b) {
  b.reset();
  expect_eq(b.read(qcore::CSR_PC, "reset"), 0, "PC after reset");
  expect_eq(b.read(qcore::CSR_STATUS, "reset"), 0, "STATUS after reset");
  expect_eq(b.read(qcore::CSR_ISA_VERSION, "reset"), qcore::ISA_VERSION, "ISA_VERSION");
  expect_eq(b.read(13, "reset"), 0, "unmapped word 13");
  expect_eq(b.read(63, "reset"), 0, "unmapped word 63");

  b.write(qcore::CSR_PC, 0x0000'0640);
  b.write(qcore::CSR_ROW_EN, 0x3);
  b.write(qcore::CSR_TOK, 42);
  b.write(qcore::CSR_POS, 7);
  expect_eq(b.read(qcore::CSR_PC, "rw"), 0x0640, "PC round trip");
  expect_eq(b.read(qcore::CSR_ROW_EN, "rw"), 3, "ROW_EN round trip");
  expect_eq(b.read(qcore::CSR_TOK, "rw"), 42, "TOK round trip");
  expect_eq(b.read(qcore::CSR_POS, "rw"), 7, "POS round trip");
  expect_eq(b.dut()->pc_q, 0x0640u, "pc_q port");
  expect_eq(b.dut()->row_en_q, 3u, "row_en_q port");
  expect_eq(b.dut()->tok_q, 42u, "tok_q port");
  expect_eq(b.dut()->pos_q, 7u, "pos_q port");

  // The four rw registers belong to the core while it runs.
  b.set_busy(true);
  b.write(qcore::CSR_PC, 0xDEAD'0000);
  b.write(qcore::CSR_TOK, 99);
  expect_eq(b.read(qcore::CSR_PC, "busy"), 0x0640, "PC ignores a write while BUSY");
  expect_eq(b.read(qcore::CSR_TOK, "busy"), 42, "TOK ignores a write while BUSY");
  b.set_busy(false);
}

void test_ctrl_pulses(Bench& b) {
  b.reset();
  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_START);
  expect(b.dut()->start == 1 && b.dut()->step == 0 && b.dut()->abort_run == 0, "START pulses alone");
  b.tick();
  expect(b.dut()->start == 0, "START is one cycle wide");

  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_STEP);
  expect(b.dut()->step == 1 && b.dut()->start == 0, "STEP pulses alone");
  b.tick();

  b.write(qcore::CSR_CTRL, (1u << qcore::CTRL_START) | (1u << qcore::CTRL_STEP));
  expect(b.dut()->start == 1 && b.dut()->step == 0, "START wins over STEP in one write");
  b.tick();

  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_ABORT);
  expect(b.dut()->abort_run == 0, "ABORT is ignored while the core is idle");
  b.tick();

  b.set_busy(true);
  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_START);
  expect(b.dut()->start == 0, "START is ignored while the core is busy");
  b.tick();
  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_ABORT);
  expect(b.dut()->abort_run == 1, "ABORT pulses while the core is busy");
  b.tick();
  b.set_busy(false);
  expect_eq(b.read(qcore::CSR_CTRL, "ctrl"), 0, "CTRL reads zero");
}

void test_status(Bench& b) {
  b.reset();
  b.dut()->done_set = 1;
  b.tick();
  b.dut()->done_set = 0;
  b.file().done = true;
  Status s{b.read(qcore::CSR_STATUS, "status")};
  expect(s.done() && !s.err(), "DONE is sticky");

  b.dut()->err_set = 1;
  b.dut()->fault_code = qcore::FAULT_ROW;
  b.dut()->fault_op = qcore::OP_GEMV;
  b.tick();
  b.dut()->err_set = 0;
  b.file().set_fault(qcore::FAULT_ROW, qcore::OP_GEMV);
  s = Status{b.read(qcore::CSR_STATUS, "fault")};
  expect(s.err() && s.fault() == qcore::FAULT_ROW && s.fault_op() == qcore::OP_GEMV,
         "ERR carries FAULT and FAULT_OP");

  b.write(qcore::CSR_STATUS, 1u << qcore::STATUS_ERR);
  s = Status{b.read(qcore::CSR_STATUS, "clear err")};
  expect(!s.err() && s.fault() == qcore::FAULT_NONE && s.fault_op() == 0,
         "writing one to ERR clears the fault fields");
  expect(s.done(), "clearing ERR leaves DONE set");
  b.write(qcore::CSR_STATUS, 1u << qcore::STATUS_DONE);
  s = Status{b.read(qcore::CSR_STATUS, "clear done")};
  expect(!s.done(), "writing one to DONE clears it");

  // BUSY is the live input, not a stored bit.
  b.set_busy(true);
  s = Status{b.read(qcore::CSR_STATUS, "busy bit")};
  expect(s.busy(), "BUSY follows busy_i");
  b.set_busy(false);

  // START clears the sticky bits.
  b.dut()->step_halted_set = 1;
  b.tick();
  b.dut()->step_halted_set = 0;
  b.file().step_halted = true;
  expect(Status{b.read(qcore::CSR_STATUS, "step halted")}.step_halted(), "STEP_HALTED is sticky");
  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_START);
  b.file().clear_status();
  b.file().take_start();
  b.tick();
  expect_eq(b.read(qcore::CSR_STATUS, "after start"), 0, "START clears the sticky bits");
}

void test_counters_and_argmax(Bench& b) {
  b.reset();
  b.dut()->argmax_we = 1;
  b.dut()->argmax_tok = 1234;
  b.dut()->argmax_val = 0x0010'0BAD;
  b.tick();
  b.dut()->argmax_we = 0;
  b.file().argmax_tok = 1234;
  b.file().argmax_val = 0x0010'0BAD;
  expect_eq(b.read(qcore::CSR_ARGMAX_TOK, "argmax"), 1234, "ARGMAX_TOK");
  expect_eq(b.read(qcore::CSR_ARGMAX_VAL, "argmax"), 0x0010'0BAD, "ARGMAX_VAL");

  const int n = 5;
  b.dut()->sat_req_inc = 3;
  b.dut()->sat_vpu_inc = 1;
  b.dut()->err_shift_inc = 2;
  b.dut()->err_bounds_inc = 255;
  for (int i = 0; i < n; i++) b.tick();
  b.dut()->sat_req_inc = 0;
  b.dut()->sat_vpu_inc = 0;
  b.dut()->err_shift_inc = 0;
  b.dut()->err_bounds_inc = 0;
  b.file().add_events(3 * n, 1 * n, 2 * n, 255 * n);
  expect_eq(b.read(qcore::CSR_SAT_REQ, "counters"), 3 * n, "SAT_REQ adds its per-cycle count");
  expect_eq(b.read(qcore::CSR_SAT_VPU, "counters"), 1 * n, "SAT_VPU adds its per-cycle count");
  expect_eq(b.read(qcore::CSR_ERR_SHIFT, "counters"), 2 * n, "ERR_SHIFT adds its per-cycle count");
  expect_eq(b.read(qcore::CSR_ERR_BOUNDS, "counters"), 255 * n, "ERR_BOUNDS adds its per-cycle count");

  b.write(qcore::CSR_CTRL, 1u << qcore::CTRL_START);
  b.file().clear_counters();
  b.file().clear_status();
  b.file().take_start();
  b.tick();
  expect_eq(b.read(qcore::CSR_SAT_REQ, "after start"), 0, "START clears SAT_REQ");
  expect_eq(b.read(qcore::CSR_ARGMAX_TOK, "after start"), 1234, "START keeps ARGMAX_TOK");
}

void test_perf_halves(Bench& b) {
  b.reset();
  uint8_t snap[128] = {};
  for (uint32_t i = 0; i < qcore::PERF_COUNT; i++) {
    uint64_t v = 0x1000'0000'0000ull * (i + 1) + i;
    memcpy(snap + 8 * i, &v, 8);
    b.file().perf_snap[i] = v;
  }
  memcpy(&b.dut()->perf_snap, snap, sizeof(snap));
  b.dut()->eval();
  for (uint32_t i = 0; i < qcore::PERF_COUNT; i++) {
    uint64_t v = 0x1000'0000'0000ull * (i + 1) + i;
    expect_eq(b.read(qcore::PERF_BASE + 2 * i, "perf"), static_cast<uint32_t>(v),
              std::string("PERF") + std::to_string(i) + "_LO");
    expect_eq(b.read(qcore::PERF_BASE + 2 * i + 1, "perf"), static_cast<uint32_t>(v >> 32),
              std::string("PERF") + std::to_string(i) + "_HI");
  }
}

void test_pc_ownership(Bench& b) {
  b.reset();
  b.write(qcore::CSR_PC, 0x1000);
  b.set_busy(true);
  b.dut()->pc_set = 1;
  b.dut()->pc_set_val = 0x1020;
  b.tick();
  b.dut()->pc_set = 0;
  b.file().pc = 0x1020;
  expect_eq(b.read(qcore::CSR_PC, "pc_set"), 0x1020, "the core advances PC while it runs");
  b.set_busy(false);
}

}  // namespace

int main(int argc, char** argv) {
  VerilatedContext ctx;
  ctx.commandArgs(argc, argv);
  Vqcore_csr dut(&ctx);
  Bench b(&dut);

  test_reset_and_rw(b);
  test_ctrl_pulses(b);
  test_status(b);
  test_counters_and_argmax(b);
  test_perf_halves(b);
  test_pc_ownership(b);

  dut.final();
  printf("csr-check: %d checks, %d failures (%llu cycles)\n", checks, failures,
         static_cast<unsigned long long>(b.cycles()));
  return failures == 0 ? 0 : 1;
}
