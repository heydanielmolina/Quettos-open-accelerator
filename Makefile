# Quettos Core -- top-level Makefile. Everything runs from the repo root.
# Every target runs except demo, demo-toolcall, demo-qwen and regen-prefix,
# which generate text and so need the VROPE and VSOFTMAX passes of
# qcore_vpu_top (see docs/ROADMAP.md).

UV       := uv
VERILATOR := verilator
YOSYS    := yosys
IVERILOG := iverilog

MODEL    ?= qwen
TOPS     ?=

# The compiled model the Verilator harness runs and the flags it runs with
# (sim/verilator, docs/PERFORMANCE.md). HARNESS_CFG selects the RTL widths.
IMAGE       ?= build/images/qwen2.5-0.5b-instruct-l2
HARNESS_CFG ?= WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096
PERF_ARGS   ?= --traffic --max-new 1

.PHONY: all help demo demo-toolcall demo-qwen regen-prefix ci test lint style cocotb synth gatesim perf waves probe harness harness-csr bringup bringup-sweep clean

all: help

help:
	@echo "Quettos Core targets:"
	@echo "  make lint          three-parser lint over rtl/*.sv and sim/cocotb/wrappers/*.sv (scripts/lint.sh)"
	@echo "  make style         ruff check and ruff format --check over every .py in the repository"
	@echo "  make test          uv run pytest -q sw/tests"
	@echo "  make cocotb        cocotb block tests on the tiny RTL configuration (sim/cocotb)"
	@echo "  make probe         Verilator speed probe (sim/probe)"
	@echo "  make harness       build the Verilator harness (sim/verilator)"
	@echo "  make harness-csr   run the harness CSR driver against rtl/qcore_csr.sv"
	@echo "  make bringup       RTL vs isa_sim on the tiny configuration, both programs (sw/quettos/compare.py)"
	@echo "  make bringup-sweep the same over random shapes at WB=64 and WB=128"
	@echo "  make clean         remove build/ and Verilator obj_dir directories"
	@echo "  make demo          Qwen 32+20 end to end; needs the VROPE and VSOFTMAX passes"
	@echo "  make demo-toolcall tool-call demo with prefix-KV reuse; needs the same two passes"
	@echo "  make demo-qwen     the recorded Qwen demo; needs the same two passes"
	@echo "  make regen-prefix  regenerate the 512-token demo prefix; needs the same two passes"
	@echo "  make ci            the CI job set, run locally"
	@echo "  make synth         Yosys synth_xilinx of every syn/synth_*.ys script; logs in build/synth/, and every syn/reports/*.md has to still reproduce"
	@echo "  make gatesim       gate-level equivalence: each Yosys netlist against the source it came from (sim/gatesim)"
	@echo "  make perf          run the harness on IMAGE and write build/perf/perf.json"
	@echo "  make waves         a VCD of a harness run into build/waves (rebuilds with tracing)"

lint:
	@TOPS="$(TOPS)" VERILATOR=$(VERILATOR) YOSYS=$(YOSYS) IVERILOG=$(IVERILOG) bash scripts/lint.sh

# The Python side of the lint: the rules and the line length live in the
# [tool.ruff] tables of pyproject.toml, and the version in uv.lock. Both
# commands are read-only; `uv run ruff format .` rewrites what the second one
# reports. Every .py in the repository is in scope -- the package, the tests,
# the benches and the scripts.
style:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

test:
	$(UV) run pytest -q sw/tests

cocotb:
	$(UV) run pytest -q sim/cocotb -x

probe:
	@test -d sim/probe || { echo "make probe: sim/probe not present"; exit 1; }
	$(MAKE) -C sim/probe

clean:
	rm -rf build
	find . -type d -name 'obj_dir*' -prune -exec rm -rf {} +

demo:
	@echo "make demo: the Qwen 32+20 run executes VROPE and VSOFTMAX; qcore_vpu_top carries VRMSNORM, VQUANT, VSILUMUL and VSUBC (docs/ROADMAP.md)"; exit 1

demo-toolcall:
	@echo "make demo-toolcall: the tool-call run executes VROPE and VSOFTMAX; qcore_vpu_top carries VRMSNORM, VQUANT, VSILUMUL and VSUBC (docs/ROADMAP.md)"; exit 1

demo-qwen:
	@echo "make demo-qwen: the recorded Qwen run executes VROPE and VSOFTMAX; qcore_vpu_top carries VRMSNORM, VQUANT, VSILUMUL and VSUBC (docs/ROADMAP.md)"; exit 1

regen-prefix:
	@echo "make regen-prefix: regenerating the demo prefix executes VROPE and VSOFTMAX; qcore_vpu_top carries VRMSNORM, VQUANT, VSILUMUL and VSUBC (docs/ROADMAP.md)"; exit 1

ci: lint style test cocotb synth gatesim harness-csr bringup
	@echo "ci: OK (the job set of .github/workflows/ci.yml, run locally)"

# One block per syn/synth_*.ys script (run from the repo root; each script names
# its block and its configurations, and ends every configuration with
# `script syn/report.ys`). scripts/synth_report.py runs them, leaves the logs in
# build/synth/ and writes syn/reports/<block>.md from what the tools printed;
# --check regenerates into a temporary directory and fails on any difference, so
# a report that no longer reproduces fails this target. Rewrite the reports with
#   uv run python scripts/synth_report.py
synth:
	$(UV) run python scripts/synth_report.py --check

# Gate-level equivalence (docs/VERIFICATION.md, layer 1). Yosys maps each block
# to Xilinx 7-series cells and writes the netlist; one Icarus bench then drives
# the source module and that netlist from the same stimulus and compares every
# output every cycle against Yosys's own cell models. A construct the two front
# ends read differently is a mismatch, and any mismatch fails the target.
# GATESIM_ARGS passes flags through, e.g. GATESIM_ARGS="--only seq_fetch_wb64".
GATESIM_ARGS ?=

gatesim:
	$(UV) run python sim/gatesim/gatesim.py $(GATESIM_ARGS)

harness:
	$(MAKE) -C sim/verilator build $(HARNESS_CFG)

harness-csr:
	$(MAKE) -C sim/verilator csr-check

# The bring-up comparison: the four-descriptor EMBED / VQUANT / GEMV(ARGMAX) /
# HALT program and the directed vector program of docs/ISA.md over a random tiny
# model, run on qcore_top and on sw/quettos/isa_sim.py and compared element by
# element. BRINGUP_TINY is the CI configuration; BRINGUP_ARGS is the wider
# sweep. Add --programs bringup or --programs vector to run just one.
BRINGUP_TINY ?= --sweep --shapes 2 --widths 16 --seed 0 --tokens 2
BRINGUP_ARGS ?= --sweep --shapes 5 --widths 64,128 --seed 0 --tokens 2

bringup:
	$(UV) run python -m quettos.compare $(BRINGUP_TINY)

bringup-sweep:
	$(UV) run python -m quettos.compare $(BRINGUP_ARGS)

perf:
	@test -f $(IMAGE)/layout.json || { \
	  echo "make perf: $(IMAGE)/layout.json not found (uv run quettos compile <model>)"; exit 1; }
	$(MAKE) -C sim/verilator run $(HARNESS_CFG) IMAGE=$(IMAGE) ARGS="$(PERF_ARGS)"

# A VCD of one harness run. The trace build is separate from the fast one, so
# this rebuilds with TRACE=1 and leaves the waveform in build/waves/.
WAVE ?= build/waves/qcore.vcd

waves:
	@mkdir -p $(dir $(WAVE))
	$(MAKE) -C sim/verilator run $(HARNESS_CFG) TRACE=1 IMAGE=$(IMAGE) ARGS="$(PERF_ARGS) --trace $(abspath $(WAVE))"
	@echo "waves: $(WAVE)"
