# Quettos Core -- top-level Makefile. Everything runs from the repo root.

UV       := uv
VERILATOR := verilator
YOSYS    := yosys
IVERILOG := iverilog

TOPS     ?=

# The compiled model the Verilator harness runs and the flags it runs with
# (sim/verilator, docs/PERFORMANCE.md). HARNESS_CFG selects the RTL widths.
IMAGE       ?= build/images/qwen2.5-0.5b-instruct-l2
HARNESS_CFG ?= WB=64 B_MAX=1 VL=4 VSRAM_WORDS=4096
PERF_ARGS   ?= --max-new 1
PREFIX_KV   ?= build/kv/prefix.kv

# The demo (scripts/demo.sh): which model it takes from the Hugging Face Hub to
# text on qcore_top, the prompt compiled into the image, and how many decode
# steps it runs before the model's own end-of-sequence id stops it. DEMO_ARGS
# goes to the script, and everything after `--` in it to the harness run.
MODEL       ?= smollm2
DEMO_PROMPT ?= prompts/chat_short.json
MAX_NEW     ?= 20
DEMO_ARGS   ?=

.PHONY: all help demo demo-qwen regen-prefix ci test lint style cocotb synth gatesim perf waves probe harness harness-csr bringup bringup-sweep stepcmp stepcmp-models stepcmp-model determinism determinism-model determinism-x clean

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
	@echo "  make bringup       RTL vs isa_sim on the tiny configuration, four programs (sw/quettos/compare.py)"
	@echo "  make bringup-sweep the same over random shapes at WB=64 and WB=128"
	@echo "  make stepcmp       whole compiled programs, descriptor by descriptor, RTL vs isa_sim (sw/quettos/stepcmp.py)"
	@echo "  make stepcmp-models the same over truncated Qwen and SmolLM2 images"
	@echo "  make stepcmp-model  the same over the complete Qwen and SmolLM2 models"
	@echo "  make determinism   one program run every way the machine allows: the ids and the state do not move (sw/quettos/determinism.py)"
	@echo "  make determinism-model the same over IMAGE and WB128_IMAGE, one model compiled at each width"
	@echo "  make determinism-x an undefined-value start (Verilator --x-initial unique)"
	@echo "  make clean         remove build/ and Verilator obj_dir directories"
	@echo "  make demo          the whole pipeline: checkpoint -> image -> text on qcore_top (MODEL, scripts/demo.sh)"
	@echo "  make demo-qwen     the same on the headline model (MODEL=qwen)"
	@echo "  make regen-prefix  prefill IMAGE's prompt and save the KV region to PREFIX_KV"
	@echo "  make ci            the CI job set, run locally"
	@echo "  make synth         Yosys synth_xilinx of every syn/synth_*.ys script; logs in build/synth/, and every syn/reports/*.md has to still reproduce"
	@echo "  make gatesim       gate-level equivalence: each Yosys netlist against the source it came from (sim/gatesim)"
	@echo "  make perf          run the harness on IMAGE and write build/perf/perf.json"
	@echo "  make waves         a VCD of the first WAVE_CYCLES cycles of a harness run into build/waves (20,000 cycles, 241 MB)"

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

# The demo, from a clean clone: sync the environment, fetch the checkpoint,
# quantize it, compile the image and the two descriptor programs, build the
# harness and run the whole model on qcore_top -- every token printed as it
# leaves the hardware with its cycle cost and its MAC utilization, then the
# summary of `quettos demo-report`: the ids against the golden model's recorded
# continuation, the counters a clean run leaves at zero, the cycle and
# utilization table and the wall clock of every stage. A counter that should be
# zero and is not fails the target. `make demo` runs the fast model and
# `make demo-qwen` the headline one.
demo:
	@bash scripts/demo.sh --model $(MODEL) --prompt $(DEMO_PROMPT) --max-new $(MAX_NEW) $(DEMO_ARGS)

demo-qwen:
	@$(MAKE) demo MODEL=qwen

# The prefix a later run restores: prefill every prompt token of IMAGE, generate
# nothing, and write the KV region out at its image layout, so the file belongs
# to the model and the port width that produced it (docs/ARCHITECTURE.md).
regen-prefix:
	@test -f $(IMAGE)/layout.json || { \
	  echo "make regen-prefix: $(IMAGE)/layout.json not found (uv run quettos compile <model>)"; exit 1; }
	@mkdir -p $(dir $(PREFIX_KV))
	$(MAKE) -C sim/verilator run $(HARNESS_CFG) IMAGE=$(IMAGE) \
	  ARGS="--max-new 0 --kv-save $(abspath $(PREFIX_KV))"
	@echo "regen-prefix: $(PREFIX_KV)"

ci: lint style test cocotb synth gatesim harness-csr bringup determinism demo
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
# HALT program of docs/ISA.md, the directed vector program, the attention step
# of decode.prog at six positions and the whole decoder layer at the same six,
# each run on qcore_top and on sw/quettos/isa_sim.py and compared element by
# element. BRINGUP_TINY is the CI configuration; BRINGUP_ARGS is the wider
# sweep. Add --programs attention (or bringup, vector, layer) to run just one.
BRINGUP_TINY ?= --sweep --shapes 2 --widths 16 --seed 0 --tokens 2
BRINGUP_ARGS ?= --sweep --shapes 5 --widths 64,128 --seed 0 --tokens 2

bringup:
	$(UV) run python -m quettos.compare $(BRINGUP_TINY)

bringup-sweep:
	$(UV) run python -m quettos.compare $(BRINGUP_ARGS)

# The per-descriptor comparison of a whole compiled program (docs/VERIFICATION.md,
# layer 4): prefill.prog and decode.prog run one CTRL.STEP per descriptor on
# qcore_top and on sw/quettos/isa_sim.py, and the VSRAM range, scale registers,
# memory regions, CSRs, PC and counters that each descriptor's dump_plan.json
# entry names are compared after every one of them. STEPCMP_TINY is five random
# tiny models at WB=16, each run over every position of a two-tile KV cache;
# STEPCMP_MODELS is the first one and two layers of Qwen and SmolLM2, which needs
# build/quant/<name>.npz (uv run quettos quantize <alias>). Both take --wb, so
# `make stepcmp STEPCMP_TINY="--sweep --shapes 5 --wb 64 --prompt-len 64
# --max-new 65"` runs the same comparison at the demo width, and --lat / --bw-div
# run it at another memory setting.
STEPCMP_TINY   ?= --sweep --shapes 5 --wb 16 --prompt-len 16 --max-new 17 --seed 0
STEPCMP_MODELS ?= --models qwen,smollm2 --layers 1,2 --wb 64 --max-ctx 128 --prompt-len 66 --max-new 2
STEPCMP_MODEL  ?= --models qwen,smollm2 --layers all --wb 64 --max-ctx 128 --prompt-len 3 --max-new 2

stepcmp:
	$(UV) run python -m quettos.stepcmp $(STEPCMP_TINY)

stepcmp-models:
	$(UV) run python -m quettos.stepcmp $(STEPCMP_MODELS)

# The same comparison on the complete models: every decoder layer, the final
# norm and the LM head of Qwen and SmolLM2, two prefill positions and two decode
# steps, so the KV cache the first decode step writes is what the second reads.
# The tile-boundary positions are the truncated compiles' job above, which run
# every position of a two-tile cache; this target is what the depth of a whole
# model adds to them.
stepcmp-model:
	$(UV) run python -m quettos.stepcmp $(STEPCMP_MODEL)

# Determinism and configuration equivalence (docs/VERIFICATION.md, layer 5): one
# program run on qcore_top every way the machine allows -- QMEM read latency
# 1 / 32 / 200, one returned beat every two cycles, one and four simulation
# threads, WB 64 against WB 128, and Verilator's initial value for a variable no
# reset reaches -- with the ids, the dumped state and the memory the program
# writes compared against the run at the configuration every measurement is
# taken at. `determinism` runs on a random tiny model compiled at both widths;
# `determinism-model` runs on IMAGE and WB128_IMAGE, which are one model
# compiled at each width; `determinism-x` is the undefined-value start.
WB128_IMAGE        ?= build/images/qwen2.5-0.5b-instruct-l2-wb128
DETERMINISM_ARGS   ?= --checks timing,width --program layer,bringup --generate 2
DETERMINISM_MODEL  ?= --checks timing,width --program layer --generate 2
DETERMINISM_X_ARGS ?= --checks x-initial --program layer,bringup --generate 2

determinism:
	$(UV) run python -m quettos.determinism $(DETERMINISM_ARGS)

determinism-model:
	@test -f $(IMAGE)/layout.json || { \
	  echo "make determinism-model: $(IMAGE)/layout.json not found (uv run quettos compile <model>)"; exit 1; }
	@test -f $(WB128_IMAGE)/layout.json || { \
	  echo "make determinism-model: $(WB128_IMAGE)/layout.json not found (uv run quettos compile <model> --wb 128 --out $(WB128_IMAGE))"; exit 1; }
	$(UV) run python -m quettos.determinism --image $(IMAGE) --image $(WB128_IMAGE) $(DETERMINISM_MODEL)

determinism-x:
	$(UV) run python -m quettos.determinism $(DETERMINISM_X_ARGS)

perf:
	@test -f $(IMAGE)/layout.json || { \
	  echo "make perf: $(IMAGE)/layout.json not found (uv run quettos compile <model>)"; exit 1; }
	$(MAKE) -C sim/verilator run $(HARNESS_CFG) IMAGE=$(IMAGE) ARGS="$(PERF_ARGS)"

# A VCD of one harness run. The trace build is separate from the fast one, so
# this rebuilds with TRACE=1 and leaves the waveform in build/waves/.
#
# A traced cycle of qcore_top is about 12 KB with --trace-structs, so the window
# is what sets the file size: WAVE_CYCLES cycles of WAVE_ARGS, from the first.
# The default is one decode token of IMAGE traced from its reset through its
# EMBED and into the first GEMV -- 20,000 cycles, 241,496,192 B -- and the run
# carries on to its HALT with the file closed. Raise WAVE_CYCLES for a longer
# window at 12 KB a cycle, or set it to 0 for the whole run.
WAVE        ?= build/waves/qcore.vcd
WAVE_CYCLES ?= 20000
WAVE_ARGS   ?= --max-new 1 --prompt-ids 1

waves:
	@mkdir -p $(dir $(WAVE))
	$(MAKE) -C sim/verilator run $(HARNESS_CFG) TRACE=1 IMAGE=$(IMAGE) \
	  ARGS="$(WAVE_ARGS) --trace $(abspath $(WAVE)) --trace-cycles $(WAVE_CYCLES)"
	@printf 'waves: %s, %s B, the first %s cycles\n' "$(WAVE)" "$$(wc -c < $(WAVE) | tr -d ' ')" "$(WAVE_CYCLES)"

