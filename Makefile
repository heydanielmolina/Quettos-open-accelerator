# Quettos Core -- top-level Makefile. Everything runs from the repo root.
# lint / test / cocotb / synth / probe / clean are implemented; demo, perf and
# waves land with the end-to-end integration (see docs/ROADMAP.md).

UV       := uv
VERILATOR := verilator
YOSYS    := yosys
IVERILOG := iverilog

MODEL    ?= qwen
TOPS     ?=

.PHONY: all help demo demo-toolcall demo-qwen regen-prefix ci test lint cocotb synth perf waves probe clean

all: help

help:
	@echo "Quettos Core targets:"
	@echo "  make lint          three-parser lint over rtl/*.sv (scripts/lint.sh)"
	@echo "  make test          uv run pytest -q sw/tests"
	@echo "  make cocotb        cocotb unit tests on the tiny RTL configuration (sim/cocotb)"
	@echo "  make probe         Verilator speed probe (sim/probe)"
	@echo "  make clean         remove build/ and Verilator obj_dir directories"
	@echo "  make demo          (not implemented yet)"
	@echo "  make demo-toolcall (not implemented yet)"
	@echo "  make demo-qwen     (not implemented yet)"
	@echo "  make regen-prefix  (not implemented yet)"
	@echo "  make ci            (not implemented yet)"
	@echo "  make synth         Yosys synth_xilinx of every syn/*.ys block script; logs in build/synth/"
	@echo "  make perf          (not implemented yet)"
	@echo "  make waves         (not implemented yet)"

lint:
	@TOPS="$(TOPS)" VERILATOR=$(VERILATOR) YOSYS=$(YOSYS) IVERILOG=$(IVERILOG) bash scripts/lint.sh

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
	@echo "make demo: not implemented yet (see docs/ROADMAP.md)"; exit 1

demo-toolcall:
	@echo "make demo-toolcall: not implemented yet (see docs/ROADMAP.md)"; exit 1

demo-qwen:
	@echo "make demo-qwen: not implemented yet (see docs/ROADMAP.md)"; exit 1

regen-prefix:
	@echo "make regen-prefix: not implemented yet (see docs/ROADMAP.md)"; exit 1

ci:
	@echo "make ci: not implemented yet (see docs/ROADMAP.md)"; exit 1

# One block per syn/*.ys script (run from the repo root; each script names its
# block and configuration). The stat table sits at the end of build/synth/<script>.log;
# syn/reports/ holds the recorded tables.
SYN_SCRIPTS := $(wildcard syn/*.ys)

synth:
	@mkdir -p build/synth
	@for s in $(SYN_SCRIPTS); do \
	  n=$$(basename $$s .ys); echo "synth: $$s -> build/synth/$$n.log"; \
	  $(YOSYS) -q -l build/synth/$$n.log -s $$s >/dev/null 2>build/synth/$$n.err || { cat build/synth/$$n.err; echo "synth: $$s FAILED"; exit 1; }; \
	  grep -v 'Resizing cell port' build/synth/$$n.err || true; \
	done
	@echo "synth: OK"

perf:
	@echo "make perf: not implemented yet (see docs/ROADMAP.md)"; exit 1

waves:
	@echo "make waves: not implemented yet (see docs/ROADMAP.md)"; exit 1
