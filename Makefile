# Quettos Core -- top-level Makefile. Everything runs from the repo root.
# lint / test / probe / clean are implemented; the rest are placeholders
# until their pieces land (see docs/ROADMAP.md).

UV       := uv
VERILATOR := verilator
YOSYS    := yosys
IVERILOG := iverilog

MODEL    ?= qwen
TOP      ?= qcore_top

.PHONY: all help demo demo-toolcall demo-qwen regen-prefix ci test lint synth perf waves probe clean

all: help

help:
	@echo "Quettos Core targets:"
	@echo "  make lint          three-parser lint over rtl/*.sv (scripts/lint.sh)"
	@echo "  make test          uv run pytest -q sw/tests"
	@echo "  make probe         Verilator speed probe (sim/probe)"
	@echo "  make clean         remove build/ and Verilator obj_dir directories"
	@echo "  make demo          (not implemented yet)"
	@echo "  make demo-toolcall (not implemented yet)"
	@echo "  make demo-qwen     (not implemented yet)"
	@echo "  make regen-prefix  (not implemented yet)"
	@echo "  make ci            (not implemented yet)"
	@echo "  make synth         (not implemented yet)"
	@echo "  make perf          (not implemented yet)"
	@echo "  make waves         (not implemented yet)"

lint:
	@TOP=$(TOP) VERILATOR=$(VERILATOR) YOSYS=$(YOSYS) IVERILOG=$(IVERILOG) bash scripts/lint.sh

test:
	$(UV) run pytest -q sw/tests

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

synth:
	@echo "make synth: not implemented yet (see docs/ROADMAP.md)"; exit 1

perf:
	@echo "make perf: not implemented yet (see docs/ROADMAP.md)"; exit 1

waves:
	@echo "make waves: not implemented yet (see docs/ROADMAP.md)"; exit 1
