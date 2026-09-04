# Quettos Core -- speed probe build/measure.
# Usage:
#   make build WB=64 T=1        build one configuration (records build time)
#   make run   WB=64 T=1        run it (TOKENS synthetic tokens)
#   make matrix                 build + run WB in {64,128} x threads in {1,2,4}
#   make lint                   three-parser lint (verilator, yosys, iverilog)
# Verilator flags: --cc --exe --build -j 0 -O3 --x-assign fast
# --x-initial fast --no-assert --no-timing -Wall -Wno-fatal --output-split 20000 -CFLAGS -O2
WB     ?= 64
T      ?= 1
TOKENS ?= 2
LAT    ?= 32
VPUW   ?= 900

HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SV      := $(wildcard $(HERE)probe_*.sv)
OBJ     := $(HERE)obj/wb$(WB)_t$(T)
RESULTS := $(HERE)results
BIN     := $(OBJ)/probe

VFLAGS := --cc --exe --build -j 0 -O3 --x-assign fast --x-initial fast --no-assert --no-timing \
          -Wall -Wno-fatal --output-split 20000 -CFLAGS -O2 -CFLAGS -DPROBE_WB=$(WB) \
          --threads $(T) -GWB=$(WB) --top-module probe_top --Mdir $(OBJ) -o probe

.PHONY: build run matrix lint clean

build:
	@mkdir -p $(RESULTS) $(HERE)obj
	@rm -rf $(OBJ)
	@echo "== build WB=$(WB) threads=$(T)"
	@bash -c 'TIMEFORMAT="build_seconds=%R"; time verilator $(VFLAGS) $(SV) $(HERE)main.cpp > $(RESULTS)/build_wb$(WB)_t$(T).log 2>&1' 2> $(RESULTS)/build_wb$(WB)_t$(T).time
	@cat $(RESULTS)/build_wb$(WB)_t$(T).time
	@grep -c "%Warning" $(RESULTS)/build_wb$(WB)_t$(T).log | sed 's/^/verilator_warnings=/' || true

run:
	@mkdir -p $(RESULTS)
	@echo "== run WB=$(WB) threads=$(T) tokens=$(TOKENS)"
	@$(BIN) --tokens $(TOKENS) --lat $(LAT) --vpu-words $(VPUW) | tee $(RESULTS)/run_wb$(WB)_t$(T).log

matrix:
	@for wb in 64 128; do for t in 1 2 4; do \
	  $(MAKE) -s -f $(HERE)Makefile build WB=$$wb T=$$t || exit 1; \
	  $(MAKE) -s -f $(HERE)Makefile run   WB=$$wb T=$$t TOKENS=$(TOKENS) || exit 1; \
	done; done
	@echo "== summary"; grep -H "^RESULT" $(RESULTS)/run_*.log; cat $(RESULTS)/build_*.time

lint:
	verilator --lint-only -Wall -Wpedantic --top-module probe_top $(SV)
	verilator --lint-only -Wall -Wpedantic --top-module probe_top -GWB=128 $(SV)
	yosys -q -p "read_verilog -sv $(SV); hierarchy -check -top probe_top; proc; opt; check -assert"
	iverilog -g2012 -Wall -o /dev/null $(SV)
	@echo "lint clean"

clean:
	rm -rf $(HERE)obj $(RESULTS)
