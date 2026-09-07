# Quettos Core harness: build and run sim/verilator against one RTL top.
# The object directory is named after a hash of the RTL (the modules and the
# includes they read), the C++ and the configuration, so an unchanged tree
# links the binary that is already there.
#
#   make -C sim/verilator build                  build the default configuration
#   make -C sim/verilator run IMAGE=<dir>        run it on a compiled model
#   make -C sim/verilator run ARGS="--traffic"   pass harness flags
#   make -C sim/verilator csr-check              CSR driver against rtl/qcore_csr.sv
#   make -C sim/verilator clean

TOP             ?= qcore_top
WB              ?= 64
B_MAX           ?= 1
VL              ?= 4
VSRAM_WORDS     ?= 4096
VPU_FIFO_BEATS  ?= 16
FIFO_BEATS      ?= 128
META_FIFO_BEATS ?= 16
ACC_W           ?= 40
MAX_BURST       ?= 64
DQ_DEPTH        ?= 8
THREADS         ?= 1
TRACE           ?= 0
IMAGE           ?= build/images/qwen2.5-0.5b-instruct-l2
ARGS            ?=

HERE  := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
REPO  := $(abspath $(HERE)/../..)
RTL   := $(REPO)/rtl

# SV goes on the Verilator command line; SVH is included by the modules, so it
# is hashed but never named.
SV   := $(sort $(wildcard $(RTL)/*.sv))
SVH  := $(sort $(wildcard $(RTL)/*.svh))
HEX  := $(sort $(wildcard $(RTL)/gen/*.hex))
HDRS := $(sort $(wildcard $(HERE)/*.hpp))
SRCS := $(HERE)/main.cpp $(HDRS)

# The lookup-table images qcore_vpu_top's ROMs load. Absolute, because Verilator
# runs $readmemh from the object directory rather than from the repository root;
# a string parameter is set quoted on the Verilator command line, so the quotes
# are escaped through the shell that runs the build.
ROMS := -GROM_FILE_SIGMOID=\"$(RTL)/gen/sigmoid.hex\" \
        -GROM_FILE_RSQRT=\"$(RTL)/gen/rsqrt.hex\" \
        -GROM_FILE_RECIP=\"$(RTL)/gen/recip.hex\"

PARAMS := -GWB=$(WB) -GB_MAX=$(B_MAX) -GVL=$(VL) -GVSRAM_WORDS=$(VSRAM_WORDS) \
          -GFIFO_BEATS=$(FIFO_BEATS) -GMETA_FIFO_BEATS=$(META_FIFO_BEATS) \
          -GACC_W=$(ACC_W) -GMAX_BURST=$(MAX_BURST) -GDQ_DEPTH=$(DQ_DEPTH) \
          -GVPU_FIFO_BEATS=$(VPU_FIFO_BEATS) $(ROMS)
DEFS := -DQCORE_WB=$(WB) -DQCORE_B_MAX=$(B_MAX) -DQCORE_VL=$(VL) \
        -DQCORE_VSRAM_WORDS=$(VSRAM_WORDS) -DQCORE_FIFO_BEATS=$(FIFO_BEATS) \
        -DQCORE_ACC_W=$(ACC_W) -DQCORE_MAX_BURST=$(MAX_BURST) \
        -DQCORE_DQ_DEPTH=$(DQ_DEPTH) -DQCORE_VPU_FIFO_BEATS=$(VPU_FIFO_BEATS) \
        -DQCORE_THREADS=$(THREADS) -DQCORE_TOP_NAME=$(TOP)
ifeq ($(TRACE),1)
DEFS      += -DQCORE_TRACE
TRACEFLAGS := --trace --trace-structs
endif
ifneq ($(THREADS),1)
THREADFLAGS := --threads $(THREADS)
endif

# Flags from docs/PERFORMANCE.md: the same set the speed probe was measured with.
VFLAGS := --cc --exe --build -j 0 -O3 --x-assign fast --x-initial fast --no-assert \
          --no-timing -Wall --output-split 20000 $(TRACEFLAGS) $(THREADFLAGS) \
          -CFLAGS "-O2 -I$(HERE) $(DEFS)" -I$(RTL) $(PARAMS) --top-module $(TOP)

CONFIG := $(TOP) $(WB) $(B_MAX) $(VL) $(VSRAM_WORDS) $(VPU_FIFO_BEATS) $(FIFO_BEATS) \
          $(META_FIFO_BEATS) $(ACC_W) $(MAX_BURST) $(DQ_DEPTH) $(THREADS) $(TRACE)
HASH := $(shell { echo "$(CONFIG)"; cat $(SV) $(SVH) $(HEX) $(SRCS); } | shasum -a 256 | cut -c1-12)
OBJ  := $(REPO)/build/verilator/$(TOP)-$(HASH)
BIN  := $(OBJ)/qcore_sim

CSR_OBJ := $(REPO)/build/verilator/qcore_csr-$(shell { cat $(RTL)/qcore_csr.sv $(RTL)/qcore_csr_defs.svh $(HERE)/csr_check.cpp $(HERE)/csr.hpp; } | shasum -a 256 | cut -c1-12)
CSR_BIN := $(CSR_OBJ)/csr_check

.PHONY: build run csr-check where clean

build: $(BIN)

# No prerequisites: the hash is in the directory name, so a binary that exists
# was built from exactly this RTL, C++ and configuration.
$(BIN):
	@mkdir -p $(OBJ)
	@echo "harness: verilating $(TOP) WB=$(WB) B_MAX=$(B_MAX) threads=$(THREADS) -> $(OBJ)"
	@cd $(REPO) && bash -c 'TIMEFORMAT="harness: build_seconds=%R"; time verilator $(VFLAGS) --Mdir $(OBJ) -o qcore_sim $(SV) $(HERE)/main.cpp'

# Run from the repository root so relative output paths land where the rest of
# the build does.
run: $(BIN)
	cd $(REPO) && $(BIN) --image $(REPO)/$(IMAGE) --threads $(THREADS) $(ARGS)

csr-check: $(CSR_BIN)
	$(CSR_BIN)

$(CSR_BIN):
	@mkdir -p $(CSR_OBJ)
	@cd $(REPO) && verilator --cc --exe --build -j 0 -O3 --x-assign fast --x-initial fast \
	  --no-assert --no-timing -Wall --output-split 20000 -CFLAGS "-O2 -I$(HERE)" -I$(RTL) \
	  --top-module qcore_csr --Mdir $(CSR_OBJ) -o csr_check $(RTL)/qcore_csr.sv $(HERE)/csr_check.cpp

where:
	@echo $(BIN)

clean:
	rm -rf $(REPO)/build/verilator
