# Gate-level equivalence

`make gatesim` synthesizes a block with Yosys, simulates the resulting netlist
against the SystemVerilog it came from, and requires the two to agree on every
output on every cycle. It answers a question the three-parser lint of
`scripts/lint.sh` cannot: the lint proves that Verilator, Yosys and Icarus all
*accept* the RTL, and this proves that Yosys and Icarus *read it the same way*.

The failure it exists to catch is a construct the two front ends parse
differently. `~25'(WB - 1)` is one: Yosys binds the complement to the size
literal and builds the mask `WB - 1`, where Verilator and Icarus build
`~(WB - 1)`. Both tools accept the line, every simulation passes, and the
bitstream computes something else.

## How it works

For each entry of `CASES` in `gatesim.py`:

1. Yosys reads the block with the case's parameters, runs
   `synth_xilinx -family xc7`, flattens, and writes the netlist and a JSON port
   list (`build/gatesim/<case>/netlist.v`, `ports.json`).
2. `gatesim.py` generates one bench from the port list
   (`build/gatesim/<case>/bench.sv`) holding both the source module and the
   netlist, wired to the same input registers.
3. Icarus compiles the bench with the RTL sources, the netlist and the Xilinx
   cell models of the Yosys that wrote it -- `xilinx/cells_sim.v` under the
   `yosys-config` beside that binary, or under the `share/yosys` beside it, so
   one build's netlist is always read against that build's own models. The
   source side is therefore read by the Icarus front end and the netlist side by
   the Yosys one.
4. The bench drives reset, then a short directed sequence chosen for the block,
   then random stimulus in four phases that sweep from free-running handshakes
   to heavy backpressure, with a mid-run reset every ~200 cycles. Wide inputs
   are drawn from a mix of zero, all-ones, small values, values astride the
   sign bit and uniform random words; a per-case `shape` steers the encoded
   fields (opcodes, row masks, N and K near a tile boundary, aligned PCs) into
   ranges the block decodes rather than faults on.
5. Every cycle it concatenates all outputs of both sides and compares them with
   `!==`. Any differing cycle is a mismatch and the target exits non-zero.

Three conditions besides a mismatch fail a case, so the check cannot quietly
stop meaning anything:

- **An output that never moves.** Every output port must toggle. The ports that
  are constant by construction are named in the case (`f_req_tag` is
  `TAG_FETCH`, `f_req_len` is the beats per request); any other silent port is
  a failure, because stimulus that no longer reaches an output proves nothing
  about it.
- **A netlist cell with no simulation model.** Yosys's library declares the hard
  block-RAM primitives with ports alone. A netlist that instantiates one is
  reported by name instead of silently comparing against a `Z`.
- **Too little of the run compared.** A cycle whose source outputs are still `X`
  carries no information -- the netlist's flops leave their `INIT` value where
  the source's are unwritten -- so it is counted and skipped. Past a tenth of the
  window the case fails.

The source side is compiled with `-DSYNTHESIS`, which Yosys also defines,
so both front ends read exactly the same text. What that leaves out is the
simulation-only protocol assertions, which the random stimulus violates by
design.

## What is covered

Nineteen configurations of fifteen blocks. `rtl/` holds eighteen files: the
package `qcore_pkg.sv` and seventeen modules, fifteen of which are in the table
below. The run writes that table: every cell count on it is one Yosys reported
for the case beside it, and the `Tool` line under it is the build that reported
them. `gatesim.py --write-table` rewrites both, and `gatesim.py --list` prints
the cases with the parameters each is elaborated with, without synthesizing
anything.

A cell count is a figure of the design and of the build that packed it, so the
table is checked against the `Tool` line. On the build the table records, a full
run holds the page to every figure on it and fails on any difference, naming the
counts that moved. On another build -- a newer Yosys, or the same release
compiled for another host -- the run holds the page to the cases and their
configurations, which are the design, and reports what that build's counts came
to. Either way the equivalence itself is checked in full: the cases, the
mismatch comparison and the three conditions above are the same on every
toolchain, and the cell counts are what the page publishes about them.

<!-- gatesim:cases -->

| Case | Block | Configuration | Cells |
|---|---|---|---|
| `mac_lane_group` | `qcore_mac_lane_group` | `ACC_W=40` | 1135 |
| `mem_arb_wb64` | `qcore_mem_arb` | `WB=64, MAX_BURST=64` | 4721 |
| `mem_arb_wb16` | `qcore_mem_arb` | `WB=16, MAX_BURST=8` | 1659 |
| `seq_fetch_wb64` | `qcore_seq_fetch` | `WB=64, DQ_DEPTH=8` | 1902 |
| `seq_fetch_wb16` | `qcore_seq_fetch` | `WB=16, DQ_DEPTH=8` | 1106 |
| `seq_dispatch_wb64` | `qcore_seq_dispatch` | `WB=64, B_MAX=1` | 2151 |
| `seq_dispatch_wb16` | `qcore_seq_dispatch` | `WB=16, B_MAX=2` | 2445 |
| `requant_tiny` | `qcore_requant` | `WB=16, B_MAX=2, ACC_W=40, VSRAM_WORDS=2048` | 10046 |
| `csr` | `qcore_csr` | defaults | 2507 |
| `perf` | `qcore_perf` | `WB=64` | 3604 |
| `kv_writer_tiny` | `qcore_kv_writer` | `WB=16, B_MAX=2, VSRAM_WORDS=2048` | 4493 |
| `row_tiny` | `qcore_row` | `WB=16, ACC_W=40, VSRAM_WORDS=2048` | 3887 |
| `stream_ctrl_tiny` | `qcore_stream_ctrl` | `WB=16, FIFO_BEATS=32, META_FIFO_BEATS=8, MAX_BURST=8` | 2877 |
| `lut_rom_exp2` | `qcore_lut_rom` | `ENTRIES=256`, `rtl/gen/exp2.hex` | 421 |
| `lut_rom_rsqrt` | `qcore_lut_rom` | `ENTRIES=512`, `rtl/gen/rsqrt.hex` | 730 |
| `lut_interp` | `qcore_lut_interp` | defaults | 99 |
| `vpu_lane` | `qcore_vpu_lane` | defaults | 1734 |
| `vpu_scalar` | `qcore_vpu_scalar` | `rtl/gen/rsqrt.hex`, `rtl/gen/recip.hex` | 2721 |
| `vpu_top_tiny` | `qcore_vpu_top` | `WB=16, B_MAX=2, VL=2, VSRAM_WORDS=2048, VPU_FIFO_BEATS=16, MAX_BURST=64`, `rtl/gen/sigmoid.hex`, `rtl/gen/exp2.hex`, `rtl/gen/rsqrt.hex`, `rtl/gen/recip.hex` | 25261 |

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

<!-- /gatesim:cases -->

A lookup table reaches the netlist as constants, so the image is part of the
configuration and each of the four is compared: `exp2` through `lut_rom_exp2`,
`rsqrt` through `lut_rom_rsqrt`, and `recip` and `sigmoid` through the units
that instantiate them, `vpu_scalar` and `vpu_top_tiny`. The run stages the
images it needs under `build/gatesim/<case>/` and both front ends read them
from there: Yosys through `chparam -set ROM_FILE`, Icarus through the same name
on the source instance, so a table the two load differently is a mismatch and a
name neither can find is an error rather than an empty ROM.

The parameter carries the bare file name for a reason worth stating. Yosys
names a parameterized submodule `$paramod$<hash of its parameters>`, and a
module name is an input to how the design is mapped, so a directory in that
value would put the directory the repository happens to sit in into the cell
count: the same tree checked out twice would report two different figures, and
the table would only ever reproduce in one of them. The bare name keeps the
count a property of the design.

## What is not covered, and why

Two of the seventeen modules are absent, both for the same reason. Yosys's
`xilinx/cells_sim.v` declares the hard block-RAM primitives with their ports and
no body, so a netlist holding one has undriven read ports and there is nothing
to compare against.

- **`qcore_vsram`.** Its 256-bit memory is inferred as `RAMB36E1` -- 32 of them
  at 4096 words and 16 at 2048, as `syn/reports/qcore_vsram.md` records. The
  block is covered by its cocotb bench instead (`sim/cocotb/tb_vsram.py`).
- **`qcore_top`.** The assembled core carries the vector SRAM of every row and
  the stream FIFOs at their demo depth, so it hits the same wall. Its blocks are
  covered individually above, and the whole core is compared against the ISA
  simulator by `make bringup` and `make bringup-sweep`.

One configuration is absent for the same reason: `qcore_stream_ctrl` at the demo
width. `WB=64, FIFO_BEATS=128` maps its FIFOs to 15 `RAMB18E1`. The tiny
configuration keeps both FIFOs in distributed RAM (`RAM32M`), which does have a
model, so that is the configuration in the table.

A configuration that starts inferring a block RAM does not fall through
silently: the case fails and names the cell.

## Running it

```sh
make gatesim                                   # every case
make gatesim GATESIM_ARGS="--only seq_fetch_wb64"
uv run python sim/gatesim/gatesim.py --list
uv run python sim/gatesim/gatesim.py --write-table   # rewrite the table above from the run
uv run python sim/gatesim/gatesim.py --rtl-dir /path/to/other/rtl
```

The full set runs eight cases at a time (`--jobs`, capped by the core count) and
takes 66.76 s, the median of five runs from an emptied `build/gatesim`
(65.94 - 67.92, n=5, on the machine `docs/PERFORMANCE.md` names). Almost all of
that is one case: `qcore_perf` has the widest output of the set, 1024 bits of
snapshot to compare every cycle, and the run prints the seconds each case took
beside its cell and cycle counts, so the eighteen others finish inside it.

`--rtl-dir` points the run at another copy of `rtl/`, which is how the check is
verified to have teeth. Two lines carry the parenthesised form,
`rtl/qcore_seq_dispatch.sv:245` and `:298`; write them back as
`& ~25'(WB - 1)` in a scratch copy and `seq_dispatch_wb64` fails from the first
compared cycle with

```
gatesim: [seq_dispatch_wb64] FAILED: 6901 mismatching cycle(s)
gatesim: MISMATCH cycle 17
gatesim:   cmd_sx_m src=0000 gate=0030
gatesim:   cmd_sreg_u32 src=00000000 gate=00000030
gatesim:   ev_macs src=0000001080 gate=000000081f
gatesim:   ev_wt_bytes src=0000001480 gate=0000000a17
```

while `seq_fetch_wb64`, which has no size cast under a unary operator, still
passes. `scripts/lint.sh` greps for the same shape, so the two checks have to
be defeated together.

Everything lands under `build/gatesim/<case>/`: the Yosys script and log, the
netlist, the port list, the generated bench and the simulation log with the
per-port toggle counts.
