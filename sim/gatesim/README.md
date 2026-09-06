# Gate-level equivalence

`make gatesim` synthesizes a block with Yosys, simulates the resulting netlist
against the SystemVerilog it came from, and requires the two to agree on every
output on every cycle. It answers a question the three-parser lint of
`scripts/lint.sh` cannot: the lint proves that Verilator, Yosys and Icarus all
*accept* the RTL, and this proves that Yosys and Icarus *read it the same way*.

The failure it exists to catch is a construct the two front ends parse
differently. `~25'(WB - 1)` is one: Yosys 0.65 binds the complement to the size
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
3. Icarus compiles the bench with the RTL sources, the netlist and Yosys's own
   Xilinx cell models (`$(yosys-config --datdir)/xilinx/cells_sim.v`). The
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

The source side is compiled with `-DSYNTHESIS`, which Yosys 0.65 also defines,
so both front ends read exactly the same text. What that leaves out is the
simulation-only protocol assertions, which the random stimulus violates by
design.

## What is covered

Thirteen configurations of ten blocks -- ten of the twelve modules under
`rtl/`. `uv run python sim/gatesim/gatesim.py --list` prints them.

| Case | Block | Configuration | Cells |
|---|---|---|---|
| `mac_lane_group` | `qcore_mac_lane_group` | `ACC_W=40` | 1135 |
| `mem_arb_wb64` | `qcore_mem_arb` | `WB=64, MAX_BURST=64` | 4721 |
| `mem_arb_wb16` | `qcore_mem_arb` | `WB=16, MAX_BURST=8` | 1659 |
| `seq_fetch_wb64` | `qcore_seq_fetch` | `WB=64, DQ_DEPTH=8` | 1902 |
| `seq_fetch_wb16` | `qcore_seq_fetch` | `WB=16, DQ_DEPTH=8` | 1106 |
| `seq_dispatch_wb64` | `qcore_seq_dispatch` | `WB=64, B_MAX=1` | 2140 |
| `seq_dispatch_wb16` | `qcore_seq_dispatch` | `WB=16, B_MAX=2` | 2440 |
| `requant_tiny` | `qcore_requant` | `WB=16, B_MAX=2, ACC_W=40, VSRAM_WORDS=2048` | 10046 |
| `csr` | `qcore_csr` | defaults | 2507 |
| `perf` | `qcore_perf` | `WB=64` | 3604 |
| `kv_writer_tiny` | `qcore_kv_writer` | `WB=16, B_MAX=2, VSRAM_WORDS=2048` | 4493 |
| `row_tiny` | `qcore_row` | `WB=16, ACC_W=40, VSRAM_WORDS=2048` | 3887 |
| `stream_ctrl_tiny` | `qcore_stream_ctrl` | `WB=16, FIFO_BEATS=32, META_FIFO_BEATS=8, MAX_BURST=8` | 2877 |

## What is not covered, and why

- **`qcore_vsram`.** Its 256-bit memory is inferred as `RAMB36E1`. Yosys 0.65's
  `xilinx/cells_sim.v` declares `RAMB18E1` and `RAMB36E1` with their ports and
  no body, so a netlist holding one has undriven read ports and there is nothing
  to compare. The block is covered by its cocotb bench instead
  (`sim/cocotb/tb_vsram.py`).
- **`qcore_stream_ctrl` at the demo width.** `WB=64, FIFO_BEATS=128` maps its
  FIFOs to 15 `RAMB18E1`, the same wall. The tiny configuration keeps both
  FIFOs in distributed RAM (`RAM32M`), which does have a model, so that is the
  configuration in the table.
- **`qcore_top`.** The assembled core carries the block RAMs of `qcore_vsram`
  and the stream FIFOs, so it hits the same wall; its blocks are covered
  individually and the whole core is covered against the ISA simulator by
  `make bringup` and `make bringup-sweep`.

A configuration that starts inferring a block RAM does not fall through
silently: the case fails and names the cell.

## Running it

```sh
make gatesim                                   # every case
make gatesim GATESIM_ARGS="--only seq_fetch_wb64"
uv run python sim/gatesim/gatesim.py --list
uv run python sim/gatesim/gatesim.py --rtl-dir /path/to/other/rtl
```

The full set takes about a minute, eight cases at a time (`--jobs`, capped by
the core count). Almost all of it is `qcore_perf`: Icarus spends the time on
its 1024-bit snapshot output.

`--rtl-dir` points the run at another copy of `rtl/`, which is how the check is
verified to have teeth: reintroduce `~25'(WB - 1)` in a scratch copy and the
`seq_fetch` and `seq_dispatch` cases fail on the first compared cycles with
`f_req_addr src=00000000 gate=0000003f` and `ev_macs src=0000001080
gate=000000081f`.

Everything lands under `build/gatesim/<case>/`: the Yosys script and log, the
netlist, the port list, the generated bench and the simulation log with the
per-port toggle counts.
