# Speed probe (`sim/probe/`)

A Verilator cost model: a full-width skeleton of the Quettos Core datapath whose
*per-cycle simulation cost* is representative of the final `qcore_top`, built
first so the demo configuration could be chosen before RTL integration. The real
RTL lives in `rtl/` and shares no code with this directory. The measured numbers
are in [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md), which also measures
how far a cycle here is from a cycle of the assembled core.

## What the probe is

`probe_top` (parameters `WB`, `B_MAX`, `VL`, `VSRAM_WORDS`, `FIFO_BEATS`, `ACC_W`,
defaults 64 / 1 / 4 / 4096 / 128 / 40) contains the structures that dominate the
final design's simulation cost, at full width and with real data flowing:

| Block | File | What it models |
|---|---|---|
| Weight FIFO | `probe_fifo.sv` | `FIFO_BEATS` x `WB*8` bits, fed from the QMEM-like `rd_data` port, popped one beat per cycle by the MAC array |
| Meta side-FIFO | `probe_fifo.sv` | 16 x `WB*8` bits; per-channel `{i32 bias, u16 Sw_m, i8 Sw_e, u8 pad}` metas, `WB/8` per beat, 8 beats per tile |
| MAC array | `probe_mac_group.sv` | `B_MAX` rows x `WB/8` groups x 8 lanes: int8 x int16 -> explicitly sized 24-bit signed product, `ACC_W`-bit accumulator with `acc <= prod + (tile_start ? 0 : acc)`, two accumulator sets swapped at tile end |
| Activation buffer | `probe_top.sv` | reads one 256-bit vsram word (port A) per 8 k, prefetched, broadcasts `A[k]` = low 16 bits of element `k % 8` |
| Vector SRAM | `probe_vsram.sv` | true-dual-port 4096 x 256-bit, 1-D unpacked array, separate read/write `always_ff`, registered reads |
| Requant | `probe_requant.sv` | drains `WB` accumulators one per cycle: `t = sat40(round_shift(acc*Sw_m, 16))`, `y = sat32(round_shift(t*Sx_m, S)) + bias`, `S = sbias-(Sw_e+Sx_e)` clamped to [0,63] with an error counter, `m == 0 -> 0`, 8 outputs per 256-bit word written to vsram port B, absmax, strict-greater argmax, saturation counter |
| VPU | `probe_vpu.sv`, `probe_vpu_lane.sv` | `VL` lanes, each with one 32x32->64 and one 32x16->48 signed multiplier, round-shift, saturate, absmax; an FSM that streams `n_words` vsram words through the lanes as an RMSNorm-like pass (sum of squares, leading-one detect, scale pass as read-modify-write on port B) |
| Perf counters | `probe_top.sv` | 64-bit cycles, busy, weight beats, MAC-active cycles |
| Checksum | `probe_top.sv` | every requant output, every VPU write-back, argmax/absmax/sat/err at op end, and (for `B_MAX > 1`) the extra rows are folded into a 32-bit checksum that the C++ prints, so Verilator cannot dead-code-eliminate any datapath |

The C++ harness (`main.cpp`) drives a synthetic decode token shaped like
Qwen2.5-0.5B: per layer `GEMV N=1152 K=896`, `N=896 K=896`, `N=9728 K=896`,
`N=896 K=4864` (x24) with a VPU pass of 900 words between GEMVs, then the LM head
`GEMV N=151936 K=896` in ARGMAX mode. Weight and meta beats come from a 64 MB
xorshift-filled buffer through a fixed-latency (default 32 cycles), one-beat-per-
cycle memory model with a 64-beat in-flight window. Per token this is
7,775,152 weight+meta beats at WB=64 (the design's 7,776,524 minus the 1,372 gamma
beats the probe does not stream), 8.22M cycles at WB=64 and 4.34M at WB=128.

## Scope

- Simulation cost only. The probe contains the datapath structures that
  dominate Verilator's per-cycle cost; the control logic (sequencer, descriptor
  decode, memory arbiter, KV writer, CSRs, LUT ROMs, RoPE/softmax/SiLU) lives in
  `rtl/`, and no `qcore_` module derives from this directory.
- Pseudo-random data. Weights, activations, metas and gammas are xorshift
  streams and the rsqrt is a placeholder mantissa; the arithmetic *shape*
  (multiplier widths, register counts, memory widths and access patterns) is
  what is measured.
- Ideal memory. Fixed latency, one beat per cycle; `--lat` above the 64-beat
  window throttles the stream.
- Its result priced a cycle of the design before `qcore_top` existed, which is
  how the demo configuration was chosen. Now that both are measured,
  [`docs/PERFORMANCE.md`](../../docs/PERFORMANCE.md) states the distance
  between them as a measured factor rather than an allowance, and every
  headline number comes from `make perf` on the real top.

## Build and run

```
make -f sim/probe/Makefile lint                 # verilator -Wall -Wpedantic, yosys check -assert, iverilog -g2012
make -f sim/probe/Makefile build WB=64 T=1      # records build time in results/
make -f sim/probe/Makefile run   WB=64 T=1 TOKENS=4
make -f sim/probe/Makefile matrix TOKENS=4      # WB in {64,128} x --threads in {1,2,4}
```

`probe --tokens N --lat N --vpu-words N --seed X --quiet`. Output ends with a
`RESULT WB=.. cycles=.. seconds=.. mcps=.. checksum=..` line. The checksum must be
identical across `--threads` values for the same `WB` (it is).

The RTL obeys the repo's SV subset: ANSI ports, `logic`, `always_ff` with
synchronous reset, `always_comb`, parameters, `generate for`, 1-D unpacked arrays
with registered reads for memories, `$signed()` multiplies into explicitly sized
wires, no unpacked-array ports, no interfaces/classes/struct literals/async
reset/`unique case`, no `return` in functions, no wildcard imports. Size casts
(`16'(x)`) are used and were verified to be accepted by Verilator 5.048,
Yosys 0.65 and Icarus 13.0.
