# Verification

## Oracle chain

```
reference_np.py   float32 numpy forward, Hugging Face Llama/Qwen2 semantics;
   |              argmax-identical to the transformers fp32 forward on the same
   |              weights (sw/tests/test_reference_np.py)  quality reference only
   |  top-1 / KL / PPL deltas, never bit-exact
   v
golden.py         op-by-op integer forward, float64 BLAS for exact integer GEMV (partials < 2^53);
   |              forward_tokens (teacher-forced, all positions) == step (one program at one
   |  bit-exact   position) at every position; program.py supplies the requant constants
   v
quality.py        golden vs reference_np over the calibration set (uv run quettos check <alias>):
   |              top-1, KL, delta-NLL +/- SE, PPL -> models/<name>/quality.json
   |
isa_sim.py        executes decode.prog / prefill.prog on image.bin at value level, with
   |  bit-exact  numerics.py as its only arithmetic; == golden (every descriptor, KV byte and
   |             counter) on synthetic shapes at WB 16 / 64 / 128, two-layer real models and
   |             the complete models over prefill + decode (sw/tests/test_isa_sim.py)
   v
RTL (Verilator)   qcore_top, all three configurations; compare.py runs the bring-up
   |              program, the directed vector program, the attention step and the whole
   |  bit-exact  decoder layer on both and reports the first differing element; the
   v             harness generates from a compiled image and the ids match isa_sim's
                 and the golden model's recorded continuation (make demo)
```

"Bit-exact" always means RTL == isa_sim == golden, our own integer model. It
never means "matches PyTorch". Quality relative to the float32 reference is a
**measured delta**, reported with standard errors at the calibration-set lengths
(up to 520 tokens).

## The eight layers

1. **Static and structural, every push.** Three-parser lint (`scripts/lint.sh`:
   Verilator `--lint-only -Wall -Wpedantic`, Yosys `hierarchy -check; proc;
   opt; check -assert`, Icarus `-g2012`) over every `rtl/*.sv` file -- the
   seventeen modules and the package -- and every `sim/cocotb/wrappers/*.sv`
   assembly as its own top, nineteen in all, zero warnings and no waivers;
   `make style`, which is `ruff check` and `ruff format --check` over
   every `.py` in the repository; lutgen / RoPE regeneration diff report;
   `layout.json` sha256s; no-literal-`$readmemh` lint; forbidden construct
   greps.
   The lint proves the three tools accept the RTL; gate-level equivalence
   (`make gatesim`, `sim/gatesim/`) proves two of them read it the same way.
   Yosys maps a block to Xilinx 7-series cells and writes the netlist, and one
   Icarus bench drives the source module and that netlist from the same
   directed and random stimulus, comparing the concatenation of their outputs
   every cycle against Yosys's own cell models. The source side is read by the
   Icarus front end and the netlist side by the Yosys one, so a construct the
   two read differently is a mismatch instead of a silent difference between
   what simulates and what synthesizes. Nineteen configurations of fifteen
   blocks are covered, fifteen of the seventeen modules under `rtl/`, and a
   case fails as well when an output port stops toggling, when the netlist
   instantiates a cell the Yosys library only declares, or
   when more than a tenth of the window is still undefined.
   `sim/gatesim/README.md` lists what is covered and the reason for each block
   that is not: `qcore_vsram`, the demo-width `qcore_stream_ctrl` FIFOs and
   `qcore_top` all infer `RAMB18E1` / `RAMB36E1`, which Yosys 0.65 declares
   without a simulation body. The lookup tables do not: `memory_libmap` picks
   the logic mapping for a write-free memory, so `qcore_lut_rom` and the whole
   of `qcore_vpu_top` are simulable.
2. **`numerics.py` pytest.** `round_shift` / `sat` / `sfloat` / `sfloat_mul`
   properties (m in range for all pairs), LUT error bounds vs float64,
   recip/rsqrt at `m in {1.0, 2.0, 4-eps}`, requant vs a big-int reference
   within 1 LSB including `S = 0 / 63` and the `m == 0` rule, the exponent
   equation, the zero-vector rule, the VQUANT clip rule (`|q| <= 32767`),
   softmax at `len = 1` / all-equal / extreme negatives, SiLU at
   `g in {-16, -8, 0, 8, 16}`, RoPE pairs, small-range exhaustive fuzz.
3. **cocotb 2.1 + Verilator block tests on the tiny config** (`make cocotb`;
   most benches stay under a million cycles, and the two whose state space is
   small enough to sweep exhaustively -- the vector lane and the vector scalar
   unit -- run a few million, scaled by `QCORE_VPU_LANE_VECTORS` and
   `QCORE_VPU_SCALAR_REQUESTS`): `vsram` on both ports with element strobes;
   lane group and row including the B=2 broadcast, which proves the batch
   parameter;
   requant fuzz including negative extremes, partial tiles, dumps and several
   rows; `stream_ctrl` at `LAT 1 / 32 / 200` with backpressure, the FIFO
   reservation bound, and a utilization case at `LAT 32` that budgets the busy
   cycles carrying no weight beat instead of taking a ratio: with the rows
   always ready the only ones it allows are the read latency and the FIFO
   registers behind it before the first beat, the eight meta beats each tile
   interleaves between its weight bursts, and the last tile's `WB` records
   still leaving the serializer -- `lat + 5 + 8*tiles + (WB - 8)` cycles, and
   `lat + 5` with `unit_meta`, which is the same statement at every port width
   where a ratio would move with `WB`; the EMBED gather and exact burst
   shapes; `mem_arb`
   routing, the reserved fetch slot and `wr_idle` timing; `kv_writer` for
   `POS in {0, 1, 63, 64, 2047}` compared byte for byte against `isa_sim`;
   `seq_fetch` restarts at every position inside a beat, stepping across beat
   boundaries and the `fetch_hold` write fence; `seq_dispatch` POS-derived
   fields against `isa_sim.gemv_dims` and `softmax_len` over every GEMV, EMBED
   and VSOFTMAX shape whose extents come from `POS`, at each of the twelve
   positions of `SWEEP_POSITIONS` -- every one that lands on a tile edge or a
   capacity -- and each row set of `SWEEP_ROWS` that survives `ROW_EN` at this
   `B_MAX`, the bench holding the descriptors it compared to that whole
   product; the auto-fence ordering, the six exclusive stall buckets, the
   zero-work rule, step mode, the write fence a step, a fault or an ABORT ends
   on, the three decode-cycle faults -- both edges of the VSOFTMAX class window
   issue and five classes outside it fault with `FAULT = CLASS`, nothing issued
   and `PC` left on the descriptor -- and the rule that a bounds event is
   counted only for a descriptor that commits;
   `csr` register semantics; `perf` bucket exclusivity and snapshot timing;
   the assembled GEMV path (`wrappers/qcore_gemv_wrap.sv`) against
   `numerics.requant` over the exact integer matmul in all three
   configurations; the lookup-table ROM and interpolator over every
   (entry, fraction) pair of all four tables against `numerics.Lut.interp`; the
   vector lane and the vector scalar unit against `numerics.py` over millions of
   elements and requests; and `qcore_vpu_top` itself against `isa_sim` on the
   tiny and the demo widths, descriptor by descriptor.
4. **Op-level RTL vs isa_sim.** `sw/quettos/compare.py` (`make bringup`,
   `make bringup-sweep`, `sw/tests/test_bringup.py`) runs four programs on
   `qcore_top` through the Verilator harness and on `isa_sim`, one `CTRL.STEP`
   per descriptor:

   - the four-descriptor bring-up program of `docs/ISA.md` (`EMBED`, `VQUANT`,
     the tied LM head as a `GEMV` in `ARGMAX_DUMP` mode, `HALT`, with no
     register loaded from the host);
   - the eight-descriptor directed vector program (`EMBED`, `VRMSNORM`,
     `VQUANT` with `USE_TRACKED`, `VSUBC`, `VSILUMUL`, `VQUANT` with `GROUP`, a
     saturating `VSILUMUL`, `HALT`);
   - the **attention step**: `decode.prog` from its `EMBED` through the last
     value `GEMV` of layer 0, with a `HALT` after it -- the input norm and its
     quantize, the `GEMV` of `Wqkv`, the `VROPE` over the contiguous q and k
     heads, the per-head quantizes of q, k and v, the two `KVWRITE` per KV
     head, and per query head the `K^T` `GEMV` whose `N` comes from the
     position, the `VSOFTMAX` whose length comes from it, and the `V` `GEMV`
     whose `K` comes from it;
   - the **whole decoder layer**: the same prefix continued through the output
     projection, the post norm, the gate-and-up `GEMV`, the `VSILUMUL`, the
     quantize of the hidden row and the down projection.

   The two prefixes are the compiler's own descriptors in the compiler's order,
   so what runs is a decode step and not a program assembled for the
   comparison. They run at six positions -- the first, one inside the first
   weight-port tile, the last position of that tile, the one that opens the
   second, one past it, and the last position the cache holds -- in order and
   on one machine, each with its own token, so the KV cache a position writes
   is what the next one reads. Every position is a different pair of derived
   extents: `N = min(roundup(POS+1, WB), n)` steps by a tile and `K = POS + 1`
   by one.

   The comparison covers the whole used VSRAM range of every bank plus the
   vector program's scratch ranges, all 32 SREG words of every bank, the `DUMP`
   region holding every int32 logit, the whole KV region as bytes at the end of
   every position, `PC`, `STATUS`, the ARGMAX registers, the four event
   counters and `DESCRIPTORS` / `MACS` / `WT_BYTES` after each descriptor. The
   first difference is reported as (position, descriptor index, opcode, field,
   element, expected, got). It runs over random tiny shapes (hidden 64-192,
   vocab 128-256, kv_heads 1-3): two shapes at WB 16 in the CI run and five at
   WB 64 and WB 128 in the sweep, each at `--lat` 1 / 32 / 200 and `--bw-div`
   2, and the RTL records must be identical across those four settings.
   `sw/tests/test_bringup.py` adds a grouped-query shape (six query heads over
   three KV heads) and a run at every position the cache holds, so the softmax
   reduces over a row with a weight in every element.

   The fault path is checked on each machine by its own bench, to the same
   expectation: a descriptor whose opcode byte is none of the twelve, and a
   VSOFTMAX whose class field leaves `[16, 30]`, stop the run with
   `FAULT = OPCODE` or `FAULT = CLASS`, the opcode byte in `FAULT_OP`, `PC` on
   the descriptor and nothing counted for it -- `sim/cocotb/tb_seq_dispatch.py`
   on the dispatcher and `sw/tests/test_isa_sim.py` on `isa_sim`. The
   `--layers N` truncated models and the KV region comparison follow through
   `isa_sim.compare_sequence`.

   **Whole programs, descriptor by descriptor.** `sw/quettos/stepcmp.py`
   (`make stepcmp`, `make stepcmp-models`, `make stepcmp-model`,
   `sw/tests/test_stepcmp.py`) takes the
   same comparison to the compiler's own programs end to end. The compiler
   writes `dump_plan.json` beside `decode.prog` and `prefill.prog` naming, per
   descriptor, the VSRAM range it writes, the scale registers it writes, the
   memory regions it touches and the CSRs; the harness runs a program one
   `CTRL.STEP` per descriptor and writes exactly that state out after each one
   (`--step --dump-ops`, with `--dump-bytes N` carrying a region of at most `N`
   bytes as bytes rather than as its FNV-1a hash), and `isa_sim` runs the same
   program on the same image and captures the same state. Compared after every
   descriptor: the VSRAM range, the scale registers as the words the bank holds,
   the memory regions, the ARGMAX registers, `PC`, `DESCRIPTORS` / `MACS` /
   `WT_BYTES` and the four event counters, the last two groups since the token's
   first descriptor -- `START` clears the counters and a `STEP` clears none of
   them, so a stepped token's cost is read against the base its first descriptor
   started from. The first difference is reported as (program, position,
   descriptor index, opcode, dataflow name, listing line, element).
   `isa_sim.check_plan` runs over the simulator's own write log first, so the
   plan is known to name every write each descriptor made and comparing the
   planned state is comparing all of it.

   The tiny sweep is five random shapes -- hidden 64 to 192, vocabulary 128 or
   256, two to six query heads over one to three KV heads, grouped-query among
   them, one and two layers, both RoPE tables and both bias settings -- at
   WB=16, each run over every position of a two-tile KV cache, so the last
   position of the first weight-port tile, the one that opens the second and the
   last position the cache holds are all in it: 32 tokens and 10,047 descriptors
   per sweep, in **3.35 s** (3.31 - 3.42, n=5) with the harness already built,
   on the machine `docs/PERFORMANCE.md` names. `make stepcmp-models` runs the
   first one and two layers of both real models at WB=64 over the same tile
   boundary, 67 positions each: 22,871 descriptors in **36.8 s**
   (36.2 - 37.3, n=3). Together they cover what the four bring-up
   programs above do not -- every layer of the program, the final norm, the LM
   head, and `prefill.prog` beside `decode.prog`.

   **On the complete models.** `make stepcmp-model` runs the same comparison on
   the whole of Qwen2.5-0.5B-Instruct and SmolLM2-135M-Instruct at WB=64 -- all
   24 and all 30 decoder layers, the final norm and the LM head over the real
   vocabulary -- at two prefill positions and two decode steps, so the KV cache
   the first decode step writes is what the second one reads:
   **11,860 descriptors** of `prefill.prog` and `decode.prog` compared element by
   element, 5,966 on Qwen and 5,894 on SmolLM2, in **20.0 s** (19.7 - 20.3, n=3)
   with the harness built and the quantized checkpoints on disk. That is the
   whole of both programs at those positions: every descriptor's VSRAM range,
   scale registers, KV bytes, ARGMAX registers, `PC` and counters. The positions
   a weight-port tile boundary falls on are the one- and two-layer compiles'
   job -- 67 positions each, above -- and the depth of a complete model is this
   target's; the images are compiled at `max_ctx = 128` so every KV region
   travels as bytes and a difference is reported at the byte it is in. The
   nightly `e2e-qwen` and `e2e-smollm2` jobs run it beside the generated-id and
   golden-model commands, so a complete model is compared as state and not only
   as ids.

   `sw/tests/test_stepcmp.py` runs the sweep, the tiny image, the four truncated
   models and the two complete ones, the last six skipped when
   `build/quant/<name>.npz` is not there.
5. **End-to-end.** A whole compiled program runs on `qcore_top` (`make perf`,
   `make demo`): every descriptor of `decode.prog` and `prefill.prog` at its
   real address, stride, meta and partial tile, over the image's own prompt and
   as many decode steps as asked for, with `WT_BYTES` and `MACS` checked
   against `layout.json`'s traffic model, `RD_BEATS` / `RD_BYTES` /
   `WR_BEATS` / `WR_BYTES` against the memory model's own counts, the six stall
   buckets required to sum to `BUSY`, and all of it recorded in
   `build/perf/perf.json`. Token equality closes the oracle chain from the top:
   the ids `qcore_top` generates are compared with `isa_sim`'s over the same
   image and prompt, and `isa_sim` reproduces the integer golden model, whose
   own generation is recorded in `models/<name>/expected_tokens.json`.

   `make demo` (`scripts/demo.sh`) runs that from the checkpoint in one
   command -- sync, download, quantize, compile, build the harness, generate on
   `qcore_top` with every token printed as it leaves the hardware -- and
   `quettos demo-report` then holds the finished run to what it has to be. The
   files first: every entry of `layout.json` that carries a `file` and a
   `sha256` -- the image, both descriptor programs, the dump plan, the token
   table, the prompt ids, the RoPE and lookup tables and the golden model's
   recorded continuation -- read again and hashed again, and held to what the
   compiler recorded; the image also at its recorded size, and that same file at
   that same byte count in the `image` and `image_bytes` the run says it mapped.
   Then the prompt: the ids `perf.json` records the loop as having been given,
   at the count and the SHA-256 `layout.json` carries for the prompt the image
   was compiled with, with one prefill token for every id but the last -- and it
   is that match which resolves the prompt file the record is looked up under.
   Then the ids: identical to the golden model's recorded continuation of that
   prompt, resolved through the `expected_tokens` entry `layout.json` carries
   and held to the per-prompt count and SHA-256 the image was compiled against,
   a run that generated nothing failing rather than matching an empty record.
   Then the counters: `SAT_REQ`, `SAT_VPU`, `ERR_SHIFT` and `ERR_BOUNDS` all
   zero; `BUSY` the sum of the six buckets; `RD_BYTES` equal to
   `RD_BEATS * WB`; the write counters equal to the memory model's own; every
   token retiring its program's full descriptor count; `CYCLES` equal to the sum
   of the tokens' own counts; and the run's `status` `ok` rather than a stop. Any one of those failing
   fails the command and is named in the report. The report computes no model
   value of its own: the ids are the hardware's and the reference is the
   checked-in record. `make demo` runs the complete SmolLM2-135M-Instruct and
   `make demo-qwen` the complete Qwen2.5-0.5B-Instruct, both to the model's own
   end-of-sequence id; `docs/PERFORMANCE.md` holds the wall clock and the stage
   table of each.

   Determinism and configuration equivalence are a runnable check of their own
   (`sw/quettos/determinism.py`, `make determinism`): one compiled program runs
   on `qcore_top` under every setting that changes the timing or the starting
   state and nothing else, and each run is compared with the run at the
   configuration every measurement is taken at, `--lat 32 --bw-div 1
   --threads 1`. A setting runs the program twice -- once descriptor by
   descriptor with every VSRAM element, SREG word, dumped region, CSR and
   counter recorded, once as the image's own prefill/decode loop -- so it is
   judged on the ids it generates as well as on the state it leaves, and a
   difference is reported the way layer 4 reports one: the position, the
   descriptor, the field and the first element it is in.

   - **Timing.** `--lat` 1, 32 and 200, one returned beat every two cycles, and
     one simulation thread against four. `--lat 200` is past the memory model's
     64-beat in-flight window, so the weight port is bandwidth-bound rather
     than saturated and the cycle count moves by more than a factor of three
     across the set while no value moves at all; four threads take the same
     cycles as one. Layer 4 holds its own records to the first four settings.
   - **Port width.** One model compiled at `WB=64` and at `WB=128` -- separate
     compiles of the same weights to the same context, with a different tiling
     and a zero-padded last tile -- generates the same ids, dumps the same int32
     logit for every vocabulary entry, and produces the same state descriptor by
     descriptor over a whole decoder layer, in different numbers of cycles. That
     is what the fallback configuration of `docs/PERFORMANCE.md` rests on.

     Both programs run at both widths. The bring-up program takes the embedding
     table and the tied head; the **layer** program takes the whole decoder layer
     of the compiled `decode.prog` -- the norm and its quantize, the `Wqkv`
     matrix-vector, the rotation, the per-head quantizes, both `KVWRITE`
     descriptors, the `K^T` matrix-vector whose length comes from the position,
     the softmax, the value matrix-vector, the output projection, the post norm,
     the gate-and-up matrix-vector, the `VSILUMUL` and the down projection -- at
     every position the two widths name in common, in order and on one machine.
     Compared after every descriptor: every element of the used VSRAM range, all
     32 SREG words of every bank, `PC`, `STATUS`, the ARGMAX registers, the four
     event counters and `DESCRIPTORS`.

     Two things about a run are the port width's own and are not compared across
     widths. `MACS` and `WT_BYTES` are the first: a partial last tile is padded
     to the width and streamed, so the padding is weight bytes the port carried
     and products the array took, and the two counters differ by exactly that.
     The KV cache is the second: both halves are stored in the weight tiling of
     the port width (`compiler.kv_sizes`), so its address, its size and the order
     of its bytes belong to the compile. What the cache holds is compared through
     the arithmetic that reads it -- the passes run in order on one machine, so
     the attention output of every position after the first is what the positions
     before it wrote. `sw/tests/test_determinism.py` measures the exemption
     rather than assuming it: over the layer program the two widths differ on
     those two counters and on nothing else, and a single flipped byte of a
     layer-0 gamma row or of the embedding table is reported at the VSRAM element
     or the dumped logit it moves.

     `make determinism` runs both programs on a random tiny model compiled at
     each width, `make determinism-model` runs the layer program on a compiled
     image at both widths, and `make bringup-sweep` holds both widths to the ISA
     simulator as well.
   - **Starting state.** `make determinism-x` builds the harness with
     Verilator's `--x-initial unique` (`XINIT=unique` in `sim/verilator`) and
     starts every variable no reset reaches at zeros, at ones, and at a value
     drawn from each of eight seeds, one run per setting, over the layer and the
     bring-up programs. Such a run is judged on what the program produces -- the
     ids, the memory it writes and the registers the host reads -- because the
     vector SRAM and the scale registers hold whatever the start put in the
     elements the program has not written.

     Twenty-one of the target's twenty-two runs produce the baseline's values.
     The twenty-second stops on a simulation-only check rather than on a value:
     `rtl/qcore_lut_interp.sv` holds its interpolated value to `[0, 65535]`
     whenever `in_valid` is high, and `rtl/qcore_vpu_scalar.sv` raises
     `in_valid` on both interpolators for every scalar request while enabling
     only the table that request needs, so the other one still holds the sample
     registers the start put there. The bring-up program's first scalar request
     is a reciprocal, so the rsqrt samples are that start value, and one of the
     eight seeds puts a pair there that leaves the range. The target reports the
     run, the module and the values it stopped on and fails; the nightly
     `x-initial-unique` job is held on it and its gate says so. The other check
     that reads unreset state, the `ifndef SYNTHESIS` block of
     `rtl/qcore_row.sv`, is qualified by `rst` like the registers beside it and
     holds from every one of the ten starts.
6. **Quality (golden vs fp32, not RTL).** `uv run quettos check <alias>`
   scores the calibration set teacher-forced (Qwen 1314 tokens and 1308 scored
   positions, SmolLM2 1181 and 1175): paired
   delta-NLL +/- SE, KL, top-1 and PPL for W8A16 and W8A8 on both models,
   written to `models/<name>/quality.json` and tabulated in `NUMERICS.md`.
   `--no-qk-smoothing` scores a build with every smoothing factor forced to 1
   into the `-nosmooth` rows of the same file, on the same ids against the same
   reference, so the fold that conditions int8 K carries a measured ablation
   rather than an assertion. The slow test re-evaluates every row of the file
   against a fresh run of the build it names.
   CI gate on SmolLM2 W8A16 (`KL <= 0.02 nats`, `top-1 >= 93%`; measured
   0.0048 and 95.66%); Qwen is reported (W8A16 measured `KL 0.0124`, `top-1
   95.57%`, with the int8 K cache conditioned by K-centering and the
   pairwise Q/K smoothing fold, see the K-centering section of
   `NUMERICS.md`). Saturation and shift-error
   counters are zero on every reported run. The table covers the calibration
   set; `docs/ROADMAP.md` lists the longer WikiText-2 run.
7. **Perf counters.** The harness asserts `busy = mac_active + stall_mem +
   stall_vpu + stall_kv + stall_seq + stall_drain`; `RD_BYTES == RD_BEATS * WB`;
   the write counters equal the C++ memory model's own counts and `RD_BEATS`
   trails it by at most the fetch unit's two outstanding bursts per token, which
   is what a burst still in flight at a `HALT` snapshot costs; `WT_BYTES` ==
   `layout.json` `traffic.<program>.wt_bytes`; `MACS` == `traffic.<program>.macs`
   plus the attention term of `traffic.attention` at the token's position
   (`ISA.md`, PERF table); `isa_sim.py` counts the same three and is the
   reference for them, compared per descriptor by `compare.py`. A run fails on
   any `SAT_*` or `ERR_*` event unless `--allow-sat` is given.
8. **CI.** `.github/workflows/ci.yml` (ubuntu-latest, 4 vCPU) and
   `nightly.yml`: `YosysHQ/setup-oss-cad-suite@v4` pinned to a dated release
   (tool versions printed into `syn/reports`), `actions/cache` for the suite,
   the harness object directory (keyed on `rtl/*.sv`, `rtl/*.svh`, the
   `rtl/gen/*.hex` lookup-table images -- a regenerated table changes the
   design, since `$readmemh` loads it into the ROMs -- and the C++), uv without
   torch, and on nightly the Hugging Face download.
   Six PR jobs with `timeout-minutes` each: the three-parser lint,
   `make style` and pytest, then the
   cocotb block tests, gate-level equivalence (`make gatesim`), the synthesis
   run (`make synth`: every block and the whole core in both configurations,
   with every `syn/reports/*.md` regenerated and checked against the run; a
   report written by a different Yosys build is held to its parameters and its
   hard-block inventory, since LUT packing and path length belong to the build),
   the tiny-configuration bring-up comparison (`bringup-tiny`: `make
   harness-csr` then `make bringup`; `docs/PERFORMANCE.md` holds that target's
   local wall clock) and the demo on the small model (`demo-smollm2`: `make
   demo`, the whole pipeline a clean clone runs -- checkpoint, quantize,
   compile, harness, the complete 30-layer SmolLM2-135M-Instruct generating on
   `qcore_top`, and layer 5's checks over the finished run), the last five gated
   on the first. The checkpoint and the harness object directory are cached, so
   the Hub fetch and the Verilator build are paid once. Nightly holds the long
   runs. Three of them run: `e2e-qwen` and `e2e-smollm2` quantize and compile the
   checkpoint, generate from the image's own prompt on `qcore_top` and on
   `isa_sim` and compare the ids, compare the state after every descriptor of
   both programs of the complete model (`make stepcmp-model`, layer 4), then run
   every descriptor of the same programs against the integer golden model --
   `RTL == isa_sim == golden` on a complete model as ids and as state, one
   generated token on Qwen and four on SmolLM2, the
   compiled program's six vector opcodes on `qcore_vpu_top`; `determinism-model`
   compiles SmolLM2 at `WB=64` and at `WB=128` and runs layer 5's timing and
   width checks over the complete model, the width half descriptor by descriptor
   over a whole decoder layer. Three are held by
   `if: false`, each gate naming what it waits for: the ECP5 stat and nextpnr
   fmax (`syn/` carries no ECP5 script), the `--x-initial unique` determinism
   run (`make determinism-x` runs it and reports the one run of its twenty-two
   that stops, on the `rtl/qcore_lut_interp.sv` range check reading the table a
   scalar request did not address -- layer 5, **Starting state**), and
   the quality regeneration over the wider set (`uv run quettos check` scores
   the calibration corpus). The whole-core synthesis is a PR job, so nightly
   carries no synthesis of its own beyond ECP5.

## What is checked into `models/<model>/`

`calib.json`, `quality.json` and `expected_tokens.json`. The last holds the
golden model's greedy continuation of `prompts/chat_short.json` and
`prompts/tool_call_weather.json` (20 tokens or until an end-of-sequence id)
with the SHA-256 of each id list,
written by `uv run quettos golden <alias> --write-expected` and reproduced by
`sw/tests/test_golden.py`; the RTL and the ISA simulator must emit exactly
these ids. `image.bin` stays in the gitignored `build/`. A clean clone must
reproduce the sha256s (checked on CI and on a second machine).
