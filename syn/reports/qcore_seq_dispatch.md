# qcore_seq_dispatch synthesis (xc7)

Every number below is read back out of `build/synth/synth_seq_dispatch.log` by
`scripts/synth_report.py`, which `make synth` runs. The run comes first and this page is
written from it, so the two cannot disagree. `--check` regenerates the page and requires it
back byte for byte on the Yosys build named below; another build has its own LUT packing and
path lengths, so it is held to the parameters and the hard-block inventory.

Tool: `Yosys 0.65 (git sha1 aec814bdf3071f7e0fd0fbe43f7f711e99d01e24, clang++ 21.0.0 -fPIC -O3)`

```sh
mkdir -p build/synth
yosys -q -l build/synth/synth_seq_dispatch.log -s syn/synth_seq_dispatch.ys
```

The cell counts are the `stat -tech xilinx` table after `synth_xilinx -family xc7`. A block
synthesized on its own has its ports on pads, so `IBUF` and `OBUF` are in its totals; inside
`qcore_top` they are internal wires.

The path depth is `ltp -noff` over the LUT fabric, cutting at the flops (`FDRE`, `FDSE`),
the clock buffer (`BUFG`), the I/O buffers (`IBUF`, `OBUF`) and the hard blocks (`DSP48E1`,
`RAMB36E1`, `RAMB18E1`, `RAM32M`, `RAM64M`, `RAM128X1D`), so it counts logic levels between
registers and means the same thing in every block. Each path below lists its two endpoints
and the named signals and source lines between them, with the position of each along the
path.

## Demo configuration

Parameters: `WB = 64`, `B_MAX = 1`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 55 |
| `DSP48E1` | 1 |
| `FDRE` | 515 |
| `IBUF` | 398 |
| `INV` | 15 |
| `LUT1` | 12 |
| `LUT2` | 141 |
| `LUT3` | 123 |
| `LUT4` | 79 |
| `LUT5` | 42 |
| `LUT6` | 113 |
| `MUXF7` | 33 |
| `MUXF8` | 7 |
| `OBUF` | 616 |

2151 cells in total: 510 LUTs, 515 flops, 55 `CARRY4`, 1014 I/O pads. Yosys estimates 357
LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 1 | `$mul` at `rtl/qcore_seq_dispatch.sv:299` |

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 43 cells.

```
 0  $techmap10152$abc$8657$auto$blifparse.cc:557:parse_blif$9156.A
10  pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:242
22  n_ru [24]
26  rtl/qcore_seq_dispatch.sv:246
36  n_pad [24]
43  wt_c [37]
```

## Tiny configuration

Parameters: `WB = 16`, `B_MAX = 2`.

| Cell | Count |
|---|---|
| `BUFG` | 1 |
| `CARRY4` | 80 |
| `DSP48E1` | 1 |
| `FDRE` | 576 |
| `IBUF` | 399 |
| `INV` | 14 |
| `LUT1` | 12 |
| `LUT2` | 203 |
| `LUT3` | 177 |
| `LUT4` | 145 |
| `LUT5` | 52 |
| `LUT6` | 98 |
| `MUXF7` | 34 |
| `MUXF8` | 5 |
| `OBUF` | 673 |

2470 cells in total: 687 LUTs, 576 flops, 80 `CARRY4`, 1072 I/O pads. Yosys estimates 472
LCs.

| Hard block | Count | Inferred from |
|---|---|---|
| `DSP48E1` | 1 | `$mul` at `rtl/qcore_seq_dispatch.sv:299` |

`memory_libmap` mapped no memory: the block holds its state in flops.

Longest topological path through the LUT fabric: 46 cells.

```
 0  $techmap21557$abc$19656$auto$blifparse.cc:557:parse_blif$20328.A
10  pos_p1 [32]
12  rtl/qcore_seq_dispatch.sv:243
13  rtl/qcore_seq_dispatch.sv:242
22  n_ru [24]
27  rtl/qcore_seq_dispatch.sv:246
37  n_pad [24]
46  wt_c [36]
```

## Notes (hand-written)

The dispatcher is a decode around one register. The 256-bit descriptor in flight drives most
of the command bundle by continuous assignment, so the bundle costs one register rather than
one per field, and the wide output is `OBUF` pads that become internal wires inside
`qcore_top`. The depth is the POS derivation: `POS + 1` is compared against the descriptor's
`N`, rounded up to a multiple of `WB`, compared again and rounded once more before it reaches
the `WT_BYTES` sum -- carry chains and comparators in series, and the same chain that sets the
depth of the whole core.
