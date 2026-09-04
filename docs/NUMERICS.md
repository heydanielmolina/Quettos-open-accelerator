# Numerics

Single source of truth: `sw/quettos/numerics.py`. The RTL knows no fixed-point
format; every shift, exponent and constant is a compiler value in a descriptor
field. `golden.py`, `quality_model.py`, `isa_sim.py`, `lutgen.py` and the
cocotb tests all import the same primitives.

Status: this is the specification. **Measured maxima and the frozen per-class
formats are TBD** until calibration freezes the numerics. Where a value is expected rather
than measured it is marked **estimate**.

## Primitives

- `round_shift(x, s) = (x + (1 << (s-1))) >>> s` for `s >= 1`, `x` for `s = 0`
  (round half toward +inf, arithmetic shift).
- `sat_N(x)` saturates to signed N bits.
- All RTL intermediates are `logic signed`; every operand is wrapped in
  `$signed()`; u16 mantissas are handled as 17-bit signed.

## sfloat

`sfloat = { u16 m in [2^15, 2^16), i8 e }`, value `m * 2^e`. Canonical zero is
`{m = 0, e = 0}`.

`sfloat_mul(a, b)`: `p = m_a * m_b`; `m = round_shift(p, 16)`; if `m >= 2^16`:
`m >>= 1, e += 1`; elif `m < 2^15`: `m <<= 1, e -= 1`; `e += e_a + e_b + 16`.
Property-tested: `m` in range for all pairs.

## Weights

int8 symmetric round-to-nearest per output channel, `Sw = absmax / 127`, RNE in
float64 offline; tiled `[N/WB][K][WB]`; zero-padded partial tiles allowed.
Per-channel meta 8 B little-endian `{i32 bias_q, u16 Sw_m, i8 Sw_e, u8 pad=0}`
(`bias_q` in the `FRAC_QKV` domain for QKV; o_proj carries the folded V bias;
else 0). No block scales in v1. SmoothQuant OFF by default (export flag exists
for ablation).

## Activations into GEMV

int16 symmetric per-token dynamic (**W8A16 default**; the lane is 8w x 16a so
this costs nothing). `VQUANT`: `a` = tracked absmax or own pass; `a_hi` = top 16
bits after LOD; `inv = recip_LUT` rounded toward zero so `|q| <= 32767` always;
`q = sat16(round_shift(x * inv, shift(e)))`;
`Sx = sfloat_mul(sfloat_norm(a), C_INV_32767)`, optionally `sfloat_mul`'ed by a
descriptor constant (`log2e/8` for q). Zero vector: `Sx = 0, q = 0`. int8 mode
only for K/V into the cache and the W8A8 ablation.

## Accumulator

40-bit signed (`ACC_W` parameter). 8x16 product is 24 b; `K <= 4864` int16
needs <= 35 b; PV over 8192 tokens needs 36 b.

## Requant (1 output per cycle)

```
t = sat40(round_shift(acc * Sw_m, s1))        s1 = descriptor sh0 (default 16; compiler lowers it to keep S >= 0)
y = sat32(round_shift(t * Sx_m, S))           S  = sbias - (Sw_e + Sx_e), sbias = descriptor sh1
```

Hardware clamps `S` to `[0, 63]` and counts any clamp in `ERR_SHIFT` (must be
0; the compiler proves the S range statically). If meta `m == 0` (padded
channel or token): output 0, no shift, no ERR/SAT count. Then
`y = sat32(y + bias_q)`; if `accumulate`: `y = sat32(y + old)`. Saturations are
counted in `SAT_REQ`.

## Per-tensor-class formats

`FRAC_X, FRAC_QKV, FRAC_S, FRAC_GU, FRAC_H, FRAC_CTX` are chosen by
`quantize.py` from calibration maxima with >= 2 bits of headroom over ~1k
tokens (including ChatML specials, `<tools>` JSON, newlines, the first token),
written into descriptor shifts, and published here.

| Class | Qwen2.5-0.5B-Instruct | SmolLM2-135M-Instruct | Calibration absmax |
|---|---|---|---|
| `FRAC_X` | TBD | TBD | measured maxima: TBD |
| `FRAC_QKV` | TBD | TBD | measured maxima: TBD |
| `FRAC_S` | TBD | TBD | measured maxima: TBD |
| `FRAC_GU` | TBD | TBD | measured maxima: TBD |
| `FRAC_H` | TBD | TBD | measured maxima: TBD |
| `FRAC_CTX` | TBD | TBD | measured maxima: TBD |
| logits | 16 (fixed) | 16 (fixed) | n/a |

Runtime counters `SAT_REQ`, `SAT_VPU`, `ERR_SHIFT`, `ERR_BOUNDS` are printed
per run; `make demo` fails on any non-zero counter unless `--allow-sat`.
`SAT_VPU` counts only saturations outside VQUANT's expected clip.

## RMSNorm

`amax -> sh = max(0, bitlen(amax) - 15)`; `ss = sum((x >> sh)^2)` (48-bit);
`ss' = ss + (eps_c >> 2sh)`, `eps_c = round(eps * d * 2^(2*FRAC_X))` from
`imm32`; LOD -> `m in [1, 4)` as two 256-entry segments (exponent parity);
`R = rsqrt_LUT(m)` (u16 Q1.15); `Rc = sfloat_mul(R, sqrt(d) constant)`;
`xhat = round_shift(x * Rc_m, S1)`; `y = sat32(round_shift(xhat * gamma[j], G))`,
gamma int16 with a per-tensor exponent; absmax tracked into SREG.

## RoPE

cos/sin int16 Q1.14 `table[pos][32 pairs]`, rotate_half convention:

```
q'_i      = sat32(round_shift(q_i * cos_i - q_{i+32} * sin_i, 14))
q'_{i+32} = sat32(round_shift(q_{i+32} * cos_i + q_i * sin_i, 14))
```

The table is generated once by `lutgen.py` with exact angle reduction (mpmath,
or exactly reduced `math`), checked in as `rope_theta1e6_2048.npy` and
`rope_theta1e5_2048.npy` (256 KB each); compiler, golden and isa_sim load the
checked-in file; sha256 recorded in `layout.json`.

## K-centering

`k' = k - c[layer][kvh]` (`VSUBC`, `c` = mean post-RoPE k over calibration
tokens). Exact for softmax since `q . c` is constant over `t`; conditions int8
K against Qwen's large k_proj biases. **Calibration gate**: int8 K vs fp K on 2k
tokens; RTL-free fallback = two K=32 half-dot GEMVs with separate per-token
scale arrays (accumulate flag). Result: TBD.

## KV cache

K int8 per (token, kv head) + sfloat scale, transposed in 64-token tiles; V
int8 (`v_raw`) per (token, kv head) + scale, row-major; 8 B meta each. The
compiler writes zero K/V meta for the whole `MAX_CTX` range into `image.bin`;
the harness zero-fills the KV region at sequence start and before prefix
restore; `compare.py` masks score elements `>= len`.

## Scores

`dot64(q_int16, k_int8)` requantized with per-token `Sk_t` and `Sq * log2e/8`
-> log2-domain score in `FRAC_S` (`1/sqrt(64)` exact as exponent -3).

## Softmax

`m = max s_t`; `d_t = clamp(s_t - m, -16 << FRAC_S, 0)`; `n = -(d_t >> FRAC_S)`,
`f` = fraction (8 index + 8 interpolation bits); `e_t = exp2_LUT(f)` (u16 Q1.15
in `[32768, 65535]`) `>> n` (`n >= 16 -> 0`); `sum` u32;
`inv = recip_LUT(LOD(sum))`; `p_t = sat(round_shift(e_t * inv_m, k))` u16 Q1.15
(`len = 1` gives exactly 32768); `e_max` = max V exponent in the row;
`w_t = sat16(round_shift(p_t * Sv_m[t], 16 + e_max - Sv_e[t]))` in `[0, 32767]`,
`w_t = 0` for `t >= len`; `SREG_out = sfloat(2^(1 + e_max))`.

**Calibration gate**: histogram of `(e_max - Sv_e)` per (layer, head); if p99 > ~6,
switch to per-(layer, kv head) static V scales or the half-dot pattern. Result:
TBD.

## SiLU

`sig = sigmoid_LUT(|g| clamped to [0, 16), 512 entries, u16 Q1.15,
interpolated)`, `sig(-x) = 32768 - sig(x)`; `silu = round_shift(g * sig, 15)`;
`h = sat32(round_shift(silu * u, sh_h))`; absmax tracked.

## LUT ROMs

exp2 256, sigmoid 512, rsqrt 2x256, recip 256 entries of `(v16, dv16)`;
`out = v + ((dv * frac8) >> 8)`; interpolation error <= 2^-16 (**verified by
pytest against float64**). 1-D initialized memories with registered
reads, dual-port serving 2 lanes each, `$readmemh` from checked-in
`rtl/gen/*.hex` via a `ROM_FILE` string parameter with no default, set as an
absolute path from the Makefile/.ys. CI lints that no literal `$readmemh("...")`
exists.

## Logits

`FRAC = 16`; argmax strict-greater ascending, so ties resolve to the lowest id.
Greedy only in the bit-exact contract.

## Quality expectations (estimates)

All rows below are **estimates** to be replaced by `uv run quettos check`
measurements before publication (overnight quality run):

| Model | Config | PPL delta | top-1 vs fp32 | KL (nats) |
|---|---|---|---|---|
| Qwen2.5-0.5B | W8A16 + int8 KV + LUT nonlinearities | +1-3% (estimate) | 94-97% (estimate) | 0.01-0.03 (estimate) |
| SmolLM2-135M | same | +1-2% (estimate) | 95-98% (estimate) | TBD |
| both | W8A8 ablation | +2-6% (estimate) | 90-95% (estimate) | TBD |

CI gate on SmolLM2: `KL <= 0.02 nats`, `top-1 >= 93%`. Qwen is reported, not
gated (expected `KL <= 0.03`, `top-1 >= 94%`). Saturation counters must be zero
on every reported run.
