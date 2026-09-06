# Numerics

Single source of truth: `sw/quettos/numerics.py`. Every integer operation the
RTL performs is defined there once, in plain Python integers and numpy `int64`
arrays; every consumer of these numbers (the table generator `lutgen.py`, the
quantizer `quantize.py`, the program constants `program.py`, the golden model
`golden.py` and the hardware unit tests) imports the same functions, and the
RTL mirrors them bit for bit. The RTL knows no fixed-point format: every shift,
exponent and constant is a compiler value in a descriptor field. This document
restates that module; where the two disagree, the module is right and this
file needs fixing.

Everything below is a definition from `numerics.py` or a value measured by a
command in this repository (`uv run quettos calibrate <alias>`,
`uv run quettos check <alias>`, `uv run pytest -q sw/tests`).

## Primitives

- `round_shift(x, s) = (x + (1 << (s-1))) >> s` for `1 <= s <= 63`, `x` for
  `s = 0` (round half toward +inf, arithmetic shift). It is the only rounding
  operation in the datapath; `s = 63` is the top of the requant clamp range.
- `sat_N(x)` saturates to signed N bits. Every saturation is counted
  (`Stats.sat` in software, the `SAT_*` CSRs in hardware) so the golden model
  and the RTL counters can be compared.
- `Stats` holds three counters: `sat` (sat40/sat32 events), `err_shift`
  (a requant stage-2 shift outside `[0, 63]`; must stay 0 on a correct
  program) and `clip` (VQUANT clips at `+-(2^(w-1) - 1)` and the softmax weight
  clip at 32767; expected, not a fault).
- Vectors are `int64` arrays, scalars are Python ints; every intermediate
  product fits in 63 bits by construction (see the width notes below).
- All RTL intermediates are `logic signed`; every operand is wrapped in
  `$signed()`; u16 mantissas are handled as 17-bit signed.

## sfloat

`sfloat = { u16 m in [2^15, 2^16), signed e }`, value `m * 2^e`. The canonical
zero is `{m = 0, e = 0}` and is the only encoding with `m` outside the mantissa
range. Weight meta stores `e` as an `i8`.

- `sfloat_from_int(a, e)`: exact when `a` has at most 16 significant bits
  (`{a << (16 - bitlen(a)), e + bitlen(a) - 16}`); otherwise
  `m = round_shift(a, bitlen(a) - 16)` with a `2^16` rounding overflow folded
  into the exponent (`{2^15, e + 1}`).
- `sfloat_from_float(x)` (offline only: weight scales, constants):
  `x = f * 2^k` with `f` in `[0.5, 1)` from `math.frexp`,
  `m = floor(f * 2^16 + 0.5)`, `e = k - 16`, same overflow fold. Deterministic
  across platforms because `frexp` is exact.
- `sfloat_mul(a, b)`: `p = m_a * m_b` lies in `[2^30, 2^32)`. If `p >= 2^31`:
  `m = round_shift(p, 16)`, `e = e_a + e_b + 16`; otherwise
  `m = round_shift(p, 15)`, `e = e_a + e_b + 15`. A rounding overflow to `2^16`
  becomes `{2^15, e + 1}`. Zero times anything is the canonical zero. There is
  one rounding, so the result is the correctly rounded (half-up) product;
  property-tested against exact rationals on every edge-mantissa/exponent
  combination and on 200k random pairs.

## Weights

int8 symmetric per output channel, rounded half up in float64 offline
(`quantize_rows_int8`): for each row `Sw = sfloat_from_float(absmax / 127)` is
encoded first, then `q = round_half_up(w / Sw)` clipped to `[-127, 127]`, so
the dequantized scale is exactly the one the hardware uses. An all-zero row
gets `q = 0` and the zero scale. Rows are grouped the way the programs stream
them: `wqkv = [q; k; v]`, `wgu = [gate; up]`, `wo`, `wdown`, and the tied
embedding / LM-head table row by row. Tiled `[N/WB][K][WB]`; zero-padded
partial tiles allowed. Per-channel meta is 8 B little-endian
`{i32 bias_q, u16 Sw_m, i8 Sw_e, u8 pad = 0}`. No block scales; the one
offline rescaling is the pairwise Q/K smoothing fold of the K-centering
section, which leaves `q . k` unchanged.

- `bias_q`: `to_fixed(bias, FRAC_QKV)` (round half up, saturating to int32)
  for the Q and K rows of `wqkv`; 0 for the V rows.
  `wo.bias_q = to_fixed(W_o @ tile(b_v), FRAC_X)` carries the V bias folded
  into `o_proj`, where `tile` repeats each KV head's bias over the query heads
  it serves; softmax weights sum to one, so
  `sum_t p_t (v_t + b_v) = sum_t p_t v_t + b_v` and the fold is exact. Every
  `bias_q` is 0 on models without QKV biases (SmolLM2).
- RMSNorm gamma: int16 with one per-tensor exponent `e <= 0`
  (`quantize_gamma`: the smallest `e` such that `max |q| <= 32767`,
  `q = floor(gamma / 2^e + 0.5)`; values above 32767 have no encoding and are
  rejected).
- K-centering rows: int32 in `FRAC_QKV` (`to_fixed`).
- Constants: `eps_c = round(eps * d * 2^(2 FRAC_X))` (Qwen `3848291`, SmolLM2
  `1546188`); `sqrt_d = sfloat_from_float(sqrt(d))` (Qwen `{61303, -11}`,
  SmolLM2 `{49152, -11}`); `log2(e)/8 = {47274, -18}`.

`uv run quettos quantize <alias>` writes all of this to
`build/quant/<name>.npz` (int8/int16/int32 arrays plus a JSON manifest); the
tests check every integer against the `numerics.py` primitive that defines it.

## Activations into GEMV (VQUANT)

int16 symmetric per token, dynamic (**W8A16 default**; the lane is 8w x 16a so
the wide activation costs nothing). int8 only for K and V into the cache and
for the W8A8 ablation. `quant(x, w, FRAC_in)` with `w` in `{8, 16}`:

```
a     = absmax(x)                       # or the absmax tracked by the producing op
a_eff = a + (a >> (w-1)) + 1            # the absmax element maps to 2^(w-1) - 1, not 2^(w-1)
a_eff = a_hi * 2^e_a,  a_hi in [2^15, 2^16),  e_a = bitlen(a_eff) - 16  (negative: shifted left)
Sx    = {a_hi, e_a - (w-1) - FRAC_in}   # a_eff / 2^(w-1) in real units, exact in sfloat
inv   = recip_q15(a_hi)                 # 1/m in Q1.15, m = a_hi / 2^15
q     = round_shift(x * inv, 31 + e_a - w)
q     = clip(q, -(2^(w-1) - 1), 2^(w-1) - 1)   # counted in `clip`, not a fault
```

since `x * 2^(w-1) / a_eff = x * inv * 2^-(31 + e_a - w)`. Using `a_eff` gives
the same `2^(w-1) - 1` levels as an `absmax / (2^(w-1) - 1)` scale without a
division, and the `+ 1` keeps the rounding of the absmax element below the
half-way point for every `a`. The clip is reachable only at the absmax element
and only through the reciprocal table's error: never for `w = 8`; for `w = 16`
the table's ~1 LSB error spans the two-level margin, so roughly one vector in
four clips its absmax element by one level (harmless, counted in `clip`; a
non-zero clip counter is the normal state). A zero vector gives `q = 0` and
the zero scale. A descriptor sfloat constant (`log2(e)/8` for q) multiplies
the scale through `sfloat_mul`. `quant_groups` applies the same per group of
64 elements (per head) for q, K and V. Widths: `x` int32, `inv` 16 bits,
product 47 bits.

## Accumulator

40-bit signed (`ACC_W` parameter). An 8x16 product is 24 b; `K <= 4864` int16
needs 36 b; PV over 8192 tokens needs 36 b.

## Requant (1 output per cycle)

The real value of an accumulator is `acc * Sw * Sx`; the output class has
`FRAC_out` fraction bits and the compiler sets `sbias = -(FRAC_out + s1)`
(plus 24 for EMBED, below):

```
t = sat40(round_shift(acc * Sw_m, s1))     s1 = descriptor sh0
S = sbias - (Sw_e + Sx_e)                  sbias = descriptor sh1; hardware clamps S to [0, 63] and counts ERR_SHIFT
y = sat32(round_shift(t * Sx_m, S))
y = sat32(y + bias_q)
y = sat32(y + old)                         only with the accumulate flag (fused residual add)
```

`ERR_SHIFT` must be 0: the compiler proves the `S` range statically with
`requant_shift(Sw, Sx, sbias) = sbias - (Sw_e + Sx_e)`. If either scale is the
canonical zero (padded channel or padded token) the dequantized term is
exactly 0 with no shift and no ERR/SAT event; the bias and accumulate adds
still apply. Saturations are counted in `SAT_REQ`. Widths: `acc` 40 bits,
mantissas 16 bits, both products 56 bits.

Stage-1 rule (`choose_s1(acc_bits, FRAC_out, Sw_e_max, Sx_e_max)`, with
`acc_bits` the signed width of the accumulator values the GEMV can produce):
`t` must fit 40 bits, so `s1 >= acc_bits + 16 - 40`; stage-1 rounding
contributes `0.5 * Sx_m * 2^-S` output LSBs, so the smallest reachable `S`
should stay at or above 16 (`<= 0.5` LSB). `S` is smallest at the largest
reachable exponents, hence `s1 <= -(FRAC_out + Sw_e_max + Sx_e_max) - 16`. The
rule returns the largest `s1 <= 16` that meets the 40-bit bound and, when
feasible, the precision bound; the compiler checks the `[0, 63]` window with
`requant_shift` at the smallest reachable exponents. When both bounds hold the
output is within one LSB of the exact value (worst case
`0.5 + 0.5 * Sx_m * 2^-16`, property-tested).

EMBED dequantizes an int8 embedding row into `FRAC_X` through the same pipe:
`acc = q << 24`, `Sx = 1.0 = {2^15, -15}`, `sbias = -(FRAC_X + s1) + 24`, with
`8 <= s1 <= 24` so the shifted product fits 40 bits and drops only zero bits;
the result is a single rounding of `q * Sw * 2^FRAC_X`.

### Program constants

`sw/quettos/program.py` (`program.build(model)`) derives one `(s1, sbias)`
pair per GEMV class of a quantized model from these rules: `acc_bits` from `K`
and the operand widths (`acc_bits_for`), the weight exponent window over the
non-zero rows of every layer, the activation exponent window between a 1-LSB
vector and the calibration absmax of the class (which the quantizer stores in
the model as `extra["absmax"]`), and two checks of the stage-2 shift `S`: the
window at the calibration extremes is the precision target (`choose_s1` sees
the calibration maximum plus one octave of activation margin, so `S >= 16`
wherever the 40-bit bound allows), and the hard minimum, evaluated with every
operand at the int32 saturation bound of its class, must be at or above 0,
which proves the hardware clamp `[0, 63]` is never reached on any input,
calibration or not (`s1` is lowered when needed, as on the scores GEMV). The
golden model and the compiler share these values; the golden counts
`ERR_SHIFT` and `SAT`, and the tests, the quality rows and the CLI writers
require both to be 0. Values for the W8A16 programs (Qwen2.5-0.5B-Instruct /
SmolLM2-135M-Instruct):

| GEMV | `K` | `acc_bits` | `s1` | `sbias` | `S` window | hard minimum |
|---|---|---|---|---|---|---|
| EMBED | 896 / 576 | 32 / 32 | 16 / 16 | -8 / -6 | [32, 35] / [31, 33] | 32 / 31 |
| QKV | 896 / 576 | 33 / 33 | 9 / 12 | -25 / -28 | [17, 50] / [17, 42] | 10 / 5 |
| o_proj | 896 / 576 | 33 / 33 | 16 / 16 | -32 / -30 | [18, 42] / [17, 40] | 6 / 5 |
| gate\|up | 896 / 576 | 33 / 33 | 12 / 11 | -28 / -27 | [17, 44] / [17, 40] | 10 / 5 |
| down | 4864 / 1536 | 36 / 34 | 12 / 10 | -28 / -24 | [15, 45] / [15, 45] | 10 / 11 |
| LM head | 896 / 576 | 33 / 33 | 14 / 14 | -30 / -30 | [17, 43] / [17, 37] | 10 / 5 |
| scores | 64 / 64 | 29 / 29 | 9 / 9 | -25 / -25 | [18, 60] / [21, 60] | 0 / 0 |
| PV | 2048 / 2048 | 34 / 34 | 10 / 14 | -26 / -30 | [17, 40] / [17, 36] | 10 / 6 |

Every window lies in `[0, 63]` and every hard minimum is at or above 0. The
down GEMV is the one class where the 40-bit bound (`s1 >= acc_bits - 24`)
beats the precision bound, so its smallest window shift is 15; the scores GEMV
is the one where the hard bound lowers `s1` (both operands at int32 saturation
give exactly `S = 0`). The QKV row follows the weight window of the smoothed
`W_q` rows (Qwen's largest QKV weight exponent is -20, its smallest -30), the
scores row the exponents of the smoothed q and centered K, and the PV row the
`SREG` window derived from the `QKV` class maximum, which bounds the V scales.
The W8A8 programs lower `s1` by 7 to 8 on the weight GEMVs (int8 activations
carry a larger scale exponent), keep the smallest window shift at 15 (down)
and keep every hard minimum at or above 5 there, with scores at 0 as above.
The PV constants use `K = MAX_CTX = 2048` whatever the size of a golden run's
cache, so a short run executes the compiled program's shifts.

## Per-tensor-class formats

`FRAC_X, FRAC_QKV, FRAC_S, FRAC_GU, FRAC_H, FRAC_CTX` are chosen by
`uv run quettos calibrate <alias>` from the float32 reference forward
(`reference_np.py`) over a calibration set of about a thousand tokens and
written to `models/<name>/calib.json`, which the quantizer reads. The rule is
`FRAC = min(16, 29 - ceil(log2(absmax)))`, which leaves at least two bits of
headroom in an int32 (`2^(31 - FRAC) >= 4 * absmax`). `FRAC_X` covers the
residual stream and the RMSNorm outputs (class `XN`) together because the norm
writes back into the residual class; `FRAC_S` is sized from the centered scores
`q . (k - c)` that the hardware holds after K-centering (the raw `q . k` maximum
is recorded for reference, and on Qwen it is almost entirely the constant
`q . c` term), must come out at 16 (the softmax consumes a 16-bit fraction) and
calibration fails otherwise; `FRAC_QKV` also covers the centered K vectors;
logits are fixed at 16. The quantizer also requires `FRAC_GU >= 13` (sigmoid table index) and
`FRAC_H <= 2 FRAC_GU` (SiLU shift).

The calibration set is the three prompt files under `prompts/` plus three
passages of plain prose rendered through the model's chat template, so ChatML
specials, a `<tools>` JSON block, newlines and a first token are all present:
1314 tokens in 6 sequences for Qwen, 1181 for SmolLM2 (the SHA-256 of the id
lists is recorded in `calib.json`). Measured maxima (largest |value| over all
tokens and layers, in real units) and the resulting formats:

| Class | Qwen2.5-0.5B-Instruct | SmolLM2-135M-Instruct |
|---|---|---|
| `FRAC_X` (residual `X`, norm output `XN`) | 16 (`X` 1710.84, `XN` 336.443) | 14 (`X` 25982.0, `XN` 47.5371) |
| `FRAC_QKV` (Q/K/V after projection, bias, RoPE and the Q/K smoothing fold; centered K 16.56 / 15.95) | 16 (332.84) | 16 (24.3374) |
| `FRAC_S` (log2-domain scores of centered K, `q . (k - c)`) | 16 (90.1764; raw `q . k` 2466.37) | 16 (54.5107; raw 88.3937) |
| `FRAC_GU` (gate, up) | 16 (101.81) | 16 (78.5393) |
| `FRAC_H` (`silu(gate) * up`) | 16 (1817.01) | 16 (3164.41) |
| `FRAC_CTX` (attention output before `o_proj`) | 16 (12.9361) | 16 (9.30108) |
| logits | 16 (31.17) | 16 (42.4522) |

The residual maxima come from a few outlier channels: on Qwen the per-layer
`X` maximum jumps from 6.8 to 827 at layer 2 and stays near 1700 through layer
21 before falling to 69 and 89 in the last two layers; on SmolLM2 it jumps from
384 to 25982 at layer 11 and stays there through layer 28 (10067 at the last
layer). `FRAC_X = 14` on SmolLM2 therefore keeps 2.3 bits of headroom, the
tightest class of either model; every other class keeps 3.4 bits or more. The
embedding rows are small (absmax 0.116 on Qwen, 1.078 on SmolLM2) and are
dequantized into `FRAC_X` by EMBED.

Runtime counters `SAT_REQ`, `SAT_VPU`, `ERR_SHIFT`, `ERR_BOUNDS` are printed
per run; `make demo` fails on any non-zero counter unless `--allow-sat`.
`SAT_VPU` counts saturations; the VQUANT and softmax clips are counted separately as clips.

## RMSNorm (VRMSNORM)

Input and output are int32 of class `FRAC_X`; `gamma_q` is int16 with a
per-tensor exponent `gamma_e <= 0`; `eps_c` (`imm32`) and `sqrt_d` (an sfloat)
are descriptor constants.

```
amax = absmax(x);  sh = max(0, bitlen(amax) - 15)
ss   = sum((x >> sh)^2)                               # <= 48 bits
ss'  = ss + (eps_c >> 2 sh)                           # (mean(x^2) + eps) * d * 2^(2 FRAC_X - 2 sh)
ss'  = m * 2^(2e), m in [1, 4):  L = bitlen(ss'); 2e = L-1 if L odd else L-2
R    = rsqrt_q15(m)                                   # m as Q2.16 in [2^16, 2^18); 1/sqrt(m) in Q1.15
Rc   = sfloat_mul(sfloat_from_int(R, -15), sqrt_d)    # sqrt(d)/sqrt(m); R is normalized first
xhat = round_shift(x * Rc_m, S1),  S1 = -(Rc_e + FRAC_X - sh - e)
y    = sat32(round_shift(xhat * gamma_q, G)),  G = -gamma_e
```

because `rsqrt(mean(x^2) + eps) = sqrt(d) * 2^(FRAC_X - sh - e) * R * 2^-15`.
`ss' = 0` gives an all-zero output. `S1` is non-negative whenever
`eps_c >= 2^(2 (FRAC_X + e_d))` with `e_d = floor(log2 sqrt(d)) - 15` (`2^10`
for `FRAC_X = 16` and `512 <= d < 1024`; both models use `eps_c > 1.5e6`); a
program that still produces `S1 < 0` gets the shift clamped to 0 and counted
in `ERR_SHIFT`. `|xhat| <= sqrt(d) * 2^FRAC_X * (1 + 2^-13)`, 21 bits at
`d = 896`. The output absmax is tracked for the VQUANT that follows. Measured against a float64 reference over 240 random
vectors (d = 896 and 576, `FRAC_X` 14 and 16, scales from 0.01 to 8000, single
outlier channels): worst relative error `0.23 * 2^-12` (test bound `2^-12`).

## RoPE (VROPE)

cos/sin are int16 Q1.14 in `table[pos][2][32]` (index 0 cos, index 1 sin) for
`head_dim = 64`, rotate_half convention; for each head and `i < 32`, with
`a = x[i]` and `b = x[i + 32]`:

```
a' = sat32(round_shift(a * cos_i - b * sin_i, 14))
b' = sat32(round_shift(b * cos_i + a * sin_i, 14))
```

`lutgen.py` generates the tables with mpmath at 128-bit precision
(`inv_freq_i = theta^(-2i/64)`, `angle = pos * inv_freq_i`, rounded half up),
checked in as `sw/quettos/tables/rope_theta1e6_2048.npy` (Qwen) and
`rope_theta1e5_2048.npy` (SmolLM2), 256 KB each, positions `0..2047`. Every
entry equals float64 `floor(cos(angle) * 2^14 + 0.5)` exactly, and the
float32 `cos`/`sin` of the reference forward agree with the table within 2 LSB
at every position.

## K-centering (VSUBC) and Q/K smoothing

`k' = sat32(k - c[layer][kvh])`, with `c` the mean post-RoPE K vector per
(layer, KV head) over the calibration tokens, stored as int32 in `FRAC_QKV`.
Exact for the softmax since `q . c` is constant over `t`; it conditions the
int8 K cache against Qwen's large `k_proj` biases.

Pairwise Q/K smoothing is the second conditioning step and is folded into the
weights offline (`quantize.smooth_qk`). For every (layer, KV head, RoPE pair
`p = (d, d + 32)`) the calibration pass computes one factor

```
s_p = (max |k_c|_p / max |q|_p) ^ (1/2)
```

with the maxima over both dimensions of the pair and all calibration tokens,
`k_c` the centered post-RoPE K of the head and `q` the post-RoPE Q of the
query heads it serves; each (layer, KV head) row is divided by its geometric
mean and clipped to `[1/16, 16]`. Rows `d` and `d + 32` of `W_k`, `b_k` and
the K-centering row are divided by `s_p`; the same rows of `W_q`, `b_q` of
every served query head are multiplied by it. `q . k` is unchanged in exact
arithmetic, and because `s_p` is constant over a RoPE pair the fold commutes
with RoPE, so it applies to the projections as stored. The factors are
recorded in `calib.json` under `qk_smoothing` (`[layers, kv_heads, 32]` at six
significant digits, applied as stored, with `alpha`, `cap` and the rule), and
the `QKV` and `K_centered` maxima of `calib.json` are those of the smoothed
model, so the program constants above are proven for the weights the hardware
streams. Measured factors: Qwen `[0.178, 13.6]`, where layer 0 spans the whole
range and layers 1 to 23 stay within `[0.44, 1.82]`; SmolLM2 `[0.54, 1.80]`.
The class maxima move from 221.4 to 332.8 (`QKV`) and from 57.7 to 16.6
(`K_centered`) on Qwen and from 22.1 to 24.3 and 15.5 to 16.0 on SmolLM2;
every `FRAC` is unchanged.

Gate (`k_centering_gate` in `calib.json`): post-RoPE K quantized to int8 per
(token, KV head) with an `absmax/127` scale, as projected (raw), after
centering, and after centering divided by the smoothing factors, and the
error of the resulting `q . k` scores over the causal window, in two units:
relative to the RMS of the exact scores, and absolute in the log2 domain the
softmax consumes (`q . k / sqrt(d) * log2(e)`; one unit halves a token's
weight), as RMS and as the largest single error:

| Model | int8 K | relative RMS | RMS (log2 units) | max (log2 units) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | raw | 0.503% | 0.613 | 22.2 |
| Qwen2.5-0.5B-Instruct | centered | 0.149% | 0.181 | 8.01 |
| Qwen2.5-0.5B-Instruct | centered + smoothed | 0.0257% | 0.0313 | 0.706 |
| SmolLM2-135M-Instruct | raw | 0.780% | 0.0657 | 0.615 |
| SmolLM2-135M-Instruct | centered | 0.306% | 0.0258 | 0.429 |
| SmolLM2-135M-Instruct | centered + smoothed | 0.290% | 0.0244 | 0.398 |

Both steps are enabled for both models. The relative figure flatters Qwen: its
raw scores are dominated by the constant `q . c` term (absmax 2466 against 90
for the centered scores), so the absolute error is the one to read. Per layer
(`layer_rms_error_log2_centered` and `_smoothed`), Qwen's centered error is
0.868 log2 units on layer 0 and between 0.020 and 0.069 on layers 1 to 23;
smoothing brings layer 0 to 0.054 and layers 1 to 23 to between 0.019 and
0.046. SmolLM2 stays between 0.009 and 0.034 on every layer (0.010 to 0.039
with centering alone). Layer 0 of Qwen, whose K pairs span two orders of
magnitude around the mean, is where the fold does its work; the Quality
section gives the end-to-end effect.

## KV cache

K int8 per (token, kv head) + sfloat scale, transposed in `WB`-token tiles; V
int8 (`v_raw`, without its bias) per (token, kv head) + scale, tiled with the
64 dimensions as channels (one row per token at `WB = 64`); 8 B meta each. The compiler writes zero K/V meta for the whole `MAX_CTX` range into
`image.bin`; the harness zero-fills the KV region at sequence start and before
prefix restore; the comparison (`isa_sim.compare_sequence`) masks score elements `>= len`.

## Scores

`dot64(q_int16, k_int8)` requantized with per-token `Sk_t` and `Sq * log2e/8`
-> log2-domain score in `FRAC_S` (`1/sqrt(64)` exact as exponent -3).

## Softmax (VSOFTMAX)

Over `scores[:len]` in class `FRAC_S` (log2 domain), with the per-token V
scales `Sv_t` of the row; returns int16 weights `w` (zero beyond `len`) and
`SREG_out` such that `sum_t w_t * V_t * SREG_out` reproduces
`sum_t p_t * Sv_t * V_t`:

```
m     = max s_t
d_t   = clamp(m - s_t, 0, 25 << FRAC_S)                  # distance in log2 units; beyond 25 the weight rounds to 0
n     = ceil(d_t / 2^FRAC_S);  g = n * 2^FRAC_S - d_t     # 2^-d = 2^-n * 2^(g / 2^FRAC_S)
e_t   = round_shift(exp2_q15(g as a 16-bit fraction) << 8, n)   # Q1.23; the max token gives 2^23
sum   = sum_t e_t                                         # <= 2^36 for 8192 tokens
sum   = sum_hi * 2^e_s;  inv = recip_q15(sum_hi)
p_t   = round_shift(e_t * inv, 7 + e_s)                   # Q1.23 probability; len = 1 gives exactly 2^23
e_max = max Sv_e over the row (zero scales excluded)
w_t   = round_shift(p_t * Sv_m[t], 24 + e_max - Sv_e[t])         in [0, 32768]; shift amounts above 40 saturate to 40 (w_t = 0)
SREG_out = 2^(1 + e_max) = {2^15, e_max - 14}
```

`g` becomes a 16-bit fraction by `g >> (FRAC_S - 16)` (or `<< (16 - FRAC_S)`).
Tokens whose V scale is the canonical zero get `w_t = 0` and do not enter
`e_max`; a row of all-zero scales returns zeros and the zero `SREG`. `w_t`
reaches 32768 only when `p_t = 1.0` (every other token at least 25 log2 units
below the maximum) and `Sv_m[t] = 65535`; that value is clipped to 32767 and
counted as a clip, like VQUANT's, never as a saturation. The exponential and
the probability carry eight extra fraction bits (Q1.23) so that tokens 13 to 23
log2 units below the maximum, the shape attention sinks produce, keep their
relative precision and the normalizer stays unbiased; the multipliers are 24 x
16 unsigned (25 x 17 in the RTL's signed convention), which the vector lanes
already have. The weights themselves are 16-bit: a token rounds to zero weight
when `p_t * Sv_t` is below `2^e_max` (half of `SREG_out`), and every weight
carries a rounding error of up to `2^e_max` in either direction, so a row of
`L` tokens has its total mass off by at most `L * 2^e_max / Sv_max <= L * 2^-15`
(3.1% at 1024 tokens in the worst case; wider weights are a roadmap item for
long contexts). Widths: `m - s_t` is a 33-bit signed difference before the
clamp, `n <= 25`, `e_t * inv` and `p_t * Sv_m` fit 39 bits, and the per-token
`w` shift lies in `[24, 54]` for real V scales (larger amounts saturate to
40 and give a zero weight). Measured against a
float64 softmax over random rows (lengths 1..600, `FRAC_S` 14/16/18, V
exponents spread over 6 octaves) and sink-dominated rows of 2048 and 8192
tokens, in units of `2^-15 * max Sv`, the per-token error stays within the
property-tested budget (`sw/tests/test_numerics.py`,
`sw/tests/test_numerics_edges.py`).

Gate (`v_scale_spread` in `calib.json`): histogram of `e_max - e_t` per
(layer, KV head, sequence) with `e_t = floor(log2(absmax(v_t)))` of the
bias-free V the cache stores:

| Model | samples | p50 | p99 | max |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 63072 | 1 | 3 | 5 |
| SmolLM2-135M-Instruct | 106290 | 1 | 3 | 7 |

A token `k` exponents below the row maximum keeps `15 - k` bits in `w_t`; with
p99 = 3 the per-token dynamic V scales stay and no static per-head V scale is
needed.

## SiLU (VSILUMUL)

`h = silu(g) * u` from class `FRAC_GU` (both inputs) into `FRAC_H`:

```
sig  = sigmoid_q15(g, FRAC_GU)             # Q1.15; sigmoid(-x) = 32768 - sigmoid(x); |g| >= 16 -> 1.0
silu = round_shift(g * sig, 15)            # class FRAC_GU
h    = sat32(round_shift(silu * u, 2 FRAC_GU - FRAC_H))
```

The sigmoid index is `floor(|g| * 32)` (bits `FRAC_GU - 5` and up; 512 entries
over `[0, 16)`) and `frac8` the next 8 bits, which needs `FRAC_GU >= 13`.
Widths: `g * sig` 47 bits, `silu * u` at most 62 bits. The output absmax is
tracked. Measured against float64 over 150 random vectors (`FRAC_GU` 13..20,
`FRAC_H` 8..20, inputs in `[-20, 20]`): worst relative error `0.10 * 2^-13`.

## Lookup tables

Four `(v16, dv16)` tables generated by `uv run python -m quettos.lutgen` with
mpmath at 128-bit precision and rounded half up, checked in as
`sw/quettos/tables/luts.json` (with a SHA-256 of the table payload in `meta`)
and as `rtl/gen/{exp2,sigmoid,rsqrt,recip}.hex` (one 32-bit word
`{v[15:0], dv[15:0]}` per line for `$readmemh`, `dv` two's complement).
`uv run python -m quettos.lutgen --check` regenerates in memory and reports any
difference. Entry `i` stores `v_i = round_half_up(f(x_i) * 2^15)` and the
forward difference `dv_i = v_{i+1} - v_i`, with `v_N` the value at the right
end of the domain. Interpolation is `out = v_i + round_shift(dv_i * frac8, 8)`
with `frac8` the 8 bits below the index.

| Table | Entries | Grid and values | Right end | Index / frac8 |
|---|---|---|---|---|
| `exp2` | 256 | `x_i = i/256` on `[0, 1)`, `f = 2^x`, `v` in `[32768, 65535]` | 65536 | `f16` in `[0, 2^16)`: bits 15..8 / bits 7..0 |
| `sigmoid` | 512 | `x_i = i/32` on `[0, 16)`, `v` in `[16384, 32768]` (32768 from entry 355 on) | 32768 | `\|x\| >> (FRAC - 5)` / `(\|x\| >> (FRAC - 13)) & 0xFF` |
| `rsqrt` | 2 x 256 | seg 0 `x_i = 1 + i/256` on `[1, 2)`; seg 1 `x_i = 2 + 2i/256` on `[2, 4)`; `v` in `(16384, 32768]` | 16384 | `m` as Q2.16: seg 0 bits 15..8 / bits 7..0; seg 1 `256 +` bits 16..9 / bits 8..1 |
| `recip` | 256 | `x_i = 1 + i/256` on `[1, 2)`, `v` in `(16384, 32768]` | 16384 | `a_hi` in `[2^15, 2^16)`: bits 14..7 / bits 6..0 `<< 1` |

Interpolation error against mpmath over the full input domain, in Q1.15 LSB:
`exp2` 0.98, `recip` 1.03, `rsqrt` 0.98 (segment 0) and 1.05 (segment 1),
`sigmoid` 1.25 at 13 fraction bits and below 2.0 at 16 and 20 (input bits
below `FRAC - 13` are dropped before the lookup). The tests bound the lookups
at 2 LSB (`2^-14` in real units; 2.25 for the sigmoid at 20 fraction bits) and
recompute every table entry independently.

Hardware: 1-D initialized memories with registered reads, dual-port serving 2
lanes each, `$readmemh` from the checked-in `rtl/gen/*.hex` via a `ROM_FILE`
string parameter with no default, set as an absolute path from the
Makefile/.ys. CI lints that no literal `$readmemh("...")` exists.

## Logits

`FRAC = 16`; argmax strict-greater ascending, so ties resolve to the lowest id.
Greedy only in the bit-exact contract.

## Float32 reference

`sw/quettos/reference_np.py` is the float32 numpy forward the integer numerics
are measured against (Hugging Face Llama/Qwen2 semantics: RMSNorm, rotate_half
RoPE with float32 `inv_freq`, grouped-query attention by repeating KV heads,
causal fp32 softmax, SwiGLU, tied LM head), with a recorder hook on every
intermediate. Against `transformers` fp32 eager attention on the same local
weights (`uv sync --group ref`, then
`uv run pytest -q sw/tests/test_reference_np.py`): argmax agreement 1.0 at
every position and max |logit difference| 3.3e-4 / 7.6e-4 on Qwen
(`chat_short` T=36 / `tool_call_weather` T=180) and 1.2e-4 / 1.2e-4 on SmolLM2
(T=37 / T=39); the test bound is 2e-2.

## Quality

Measured by `uv run quettos check <alias>` (`sw/quettos/quality.py`), which
runs the calibration set (the three prompt files under `prompts/` and three
prose passages of `calibrate.py`, rendered through the chat template)
teacher-forced through the integer golden model and the float32 reference and
scores every position that has a next token: top-1 agreement of the argmax,
mean `KL(p_fp32 || p_int)` from float64 log-softmax, the paired next-token
NLL difference `NLL_int - NLL_fp32` with its standard error, and the
perplexities `exp(mean NLL)`. The results, with per-sequence values, are
written to `models/<name>/quality.json`; the token ids are the ones hashed in
`calib.json`.

| Model | Config | Tokens | top-1 vs fp32 | mean KL (nats) | delta-NLL +/- SE (nats) | PPL fp32 -> int |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | W8A16 | 1308 | 95.57% (1250) | 0.0124 | +0.0026 +/- 0.0066 | 34.97 -> 35.06 |
| Qwen2.5-0.5B-Instruct | W8A8 | 1308 | 86.39% (1130) | 0.110 | +0.0111 +/- 0.0208 | 34.97 -> 35.36 |
| SmolLM2-135M-Instruct | W8A16 | 1175 | 95.66% (1124) | 0.00485 | -0.0023 +/- 0.0030 | 35.91 -> 35.82 |
| SmolLM2-135M-Instruct | W8A8 | 1175 | 89.19% (1048) | 0.0454 | +0.0302 +/- 0.0113 | 35.91 -> 37.01 |

W8A16 is the default (int16 activations into every weight GEMV, int8 K/V
cache, table-driven nonlinearities, K-centering and the Q/K smoothing fold);
W8A8 feeds int8 activations to the weight GEMVs with everything else
unchanged. `sat` and `err_shift` are 0 on every row; the VQUANT and softmax
clip counts are 243621 / 194390 on Qwen (W8A16 / W8A8) and 171687 / 140157 on
SmolLM2. With K-centering alone (every smoothing factor 1, everything else
equal) the Qwen W8A16 row measures 93.65% top-1, KL 0.0556 and delta-NLL
+0.0492 +/- 0.0132 (PPL 36.73), so the fold recovers 1.9 points of top-1 and
4.5x in KL on the model with large `k_proj` biases and leaves SmolLM2 within
noise (95.66%, KL 0.00464 with centering alone). On Qwen the KL is largest on
the two short prompts (0.038 per position at T = 36 and 0.034 at T = 180,
W8A16) while the four longer sequences lie between 0.0054 and 0.0117; on
SmolLM2 every sequence lies between 0.0040 and 0.0080. The W8A8 rows lose 9.2
(Qwen) and 6.5 (SmolLM2) points of top-1 and raise the KL about 9x, with
delta-NLL +0.011 +/- 0.021 and +0.030 +/- 0.011: the int8 activations perturb
the distribution far more than they shift the likelihood of the true token.

`sw/tests/test_quality.py` gates SmolLM2 W8A16 on this set at `KL <= 0.02`
nats and `top-1 >= 93%`, and checks that `quality.json` agrees with a fresh
evaluation (integers exactly, floats within `2e-3` absolute plus `1e-3`
relative, the room float32 summation order needs across BLAS libraries); Qwen
is reported.
