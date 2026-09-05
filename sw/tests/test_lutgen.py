"""The checked-in lookup tables and RoPE tables against ``quettos.lutgen`` and ``numerics``.

Every table value is recomputed here with an independent mpmath evaluation at
128-bit precision on the grid documented in :mod:`quettos.numerics`, so a
change to either the generator or the checked-in files shows up as a diff.
The RoPE tables are spot-checked (their full regeneration is slow); the LUTs
are regenerated in full.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys

import mpmath
import numpy as np
import pytest
from quettos import lutgen, numerics
from quettos.numerics import TABLE_SPECS

TABLE_NAMES = tuple(TABLE_SPECS)
Q14_ONE = 1 << 14
Q15_ONE = numerics.Q15_ONE

# Right-end value of each table's domain, rounded to Q1.15 (documented in numerics.py).
RIGHT_END = {"exp2": 65536, "sigmoid": None, "rsqrt": 16384, "recip": 16384}

HEX_LINE = re.compile(r"^[0-9a-f]{8}$")


def q15_half_up(value: mpmath.mpf) -> int:
    """``round_half_up(value * 2**15)`` computed in mpmath."""
    return int(mpmath.floor(value * Q15_ONE + mpmath.mpf(1) / 2))


def q14_half_up(value: mpmath.mpf) -> int:
    return int(mpmath.floor(value * Q14_ONE + mpmath.mpf(1) / 2))


def grid_and_function(name: str):
    """The documented sample points ``x_i`` and ``f`` for one table, as mpmath values."""
    one = mpmath.mpf(1)
    if name == "exp2":
        xs = [mpmath.mpf(i) / 256 for i in range(256)]
        return xs, (lambda x: mpmath.exp(x * mpmath.log(2))), one
    if name == "sigmoid":
        xs = [mpmath.mpf(i) / 32 for i in range(512)]
        return xs, (lambda x: one / (one + mpmath.exp(-x))), mpmath.mpf(16)
    if name == "rsqrt":
        xs = [one + mpmath.mpf(i) / 256 for i in range(256)]
        xs += [mpmath.mpf(2) + mpmath.mpf(2 * i) / 256 for i in range(256)]
        return xs, (lambda x: mpmath.power(x, -mpmath.mpf(1) / 2)), mpmath.mpf(4)
    if name == "recip":
        xs = [one + mpmath.mpf(i) / 256 for i in range(256)]
        return xs, (lambda x: one / x), mpmath.mpf(2)
    raise ValueError(name)


@pytest.fixture(scope="module")
def disk() -> dict:
    with open(numerics.LUTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def built() -> dict:
    return lutgen.build_luts()


@pytest.fixture(scope="module")
def tables() -> numerics.Tables:
    return numerics.load_tables()


def arr(table: dict, key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=np.int64)


# --------------------------------------------------------------------------- generator vs files


def test_check_cli_matches_checked_in_luts() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "quettos.lutgen", "--check", "--no-rope"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    # luts.json plus one hex file per table, every one reported "ok".
    assert len(lines) == 1 + len(TABLE_NAMES)
    assert all(ln.startswith("ok") for ln in lines), proc.stdout


def test_build_luts_equals_json_on_disk(built: dict, disk: dict) -> None:
    assert set(built) == set(disk) == set(TABLE_NAMES) | {"meta"}
    for name in TABLE_NAMES:
        assert built[name]["v"] == disk[name]["v"], name
        assert built[name]["dv"] == disk[name]["dv"], name
    assert built["meta"] == disk["meta"]
    assert lutgen.json_text(built).encode() == numerics.LUTS_JSON.read_bytes()


def test_meta_fields(disk: dict) -> None:
    meta = disk["meta"]
    assert meta["generator"] == "quettos.lutgen"
    assert meta["mp_prec_bits"] == 128
    assert "interp = v + ((dv*frac8 + 128)>>8)" in meta["format"]
    assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"])


def test_meta_sha256_is_hash_of_compact_tables(disk: dict, built: dict) -> None:
    payload = json.dumps(
        {name: {"v": disk[name]["v"], "dv": disk[name]["dv"]} for name in TABLE_NAMES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()
    assert disk["meta"]["sha256"] == digest
    assert built["meta"]["sha256"] == digest


def test_load_tables_matches_json(tables: numerics.Tables, disk: dict) -> None:
    for name in TABLE_NAMES:
        lut: numerics.Lut = getattr(tables, name)
        assert lut.name == name
        assert lut.v.dtype == np.int64 and lut.dv.dtype == np.int64
        np.testing.assert_array_equal(lut.v, arr(disk[name], "v"))
        np.testing.assert_array_equal(lut.dv, arr(disk[name], "dv"))
    assert tables.meta == disk["meta"]


# --------------------------------------------------------------------------- hex files


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_hex_file_decodes_to_json(name: str, disk: dict) -> None:
    path = lutgen.HEX_DIR / f"{name}.hex"
    assert path.is_file(), path
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    lines = text.split("\n")[:-1]
    assert len(lines) == TABLE_SPECS[name]["entries"]
    assert all(HEX_LINE.match(ln) for ln in lines), lines[:3]
    words = [int(ln, 16) for ln in lines]
    v = [w >> 16 for w in words]
    dv = [(w & 0xFFFF) - (0x10000 if w & 0x8000 else 0) for w in words]
    assert v == disk[name]["v"]
    assert dv == disk[name]["dv"]
    assert lutgen.hex_lines(disk[name]["v"], disk[name]["dv"]) == text


def test_hex_lines_encoding() -> None:
    assert lutgen.hex_lines([1], [-1]) == "0001ffff\n"
    assert lutgen.hex_lines([0xFFFF], [-32768]) == "ffff8000\n"
    assert lutgen.hex_lines([32768, 0], [89, 32767]) == "80000059\n00007fff\n"
    with pytest.raises(ValueError):
        lutgen.hex_lines([1, 2], [1])


# --------------------------------------------------------------------------- shape, ranges, deltas


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_shape_and_widths(name: str, disk: dict) -> None:
    n = TABLE_SPECS[name]["entries"]
    v, dv = arr(disk[name], "v"), arr(disk[name], "dv")
    assert v.shape == dv.shape == (n,)
    assert 0 <= v.min() and v.max() <= 0xFFFF
    assert -(1 << 15) <= dv.min() and dv.max() <= numerics.I16_MAX


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_dv_is_forward_difference(name: str, disk: dict) -> None:
    v, dv = arr(disk[name], "v"), arr(disk[name], "dv")
    np.testing.assert_array_equal(dv[:-1], np.diff(v))
    right = RIGHT_END[name]
    if right is None:
        with mpmath.workprec(128):
            right = q15_half_up(1 / (1 + mpmath.exp(-mpmath.mpf(16))))
        assert right == 32768  # sigmoid(16) rounds to 1.0 in Q1.15
    assert int(dv[-1]) == right - int(v[-1])


def test_exp2_range_and_monotonicity(disk: dict) -> None:
    v, dv = arr(disk["exp2"], "v"), arr(disk["exp2"], "dv")
    assert int(v[0]) == Q15_ONE
    assert v.max() <= 65535
    assert np.all(np.diff(v) > 0)
    assert np.all(dv > 0)
    assert int(dv[-1]) == 65536 - int(v[-1])


def test_sigmoid_range_and_monotonicity(disk: dict) -> None:
    v, dv = arr(disk["sigmoid"], "v"), arr(disk["sigmoid"], "dv")
    assert int(v[0]) == 16384  # sigmoid(0) = 0.5
    assert v.min() == 16384
    assert v.max() <= Q15_ONE
    assert np.all(np.diff(v) >= 0)
    assert np.all(dv >= 0)
    # The step sigmoid'(x)/32 exceeds one Q1.15 LSB up to x = ln(1023) ~ 6.93, so the
    # table is strictly increasing over [0, 6]; it reaches 1.0 once 1 - sigmoid(x) < 2**-16,
    # i.e. from x = ln(2**16 - 1) ~ 11.09 on, and stays there.
    assert np.all(np.diff(v[: 6 * 32 + 1]) > 0)
    knee = int(np.argmax(v >= Q15_ONE))
    assert knee == 355  # first x_i = i/32 above ln(2**16 - 1)
    assert np.all(v[knee:] == Q15_ONE)


def test_rsqrt_range_monotonicity_and_segment_boundary(disk: dict) -> None:
    v, dv = arr(disk["rsqrt"], "v"), arr(disk["rsqrt"], "dv")
    assert int(v[0]) == Q15_ONE  # 1/sqrt(1)
    assert v.min() > 16384
    assert np.all(np.diff(v) < 0)
    assert np.all(dv < 0)
    assert int(dv[-1]) == 16384 - int(v[-1])
    with mpmath.workprec(128):
        rsqrt2 = q15_half_up(1 / mpmath.sqrt(2))
    assert rsqrt2 == 23170
    assert int(v[256]) == rsqrt2  # first entry of segment 1 is m = 2
    assert int(v[255]) + int(dv[255]) == int(v[256])  # segment 0 interpolates into segment 1
    # Segment 1 steps in x are twice as wide as segment 0 steps but the curve flattens,
    # so the step magnitude stays in a narrow band.
    assert dv[:256].min() >= -64 and dv[256:].max() <= -16


def test_recip_range_and_monotonicity(disk: dict) -> None:
    v, dv = arr(disk["recip"], "v"), arr(disk["recip"], "dv")
    assert int(v[0]) == Q15_ONE  # 1/1
    assert v.min() > 16384
    assert np.all(np.diff(v) < 0)
    assert np.all(dv < 0)
    assert int(dv[-1]) == 16384 - int(v[-1])
    assert int(dv[0]) == -128  # slope -1 at x = 1 in units of 2**-15 per 1/256


# --------------------------------------------------------------------------- exact recomputation


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_values_match_mpmath_128(name: str, disk: dict) -> None:
    with mpmath.workprec(128):
        xs, f, right = grid_and_function(name)
        expect_v = [q15_half_up(f(x)) for x in xs]
        expect_end = q15_half_up(f(right))
    assert expect_v == disk[name]["v"]
    n = len(xs)
    expect_dv = [expect_v[i + 1] - expect_v[i] for i in range(n - 1)] + [expect_end - expect_v[-1]]
    assert expect_dv == disk[name]["dv"]


def test_q15_rounds_half_up() -> None:
    with mpmath.workprec(128):
        assert lutgen._q15(mpmath.mpf(1)) == 32768
        assert lutgen._q15(mpmath.mpf(1) / 2) == 16384
        assert lutgen._q15(mpmath.mpf(65537) / 65536) == 32769  # 32768.5 -> up
        assert lutgen._q15(mpmath.mpf(65535) / 65536) == 32768  # 32767.5 -> up
        assert lutgen._q15(mpmath.mpf(131069) / 131072) == 32767  # 32767.25 -> down


# --------------------------------------------------------------------------- interpolation accuracy


def test_interpolated_exp2_within_one_lsb(tables: numerics.Tables) -> None:
    f16 = np.arange(1 << 16, dtype=np.int64)
    got = numerics.exp2_q15(f16, tables)
    exact = np.floor(np.exp2(f16 / 65536.0) * Q15_ONE + 0.5)
    assert np.abs(got - exact).max() <= 1
    assert got.min() == Q15_ONE and got.max() <= 65535


def test_interpolated_recip_within_one_lsb(tables: numerics.Tables) -> None:
    a_hi = np.arange(1 << 15, 1 << 16, dtype=np.int64)
    got = np.array([numerics.recip_q15(int(a), tables) for a in a_hi])
    exact = np.floor((Q15_ONE / a_hi) * Q15_ONE + 0.5)
    assert np.abs(got - exact).max() <= 1
    assert got.min() == 16384 and got.max() == Q15_ONE  # a_hi = 65535 rounds to 0.5 exactly


def test_interpolated_rsqrt_within_one_lsb(tables: numerics.Tables) -> None:
    m_q16 = np.arange(1 << 16, 1 << 18, dtype=np.int64)
    got = np.array([numerics.rsqrt_q15(int(m), tables) for m in m_q16])
    exact = np.floor((1.0 / np.sqrt(m_q16 / 65536.0)) * Q15_ONE + 0.5)
    assert np.abs(got - exact).max() <= 1
    assert got.min() == 16384 and got.max() == Q15_ONE  # m -> 4 rounds to 0.5 exactly


def test_interpolated_sigmoid_within_one_lsb(tables: numerics.Tables) -> None:
    frac = 13
    x_q = np.arange(0, 16 << frac, dtype=np.int64)
    got = numerics.sigmoid_q15(x_q, frac, tables)
    exact = np.floor(1.0 / (1.0 + np.exp(-x_q / 2.0**frac)) * Q15_ONE + 0.5)
    assert np.abs(got - exact).max() <= 1
    assert got.min() == 16384 and got.max() == Q15_ONE
    neg = numerics.sigmoid_q15(-x_q, frac, tables)
    np.testing.assert_array_equal(neg, Q15_ONE - got)


# --------------------------------------------------------------------------- RoPE tables


@pytest.mark.parametrize("theta", lutgen.ROPE_THETAS)
def test_rope_table_file(theta: float) -> None:
    path = numerics.rope_table_path(theta, lutgen.ROPE_MAX_POS)
    exponent = int(round(np.log10(theta)))
    assert path.name == f"rope_theta1e{exponent}_2048.npy"
    assert path.parent == numerics.TABLES_DIR
    assert path.is_file(), path
    raw = np.load(path)
    assert raw.dtype == np.int16
    assert raw.shape == (lutgen.ROPE_MAX_POS, 2, lutgen.HEAD_DIM // 2)
    assert np.all(raw[0, 0] == Q14_ONE)  # cos(0) = 1.0 in Q1.14
    assert np.all(raw[0, 1] == 0)  # sin(0) = 0
    assert np.abs(raw.astype(np.int64)).max() <= Q14_ONE
    c = raw[:, 0, :].astype(np.int64)
    s = raw[:, 1, :].astype(np.int64)
    # cos^2 + sin^2 = 1 up to the two half-LSB roundings.
    assert np.abs(c * c + s * s - (1 << 28)).max() <= (1 << 15)
    loaded = numerics.load_rope_table(theta, lutgen.ROPE_MAX_POS)
    assert loaded.dtype == np.int64
    assert loaded.shape == raw.shape
    np.testing.assert_array_equal(loaded, raw.astype(np.int64))


@pytest.mark.parametrize("theta", lutgen.ROPE_THETAS)
def test_rope_entries_match_mpmath_128(theta: float) -> None:
    table = numerics.load_rope_table(theta, lutgen.ROPE_MAX_POS)
    rng = np.random.default_rng(int(theta) % 1000003)
    half = lutgen.HEAD_DIM // 2
    positions = rng.integers(0, lutgen.ROPE_MAX_POS, size=300)
    dims = rng.integers(0, half, size=300)
    # Always include the last position and the fastest / slowest frequencies.
    positions[:3] = lutgen.ROPE_MAX_POS - 1
    dims[:3] = (0, half - 1, half // 2)
    with mpmath.workprec(128):
        th = mpmath.mpf(theta)
        for pos, i in zip(positions.tolist(), dims.tolist(), strict=True):
            inv_freq = mpmath.power(th, -mpmath.mpf(2 * i) / lutgen.HEAD_DIM)
            angle = mpmath.mpf(pos) * inv_freq
            assert int(table[pos, 0, i]) == q14_half_up(mpmath.cos(angle)), (theta, pos, i)
            assert int(table[pos, 1, i]) == q14_half_up(mpmath.sin(angle)), (theta, pos, i)


@pytest.mark.parametrize("theta", lutgen.ROPE_THETAS)
def test_build_rope_prefix_matches_checked_in(theta: float) -> None:
    prefix = 8
    built = lutgen.build_rope(theta, max_pos=prefix)
    assert built.dtype == np.int16 and built.shape == (prefix, 2, lutgen.HEAD_DIM // 2)
    on_disk = np.load(numerics.rope_table_path(theta, lutgen.ROPE_MAX_POS))
    np.testing.assert_array_equal(built, on_disk[:prefix])


def test_rope_thetas_differ() -> None:
    a = numerics.load_rope_table(1e6)
    b = numerics.load_rope_table(1e5)
    # Position 1 at i = 0 is the same for both (inv_freq = 1); higher i diverge.
    assert a[1, 0, 0] == b[1, 0, 0] and a[1, 1, 0] == b[1, 1, 0]
    assert np.any(a[1] != b[1])
    assert np.any(a[1:] != b[1:])


def test_rope_table_path_naming() -> None:
    assert numerics.rope_table_path(1e6).name == "rope_theta1e6_2048.npy"
    assert numerics.rope_table_path(1e5).name == "rope_theta1e5_2048.npy"
    assert numerics.rope_table_path(1e6, 4096).name == "rope_theta1e6_4096.npy"
    assert numerics.rope_table_path(500000.0).name == "rope_theta5e5_2048.npy"
