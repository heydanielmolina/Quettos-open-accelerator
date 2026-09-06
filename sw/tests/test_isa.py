"""Descriptor ISA: the documented bit layout, encode/decode round trips at every field extreme,
the stability of the ``.lst`` disassembly, the opcode helpers and the CSR map against the
generated SystemVerilog and C++ headers."""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess

import numpy as np
import pytest
from quettos import cli, csrgen, isa
from quettos.isa import Descriptor, Opcode, OutMode
from quettos.numerics import SFloat

# docs/ISA.md, descriptor bit layout: field -> (lsb, width)
DOCUMENTED_FIELDS = {
    "opcode": (0, 8),
    "flags": (8, 8),
    "row_mask": (16, 8),
    "accumulate": (24, 1),
    "unit_meta": (25, 1),
    "n_from_pos": (26, 1),
    "k_from_pos": (27, 1),
    "out_mode": (28, 2),
    "len_from_pos": (30, 1),
    "track_absmax": (31, 1),
    "addr_a": (32, 32),
    "addr_m": (64, 32),
    "n": (96, 24),
    "k": (120, 16),
    "vs_src": (136, 16),
    "vs_dst": (152, 16),
    "vs_aux": (168, 16),
    "sreg_src": (184, 8),
    "sreg_dst": (192, 8),
    "src_row": (200, 4),
    "dst_row": (204, 4),
    "sh0": (208, 8),
    "sh1": (216, 8),
    "imm32": (224, 32),
}
DOCUMENTED_OPCODES = {
    "NOP": 0x00,
    "HALT": 0x01,
    "GEMV": 0x10,
    "EMBED": 0x11,
    "VRMSNORM": 0x20,
    "VQUANT": 0x21,
    "VROPE": 0x22,
    "VSILUMUL": 0x23,
    "VSOFTMAX": 0x24,
    "VSUBC": 0x25,
    "KVWRITE": 0x30,
    "FENCE": 0x31,
}
# docs/ISA.md, CSR table: register -> word offset
DOCUMENTED_CSRS = {
    "CTRL": 0,
    "STATUS": 1,
    "PC": 2,
    "ROW_EN": 3,
    "TOK": 4,
    "POS": 5,
    "ARGMAX_TOK": 6,
    "ARGMAX_VAL": 7,
    "SAT_REQ": 8,
    "SAT_VPU": 9,
    "ERR_SHIFT": 10,
    "ERR_BOUNDS": 11,
    "ISA_VERSION": 12,
    "PERF0_LO": 16,
    "PERF0_HI": 17,
    "PERF15_LO": 46,
    "PERF15_HI": 47,
}

SV_LINE = re.compile(r"^`define QCORE_(\w+) (\d+)$")
HPP_LINE = re.compile(r"^constexpr uint32_t (\w+) = (\d+)u;(?:\s*//.*)?$")


def _random_descriptor(rng: np.random.Generator) -> Descriptor:
    kw = {}
    for f in isa.FIELDS:
        lo, hi = f.bounds
        if f.name == "opcode":
            kw[f.name] = int(rng.choice(list(Opcode)))
        else:
            kw[f.name] = int(rng.integers(lo, hi + 1))
    return Descriptor(**kw)


def _program() -> list[Descriptor]:
    """A representative slice of a decode program (Qwen-shaped addresses)."""
    return [
        isa.embed(addr_a=0x00300000, addr_m=0x00400000, k=896, vs_dst=0, s1=16, sbias=-8),
        isa.vrmsnorm(
            vs_src=0,
            vs_dst=896,
            n=896,
            addr_a=0x00500000,
            eps_c=3848291,
            frac_x=16,
            g=15,
            sqrt_d=SFloat(61303, -11),
            sreg_dst=0,
        ),
        isa.vquant(vs_src=896, vs_dst=1792, n=896, width=16, frac_in=16, sreg_dst=1),
        isa.gemv(
            addr_a=0x00200000,
            addr_m=0x002FC000,
            n=1152,
            k=896,
            vs_src=1792,
            vs_dst=2688,
            sreg_src=1,
            s1=9,
            sbias=-25,
        ),
        isa.vrope(vs_src=2688, n=896, addr_a=0x00100000),
        isa.vquant(
            vs_src=2688,
            vs_dst=2688,
            n=896,
            width=16,
            frac_in=16,
            sreg_dst=2,
            group=64,
            scale_mul=SFloat(47274, -18),
        ),
        isa.vsubc(vs_src=3584, vs_dst=3584, n=128, addr_a=0x00140000),
        isa.vquant(vs_src=3584, vs_dst=3584, n=128, width=8, frac_in=16, sreg_dst=16, group=64),
        isa.kvwrite(
            vs_src=3584,
            addr_a=0x10000000,
            addr_m=0x10040000,
            sreg_src=16,
            transposed=True,
            max_ctx=2048,
        ),
        isa.kvwrite(
            vs_src=3712,
            addr_a=0x10020000,
            addr_m=0x10042000,
            sreg_src=18,
            transposed=False,
            max_ctx=2048,
        ),
        isa.gemv(
            addr_a=0x10000000,
            addr_m=0x10040000,
            n=2048,
            k=64,
            vs_src=2688,
            vs_dst=20224,
            sreg_src=2,
            s1=9,
            sbias=-25,
            n_from_pos=True,
        ),
        isa.vsoftmax(vs_src=20224, vs_dst=22272, n=2048, addr_a=0x10042000, frac_s=16, sreg_dst=20),
        isa.gemv(
            addr_a=0x10020000,
            n=64,
            k=2048,
            vs_src=22272,
            vs_dst=3840,
            sreg_src=20,
            s1=10,
            sbias=-26,
            unit_meta=True,
            k_from_pos=True,
        ),
        isa.vsilumul(
            vs_src=5632, vs_aux=10496, vs_dst=15360, n=4864, frac_gu=16, sh_h=16, sreg_dst=3
        ),
        isa.gemv(
            addr_a=0x00200000,
            addr_m=0x002FC000,
            n=151936,
            k=896,
            vs_src=1792,
            vs_dst=0,
            sreg_src=1,
            s1=14,
            sbias=-30,
            out_mode=OutMode.ARGMAX_DUMP,
            addr_c=0x20000000,
        ),
        isa.fence(),
        isa.halt(),
    ]


EXPECTED_LISTING = """\
    0  EMBED    addr_a=0x00300000 addr_m=0x00400000 n=896 k=896 sh0=16 sh1=-8
    1  VRMSNORM track_absmax addr_a=0x00500000 sqrt_d={61303,-11} n=896 vs_dst=896 sh0=16 sh1=15 eps_c=3848291
    2  VQUANT   n=896 vs_src=896 vs_dst=1792 sreg_dst=1 sh0=16
    3  GEMV     addr_a=0x00200000 addr_m=0x002fc000 n=1152 k=896 vs_src=1792 vs_dst=2688 sreg_src=1 sh0=9 sh1=-25
    4  VROPE    addr_a=0x00100000 n=896 vs_src=2688
    5  VQUANT   flags=GROUP|SCALE_MUL n=896 vs_src=2688 vs_dst=2688 vs_aux=64 sreg_dst=2 sh0=16 scale_mul={47274,-18}
    6  VSUBC    addr_a=0x00140000 n=128 vs_src=3584 vs_dst=3584
    7  VQUANT   flags=W8|GROUP n=128 vs_src=3584 vs_dst=3584 vs_aux=64 sreg_dst=16 sh0=16
    8  KVWRITE  flags=TRANSPOSED addr_a=0x10000000 addr_m=0x10040000 n=64 k=2048 vs_src=3584 sreg_src=16
    9  KVWRITE  addr_a=0x10020000 addr_m=0x10042000 n=64 k=2048 vs_src=3712 sreg_src=18
   10  GEMV     n_from_pos addr_a=0x10000000 addr_m=0x10040000 n=2048 k=64 vs_src=2688 vs_dst=20224 sreg_src=2 sh0=9 sh1=-25
   11  VSOFTMAX len_from_pos addr_a=0x10042000 n=2048 vs_src=20224 vs_dst=22272 sreg_dst=20 sh0=16
   12  GEMV     unit_meta k_from_pos addr_a=0x10020000 n=64 k=2048 vs_src=22272 vs_dst=3840 sreg_src=20 sh0=10 sh1=-26
   13  VSILUMUL track_absmax n=4864 vs_src=5632 vs_dst=15360 vs_aux=10496 sreg_dst=3 sh0=16 sh1=16
   14  GEMV     out_mode=ARGMAX_DUMP addr_a=0x00200000 addr_m=0x002fc000 n=151936 k=896 vs_src=1792 sreg_src=1 sh0=14 sh1=-30 addr_c=0x20000000
   15  FENCE
   16  HALT
"""  # noqa: E501


# --------------------------------------------------------------------------- layout


def test_field_layout_matches_the_documented_table() -> None:
    isa.check_tables()
    assert {f.name: (f.lsb, f.width) for f in isa.FIELDS} == DOCUMENTED_FIELDS
    assert [f.name for f in isa.FIELDS] == list(DOCUMENTED_FIELDS)
    assert [f.name for f in isa.FIELDS if f.signed] == ["sh1"]
    assert isa.FIELD_BY_NAME["sh1"].bounds == (-128, 127)
    assert isa.FIELD_BY_NAME["sh0"].bounds == (0, 255)
    assert isa.FIELD_BY_NAME["n"].bounds == (0, (1 << 24) - 1)
    assert sum(f.width for f in isa.FIELDS) == isa.DESC_BITS == 8 * isa.DESC_BYTES == 256
    assert {op.name: op.value for op in Opcode} == DOCUMENTED_OPCODES
    assert [m.value for m in OutMode] == [0, 1, 2, 3]
    assert isa.ISA_VERSION == 1 and isa.PROGRAM_ALIGN == 2 * isa.DESC_BYTES


def test_each_field_lands_on_its_bits() -> None:
    """Setting one field to its maximum sets exactly the bits [lsb + width - 1 : lsb]."""
    for f in isa.FIELDS:
        if f.name == "opcode":
            continue
        _, hi = f.bounds
        value = -1 if f.signed else hi  # all ones in the field
        d = Descriptor(**{f.name: value})
        word = d.word() ^ Descriptor().word()  # remove the default row_mask bit
        expect = ((1 << f.width) - 1) << f.lsb
        if f.name == "row_mask":
            expect ^= 1 << f.lsb
        assert word == expect, f.name
        data, full = isa.encode(d), d.word()
        for i in range(isa.DESC_BITS):  # bit i of the descriptor is bit i % 8 of byte i // 8
            assert (data[i // 8] >> (i % 8)) & 1 == (full >> i) & 1, (f.name, i)
    d = Descriptor(opcode=Opcode.KVWRITE)
    assert isa.encode(d)[0] == 0x30 and isa.encode(d)[2] == 1 and isa.encode(d)[3:] == bytes(29)


def test_round_trip_random_descriptors() -> None:
    rng = np.random.default_rng(2024)
    for _ in range(3000):
        d = _random_descriptor(rng)
        data = isa.encode(d)
        assert len(data) == isa.DESC_BYTES
        back = isa.decode(data)
        assert back == d
        assert isa.encode(back) == data


def test_round_trip_at_the_extremes() -> None:
    maxed = Descriptor(
        **{f.name: (f.bounds[1] if f.name != "opcode" else Opcode.FENCE) for f in isa.FIELDS}
    )
    assert isa.decode(isa.encode(maxed)) == maxed
    assert isa.encode(maxed)[27] == 0x7F and isa.encode(maxed)[28:] == b"\xff" * 4
    lowest = Descriptor(sh1=-128, row_mask=0)
    assert isa.encode(lowest)[27] == 0x80 and isa.decode(isa.encode(lowest)).sh1 == -128
    assert isa.encode(Descriptor(sh1=-1))[27] == 0xFF
    assert isa.decode(isa.encode(Descriptor())) == Descriptor() == isa.DEFAULT
    for bad in (
        {"n": 1 << 24},
        {"k": 1 << 16},
        {"sh0": -1},
        {"sh0": 256},
        {"sh1": 128},
        {"sh1": -129},
        {"out_mode": 4},
        {"opcode": 0x02},
        {"row_mask": 256},
        {"src_row": 16},
        {"addr_a": 1 << 32},
        {"imm32": -1},
    ):
        with pytest.raises(ValueError):
            Descriptor(**bad)
    with pytest.raises(TypeError):
        Descriptor(n=True)


def test_decode_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        isa.decode(bytes(31))
    with pytest.raises(ValueError):
        isa.decode(bytes([0x02]) + bytes(31))  # unknown opcode
    with pytest.raises(ValueError):
        isa.parse(bytes(33))
    assert isa.parse(b"") == []
    # the simulator's STATUS.ERR path: probe the opcode byte before decoding
    assert isa.opcode_of(bytes([0x7F]) + bytes(31)) == 0x7F and not isa.is_opcode(0x7F)
    assert isa.is_opcode(isa.opcode_of(isa.encode(isa.halt())))
    assert all(isa.is_opcode(op) for op in Opcode) and not isa.is_opcode(0x02)
    with pytest.raises(ValueError):
        isa.opcode_of(bytes(4))


def test_one_bit_fields_and_enums_are_normalized() -> None:
    a = Descriptor(accumulate=1, out_mode=2, opcode=0x10, track_absmax=True)
    b = Descriptor(
        accumulate=True, out_mode=OutMode.ARGMAX_DUMP, opcode=Opcode.GEMV, track_absmax=1
    )
    assert a == b and a.accumulate is True and a.out_mode is OutMode.ARGMAX_DUMP
    assert a.opcode is Opcode.GEMV and isinstance(a.n, int)
    assert dataclasses.replace(a, accumulate=0).accumulate is False


# --------------------------------------------------------------------------- disassembly


def test_disassembly_is_stable() -> None:
    prog = _program()
    listing = isa.disassemble(prog)
    assert listing == EXPECTED_LISTING
    blob = isa.assemble(prog)
    assert len(blob) == len(prog) * isa.DESC_BYTES
    assert isa.parse(blob) == prog
    assert isa.disassemble(isa.parse(blob)) == listing
    assert isa.disassemble([]) == ""
    assert isa.disassemble_one(Descriptor()) == "NOP"
    assert (
        isa.disassemble_one(Descriptor(opcode=Opcode.VQUANT, flags=0xF0)) == "VQUANT   flags=0xf0"
    )
    assert isa.disassemble_one(Descriptor(row_mask=3)) == "NOP      row_mask=3"
    assert (
        isa.disassemble_one(Descriptor(opcode=Opcode.VSOFTMAX, imm32=7, n=64))
        == "VSOFTMAX n=64 len=7"
    )
    assert isa.disassemble_one(Descriptor(opcode=Opcode.FENCE, imm32=0xABC)).endswith(
        "imm32=0x00000abc"
    )
    # malformed sfloat immediates disassemble as the raw field, so any decodable program lists
    for bad in (1 << 24, 5, (1 << 24) | 40000):
        rms = Descriptor(opcode=Opcode.VRMSNORM, addr_m=bad, n=8)
        assert isa.disassemble_one(rms) == f"VRMSNORM addr_m=0x{bad:08x} n=8"
        vq = Descriptor(opcode=Opcode.VQUANT, flags=isa.VquantFlag.SCALE_MUL, imm32=bad)
        assert isa.disassemble_one(vq) == f"VQUANT   flags=SCALE_MUL imm32=0x{bad:08x}"
        assert isa.parse(isa.assemble([rms, vq])) == [rms, vq]
    # every listed field is recoverable: numbers in the listing are the descriptor's values
    for line, d in zip(listing.splitlines(), prog, strict=True):
        assert line[7:16].strip() == d.opcode.name
        for f in isa.FIELDS:
            v = getattr(d, f.name)
            if f.name in ("opcode", "flags", "out_mode", "imm32", "addr_m") or f.width == 1:
                continue
            if v != getattr(isa.DEFAULT, f.name):
                text = f"{f.name}=0x{v:08x}" if f.name == "addr_a" else f"{f.name}={int(v)}"
                assert f" {text}" in line, (f.name, line)


# --------------------------------------------------------------------------- helpers


def test_helpers_place_the_documented_fields() -> None:
    g = isa.gemv(
        addr_a=0x1000, addr_m=0x2000, n=100, k=64, vs_src=8, vs_dst=16, sreg_src=3, s1=9, sbias=-25
    )
    assert (g.opcode, g.sh0, g.sh1, g.imm32, g.out_mode) == (Opcode.GEMV, 9, -25, 0, OutMode.VSRAM)
    assert (g.n, g.k, g.vs_src, g.vs_dst, g.sreg_src, g.row_mask) == (100, 64, 8, 16, 3, 1)
    base = dict(addr_a=0, n=1, k=1, vs_src=0, vs_dst=0, sreg_src=0, s1=0, sbias=0)
    dump = isa.gemv(**base, out_mode=OutMode.VSRAM_DUMP, addr_c=0xDE00)
    assert dump.imm32 == 0xDE00 and dump.out_mode is OutMode.VSRAM_DUMP
    with pytest.raises(ValueError):
        isa.gemv(**base, addr_c=64)  # addr_c without a DUMP mode
    for addr_c in (0, 0xDEAD):  # a DUMP needs a non-zero, beat-aligned address
        with pytest.raises(ValueError):
            isa.gemv(**base, out_mode=OutMode.ARGMAX_DUMP, addr_c=addr_c)
    with pytest.raises(ValueError):
        isa.gemv(**{**base, "s1": 64})
    with pytest.raises(ValueError):  # SREG has 32 registers
        isa.gemv(**{**base, "sreg_src": 32})

    e = isa.embed(addr_a=0x10, addr_m=0x20, k=64, vs_dst=0, s1=16, sbias=-8)
    assert (e.n, e.k, e.sh0, e.sh1, e.opcode) == (64, 64, 16, -8, Opcode.EMBED)
    for s1 in (7, 25):
        with pytest.raises(ValueError):
            isa.embed(addr_a=0, addr_m=0, k=64, vs_dst=0, s1=s1, sbias=0)

    sq = SFloat(61303, -11)
    r = isa.vrmsnorm(
        vs_src=0, vs_dst=64, n=64, addr_a=0x40, eps_c=12345, frac_x=16, g=15, sqrt_d=sq, sreg_dst=2
    )
    assert (r.sh0, r.sh1, r.imm32, r.track_absmax) == (16, 15, 12345, True)
    assert isa.sfloat_from_imm(r.addr_m) == sq and r.addr_m >> 24 == 0
    for g in (-1, 64):  # G is a shift amount in [0, 63]
        with pytest.raises(ValueError):
            isa.vrmsnorm(
                vs_src=0, vs_dst=0, n=64, addr_a=0, eps_c=1, frac_x=16, g=g, sqrt_d=sq, sreg_dst=0
            )

    q = isa.vquant(vs_src=0, vs_dst=64, n=128, width=8, frac_in=14, sreg_dst=4, group=64)
    assert q.flags == isa.VquantFlag.W8 | isa.VquantFlag.GROUP and q.vs_aux == 64 and q.sh0 == 14
    q16 = isa.vquant(
        vs_src=0,
        vs_dst=0,
        n=64,
        width=16,
        frac_in=16,
        sreg_dst=1,
        use_tracked=True,
        sreg_src=9,
        scale_mul=SFloat(47274, -18),
    )
    assert q16.flags == isa.VquantFlag.USE_TRACKED | isa.VquantFlag.SCALE_MUL
    assert isa.sfloat_from_imm(q16.imm32) == SFloat(47274, -18) and q16.sreg_src == 9
    with pytest.raises(ValueError):
        isa.vquant(vs_src=0, vs_dst=0, n=100, width=8, frac_in=16, sreg_dst=0, group=64)
    with pytest.raises(ValueError):
        isa.vquant(vs_src=0, vs_dst=0, n=64, width=4, frac_in=16, sreg_dst=0)
    with pytest.raises(ValueError):  # 14 group scales from SREG[20] leave the 32 registers
        isa.vquant(vs_src=0, vs_dst=0, n=896, width=16, frac_in=16, sreg_dst=20, group=64)

    with pytest.raises(ValueError):
        isa.vrope(vs_src=0, n=100, addr_a=0)
    assert isa.vrope(vs_src=8, n=128, addr_a=0x100).vs_src == 8

    s = isa.vsilumul(vs_src=0, vs_aux=64, vs_dst=128, n=64, frac_gu=16, sh_h=16, sreg_dst=5)
    assert (s.sh0, s.sh1, s.vs_aux, s.track_absmax) == (16, 16, 64, True)
    with pytest.raises(ValueError):
        isa.vsilumul(vs_src=0, vs_aux=64, vs_dst=128, n=64, frac_gu=16, sh_h=64, sreg_dst=5)

    sm = isa.vsoftmax(vs_src=0, vs_dst=64, n=2048, addr_a=0x8, frac_s=16, sreg_dst=6)
    assert sm.len_from_pos and sm.imm32 == 0 and sm.n == 2048 and sm.sh0 == 16
    fixed = isa.vsoftmax(vs_src=0, vs_dst=64, n=2048, addr_a=0x8, frac_s=16, sreg_dst=6, length=5)
    assert not fixed.len_from_pos and fixed.imm32 == 5
    with pytest.raises(ValueError):
        isa.vsoftmax(vs_src=0, vs_dst=0, n=8, addr_a=0, frac_s=16, sreg_dst=0, length=9)

    c = isa.vsubc(vs_src=1, vs_dst=2, n=64, addr_a=0x140000)
    assert (c.opcode, c.vs_src, c.vs_dst, c.n, c.addr_a) == (Opcode.VSUBC, 1, 2, 64, 0x140000)

    kw = dict(vs_src=0, addr_a=0x100, addr_m=0x200, sreg_src=7, max_ctx=2048)
    kt = isa.kvwrite(**kw, transposed=True)
    kv = isa.kvwrite(**kw, transposed=False)
    assert kt.flags == isa.KvwriteFlag.TRANSPOSED and kv.flags == 0 and kt.n == kv.n == 64
    assert kt.k == kv.k == 2048  # the token capacity of the region
    with pytest.raises(ValueError):
        isa.kvwrite(**kw, transposed=True, n=32)
    with pytest.raises(ValueError):
        isa.kvwrite(**{**kw, "max_ctx": 0}, transposed=True)
    with pytest.raises(ValueError):
        isa.kvwrite(**{**kw, "sreg_src": 40}, transposed=False)
    with pytest.raises(TypeError):
        isa.kvwrite(vs_src=0, addr_a=0, addr_m=0, sreg_src=0, transposed=True)

    assert isa.fence().opcode is Opcode.FENCE and isa.halt().opcode is Opcode.HALT
    assert isa.nop() == Descriptor()


def test_sfloat_immediate_encoding() -> None:
    rng = np.random.default_rng(7)
    for _ in range(500):
        s = SFloat(int(rng.integers(1 << 15, 1 << 16)), int(rng.integers(-128, 128)))
        v = isa.sfloat_imm(s)
        assert v >> 24 == 0 and v & 0xFFFF == s.m and isa.sfloat_from_imm(v) == s
    assert isa.sfloat_imm(SFloat(0, 0)) == 0 and isa.sfloat_from_imm(0) == SFloat(0, 0)
    assert isa.sfloat_imm(SFloat(1 << 15, -1)) == (0xFF << 16) | (1 << 15)
    with pytest.raises(ValueError):
        isa.sfloat_imm(SFloat(1 << 15, 128))
    with pytest.raises(ValueError):
        isa.sfloat_from_imm(1 << 24)


# --------------------------------------------------------------------------- CSR map


def test_csr_map() -> None:
    words = {c.name: c.word for c in isa.CSRS}
    for name, word in DOCUMENTED_CSRS.items():
        assert words[name] == word, name
    assert len(words) == len(isa.CSRS) == 13 + 2 * isa.PERF_COUNT
    assert len(set(words.values())) == len(words) and max(words.values()) < isa.CSR_WORDS
    assert isa.CSR_WORDS == 64
    for i in range(isa.PERF_COUNT):
        lo, hi = isa.perf_words(i)
        assert (lo, hi) == (words[f"PERF{i}_LO"], words[f"PERF{i}_HI"]) == (16 + 2 * i, 17 + 2 * i)
    with pytest.raises(ValueError):
        isa.perf_words(16)
    assert isa.CSR_BY_NAME["CTRL"].access == "w1p" and isa.CSR_BY_NAME["PC"].access == "rw"
    assert all(isa.CSR_BY_NAME[n].access == "ro" for n in ("STATUS", "SAT_REQ", "PERF3_HI"))
    assert isa.CTRL_BITS == {"START": 0, "STEP": 1, "ABORT": 2}
    assert isa.STATUS_BITS == {"DONE": 0, "BUSY": 1, "STEP_HALTED": 2, "ERR": 3}
    assert sorted(isa.PERF_INDEX.values()) == list(range(16))
    assert isa.PERF_INDEX["CYCLES"] == 0 and isa.PERF_INDEX["BUSY"] == 1


def _parse(text: str, pattern: re.Pattern[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            assert m.group(1) not in out, m.group(1)
            out[m.group(1)] = int(m.group(2))
    return out


def test_generated_headers_agree_with_isa() -> None:
    """rtl/qcore_csr_defs.svh, sim/verilator/csr_defs.hpp and isa.definitions() are one table."""
    defs = dict(isa.definitions())
    assert len(defs) == len(isa.definitions())
    assert csrgen.check() == {csrgen.SVH_PATH: "ok", csrgen.HPP_PATH: "ok"}
    svh = _parse(csrgen.SVH_PATH.read_text(encoding="utf-8"), SV_LINE)
    hpp = _parse(csrgen.HPP_PATH.read_text(encoding="utf-8"), HPP_LINE)
    assert svh == defs and hpp == defs
    assert defs["ISA_VERSION"] == isa.ISA_VERSION == 1
    assert defs["CSR_CTRL"] == 0 and defs["CSR_PERF15_HI"] == 47 and defs["CSR_WORDS"] == 64
    assert defs["OP_GEMV"] == 0x10 and defs["DESC_SH1_LSB"] == 216 and defs["DESC_IMM32_W"] == 32
    assert defs["VQ_W8"] == 1 and defs["KVW_TRANSPOSED"] == 1 and defs["OUT_ARGMAX_DUMP"] == 2
    assert "KV_TILE_TOKENS" not in defs and defs["DUMP_ALIGN"] == 64
    # the include is macros only (a module or package restates what it uses), guarded, no imports
    sv_text = csrgen.svh_text()
    body = [ln for ln in sv_text.splitlines() if ln and not ln.startswith("//")]
    assert body[0] == "`ifndef QCORE_CSR_DEFS_SVH" and body[1] == "`define QCORE_CSR_DEFS_SVH"
    assert body[-1] == "`endif" and all(SV_LINE.match(ln) for ln in body[2:-1])
    assert "import" not in sv_text and "localparam" not in sv_text
    assert sv_text.startswith("// Generated")
    hpp_text = csrgen.hpp_text()
    assert "#pragma once" in hpp_text and "namespace qcore {" in hpp_text


SVH_PKG = """\
`include "qcore_csr_defs.svh"
package qcore_probe_pkg;
  localparam int OP_GEMV = `QCORE_OP_GEMV;
  localparam int CSR_CTRL = `QCORE_CSR_CTRL;
endpackage
"""
SVH_TOP = """\
module qcore_probe_top (
  input  logic       clk,
  input  logic [7:0] op,
  input  logic [5:0] word,
  output logic       is_gemv,
  output logic       is_ctrl,
  output logic       acc
);
  `include "qcore_csr_defs.svh"
  always_ff @(posedge clk) begin
    is_gemv <= (op == 8'(qcore_probe_pkg::OP_GEMV));
    is_ctrl <= (word == 6'(qcore_probe_pkg::CSR_CTRL));
    acc     <= op[`QCORE_DESC_ACCUMULATE_LSB - 24];
  end
endmodule
"""


def test_generated_svh_passes_the_three_parsers(tmp_path) -> None:
    """rtl/qcore_csr_defs.svh, included by a package and a module, passes the lint recipe."""
    tools = {t: shutil.which(t) for t in ("verilator", "yosys", "iverilog")}
    if not all(tools.values()):
        pytest.skip(f"lint tools not on PATH: {[t for t, p in tools.items() if p is None]}")
    (tmp_path / "qcore_probe_pkg.sv").write_text(SVH_PKG)
    (tmp_path / "qcore_probe_top.sv").write_text(SVH_TOP)
    inc = str(csrgen.SVH_PATH.parent)
    files = [str(tmp_path / "qcore_probe_pkg.sv"), str(tmp_path / "qcore_probe_top.sv")]
    runs = {
        "verilator": [
            tools["verilator"],
            "--lint-only",
            "-Wall",
            "-Wpedantic",
            "--top-module",
            "qcore_probe_top",
            f"-I{inc}",
            *files,
        ],
        "yosys": [
            tools["yosys"],
            "-q",
            "-p",
            f"read_verilog -sv -I{inc} {' '.join(files)}; hierarchy -check -top qcore_probe_top; "
            "proc; opt; check -assert",
        ],
        "iverilog": [
            tools["iverilog"],
            "-g2012",
            f"-I{inc}",
            "-s",
            "qcore_probe_top",
            "-o",
            str(tmp_path / "probe.vvp"),
            *files,
        ],
    }
    for name, cmd in runs.items():
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, f"{name}: {res.stdout}\n{res.stderr}"


def test_csr_defs_cli_check(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["csr-defs", "--check"]) == 0
    out = capsys.readouterr().out
    assert "ok       rtl/qcore_csr_defs.svh" in out and "ok       sim/verilator/csr_defs.hpp" in out
    assert csrgen.main(["--check"]) == 0


def test_helpers_reject_out_of_range_fixed_point_classes() -> None:
    import pytest
    from quettos import isa

    with pytest.raises(ValueError):
        isa.vsilumul(vs_src=0, vs_aux=64, vs_dst=128, n=64, frac_gu=5, sh_h=10, sreg_dst=0)
    with pytest.raises(ValueError):
        isa.vsoftmax(vs_src=0, vs_dst=64, n=64, frac_s=3, addr_a=64, sreg_dst=0)
