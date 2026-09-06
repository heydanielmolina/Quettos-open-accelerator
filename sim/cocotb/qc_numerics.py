"""Adapter between the cocotb tests and sw/quettos/numerics.py.

Exposes the integer primitives the RTL mirrors (``round_shift``, ``sat``,
``requant``, ``sfloat_mul``, the lookup tables) and the bit-level packing of
descriptor fields, SREG words, meta records and QMEM beats as the RTL sees
them (``docs/RTL.md``).  Everything returns Python ints.
"""

from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np
from quettos import isa, numerics
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat, Stats

round_shift = numerics.round_shift
sat = numerics.sat
sfloat_mul = numerics.sfloat_mul
sfloat_from_int = numerics.sfloat_from_int
bitlen = numerics.bitlen
absmax = numerics.absmax
encode_descriptor = isa.encode
Descriptor = isa.Descriptor

__all__ = [
    "Descriptor",
    "SFLOAT_ONE",
    "SFLOAT_ZERO",
    "SFloat",
    "Stats",
    "absmax",
    "beat_to_int8s",
    "bitlen",
    "descriptor_word",
    "encode_descriptor",
    "from_signed",
    "int8s_to_beat",
    "meta_bytes",
    "meta_record56",
    "requant",
    "round_shift",
    "sat",
    "sfloat_from_int",
    "sfloat_mul",
    "sreg_pack",
    "sreg_unpack",
    "tables",
    "to_signed",
]


@lru_cache(maxsize=1)
def tables() -> numerics.Tables:
    """The checked-in lookup tables (``sw/quettos/tables/luts.json``)."""
    return numerics.load_tables()


def requant(
    acc: int,
    sw: SFloat,
    sx: SFloat,
    s1: int,
    sbias: int,
    bias_q: int = 0,
    old: int | None = None,
    stats: Stats | None = None,
) -> int:
    """``numerics.requant`` on one accumulator, as a Python int."""
    return int(numerics.requant(acc, sw, sx, s1, sbias, bias_q, old, stats))


def to_signed(value: int, bits: int) -> int:
    """Two's-complement interpretation of the low ``bits`` of ``value``."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def from_signed(value: int, bits: int) -> int:
    """The low ``bits`` of a signed integer as an unsigned field."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= value <= hi:
        raise ValueError(f"{value} does not fit {bits} signed bits")
    return value & ((1 << bits) - 1)


def sreg_pack(s: SFloat | int) -> int:
    """SREG word: an sfloat as ``{8'0, e[7:0], m[15:0]}`` or a tracked absmax as a u32."""
    if isinstance(s, SFloat):
        return s.m | (from_signed(s.e, 8) << 16)
    if not 0 <= s < 1 << 32:
        raise ValueError("tracked absmax must be a u32")
    return int(s)


def sreg_unpack(word: int) -> SFloat:
    """The sfloat held by an SREG word (``m`` in [15:0], ``e`` as an i8 in [23:16])."""
    m = word & 0xFFFF
    e = to_signed(word >> 16, 8)
    return SFLOAT_ZERO if m == 0 else SFloat(m, e)


def meta_bytes(bias_q: int, s: SFloat) -> bytes:
    """The 8-byte QMEM meta record ``{i32 bias_q, u16 m, i8 e, u8 pad}``."""
    return struct.pack("<iHbB", bias_q, s.m, s.e, 0)


def meta_record56(bias_q: int, s: SFloat) -> int:
    """The 56-bit meta side-stream record ``{e[7:0], m[15:0], bias_q[31:0]}``."""
    return from_signed(bias_q, 32) | (s.m << 32) | (from_signed(s.e, 8) << 48)


def int8s_to_beat(values: np.ndarray | list[int]) -> int:
    """A WB-byte beat from int8 lanes: lane ``j`` is byte ``j`` (bits ``[8j+7:8j]``)."""
    arr = np.asarray(values, dtype=np.int64)
    return int.from_bytes(arr.astype(np.int8).tobytes(), "little")


def beat_to_int8s(beat: int, wb: int) -> np.ndarray:
    """Inverse of :func:`int8s_to_beat` for a ``wb``-byte beat."""
    return np.frombuffer(beat.to_bytes(wb, "little"), dtype=np.int8).astype(np.int64)


def descriptor_word(d: Descriptor) -> int:
    """The 256-bit descriptor as one integer (bit ``i`` of the word is descriptor bit ``i``)."""
    return d.word()
