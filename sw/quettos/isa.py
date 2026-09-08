"""Descriptor ISA of Quettos Core: fields, opcodes, encoder, disassembler and the CSR map.

A program is a straight-line array of 256-bit little-endian descriptors ending
in HALT.  :class:`Descriptor` holds every field with its range checked,
:func:`encode` / :func:`decode` round-trip the 32 bytes, :func:`disassemble`
writes the ``.lst`` listing, and the per-opcode helpers (:func:`gemv`,
:func:`vquant`, ...) take the field meanings of ``docs/ISA.md`` as keyword
arguments so a compiler reads like the dataflow.  :data:`CSRS` is the register
map and :func:`definitions` the flat name/value table that
:mod:`quettos.csrgen` writes for the RTL and the harness.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum, IntFlag

from quettos.numerics import (
    E8_MAX,
    E8_MIN,
    SOFTMAX_FRAC_MAX,
    SOFTMAX_FRAC_MIN,
    SFloat,
    quant_scale_exponents,
)

ISA_VERSION = 1
DESC_BITS = 256
DESC_BYTES = DESC_BITS // 8
PROGRAM_ALIGN = 64  # programs start on a 64-byte beat, two descriptors per beat
SREG_COUNT = 32  # scale registers per row
VSRAM_WORD_ELEMS = 8  # int32 elements per 256-bit VSRAM word
META_BYTES = 8  # per-channel / per-token meta {i32 bias_q, u16 m, i8 e, u8 pad}
ROPE_ROW_BYTES = 128  # one RoPE table row: 32 cos then 32 sin, int16 Q1.14
ROPE_SHIFT = 14
HEAD_DIM = 64
EMBED_S1_RANGE = (8, 24)  # numerics.embed_dequant window for the EMBED stage-1 shift
SHIFT_MAX = 63  # every shift amount a descriptor carries lies in [0, SHIFT_MAX]
DUMP_ALIGN = 64  # a DUMP address is a whole beat


# --------------------------------------------------------------------------- enums


class Opcode(IntEnum):
    NOP = 0x00
    HALT = 0x01
    GEMV = 0x10
    EMBED = 0x11
    VRMSNORM = 0x20
    VQUANT = 0x21
    VROPE = 0x22
    VSILUMUL = 0x23
    VSOFTMAX = 0x24
    VSUBC = 0x25
    KVWRITE = 0x30
    FENCE = 0x31


class OutMode(IntEnum):
    """GEMV / EMBED output destination (descriptor bits ``[29:28]``)."""

    VSRAM = 0
    ARGMAX = 1
    ARGMAX_DUMP = 2
    VSRAM_DUMP = 3


class VquantFlag(IntFlag):
    """``flags`` bits of VQUANT."""

    W8 = 1 << 0  # int8 output (clear: int16)
    USE_TRACKED = 1 << 1  # absmax from SREG[sreg_src] instead of a pass over the data
    GROUP = 1 << 2  # one scale per vs_aux elements into SREG[sreg_dst + g]
    SCALE_MUL = 1 << 3  # every scale multiplied by the sfloat in imm32


class KvwriteFlag(IntFlag):
    """``flags`` bits of KVWRITE."""

    TRANSPOSED = 1 << 0  # K^T byte scatter (clear: one V row)


# --------------------------------------------------------------------------- descriptor fields


@dataclass(frozen=True)
class Field:
    """One descriptor field: bits ``[lsb + width - 1 : lsb]``; ``sh1`` is the only signed one."""

    name: str
    lsb: int
    width: int
    signed: bool = False

    @property
    def msb(self) -> int:
        return self.lsb + self.width - 1

    @property
    def bounds(self) -> tuple[int, int]:
        """Inclusive ``(min, max)`` of the field's value."""
        if self.signed:
            return -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        return 0, (1 << self.width) - 1


FIELDS: tuple[Field, ...] = (
    Field("opcode", 0, 8),
    Field("flags", 8, 8),
    Field("row_mask", 16, 8),
    Field("accumulate", 24, 1),
    Field("unit_meta", 25, 1),
    Field("n_from_pos", 26, 1),
    Field("k_from_pos", 27, 1),
    Field("out_mode", 28, 2),
    Field("len_from_pos", 30, 1),
    Field("track_absmax", 31, 1),
    Field("addr_a", 32, 32),
    Field("addr_m", 64, 32),
    Field("n", 96, 24),
    Field("k", 120, 16),
    Field("vs_src", 136, 16),
    Field("vs_dst", 152, 16),
    Field("vs_aux", 168, 16),
    Field("sreg_src", 184, 8),
    Field("sreg_dst", 192, 8),
    Field("src_row", 200, 4),
    Field("dst_row", 204, 4),
    Field("sh0", 208, 8),
    Field("sh1", 216, 8, signed=True),
    Field("imm32", 224, 32),
)
FIELD_BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}
_OPCODE_VALUES = frozenset(op.value for op in Opcode)


@dataclass(frozen=True)
class Descriptor:
    """One 256-bit descriptor; every field is range-checked on construction.

    One-bit fields are ``bool``, ``opcode`` and ``out_mode`` are their enums,
    ``sh1`` is a signed byte and everything else an unsigned integer of the
    field's width (``docs/ISA.md``, descriptor bit layout).  ``row_mask``
    defaults to row 0.
    """

    opcode: Opcode = Opcode.NOP
    flags: int = 0
    row_mask: int = 1
    accumulate: bool = False
    unit_meta: bool = False
    n_from_pos: bool = False
    k_from_pos: bool = False
    out_mode: OutMode = OutMode.VSRAM
    len_from_pos: bool = False
    track_absmax: bool = False
    addr_a: int = 0
    addr_m: int = 0
    n: int = 0
    k: int = 0
    vs_src: int = 0
    vs_dst: int = 0
    vs_aux: int = 0
    sreg_src: int = 0
    sreg_dst: int = 0
    src_row: int = 0
    dst_row: int = 0
    sh0: int = 0
    sh1: int = 0
    imm32: int = 0

    def __post_init__(self) -> None:
        for f in FIELDS:
            raw = getattr(self, f.name)
            if isinstance(raw, bool) and f.width != 1:
                raise TypeError(f"{f.name}: expected an integer, got a bool")
            value = int(raw)
            lo, hi = f.bounds
            if value < lo or value > hi:
                raise ValueError(f"{f.name} = {value} outside [{lo}, {hi}]")
            if f.name == "opcode":
                value = Opcode(value)
            elif f.name == "out_mode":
                value = OutMode(value)
            elif f.width == 1:
                value = bool(value)
            object.__setattr__(self, f.name, value)

    def word(self) -> int:
        """The descriptor as one 256-bit integer (bit ``i`` of the descriptor is bit ``i``)."""
        w = 0
        for f in FIELDS:
            v = int(getattr(self, f.name)) & ((1 << f.width) - 1)
            w |= v << f.lsb
        return w


DEFAULT = Descriptor()


def encode(d: Descriptor) -> bytes:
    """The 32 little-endian bytes: descriptor bit ``i`` is bit ``i % 8`` of byte ``i // 8``."""
    return d.word().to_bytes(DESC_BYTES, "little")


def opcode_of(data: bytes) -> int:
    """The opcode byte of an encoded descriptor (byte 0), known or not."""
    if len(data) != DESC_BYTES:
        raise ValueError(f"opcode_of: {len(data)} bytes, expected {DESC_BYTES}")
    return data[0]


def is_opcode(value: int) -> bool:
    """Whether ``value`` is one of the twelve defined opcodes."""
    return value in _OPCODE_VALUES


def decode(data: bytes) -> Descriptor:
    """Inverse of :func:`encode`; rejects a wrong length or an unknown opcode.

    A program that may carry an unknown opcode (the ``STATUS.ERR`` path) tests
    :func:`opcode_of` with :func:`is_opcode` before decoding.
    """
    if len(data) != DESC_BYTES:
        raise ValueError(f"decode: {len(data)} bytes, expected {DESC_BYTES}")
    w = int.from_bytes(data, "little")
    kw: dict[str, int] = {}
    for f in FIELDS:
        v = (w >> f.lsb) & ((1 << f.width) - 1)
        if f.signed and v >= 1 << (f.width - 1):
            v -= 1 << f.width
        kw[f.name] = v
    if kw["opcode"] not in _OPCODE_VALUES:
        raise ValueError(f"decode: unknown opcode 0x{kw['opcode']:02x}")
    return Descriptor(**kw)


def assemble(program: Iterable[Descriptor]) -> bytes:
    """Concatenated encodings of ``program`` (the ``.prog`` file content)."""
    return b"".join(encode(d) for d in program)


def parse(data: bytes) -> list[Descriptor]:
    """Split a ``.prog`` image into descriptors; the length must be a multiple of 32."""
    if len(data) % DESC_BYTES:
        raise ValueError(f"parse: {len(data)} bytes is not a multiple of {DESC_BYTES}")
    return [decode(data[i : i + DESC_BYTES]) for i in range(0, len(data), DESC_BYTES)]


# --------------------------------------------------------------------------- sfloat immediates


def sfloat_imm(s: SFloat) -> int:
    """sfloat in a descriptor word: ``m`` in bits ``[15:0]``, ``e`` as an i8 in ``[23:16]``."""
    if not -128 <= s.e <= 127:
        raise ValueError(f"sfloat_imm: exponent {s.e} does not fit an i8")
    return s.m | ((s.e & 0xFF) << 16)


def sfloat_from_imm(v: int) -> SFloat:
    """Inverse of :func:`sfloat_imm` (bits above 23 must be zero)."""
    if v >> 24:
        raise ValueError(f"sfloat_from_imm: bits above 23 set in 0x{v:08x}")
    e = (v >> 16) & 0xFF
    return SFloat(v & 0xFFFF, e - 256 if e >= 128 else e)


# --------------------------------------------------------------------------- disassembler

_HEX_FIELDS = frozenset({"addr_a", "addr_m"})


def _flags_text(d: Descriptor) -> str:
    if d.opcode == Opcode.VQUANT:
        names = [f.name for f in VquantFlag if d.flags & f.value]
        rest = d.flags & ~sum(f.value for f in VquantFlag)
    elif d.opcode == Opcode.KVWRITE:
        names = [f.name for f in KvwriteFlag if d.flags & f.value]
        rest = d.flags & ~sum(f.value for f in KvwriteFlag)
    else:
        names, rest = [], d.flags
    if rest:
        names.append(f"0x{rest:02x}")
    return "|".join(names) if names else "0"


def _sfloat_text(name: str, field: str, v: int) -> str:
    """``name={m,e}`` for a well-formed sfloat immediate, else the raw field in hex."""
    try:
        s = sfloat_from_imm(v)
    except ValueError:
        return f"{field}=0x{v:08x}"
    return f"{name}={{{s.m},{s.e}}}"


def _imm_text(d: Descriptor) -> str:
    if d.opcode in (Opcode.GEMV, Opcode.EMBED):
        return f"addr_c=0x{d.imm32:08x}"
    if d.opcode == Opcode.VRMSNORM:
        return f"eps_c={d.imm32}"
    if d.opcode == Opcode.VQUANT and d.flags & VquantFlag.SCALE_MUL:
        return _sfloat_text("scale_mul", "imm32", d.imm32)
    if d.opcode == Opcode.VSOFTMAX and not d.len_from_pos:
        return f"len={d.imm32}"
    return f"imm32=0x{d.imm32:08x}"


def disassemble_one(d: Descriptor) -> str:
    """Opcode name followed by every field that differs from :data:`DEFAULT`."""
    parts: list[str] = []
    for f in FIELDS:
        if f.name == "opcode":
            continue
        v = getattr(d, f.name)
        if v == getattr(DEFAULT, f.name):
            continue
        if f.name == "flags":
            parts.append(f"flags={_flags_text(d)}")
        elif f.name == "out_mode":
            parts.append(f"out_mode={OutMode(v).name}")
        elif f.name == "addr_m" and d.opcode == Opcode.VRMSNORM:
            parts.append(_sfloat_text("sqrt_d", "addr_m", v))
        elif f.name in _HEX_FIELDS:
            parts.append(f"{f.name}=0x{v:08x}")
        elif f.name == "imm32":
            parts.append(_imm_text(d))
        elif f.width == 1:
            parts.append(f.name)
        else:
            parts.append(f"{f.name}={int(v)}")
    return f"{d.opcode.name:<9}" + " ".join(parts) if parts else d.opcode.name


def disassemble(program: Sequence[Descriptor]) -> str:
    """The ``.lst`` listing: one ``index  OPCODE fields`` line per descriptor."""
    return "".join(f"{i:5d}  {disassemble_one(d)}\n" for i, d in enumerate(program))


# --------------------------------------------------------------------------- class windows

# The window an opcode's class field (``sh0``) is defined over.  Outside it the
# reference and the datapath are different functions, so the descriptor is
# refused at decode with ``FAULT = CLASS`` and the opcode byte in ``FAULT_OP``,
# the way an unknown opcode is refused; an opcode absent from this table has no
# class field the decoder checks.
CLASS_WINDOW: dict[Opcode, tuple[int, int]] = {
    Opcode.VSOFTMAX: (SOFTMAX_FRAC_MIN, SOFTMAX_FRAC_MAX),
}


def class_fault(d: Descriptor) -> bool:
    """True when ``d``'s class field leaves :data:`CLASS_WINDOW` (``FAULT = CLASS``)."""
    window = CLASS_WINDOW.get(Opcode(d.opcode))
    return window is not None and not window[0] <= d.sh0 <= window[1]


# --------------------------------------------------------------------------- opcode helpers


def _check_shift(name: str, value: int, lo: int, hi: int) -> None:
    if not lo <= value <= hi:
        raise ValueError(f"{name} = {value} outside [{lo}, {hi}]")


def _check_sreg(**indices: int) -> None:
    for name, value in indices.items():
        if not 0 <= value < SREG_COUNT:
            raise ValueError(f"{name} = {value} outside [0, {SREG_COUNT})")


def _check_dump(op: str, out_mode: OutMode, addr_c: int) -> None:
    """A DUMP needs a non-zero, beat-aligned ``addr_c``; the other modes carry ``addr_c = 0``."""
    if out_mode in (OutMode.ARGMAX_DUMP, OutMode.VSRAM_DUMP):
        if addr_c == 0 or addr_c % DUMP_ALIGN:
            raise ValueError(
                f"{op}: DUMP addr_c = 0x{addr_c:08x} must be a non-zero multiple of 64"
            )
    elif addr_c:
        raise ValueError(f"{op}: addr_c needs a DUMP out_mode")


def gemv(
    *,
    addr_a: int,
    n: int,
    k: int,
    vs_src: int,
    vs_dst: int,
    sreg_src: int,
    s1: int,
    sbias: int,
    addr_m: int = 0,
    accumulate: bool = False,
    unit_meta: bool = False,
    n_from_pos: bool = False,
    k_from_pos: bool = False,
    out_mode: OutMode = OutMode.VSRAM,
    addr_c: int = 0,
    track_absmax: bool = False,
    sreg_dst: int = 0,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """``y[n] = requant(W[n] . A, meta[n], SREG[sreg_src], s1, sbias)`` over the tiled int8 matrix.

    ``addr_a`` is the ``[N/WB][K][WB]`` weight base, ``addr_m`` the per-channel
    meta base (ignored with ``unit_meta``: unit scale, zero bias), ``A`` the low
    16 bits of ``vsram[vs_src .. vs_src+K-1]``.  ``n_from_pos`` / ``k_from_pos``
    derive N / K from POS at run time; the fields then carry the capacity.
    ``out_mode`` VSRAM writes ``vsram[vs_dst ..]`` (read-modify-write with
    ``accumulate``), ARGMAX updates the ARGMAX CSRs, the DUMP variants also
    write the int32 outputs to ``addr_c``.  ``s1``/``sbias`` are the requant
    shifts (``sh0``/``sh1``).
    """
    _check_shift("s1", s1, 0, SHIFT_MAX)
    _check_sreg(sreg_src=sreg_src, sreg_dst=sreg_dst)
    _check_dump("gemv", out_mode, addr_c)
    return Descriptor(
        opcode=Opcode.GEMV,
        row_mask=row_mask,
        accumulate=accumulate,
        unit_meta=unit_meta,
        n_from_pos=n_from_pos,
        k_from_pos=k_from_pos,
        out_mode=out_mode,
        track_absmax=track_absmax,
        addr_a=addr_a,
        addr_m=addr_m,
        n=n,
        k=k,
        vs_src=vs_src,
        vs_dst=vs_dst,
        sreg_src=sreg_src,
        sreg_dst=sreg_dst,
        src_row=src_row,
        dst_row=dst_row,
        sh0=s1,
        sh1=sbias,
        imm32=addr_c,
    )


def embed(
    *,
    addr_a: int,
    addr_m: int,
    k: int,
    vs_dst: int,
    s1: int,
    sbias: int,
    out_mode: OutMode = OutMode.VSRAM,
    addr_c: int = 0,
    track_absmax: bool = False,
    sreg_dst: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """Gather the ``k`` int8 bytes of row TOK from the tiled table at ``addr_a`` into ``vs_dst``.

    Each byte is dequantized through requant with ``acc = q << 24``, ``Sw`` from
    ``meta[addr_m + TOK*8]``, ``Sx = 1.0`` and the shifts ``s1`` (in
    ``[8, 24]``) and ``sbias = -(FRAC_X + s1) + 24``; ``n`` equals ``k``.
    """
    _check_shift("s1", s1, *EMBED_S1_RANGE)
    _check_sreg(sreg_dst=sreg_dst)
    _check_dump("embed", out_mode, addr_c)
    return Descriptor(
        opcode=Opcode.EMBED,
        row_mask=row_mask,
        out_mode=out_mode,
        track_absmax=track_absmax,
        addr_a=addr_a,
        addr_m=addr_m,
        n=k,
        k=k,
        vs_dst=vs_dst,
        sreg_dst=sreg_dst,
        dst_row=dst_row,
        sh0=s1,
        sh1=sbias,
        imm32=addr_c,
    )


def vrmsnorm(
    *,
    vs_src: int,
    vs_dst: int,
    n: int,
    addr_a: int,
    eps_c: int,
    frac_x: int,
    g: int,
    sqrt_d: SFloat,
    sreg_dst: int,
    track_absmax: bool = True,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """``vsram[vs_dst ..] = rmsnorm(vsram[vs_src .. +n], gamma at addr_a, eps_c, sqrt_d)``.

    ``sh0 = frac_x``, ``sh1 = g = -gamma_e`` (a shift, in ``[0, 63]``),
    ``imm32 = eps_c``, ``addr_m`` the sfloat ``sqrt_d`` (:func:`sfloat_imm`);
    the output absmax lands in ``SREG[sreg_dst]`` when ``track_absmax`` is set.
    """
    _check_shift("frac_x", frac_x, 0, 30)
    _check_shift("g", g, 0, SHIFT_MAX)
    _check_sreg(sreg_dst=sreg_dst)
    return Descriptor(
        opcode=Opcode.VRMSNORM,
        row_mask=row_mask,
        track_absmax=track_absmax,
        addr_a=addr_a,
        addr_m=sfloat_imm(sqrt_d),
        n=n,
        vs_src=vs_src,
        vs_dst=vs_dst,
        sreg_dst=sreg_dst,
        src_row=src_row,
        dst_row=dst_row,
        sh0=frac_x,
        sh1=g,
        imm32=eps_c,
    )


def vquant_scales(d: Descriptor) -> int:
    """Scale registers a VQUANT writes: ``SREG[sreg_dst .. sreg_dst + count - 1]``.

    ``ceil(n / vs_aux)`` in the ``GROUP`` form -- a group length that does not
    divide ``n`` leaves the last group short and still writes a scale for it --
    and one otherwise.  Zero for any other opcode.
    """
    if d.opcode is not Opcode.VQUANT:
        return 0
    if (d.flags & VquantFlag.GROUP) and d.vs_aux:
        return -(-d.n // d.vs_aux)
    return 1


def vquant(
    *,
    vs_src: int,
    vs_dst: int,
    n: int,
    width: int,
    frac_in: int,
    sreg_dst: int,
    use_tracked: bool = False,
    sreg_src: int = 0,
    group: int = 0,
    scale_mul: SFloat | None = None,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """``quant(vsram[vs_src .. +n], width, frac_in)`` into ``vs_dst`` with the scale in SREG.

    ``width`` is 16 or 8 (flag ``W8``); ``use_tracked`` takes the absmax from
    ``SREG[sreg_src]``; ``group > 0`` quantizes every ``group`` elements on
    their own (``vs_aux = group``) into ``SREG[sreg_dst + g]``; ``scale_mul``
    multiplies each scale by the sfloat carried in ``imm32``.  A group form
    writes at most the :data:`SREG_COUNT` scales the register file holds
    (:func:`vquant_scales`), and the scale exponent is an i8 in the SREG word
    the descriptor writes, so the fields have to keep it there for every input
    vector (:func:`numerics.quant_scale_exponents`).
    """
    _check_shift("frac_in", frac_in, 0, 30)
    if width not in (8, 16):
        raise ValueError("vquant: width must be 8 or 16")
    _check_sreg(sreg_src=sreg_src, sreg_dst=sreg_dst)
    flags = VquantFlag(0)
    if width == 8:
        flags |= VquantFlag.W8
    if use_tracked:
        flags |= VquantFlag.USE_TRACKED
    scales = 1
    if group:
        if group <= 0 or n % group:
            raise ValueError(f"vquant: n = {n} is not a multiple of group {group}")
        flags |= VquantFlag.GROUP
        scales = n // group
    if sreg_dst + scales > SREG_COUNT:
        raise ValueError(
            f"vquant: the group form writes {scales} scales into "
            f"SREG[{sreg_dst}..{sreg_dst + scales - 1}], past the {SREG_COUNT} "
            "registers the file holds"
        )
    lo, hi = quant_scale_exponents(width, frac_in, scale_mul)
    if lo < E8_MIN or hi > E8_MAX:
        raise ValueError(
            f"vquant: the scale exponent reaches [{lo}, {hi}], outside the i8 "
            f"[{E8_MIN}, {E8_MAX}] the descriptor and the SREG word carry"
        )
    imm = 0
    if scale_mul is not None:
        flags |= VquantFlag.SCALE_MUL
        imm = sfloat_imm(scale_mul)
    return Descriptor(
        opcode=Opcode.VQUANT,
        flags=int(flags),
        row_mask=row_mask,
        n=n,
        vs_src=vs_src,
        vs_dst=vs_dst,
        vs_aux=group,
        sreg_src=sreg_src,
        sreg_dst=sreg_dst,
        src_row=src_row,
        dst_row=dst_row,
        sh0=frac_in,
        imm32=imm,
    )


def vrope(*, vs_src: int, n: int, addr_a: int, src_row: int = 0, row_mask: int = 1) -> Descriptor:
    """RoPE in place on ``vsram[vs_src .. +n]`` (``n`` = heads x 64), row ``addr_a + POS*128``."""
    if n <= 0 or n % HEAD_DIM:
        raise ValueError(f"vrope: n = {n} is not a multiple of {HEAD_DIM}")
    return Descriptor(
        opcode=Opcode.VROPE, row_mask=row_mask, addr_a=addr_a, n=n, vs_src=vs_src, src_row=src_row
    )


def vsilumul(
    *,
    vs_src: int,
    vs_aux: int,
    vs_dst: int,
    n: int,
    frac_gu: int,
    sh_h: int,
    sreg_dst: int,
    track_absmax: bool = True,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """``vsram[vs_dst ..] = silu(vsram[vs_src ..]) * vsram[vs_aux ..]`` over ``n`` elements.

    ``sh0 = frac_gu`` (sigmoid index), ``sh1 = sh_h = 2 FRAC_GU - FRAC_H`` (a
    shift, in ``[0, 63]``); the output absmax lands in ``SREG[sreg_dst]`` when
    ``track_absmax`` is set.
    """
    _check_shift("frac_gu", frac_gu, 13, 30)
    _check_shift("sh_h", sh_h, 0, SHIFT_MAX)
    _check_sreg(sreg_dst=sreg_dst)
    return Descriptor(
        opcode=Opcode.VSILUMUL,
        row_mask=row_mask,
        track_absmax=track_absmax,
        n=n,
        vs_src=vs_src,
        vs_dst=vs_dst,
        vs_aux=vs_aux,
        sreg_dst=sreg_dst,
        src_row=src_row,
        dst_row=dst_row,
        sh0=frac_gu,
        sh1=sh_h,
    )


def vsoftmax(
    *,
    vs_src: int,
    vs_dst: int,
    n: int,
    addr_a: int,
    frac_s: int,
    sreg_dst: int,
    length: int | None = None,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """Softmax over ``vsram[vs_src .. +len]`` with the V-scale meta at ``addr_a``.

    ``len`` is ``POS + 1`` (``len_from_pos``, the default) or the immediate
    ``length``; ``n`` is the row capacity (``MAX_CTX``).  int16 weights go to
    ``vs_dst`` (zeros from ``len`` to ``n``) and ``SREG_out`` to
    ``SREG[sreg_dst]``; ``sh0 = frac_s``.
    """
    _check_shift("frac_s", frac_s, SOFTMAX_FRAC_MIN, SOFTMAX_FRAC_MAX)
    if length is not None and not 1 <= length <= n:
        raise ValueError(f"vsoftmax: length {length} outside [1, {n}]")
    _check_sreg(sreg_dst=sreg_dst)
    return Descriptor(
        opcode=Opcode.VSOFTMAX,
        row_mask=row_mask,
        len_from_pos=length is None,
        addr_a=addr_a,
        n=n,
        vs_src=vs_src,
        vs_dst=vs_dst,
        sreg_dst=sreg_dst,
        src_row=src_row,
        dst_row=dst_row,
        sh0=frac_s,
        imm32=0 if length is None else length,
    )


def vsubc(
    *,
    vs_src: int,
    vs_dst: int,
    n: int,
    addr_a: int,
    src_row: int = 0,
    dst_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """``dst = sat32(src - c)`` over ``n`` elements, ``c`` the int32 row at ``addr_a``."""
    return Descriptor(
        opcode=Opcode.VSUBC,
        row_mask=row_mask,
        addr_a=addr_a,
        n=n,
        vs_src=vs_src,
        vs_dst=vs_dst,
        src_row=src_row,
        dst_row=dst_row,
    )


def kvwrite(
    *,
    vs_src: int,
    addr_a: int,
    addr_m: int,
    sreg_src: int,
    transposed: bool,
    max_ctx: int,
    n: int = HEAD_DIM,
    src_row: int = 0,
    row_mask: int = 1,
) -> Descriptor:
    """Write the ``n = 64`` int8 values at ``vs_src`` and the meta ``{0, SREG[sreg_src]}``.

    ``k = max_ctx`` is the token capacity of the region.  ``transposed``
    scatters byte ``d`` into the K^T tiles at ``addr_a + (POS/WB)*64*WB + d*WB
    + POS%WB``; otherwise the V tiles receive byte ``d`` at
    ``addr_a + ((d/WB)*max_ctx + POS)*WB + d%WB`` (one zero-padded WB-byte row
    per token when ``WB >= 64``).  The meta goes to ``addr_m + POS*8``.  A
    ``POS`` at or above ``max_ctx`` writes nothing and counts in ``ERR_BOUNDS``.
    """
    if n != HEAD_DIM:
        raise ValueError(f"kvwrite: n must be {HEAD_DIM}")
    if max_ctx < 1:
        raise ValueError("kvwrite: max_ctx must be positive")
    _check_sreg(sreg_src=sreg_src)
    return Descriptor(
        opcode=Opcode.KVWRITE,
        flags=int(KvwriteFlag.TRANSPOSED) if transposed else 0,
        row_mask=row_mask,
        addr_a=addr_a,
        addr_m=addr_m,
        n=n,
        k=max_ctx,
        vs_src=vs_src,
        sreg_src=sreg_src,
        src_row=src_row,
    )


def fence() -> Descriptor:
    """Wait until every issued write has been acknowledged."""
    return Descriptor(opcode=Opcode.FENCE)


def halt() -> Descriptor:
    """End of program: sets ``STATUS.DONE`` and snapshots the PERF counters."""
    return Descriptor(opcode=Opcode.HALT)


def nop() -> Descriptor:
    return Descriptor(opcode=Opcode.NOP)


# --------------------------------------------------------------------------- CSR map


@dataclass(frozen=True)
class Csr:
    """One 32-bit control/status register at word offset ``word`` (byte ``4 * word``)."""

    name: str
    word: int
    access: str  # "rw" host read/write, "ro" read-only, "w1p" write-one-to-pulse
    doc: str
    width: int = 32


CSR_WORDS = 64  # the CSR window: 64 words = 256 bytes
PERF_BASE = 16  # PERF[i] low half at PERF_BASE + 2 i, high half at PERF_BASE + 2 i + 1
PERF_COUNT = 16


class Fault(IntEnum):
    """``STATUS.FAULT``: why the sequencer stopped the program (0 while it runs)."""

    NONE = 0
    OPCODE = 1  # the opcode byte is none of the twelve
    ROW = 2  # a participating row addresses a VSRAM / SREG bank at or above B_MAX
    PC_ALIGN = 3  # START or STEP with a PC that is not a multiple of DESC_BYTES
    CLASS = 4  # a class field outside CLASS_WINDOW, the set the opcode is defined over


CTRL_BITS: dict[str, int] = {"START": 0, "STEP": 1, "ABORT": 2}
STATUS_BITS: dict[str, int] = {"DONE": 0, "BUSY": 1, "STEP_HALTED": 2, "ERR": 3}
# multi-bit STATUS fields, valid while ERR is set: name -> (lsb, width)
STATUS_FIELDS: dict[str, tuple[int, int]] = {"FAULT": (4, 4), "FAULT_OP": (8, 8)}


def status_word(
    *,
    done: bool = False,
    busy: bool = False,
    step_halted: bool = False,
    err: bool = False,
    fault: Fault = Fault.NONE,
    fault_op: int = 0,
) -> int:
    """The 32-bit ``STATUS`` word from its bits and fault fields (``docs/ISA.md``, CSR table)."""
    if not 0 <= int(fault_op) < 256:
        raise ValueError(f"fault_op = {fault_op} does not fit a byte")
    word = 0
    for name, value in (
        ("DONE", done),
        ("BUSY", busy),
        ("STEP_HALTED", step_halted),
        ("ERR", err),
    ):
        word |= int(bool(value)) << STATUS_BITS[name]
    word |= int(Fault(fault)) << STATUS_FIELDS["FAULT"][0]
    word |= int(fault_op) << STATUS_FIELDS["FAULT_OP"][0]
    return word


def status_fault(word: int) -> tuple[Fault, int]:
    """``(fault, opcode byte)`` of a ``STATUS`` word; ``(Fault.NONE, 0)`` when no fault is set."""
    lsb, width = STATUS_FIELDS["FAULT"]
    code = Fault((word >> lsb) & ((1 << width) - 1))
    lsb, width = STATUS_FIELDS["FAULT_OP"]
    return code, (word >> lsb) & ((1 << width) - 1)


PERF_INDEX: dict[str, int] = {
    "CYCLES": 0,
    "BUSY": 1,
    "MAC_ACTIVE": 2,
    "STALL_MEM": 3,
    "STALL_VPU": 4,
    "STALL_KV": 5,
    "STALL_SEQ": 6,
    "STALL_DRAIN": 7,
    "RD_BEATS": 8,
    "RD_BYTES": 9,
    "WT_BYTES": 10,
    "WR_BEATS": 11,
    "WR_BYTES": 12,
    "MACS": 13,
    "DESCRIPTORS": 14,
    "FETCH_BEATS": 15,
}


def _perf_csrs() -> tuple[Csr, ...]:
    out = []
    for i in range(PERF_COUNT):
        out.append(Csr(f"PERF{i}_LO", PERF_BASE + 2 * i, "ro", f"PERF[{i}] bits [31:0]"))
        out.append(Csr(f"PERF{i}_HI", PERF_BASE + 2 * i + 1, "ro", f"PERF[{i}] bits [63:32]"))
    return tuple(out)


CSRS: tuple[Csr, ...] = (
    Csr("CTRL", 0, "w1p", "bit 0 START, bit 1 STEP, bit 2 ABORT"),
    Csr(
        "STATUS",
        1,
        "w1c",
        "bit 0 DONE, bit 1 BUSY, bit 2 STEP_HALTED, bit 3 ERR, [7:4] FAULT, [15:8] FAULT_OP",
    ),
    Csr("PC", 2, "rw", "byte address of the next descriptor"),
    Csr("ROW_EN", 3, "rw", "bit r enables activation row r"),
    Csr("TOK", 4, "rw", "token id for EMBED"),
    Csr("POS", 5, "rw", "position of the token"),
    Csr("ARGMAX_TOK", 6, "ro", "index of the largest ARGMAX-mode output"),
    Csr("ARGMAX_VAL", 7, "ro", "that output (int32)"),
    Csr("SAT_REQ", 8, "ro", "requant saturations"),
    Csr("SAT_VPU", 9, "ro", "vector-unit saturations"),
    Csr("ERR_SHIFT", 10, "ro", "shifts clamped to [0, 63]"),
    Csr(
        "ERR_BOUNDS",
        11,
        "ro",
        "POS-derived values above their field, VSOFTMAX len outside [1, n], KVWRITE at POS >= k, "
        "VSRAM ranges past the end, SREG indices >= 32",
    ),
    Csr("ISA_VERSION", 12, "ro", "constant ISA_VERSION"),
    *_perf_csrs(),
)
CSR_BY_NAME: dict[str, Csr] = {c.name: c for c in CSRS}


def perf_words(index: int) -> tuple[int, int]:
    """Word offsets ``(lo, hi)`` of ``PERF[index]``."""
    if not 0 <= index < PERF_COUNT:
        raise ValueError(f"perf_words: index {index} outside [0, {PERF_COUNT})")
    return PERF_BASE + 2 * index, PERF_BASE + 2 * index + 1


def definitions() -> list[tuple[str, int]]:
    """Every ISA constant as ``(NAME, value)``, the table the generated headers are written from.

    The SystemVerilog side receives them as ```define QCORE_<NAME>`` macros
    (``rtl/qcore_csr_defs.svh``), the C++ side as ``constexpr`` values.
    Descriptor fields appear as ``DESC_<FIELD>_LSB`` / ``DESC_<FIELD>_W``,
    opcodes as ``OP_<NAME>``, output modes as ``OUT_<NAME>``, VQUANT and
    KVWRITE flag masks as ``VQ_<NAME>`` / ``KVW_<NAME>``, CTRL and STATUS bits
    as ``CTRL_<NAME>`` / ``STATUS_<NAME>``, register offsets as ``CSR_<NAME>``
    and PERF indices as ``PERF_<NAME>``.
    """
    defs: list[tuple[str, int]] = [
        ("ISA_VERSION", ISA_VERSION),
        ("DESC_BYTES", DESC_BYTES),
        ("PROGRAM_ALIGN", PROGRAM_ALIGN),
        ("SREG_COUNT", SREG_COUNT),
        ("VSRAM_WORD_ELEMS", VSRAM_WORD_ELEMS),
        ("META_BYTES", META_BYTES),
        ("ROPE_ROW_BYTES", ROPE_ROW_BYTES),
        ("ROPE_SHIFT", ROPE_SHIFT),
        ("HEAD_DIM", HEAD_DIM),
        ("DUMP_ALIGN", DUMP_ALIGN),
        ("SOFTMAX_FRAC_MIN", SOFTMAX_FRAC_MIN),
        ("SOFTMAX_FRAC_MAX", SOFTMAX_FRAC_MAX),
    ]
    for f in FIELDS:
        defs.append((f"DESC_{f.name.upper()}_LSB", f.lsb))
        defs.append((f"DESC_{f.name.upper()}_W", f.width))
    defs += [(f"OP_{op.name}", op.value) for op in Opcode]
    defs += [(f"OUT_{m.name}", m.value) for m in OutMode]
    defs += [(f"VQ_{fl.name}", fl.value) for fl in VquantFlag]
    defs += [(f"KVW_{fl.name}", fl.value) for fl in KvwriteFlag]
    defs += [(f"CTRL_{name}", bit) for name, bit in CTRL_BITS.items()]
    defs += [(f"STATUS_{name}", bit) for name, bit in STATUS_BITS.items()]
    for name, (lsb, width) in STATUS_FIELDS.items():
        defs += [(f"STATUS_{name}_LSB", lsb), (f"STATUS_{name}_W", width)]
    defs += [(f"FAULT_{f.name}", f.value) for f in Fault]
    defs.append(("CSR_WORDS", CSR_WORDS))
    defs += [(f"CSR_{c.name}", c.word) for c in CSRS]
    defs += [("PERF_BASE", PERF_BASE), ("PERF_COUNT", PERF_COUNT)]
    defs += [(f"PERF_{name}", idx) for name, idx in PERF_INDEX.items()]
    names = [n for n, _ in defs]
    if len(set(names)) != len(names):
        raise AssertionError("definitions: duplicate name")
    return defs


def check_tables() -> None:
    """Structural invariants of the field list and the CSR map (called by the tests)."""
    if [f.name for f in FIELDS] != [f.name for f in fields(Descriptor)]:
        raise AssertionError("FIELDS and Descriptor disagree")
    pos = 0
    for f in FIELDS:
        if f.lsb != pos:
            raise AssertionError(f"{f.name}: gap or overlap at bit {pos}")
        pos += f.width
    if pos != DESC_BITS:
        raise AssertionError(f"fields cover {pos} bits, not {DESC_BITS}")
    words = [c.word for c in CSRS]
    if len(set(words)) != len(words) or max(words) >= CSR_WORDS:
        raise AssertionError("CSR offsets collide or leave the window")
    if any(c.access not in ("rw", "ro", "w1p", "w1c") for c in CSRS):
        raise AssertionError("unknown CSR access")
    taken = set(STATUS_BITS.values())
    for name, (lsb, width) in STATUS_FIELDS.items():
        bits = set(range(lsb, lsb + width))
        if bits & taken or lsb + width > 32:
            raise AssertionError(f"STATUS field {name} overlaps or leaves the word")
        taken |= bits
    if sorted(PERF_INDEX.values()) != list(range(PERF_COUNT)):
        raise AssertionError("PERF_INDEX must enumerate 0 .. PERF_COUNT-1")
