"""Compiler: :class:`~quettos.quantize.QuantModel` -> memory image, programs and layout.

:func:`compile` lays a quantized model out in QMEM (``image.bin``: programs,
RoPE table, K-centering rows, tiled int8 weights with per-channel meta, int16
gammas, the tied embedding / LM-head table and the zeroed KV region), allocates
the VSRAM element map and the scale registers, emits ``decode.prog`` and
``prefill.prog`` from the :mod:`quettos.isa` helpers with the requant constants
of :mod:`quettos.program`, and writes ``layout.json``, ``dump_plan.json``, the
listings, ``tokens.bin`` and ``prompt.tokens``.  Every output is a pure
function of the model and the parameters.  :class:`Image` reads an image back.
Address map and file formats: ``docs/MEMORY_MAP.md``; descriptors: ``docs/ISA.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quettos import golden, isa, numerics, program
from quettos.isa import Descriptor, Opcode, OutMode, VquantFlag
from quettos.model import BUILD_DIR, REPO_ROOT, ModelSpec
from quettos.quantize import QuantLinear, QuantModel, QuantNorm
from quettos.tokenizer_io import prompt_tokens, write_tokens_bin

WB = 64  # weight-port width in bytes: tile width of every weight matrix and of the K^T cache
VSRAM_WORDS = 4096  # 256-bit words; 8 int32 elements each
ALIGN = 64  # every QMEM region starts on a 64-byte beat
MAX_CTX = program.MAX_CTX
HEAD_DIM = isa.HEAD_DIM
PROG_BASE = 0x0000_0000
ROPE_BASE = 0x0010_0000
CONST_BASE = 0x0014_0000
WEIGHT_BASE = 0x0020_0000
IMAGES_DIR = BUILD_DIR / "images"
META_DTYPE = np.dtype([("bias_q", "<i4"), ("m", "<u2"), ("e", "i1"), ("pad", "u1")])
TILE_CHUNK_BYTES = 16 << 20  # weight bytes tiled per numpy call (bounds the temporaries)
_ZERO_CHUNK = bytes(1 << 20)
VSRAM_ORDER: tuple[str, ...] = ("X", "XN", "A", "QKV", "CTX", "CTXQ", "GU", "HQ", "S", "W")
LINEARS: tuple[str, ...] = ("wqkv", "wo", "wgu", "wdown")
FILES: dict[str, str] = {
    "image": "image.bin",
    "decode": "decode.prog",
    "prefill": "prefill.prog",
    "decode_lst": "program.lst",
    "prefill_lst": "prefill.lst",
    "layout": "layout.json",
    "dump_plan": "dump_plan.json",
    "tokens_bin": "tokens.bin",
    "prompt": "prompt.tokens",
}
ARGMAX_CSRS: tuple[str, ...] = ("ARGMAX_TOK", "ARGMAX_VAL")


def align_up(x: int, a: int = ALIGN) -> int:
    return -(-x // a) * a


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_text(obj: Any) -> str:
    """Canonical JSON of an output file: sorted keys, indent 1, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=1) + "\n"


# --------------------------------------------------------------------------- tiling and meta


def tiles_for(n: int, wb: int) -> int:
    """Number of ``wb``-wide output tiles of an ``n``-channel matrix (the last may be partial)."""
    return -(-n // wb)


def tiled_bytes(n: int, k: int, wb: int) -> int:
    """Bytes of an int8 ``[n, k]`` matrix tiled ``[tiles][k][wb]`` with zero-padded last tile."""
    return tiles_for(n, wb) * k * wb


def iter_weight_tiles(
    q: np.ndarray, wb: int, chunk_bytes: int = TILE_CHUNK_BYTES
) -> Iterator[bytes]:
    """The ``[N/WB][K][WB]`` tiling of int8 ``q[N, K]`` as address-ordered byte chunks.

    Byte ``(tile*K + k)*WB + j`` holds ``q[tile*WB + j, k]``; rows past ``N`` in
    the last tile are zero.  Each chunk covers whole tiles and at most about
    ``chunk_bytes`` bytes.
    """
    q = np.asarray(q)
    if q.dtype != np.int8 or q.ndim != 2:
        raise ValueError("iter_weight_tiles: q must be an int8 [N, K] matrix")
    n, k = q.shape
    tiles = tiles_for(n, wb)
    per = max(1, chunk_bytes // (k * wb))
    for t0 in range(0, tiles, per):
        t1 = min(tiles, t0 + per)
        rows = np.zeros(((t1 - t0) * wb, k), dtype=np.int8)
        src = q[t0 * wb : min(n, t1 * wb)]
        rows[: src.shape[0]] = src
        yield np.ascontiguousarray(rows.reshape(t1 - t0, wb, k).transpose(0, 2, 1)).tobytes()


def tile_weights(q: np.ndarray, wb: int) -> bytes:
    return b"".join(iter_weight_tiles(q, wb))


def untile_weights(data: bytes, n: int, k: int, wb: int) -> np.ndarray:
    """Inverse of :func:`tile_weights`: the int8 ``[n, k]`` matrix (padding rows dropped)."""
    tiles = tiles_for(n, wb)
    if len(data) != tiles * k * wb:
        raise ValueError(f"untile_weights: {len(data)} bytes, expected {tiles * k * wb}")
    arr = np.frombuffer(data, dtype=np.int8).reshape(tiles, k, wb).transpose(0, 2, 1)
    return np.ascontiguousarray(arr.reshape(tiles * wb, k)[:n])


def pack_meta(bias_q: np.ndarray, m: np.ndarray, e: np.ndarray, count: int) -> bytes:
    """``count`` little-endian ``{i32 bias_q, u16 m, i8 e, u8 0}`` records; padded ones are zero.

    ``m`` is a valid sfloat mantissa (``[2^15, 2^16)`` or 0 with ``e = 0``) and
    ``e`` fits an i8; the arrays give the first ``len(m)`` channels.
    """
    b = np.asarray(bias_q, dtype=np.int64)
    mm = np.asarray(m, dtype=np.int64)
    ee = np.asarray(e, dtype=np.int64)
    n = mm.size
    if b.shape != (n,) or ee.shape != (n,) or n > count:
        raise ValueError("pack_meta: arrays disagree or exceed the record count")
    zero = mm == 0
    if np.any(ee[zero] != 0) or np.any((mm[~zero] < 1 << 15) | (mm[~zero] >= 1 << 16)):
        raise ValueError("pack_meta: mantissa outside [2^15, 2^16) or a non-canonical zero")
    if np.any(ee < -128) or np.any(ee > 127):
        raise ValueError("pack_meta: exponent does not fit an i8")
    if np.any(b < -(1 << 31)) or np.any(b > numerics.I32_MAX):
        raise ValueError("pack_meta: bias does not fit an int32")
    out = np.zeros(count, dtype=META_DTYPE)
    out["bias_q"][:n] = b
    out["m"][:n] = mm
    out["e"][:n] = ee
    return out.tobytes()


def unpack_meta(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(bias_q, m, e)`` int64 arrays of a meta region."""
    if len(data) % META_DTYPE.itemsize:
        raise ValueError("unpack_meta: length is not a multiple of 8")
    arr = np.frombuffer(data, dtype=META_DTYPE)
    return (
        arr["bias_q"].astype(np.int64),
        arr["m"].astype(np.int64),
        arr["e"].astype(np.int64),
    )


def rope_bytes(theta: float, max_ctx: int) -> bytes:
    """Rows ``0 .. max_ctx-1`` of the checked-in table: 32 cos then 32 sin int16 per row."""
    table = numerics.load_rope_table(theta)
    if max_ctx > table.shape[0]:
        raise ValueError(f"rope_bytes: {max_ctx} rows exceed the table ({table.shape[0]})")
    return np.ascontiguousarray(table[:max_ctx], dtype="<i2").tobytes()


# --------------------------------------------------------------------------- on-chip maps


@dataclass(frozen=True)
class VsramMap:
    """Element ranges ``name -> (start, count)`` in :data:`VSRAM_ORDER`; every start 8-aligned."""

    regions: dict[str, tuple[int, int]]
    elements: int
    used: int

    def start(self, name: str) -> int:
        return self.regions[name][0]

    def as_json(self) -> dict[str, Any]:
        return {
            "elements": self.elements,
            "words": self.elements // isa.VSRAM_WORD_ELEMS,
            "used": self.used,
            "map": [{"name": n, "start": s, "count": c} for n, (s, c) in self.regions.items()],
        }


def vsram_map(model: QuantModel, max_ctx: int, vsram_words: int = VSRAM_WORDS) -> VsramMap:
    """Allocate ``X, XN, A, QKV, CTX, CTXQ, GU, HQ, S, W`` in that order; raise on overflow."""
    hid, hd, kvd = model.hidden, model.heads * HEAD_DIM, model.kv_heads * HEAD_DIM
    sizes = {
        "X": hid,
        "XN": hid,
        "A": hid,
        "QKV": hd + 2 * kvd,
        "CTX": hd,
        "CTXQ": hd,
        "GU": 2 * model.intermediate,
        "HQ": model.intermediate,
        "S": max_ctx,
        "W": max_ctx,
    }
    regions: dict[str, tuple[int, int]] = {}
    pos = 0
    for name in VSRAM_ORDER:
        pos = align_up(pos, isa.VSRAM_WORD_ELEMS)
        regions[name] = (pos, sizes[name])
        pos += sizes[name]
    elements = vsram_words * isa.VSRAM_WORD_ELEMS
    if pos > elements:
        raise ValueError(f"VSRAM: {pos} elements needed, {elements} available")
    return VsramMap(regions, elements, pos)


@dataclass(frozen=True)
class SregMap:
    """Scale-register assignment: fixed slots, then one per query head and per KV head (K, V)."""

    absmax: int
    a: int
    ctx: int
    h: int
    w: int
    q0: int
    k0: int
    v0: int
    heads: int
    kv_heads: int

    @property
    def count(self) -> int:
        return self.v0 + self.kv_heads

    def as_json(self) -> dict[str, Any]:
        return {
            "ABSMAX": self.absmax,
            "A": self.a,
            "CTX": self.ctx,
            "H": self.h,
            "W": self.w,
            "Q": list(range(self.q0, self.q0 + self.heads)),
            "K": list(range(self.k0, self.k0 + self.kv_heads)),
            "V": list(range(self.v0, self.v0 + self.kv_heads)),
            "used": self.count,
            "count": isa.SREG_COUNT,
        }


def sreg_map(model: QuantModel) -> SregMap:
    q0 = 5
    k0 = q0 + model.heads
    v0 = k0 + model.kv_heads
    m = SregMap(0, 1, 2, 3, 4, q0, k0, v0, model.heads, model.kv_heads)
    if m.count > isa.SREG_COUNT:
        raise ValueError(f"SREG: {m.count} registers needed, {isa.SREG_COUNT} available")
    return m


# --------------------------------------------------------------------------- QMEM regions


@dataclass(frozen=True)
class Region:
    """A QMEM region: ``size`` payload bytes at the 64-byte-aligned ``addr``; ``info`` per kind."""

    name: str
    addr: int
    size: int
    kind: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.addr + self.size

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "addr": self.addr,
            "size": self.size,
            "kind": self.kind,
            **self.info,
        }


class _Allocator:
    def __init__(self, base: int) -> None:
        self.next = base
        self.regions: list[Region] = []

    def jump(self, base: int, what: str) -> None:
        if self.next > base:
            raise ValueError(f"{what} end 0x{self.next:08x} past the next base 0x{base:08x}")
        self.next = base

    def add(self, name: str, size: int, kind: str, **info: Any) -> Region:
        r = Region(name, self.next, size, kind, dict(info))
        self.regions.append(r)
        self.next = align_up(r.end)
        return r


def kv_sizes(max_ctx: int, wb: int) -> dict[str, int]:
    """Bytes per (layer, KV head): K^T tiles, V tiles, K meta and V meta.

    Both caches are the weight tiling: K^T has the tokens as output channels
    and ``K = 64`` (``[max_ctx/WB][64][WB]``), V has the 64 dimensions as
    channels and the tokens as ``K`` (``[ceil(64/WB)][max_ctx][WB]``).
    """
    return {
        "kt": tiles_for(max_ctx, wb) * HEAD_DIM * wb,
        "v": tiles_for(HEAD_DIM, wb) * max_ctx * wb,
        "k_meta": max_ctx * isa.META_BYTES,
        "v_meta": max_ctx * isa.META_BYTES,
    }


def plan_regions(model: QuantModel, max_ctx: int, wb: int) -> list[Region]:
    """Every region above the programs, in address order (``docs/MEMORY_MAP.md``)."""
    a = _Allocator(ROPE_BASE)
    a.add("rope", max_ctx * isa.ROPE_ROW_BYTES, "table", rows=max_ctx, theta=model.rope_theta)
    a.jump(CONST_BASE, "RoPE table")
    kv, hid = model.kv_heads, model.hidden
    for i in range(model.n_layers):
        a.add(f"kcenter.{i}", kv * HEAD_DIM * 4, "const", layer=i, shape=[kv, HEAD_DIM])
    a.jump(WEIGHT_BASE, "constants")

    def linear(name: str, lin: QuantLinear) -> None:
        n, k = lin.q.shape
        t = tiles_for(n, wb)
        a.add(name, tiled_bytes(n, k, wb), "weights", shape=[n, k], tiles=t)
        a.add(f"{name}.meta", t * wb * isa.META_BYTES, "meta", count=t * wb, channels=n)

    def gamma(name: str, norm: QuantNorm) -> None:
        a.add(name, hid * 2, "gamma", count=hid, gamma_e=int(norm.gamma_e))

    for i, lay in enumerate(model.layers):
        for lin_name in LINEARS:
            linear(f"layer.{i}.{lin_name}", getattr(lay, lin_name))
        gamma(f"layer.{i}.gamma_in", lay.norm_in)
        gamma(f"layer.{i}.gamma_post", lay.norm_post)
    linear("embed", model.embed)
    gamma("gamma_final", model.norm_final)
    sizes = kv_sizes(max_ctx, wb)
    for i in range(model.n_layers):
        for g in range(kv):
            for part, size in sizes.items():
                a.add(f"kv.{i}.{g}.{part}", size, "kv", layer=i, kv_head=g, part=part)
    return a.regions


def _linear_of(model: QuantModel, name: str) -> QuantLinear:
    if name == "embed":
        return model.embed
    _, layer, lin = name.split(".")
    return getattr(model.layers[int(layer)], lin)


def _norm_of(model: QuantModel, name: str) -> QuantNorm:
    if name == "gamma_final":
        return model.norm_final
    _, layer, which = name.split(".")
    return (
        model.layers[int(layer)].norm_in
        if which == "gamma_in"
        else model.layers[int(layer)].norm_post
    )


def region_chunks(
    model: QuantModel, r: Region, wb: int, max_ctx: int, programs: dict[str, bytes]
) -> Iterable[bytes]:
    """The payload of ``r`` as byte chunks (empty for the zero-filled KV region)."""
    if r.kind == "program":
        return [programs[r.name]]
    if r.kind == "table":
        return [rope_bytes(model.rope_theta, max_ctx)]
    if r.kind == "const":
        row = model.k_center[r.info["layer"]].reshape(-1)
        return [np.ascontiguousarray(row, dtype="<i4").tobytes()]
    if r.kind == "weights":
        return iter_weight_tiles(_linear_of(model, r.name).q, wb)
    if r.kind == "meta":
        lin = _linear_of(model, r.name[: -len(".meta")])
        return [pack_meta(lin.bias_q, lin.scale_m, lin.scale_e, r.info["count"])]
    if r.kind == "gamma":
        return [np.ascontiguousarray(_norm_of(model, r.name).gamma_q, dtype="<i2").tobytes()]
    if r.kind == "kv":
        return []
    raise ValueError(f"region {r.name}: unknown kind {r.kind}")


def write_image(path: Path | str, regions: Sequence[Region], chunks_of, total: int) -> str:
    """Write the regions in address order with zero gaps, extend to ``total``; return the sha256."""
    h = hashlib.sha256()
    pos = 0
    with open(path, "wb") as f:

        def put(data: bytes) -> None:
            nonlocal pos
            f.write(data)
            h.update(data)
            pos += len(data)

        def zeros(n: int) -> None:
            while n > 0:
                put(_ZERO_CHUNK[: min(n, len(_ZERO_CHUNK))])
                n -= min(n, len(_ZERO_CHUNK))

        for r in regions:
            if r.addr < pos or r.addr % ALIGN:
                raise ValueError(f"region {r.name} at 0x{r.addr:08x} overlaps or is misaligned")
            zeros(r.addr - pos)
            for c in chunks_of(r):
                put(c)
            if r.kind == "kv":
                zeros(r.end - pos)
            if pos != r.end:
                raise ValueError(f"region {r.name}: payload {pos - r.addr} bytes, size {r.size}")
        if pos > total:
            raise ValueError("image: regions extend past the total size")
        zeros(total - pos)
    return h.hexdigest()


# --------------------------------------------------------------------------- programs


@dataclass(frozen=True)
class Effects:
    """What one descriptor writes: a VSRAM range, SREG ids, and whether POS sets the extent."""

    vsram: tuple[int, int] | None
    sreg: tuple[int, ...]
    pos_dependent: bool


def descriptor_effects(d: Descriptor) -> Effects:
    """The written VSRAM range ``(start, count)`` and SREG ids implied by the fields of ``d``.

    Capacity fields stand in for POS-derived extents (scores GEMV ``n``,
    VSOFTMAX ``n``), flagged ``pos_dependent``; KVWRITE writes memory only.
    """
    op = d.opcode
    tracked = (d.sreg_dst,) if d.track_absmax else ()
    to_vsram = d.out_mode in (OutMode.VSRAM, OutMode.VSRAM_DUMP)
    if op == Opcode.GEMV:
        return Effects((d.vs_dst, d.n) if to_vsram else None, tracked, d.n_from_pos)
    if op == Opcode.EMBED:
        return Effects((d.vs_dst, d.k) if to_vsram else None, tracked, False)
    if op in (Opcode.VRMSNORM, Opcode.VSILUMUL):
        return Effects((d.vs_dst, d.n), tracked, False)
    if op == Opcode.VQUANT:
        scales = d.n // d.vs_aux if d.flags & VquantFlag.GROUP else 1
        return Effects((d.vs_dst, d.n), tuple(range(d.sreg_dst, d.sreg_dst + scales)), False)
    if op == Opcode.VROPE:
        return Effects((d.vs_src, d.n), (), False)
    if op == Opcode.VSOFTMAX:
        return Effects((d.vs_dst, d.n), (d.sreg_dst,), d.len_from_pos)
    if op == Opcode.VSUBC:
        return Effects((d.vs_dst, d.n), (), False)
    if op == Opcode.KVWRITE:
        return Effects(None, (), True)
    return Effects(None, (), False)


class _Emitter:
    def __init__(self) -> None:
        self.descs: list[Descriptor] = []
        self.plan: list[dict[str, Any]] = []

    def emit(
        self,
        d: Descriptor,
        name: str,
        *,
        layer: int | None = None,
        head: int | None = None,
        kv_head: int | None = None,
        mem: Sequence[Region] = (),
        csr: Sequence[str] = (),
    ) -> None:
        eff = descriptor_effects(d)
        self.plan.append(
            {
                "index": len(self.descs),
                "op": d.opcode.name,
                "name": name,
                "layer": layer,
                "head": head,
                "kv_head": kv_head,
                "vsram": None
                if eff.vsram is None
                else {"start": eff.vsram[0], "count": eff.vsram[1]},
                "sreg": list(eff.sreg),
                "mem": [{"name": r.name, "addr": r.addr, "size": r.size} for r in mem],
                "csr": list(csr),
                "pos_dependent": eff.pos_dependent,
            }
        )
        self.descs.append(d)

    def extend(self, other: _Emitter) -> None:
        for d, entry in zip(other.descs, other.plan, strict=True):
            self.descs.append(d)
            self.plan.append(dict(entry))


@dataclass(frozen=True)
class _Ctx:
    model: QuantModel
    consts: program.ProgramConstants
    regions: dict[str, Region]
    vs: VsramMap
    sr: SregMap
    max_ctx: int
    a_bits: int


def _emit_norm_quant(e: _Emitter, c: _Ctx, gamma: str, eps_key: str, names: tuple[str, str], layer):
    m, vs, sr = c.model, c.vs, c.sr
    norm = _norm_of(m, gamma)
    e.emit(
        isa.vrmsnorm(
            vs_src=vs.start("X"),
            vs_dst=vs.start("XN"),
            n=m.hidden,
            addr_a=c.regions[gamma].addr,
            eps_c=m.eps_c[eps_key],
            frac_x=m.frac["X"],
            g=-norm.gamma_e,
            sqrt_d=m.sqrt_d,
            sreg_dst=sr.absmax,
        ),
        names[0],
        layer=layer,
    )
    e.emit(
        isa.vquant(
            vs_src=vs.start("XN"),
            vs_dst=vs.start("A"),
            n=m.hidden,
            width=c.a_bits,
            frac_in=m.frac["X"],
            sreg_dst=sr.a,
            use_tracked=True,
            sreg_src=sr.absmax,
        ),
        names[1],
        layer=layer,
    )


def _emit_layer(e: _Emitter, c: _Ctx, i: int) -> None:
    m, vs, sr, pc = c.model, c.vs, c.sr, c.consts
    frac = m.frac
    hid, inter = m.hidden, m.intermediate
    hd, kvd = m.heads * HEAD_DIM, m.kv_heads * HEAD_DIM
    n_rep = m.heads // m.kv_heads
    reg = c.regions
    x, a, qkv = vs.start("X"), vs.start("A"), vs.start("QKV")
    ctx, ctxq, gu, hq = vs.start("CTX"), vs.start("CTXQ"), vs.start("GU"), vs.start("HQ")
    s, w = vs.start("S"), vs.start("W")
    q_at, k_at, v_at = qkv, qkv + hd, qkv + hd + kvd

    def gemv(lin: str, g: program.GemvConstants, **kw: Any) -> Descriptor:
        return isa.gemv(
            addr_a=reg[f"layer.{i}.{lin}"].addr,
            addr_m=reg[f"layer.{i}.{lin}.meta"].addr,
            s1=g.s1,
            sbias=g.sbias,
            **kw,
        )

    _emit_norm_quant(e, c, f"layer.{i}.gamma_in", "input", ("rmsnorm_in", "quant_in"), i)
    e.emit(
        gemv("wqkv", pc["qkv"], n=hd + 2 * kvd, k=hid, vs_src=a, vs_dst=qkv, sreg_src=sr.a),
        "gemv_qkv",
        layer=i,
    )
    e.emit(isa.vrope(vs_src=q_at, n=hd + kvd, addr_a=reg["rope"].addr), "rope", layer=i)
    e.emit(
        isa.vquant(
            vs_src=q_at,
            vs_dst=q_at,
            n=hd,
            width=16,
            frac_in=frac["QKV"],
            sreg_dst=sr.q0,
            group=HEAD_DIM,
            scale_mul=m.log2e_over_8,
        ),
        "quant_q",
        layer=i,
    )
    e.emit(
        isa.vsubc(vs_src=k_at, vs_dst=k_at, n=kvd, addr_a=reg[f"kcenter.{i}"].addr),
        "subc_k",
        layer=i,
    )
    e.emit(
        isa.vquant(
            vs_src=k_at,
            vs_dst=k_at,
            n=kvd,
            width=8,
            frac_in=frac["QKV"],
            sreg_dst=sr.k0,
            group=HEAD_DIM,
        ),
        "quant_k",
        layer=i,
    )
    e.emit(
        isa.vquant(
            vs_src=v_at,
            vs_dst=v_at,
            n=kvd,
            width=8,
            frac_in=frac["QKV"],
            sreg_dst=sr.v0,
            group=HEAD_DIM,
        ),
        "quant_v",
        layer=i,
    )
    kv_reg = [
        {part: reg[f"kv.{i}.{g}.{part}"] for part in ("kt", "v", "k_meta", "v_meta")}
        for g in range(m.kv_heads)
    ]
    for g, kr in enumerate(kv_reg):
        e.emit(
            isa.kvwrite(
                vs_src=k_at + g * HEAD_DIM,
                addr_a=kr["kt"].addr,
                addr_m=kr["k_meta"].addr,
                sreg_src=sr.k0 + g,
                transposed=True,
                max_ctx=c.max_ctx,
            ),
            "kvwrite_k",
            layer=i,
            kv_head=g,
            mem=[kr["kt"], kr["k_meta"]],
        )
        e.emit(
            isa.kvwrite(
                vs_src=v_at + g * HEAD_DIM,
                addr_a=kr["v"].addr,
                addr_m=kr["v_meta"].addr,
                sreg_src=sr.v0 + g,
                transposed=False,
                max_ctx=c.max_ctx,
            ),
            "kvwrite_v",
            layer=i,
            kv_head=g,
            mem=[kr["v"], kr["v_meta"]],
        )
    g_sc, g_pv = pc["scores"], pc["pv"]
    for h in range(m.heads):
        kr = kv_reg[h // n_rep]
        e.emit(
            isa.gemv(
                addr_a=kr["kt"].addr,
                addr_m=kr["k_meta"].addr,
                n=c.max_ctx,
                k=HEAD_DIM,
                vs_src=q_at + h * HEAD_DIM,
                vs_dst=s,
                sreg_src=sr.q0 + h,
                s1=g_sc.s1,
                sbias=g_sc.sbias,
                n_from_pos=True,
            ),
            "gemv_scores",
            layer=i,
            head=h,
        )
        e.emit(
            isa.vsoftmax(
                vs_src=s,
                vs_dst=w,
                n=c.max_ctx,
                addr_a=kr["v_meta"].addr,
                frac_s=frac["S"],
                sreg_dst=sr.w,
            ),
            "softmax",
            layer=i,
            head=h,
        )
        e.emit(
            isa.gemv(
                addr_a=kr["v"].addr,
                n=HEAD_DIM,
                k=c.max_ctx,
                vs_src=w,
                vs_dst=ctx + h * HEAD_DIM,
                sreg_src=sr.w,
                s1=g_pv.s1,
                sbias=g_pv.sbias,
                unit_meta=True,
                k_from_pos=True,
            ),
            "gemv_pv",
            layer=i,
            head=h,
        )
    e.emit(
        isa.vquant(
            vs_src=ctx, vs_dst=ctxq, n=hd, width=c.a_bits, frac_in=frac["CTX"], sreg_dst=sr.ctx
        ),
        "quant_ctx",
        layer=i,
    )
    e.emit(
        gemv("wo", pc["o"], n=hid, k=hd, vs_src=ctxq, vs_dst=x, sreg_src=sr.ctx, accumulate=True),
        "gemv_o",
        layer=i,
    )
    _emit_norm_quant(e, c, f"layer.{i}.gamma_post", "post", ("rmsnorm_post", "quant_post"), i)
    e.emit(
        gemv("wgu", pc["gu"], n=2 * inter, k=hid, vs_src=a, vs_dst=gu, sreg_src=sr.a),
        "gemv_gu",
        layer=i,
    )
    e.emit(
        isa.vsilumul(
            vs_src=gu,
            vs_aux=gu + inter,
            vs_dst=hq,
            n=inter,
            frac_gu=frac["GU"],
            sh_h=2 * frac["GU"] - frac["H"],
            sreg_dst=sr.absmax,
        ),
        "silu_mul",
        layer=i,
    )
    e.emit(
        isa.vquant(
            vs_src=hq,
            vs_dst=hq,
            n=inter,
            width=c.a_bits,
            frac_in=frac["H"],
            sreg_dst=sr.h,
            use_tracked=True,
            sreg_src=sr.absmax,
        ),
        "quant_h",
        layer=i,
    )
    e.emit(
        gemv(
            "wdown", pc["down"], n=hid, k=inter, vs_src=hq, vs_dst=x, sreg_src=sr.h, accumulate=True
        ),
        "gemv_down",
        layer=i,
    )


def build_programs(
    model: QuantModel,
    consts: program.ProgramConstants,
    regions: dict[str, Region],
    vs: VsramMap,
    sr: SregMap,
    max_ctx: int,
    a_bits: int,
) -> tuple[_Emitter, _Emitter]:
    """``(decode, prefill)`` emitters: descriptors plus dump-plan entries, in program order.

    Both start with EMBED and the layer body; ``decode`` adds the final norm,
    its VQUANT and the ARGMAX LM-head GEMV before HALT, ``prefill`` halts after
    the last layer (``docs/ARCHITECTURE.md``, decode step dataflow).
    """
    c = _Ctx(model, consts, regions, vs, sr, max_ctx, a_bits)
    body = _Emitter()
    g_em = consts["embed"]
    body.emit(
        isa.embed(
            addr_a=regions["embed"].addr,
            addr_m=regions["embed.meta"].addr,
            k=model.hidden,
            vs_dst=vs.start("X"),
            s1=g_em.s1,
            sbias=g_em.sbias,
        ),
        "embed",
    )
    for i in range(model.n_layers):
        _emit_layer(body, c, i)
    decode = _Emitter()
    decode.extend(body)
    _emit_norm_quant(decode, c, "gamma_final", "final", ("rmsnorm_final", "quant_final"), None)
    g_lm = consts["lm_head"]
    decode.emit(
        isa.gemv(
            addr_a=regions["embed"].addr,
            addr_m=regions["embed.meta"].addr,
            n=model.vocab,
            k=model.hidden,
            vs_src=vs.start("A"),
            vs_dst=0,
            sreg_src=sr.a,
            s1=g_lm.s1,
            sbias=g_lm.sbias,
            out_mode=OutMode.ARGMAX,
        ),
        "gemv_lm_head",
        csr=ARGMAX_CSRS,
    )
    decode.emit(isa.halt(), "halt")
    prefill = _Emitter()
    prefill.extend(body)
    prefill.emit(isa.halt(), "halt")
    return decode, prefill


def descriptor_counts(model: QuantModel) -> tuple[int, int]:
    """``(decode, prefill)`` descriptor counts: ``1 + L (16 + 2 KV + 3 H) + 4`` and three fewer."""
    per_layer = 16 + 2 * model.kv_heads + 3 * model.heads
    decode = 1 + model.n_layers * per_layer + 4
    return decode, decode - 3


# --------------------------------------------------------------------------- compile


@dataclass
class Compiled:
    """The outputs of :func:`compile`: files under ``out_dir`` and their in-memory form."""

    out_dir: Path
    layout: dict[str, Any]
    dump_plan: dict[str, Any]
    decode: list[Descriptor]
    prefill: list[Descriptor]
    regions: list[Region]
    vsram: VsramMap
    sreg: SregMap

    @property
    def image_path(self) -> Path:
        return self.out_dir / FILES["image"]


def truncate_layers(model: QuantModel, layers: int) -> QuantModel:
    """The first ``layers`` decoder layers of ``model`` (norm, embedding and constants kept)."""
    if not 1 <= layers <= model.n_layers:
        raise ValueError(f"truncate_layers: {layers} not in [1, {model.n_layers}]")
    return dataclasses.replace(
        model, layers=model.layers[:layers], k_center=model.k_center[:layers]
    )


def _check_params(model: QuantModel, max_ctx: int, wb: int, vsram_words: int, a_bits: int) -> None:
    if model.head_dim != HEAD_DIM:
        raise ValueError(f"compile: head_dim {model.head_dim} is not {HEAD_DIM}")
    if wb <= 0 or wb % 8:
        raise ValueError(f"compile: WB {wb} must be a positive multiple of 8")
    if max_ctx < 1 or max_ctx > program.MAX_CTX or max_ctx % wb:
        raise ValueError(
            f"compile: max_ctx {max_ctx} must be a multiple of WB in [1, {program.MAX_CTX}]"
        )
    if vsram_words < 1:
        raise ValueError("compile: vsram_words must be positive")
    if a_bits not in (8, 16):
        raise ValueError("compile: a_bits must be 8 or 16")


def _traffic(regions: Sequence[Region], model: QuantModel, wb: int) -> dict[str, Any]:
    """Per-token weight-stream bytes and MACs, position-independent, plus the attention rates.

    ``total`` is weights + meta + gammas (the bytes the program streams),
    ``wt_bytes`` the ``WT_BYTES`` counter (weights + meta + the EMBED gather:
    ``hidden`` table bytes and one meta record) and ``macs`` the ``MACS`` of
    the weight GEMVs; ``attention`` gives the per-head rates that complete
    ``MACS`` at a position (``docs/ISA.md``, PERF table).
    """
    weights = meta = gammas = 0
    macs_layers = 0
    for r in regions:
        if r.name.startswith("layer."):
            if r.kind == "weights":
                weights += r.size
                macs_layers += r.info["tiles"] * wb * r.info["shape"][1]
            elif r.kind == "meta":
                meta += r.size
            elif r.kind == "gamma":
                gammas += r.size
    by_name = {r.name: r for r in regions}
    head = by_name["embed"].size + by_name["embed.meta"].size + by_name["gamma_final"].size
    macs_head = by_name["embed"].info["tiles"] * wb * model.hidden
    body = weights + meta + gammas

    embed_bytes = model.hidden + isa.META_BYTES

    def row(total: int, macs: int, lm_head: bool) -> dict[str, int]:
        w = weights + (by_name["embed"].size if lm_head else 0)
        m = meta + (by_name["embed.meta"].size if lm_head else 0)
        return {
            "weights": w,
            "meta": m,
            "gammas": gammas + (by_name["gamma_final"].size if lm_head else 0),
            "total": total,
            "beats": -(-total // wb),
            "macs": macs,
            "wt_bytes": w + m + embed_bytes,
        }

    return {
        "decode": row(body + head, macs_layers + macs_head, True),
        "prefill": row(body, macs_layers, False),
        "embed_gather_beats": model.hidden,
        "attention": {
            "head_layers": model.n_layers * model.heads,
            "scores_macs_per_tile": HEAD_DIM * wb,
            "pv_macs_per_token": tiles_for(HEAD_DIM, wb) * wb,
        },
    }


def _expected_tokens_entry(model: QuantModel, a_bits: int) -> dict[str, Any] | None:
    path = golden.expected_tokens_path(model)
    if not path.is_file():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    if (
        rep.get("model", {}).get("layers") != model.n_layers
        or rep.get("calib_tokens_sha256") != model.calib_tokens_sha256
        or rep.get("a_bits") != a_bits
    ):
        return None
    return {
        "file": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(path),
        "prompts": {
            k: {"count": len(v["generated_ids"]), "sha256": v["sha256"]}
            for k, v in rep["prompts"].items()
        },
    }


def compile(
    model: QuantModel,
    spec: ModelSpec | None = None,
    *,
    out_dir: Path | str,
    max_ctx: int = MAX_CTX,
    wb: int = WB,
    a_bits: int = 16,
    vsram_words: int = VSRAM_WORDS,
    prompt: Path | str | None = None,
) -> Compiled:
    """Lay ``model`` out for a ``wb``-byte weight port with a ``max_ctx``-position KV region.

    Writes ``image.bin``, ``decode.prog``, ``prefill.prog``, ``program.lst``,
    ``prefill.lst``, ``layout.json`` and ``dump_plan.json`` into ``out_dir``,
    plus ``tokens.bin`` when ``spec`` has a tokenizer and ``prompt.tokens`` for a
    prompt file.  The requant constants are :func:`program.build` at
    ``program.MAX_CTX`` for every ``max_ctx`` (``max_ctx`` sizes the KV region,
    the RoPE rows and the capacity fields), so the golden model's defaults
    reproduce the program.  Raises on any static check: requant windows,
    VSRAM and SREG fit, region bases, field ranges.
    """
    _check_params(model, max_ctx, wb, vsram_words, a_bits)
    if prompt is not None and spec is None:
        raise ValueError("compile: a prompt file needs the model spec")
    out = Path(out_dir)
    consts = program.build(model, a_bits=a_bits)
    for g in consts.gemvs.values():
        g.check_window()
    vs = vsram_map(model, max_ctx, vsram_words)
    sr = sreg_map(model)
    regions = plan_regions(model, max_ctx, wb)
    by_name = {r.name: r for r in regions}
    decode, prefill = build_programs(model, consts, by_name, vs, sr, max_ctx, a_bits)
    for d in decode.descs:
        if d.opcode in (Opcode.GEMV, Opcode.EMBED) and d.vs_dst % isa.VSRAM_WORD_ELEMS:
            raise ValueError(f"GEMV output at element {d.vs_dst} is not 8-aligned")
    progs = {"prog.decode": isa.assemble(decode.descs), "prog.prefill": isa.assemble(prefill.descs)}
    prog_regions = [
        Region(
            "prog.decode",
            PROG_BASE,
            len(progs["prog.decode"]),
            "program",
            {"descriptors": len(decode.descs)},
        )
    ]
    prog_regions.append(
        Region(
            "prog.prefill",
            align_up(prog_regions[0].end),
            len(progs["prog.prefill"]),
            "program",
            {"descriptors": len(prefill.descs)},
        )
    )
    if prog_regions[-1].end > ROPE_BASE:
        raise ValueError("programs exceed the 1 MB program window")
    regions = prog_regions + regions
    total = align_up(regions[-1].end)

    out.mkdir(parents=True, exist_ok=True)
    image_sha = write_image(
        out / FILES["image"], regions, lambda r: region_chunks(model, r, wb, max_ctx, progs), total
    )
    (out / FILES["decode"]).write_bytes(progs["prog.decode"])
    (out / FILES["prefill"]).write_bytes(progs["prog.prefill"])
    (out / FILES["decode_lst"]).write_text(isa.disassemble(decode.descs), encoding="utf-8")
    (out / FILES["prefill_lst"]).write_text(isa.disassemble(prefill.descs), encoding="utf-8")

    tokens_entry = None
    if spec is not None and spec.path("tokenizer.json").is_file():
        n = write_tokens_bin(spec, out / FILES["tokens_bin"])
        tokens_entry = {
            "file": FILES["tokens_bin"],
            "bytes": n,
            "count": spec.vocab,
            "sha256": sha256_file(out / FILES["tokens_bin"]),
        }
    prompt_entry = None
    if prompt is not None and spec is not None:
        ids = prompt_tokens(spec, prompt)
        (out / FILES["prompt"]).write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        prompt_entry = {
            "file": FILES["prompt"],
            "source": golden.prompt_key(prompt),
            "count": len(ids),
            "sha256": golden.ids_sha256(ids),
        }

    dump_plan = {
        "format": "quettos-dump-plan",
        "isa_version": isa.ISA_VERSION,
        "model": model.name,
        "wb": wb,
        "max_ctx": max_ctx,
        "decode": decode.plan,
        "prefill": prefill.plan,
    }
    (out / FILES["dump_plan"]).write_text(json_text(dump_plan), encoding="utf-8")
    rope_path = numerics.rope_table_path(model.rope_theta)
    layout: dict[str, Any] = {
        "format": "quettos-layout",
        "isa_version": isa.ISA_VERSION,
        "numerics": model.numerics_version,
        "model": {
            "name": model.name,
            "repo_id": model.repo_id,
            "arch": model.arch,
            "layers": model.n_layers,
            "hidden": model.hidden,
            "heads": model.heads,
            "kv_heads": model.kv_heads,
            "head_dim": model.head_dim,
            "intermediate": model.intermediate,
            "vocab": model.vocab,
            "has_qkv_bias": model.has_qkv_bias,
            "rope_theta": model.rope_theta,
        },
        "calib_tokens_sha256": model.calib_tokens_sha256,
        "a_bits": a_bits,
        "wb": wb,
        "max_ctx": max_ctx,
        "align": ALIGN,
        "bases": {
            "programs": PROG_BASE,
            "rope": ROPE_BASE,
            "constants": CONST_BASE,
            "weights": WEIGHT_BASE,
            "embed": by_name["embed"].addr,
            "kv": by_name["kv.0.0.kt"].addr,
        },
        "kv": {
            "per_head": kv_sizes(max_ctx, wb),
            "order": "layer-major, KV head, then kt v k_meta v_meta",
            "kt_tiles": tiles_for(max_ctx, wb),
            "v_tiles": tiles_for(HEAD_DIM, wb),
        },
        "regions": [r.as_json() for r in regions],
        "image": {"file": FILES["image"], "size": total, "sha256": image_sha},
        "programs": {
            "decode": {
                "file": FILES["decode"],
                "listing": FILES["decode_lst"],
                "addr": prog_regions[0].addr,
                "size": prog_regions[0].size,
                "descriptors": len(decode.descs),
                "sha256": sha256_bytes(progs["prog.decode"]),
            },
            "prefill": {
                "file": FILES["prefill"],
                "listing": FILES["prefill_lst"],
                "addr": prog_regions[1].addr,
                "size": prog_regions[1].size,
                "descriptors": len(prefill.descs),
                "sha256": sha256_bytes(progs["prog.prefill"]),
            },
        },
        "dump_plan": {"file": FILES["dump_plan"], "sha256": sha256_file(out / FILES["dump_plan"])},
        "vsram": vs.as_json(),
        "sreg": sr.as_json(),
        "frac": dict(sorted(model.frac.items())),
        "constants": {
            **consts.as_dict(),
            "eps_c": dict(sorted(model.eps_c.items())),
            "sqrt_d": {"m": model.sqrt_d.m, "e": model.sqrt_d.e},
            "log2e_over_8": {"m": model.log2e_over_8.m, "e": model.log2e_over_8.e},
            "gamma_e": {
                "final": model.norm_final.gamma_e,
                "layers": [
                    {"norm_in": lay.norm_in.gamma_e, "norm_post": lay.norm_post.gamma_e}
                    for lay in model.layers
                ],
            },
        },
        "traffic": _traffic(regions, model, wb),
        "csr": {
            "words": isa.CSR_WORDS,
            "registers": {c.name: c.word for c in isa.CSRS},
            "ctrl_bits": dict(isa.CTRL_BITS),
            "status_bits": dict(isa.STATUS_BITS),
            "perf_index": dict(isa.PERF_INDEX),
        },
        "tables": {
            "rope": {
                "file": rope_path.relative_to(REPO_ROOT).as_posix(),
                "rows": max_ctx,
                "sha256": sha256_file(rope_path),
            },
            "luts": {
                "file": numerics.LUTS_JSON.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(numerics.LUTS_JSON),
            },
        },
        "tokens_bin": tokens_entry,
        "prompt": prompt_entry,
        "expected_tokens": _expected_tokens_entry(model, a_bits),
    }
    (out / FILES["layout"]).write_text(json_text(layout), encoding="utf-8")
    return Compiled(out, layout, dump_plan, decode.descs, prefill.descs, regions, vs, sr)


# --------------------------------------------------------------------------- reading back


def load_layout(out_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(out_dir) / FILES["layout"]).read_text(encoding="utf-8"))


def load_dump_plan(out_dir: Path | str) -> dict[str, Any]:
    return json.loads((Path(out_dir) / FILES["dump_plan"]).read_text(encoding="utf-8"))


class Image:
    """Read-side view of a compiled directory: ``image.bin`` regions decoded via ``layout.json``."""

    def __init__(self, out_dir: Path | str) -> None:
        self.dir = Path(out_dir)
        self.layout = load_layout(self.dir)
        self.regions: dict[str, dict[str, Any]] = {r["name"]: r for r in self.layout["regions"]}
        self.wb: int = int(self.layout["wb"])
        self.path = self.dir / self.layout["image"]["file"]

    def read(self, name: str) -> bytes:
        r = self.regions[name]
        with open(self.path, "rb") as f:
            f.seek(r["addr"])
            data = f.read(r["size"])
        if len(data) != r["size"]:
            raise ValueError(f"{name}: short read")
        return data

    def weights(self, name: str) -> np.ndarray:
        """int8 ``[N, K]`` of a weights region (``embed`` or ``layer.<i>.<lin>``)."""
        n, k = self.regions[name]["shape"]
        return untile_weights(self.read(name), n, k, self.wb)

    def meta(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(bias_q, m, e)`` over the padded channel count of a meta region."""
        return unpack_meta(self.read(name))

    def gamma(self, name: str) -> np.ndarray:
        return np.frombuffer(self.read(name), dtype="<i2").astype(np.int16)

    def rope(self) -> np.ndarray:
        """int16 ``[rows, 2, 32]``: cos then sin per position."""
        rows = self.regions["rope"]["rows"]
        return np.frombuffer(self.read("rope"), dtype="<i2").reshape(rows, 2, HEAD_DIM // 2)

    def kcenter(self, layer: int) -> np.ndarray:
        """int32 ``[kv_heads, 64]`` K-centering rows of ``layer``."""
        r = self.regions[f"kcenter.{layer}"]
        return np.frombuffer(self.read(r["name"]), dtype="<i4").reshape(r["shape"])

    def program(self, which: str) -> list[Descriptor]:
        """The descriptors of ``decode`` or ``prefill`` as stored in the image."""
        return isa.parse(self.read(f"prog.{which}"))
