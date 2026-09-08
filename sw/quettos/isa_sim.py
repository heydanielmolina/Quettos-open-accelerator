"""ISA-level simulator: descriptor programs on a memory image, value by value, as the RTL sees them.

:class:`Machine` holds the QMEM image, the VSRAM rows, the scale registers,
the CSRs and the counters; :func:`execute` runs one descriptor with
:mod:`quettos.numerics` as its only arithmetic, :func:`run_program` a whole
program, :func:`run_token` one token and :func:`generate` the prefill/decode
loop of :func:`quettos.golden.generate`.  :func:`written_ranges`,
:func:`record_program` and :func:`check_plan` produce and check the
per-descriptor blobs of the RTL comparison against ``dump_plan.json``;
:func:`compare_sequence` and :func:`compare_generate` check a program against
the golden model and name the first differing descriptor.
Semantics: ``docs/ISA.md``; layouts: ``docs/MEMORY_MAP.md``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quettos import golden, isa, numerics, program
from quettos.isa import Descriptor, Fault, KvwriteFlag, Opcode, OutMode, VquantFlag
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat, Stats
from quettos.quantize import QuantModel

WB = 64  # weight-port width in bytes (lanes); the image is tiled per WB
B_MAX = 1  # activation rows
VSRAM_WORDS = 4096
VSRAM_ELEMS = VSRAM_WORDS * isa.VSRAM_WORD_ELEMS
HEAD_DIM = isa.HEAD_DIM
GEMV_CHUNK_ELEMS = 1 << 22  # weight elements converted per exact matmul block
ACC_LIMIT = 1 << (program.ACC_W - 1)  # the hardware accumulator never wraps
EXACT_BITS = 53  # float64 integer exactness bound for every partial sum
W_ABS_MAX = 128
MASK32 = 0xFFFFFFFF
META_DTYPE = np.dtype([("bias", "<i4"), ("m", "<u2"), ("e", "<i1"), ("pad", "u1")])
PERF_COUNTED: tuple[str, ...] = ("DESCRIPTORS", "MACS", "WT_BYTES")  # the value-level PERF indices

SregValue = SFloat | int  # a scale, or a tracked absmax (non-negative int)
RetireFn = Callable[[int, Descriptor], None]
Source = Sequence[Descriptor] | bytes | None  # descriptors, .prog bytes, or the image at PC


# --------------------------------------------------------------------------- state


@dataclass(frozen=True)
class Write:
    """One state write of a descriptor; ``kind`` is ``vsram``, ``sreg``, ``mem`` or ``csr``.

    ``vsram``: ``count`` elements from ``start`` of row ``row``; ``sreg``: register
    ``start`` of row ``row``; ``mem``: ``count`` bytes at byte address ``start``;
    ``csr``: the register ``name``.
    """

    kind: str
    row: int = 0
    start: int = 0
    count: int = 1
    name: str = ""

    def key(self) -> tuple:
        if self.kind == "csr":
            return ("csr", self.name)
        if self.kind == "sreg":
            return ("sreg", self.row, self.start)
        return (self.kind, self.row, self.start, self.count)


class Machine:
    """QMEM image, VSRAM, SREG, CSRs and counters of one core.

    ``vsram[r]`` is the int64 element vector of row ``r`` (``vsram_words * 8``
    elements); ``sreg[r][i]`` holds an sfloat or a tracked absmax; ``csr`` maps
    every register of :data:`quettos.isa.CSRS` to its 32-bit word.  ``ROW_EN``
    starts at 1 (row 0 enabled).  ``stats_req`` counts the requant events
    (``SAT_REQ``), ``stats_vpu`` the vector-unit events (``SAT_VPU``); both feed
    ``ERR_SHIFT`` and :meth:`stats` is their sum, the golden model's
    :class:`Stats`.  ``perf`` holds the 16 PERF counters; the simulator counts
    ``DESCRIPTORS``, ``MACS`` (lanes times beats issued, padded lanes included)
    and ``WT_BYTES`` (tile and meta bytes of GEMVs without POS-derived
    dimensions, plus the ``K + 8`` bytes an EMBED consumes); the cycle and beat
    counters read zero.  ``argmax_out`` keeps the int32 outputs of the most
    recent ARGMAX-mode GEMV since START (an observation, not hardware state).
    ``err_bounds`` counts the ``ERR_BOUNDS`` events: a POS-derived value above
    its capacity field, a KVWRITE at or past its capacity, a VSRAM range or an
    SREG index past the end (reads give 0, writes are dropped).
    ``log``, when a list, receives a :class:`Write` for every state change.
    """

    def __init__(
        self,
        image: bytes | bytearray,
        *,
        wb: int = WB,
        b_max: int = B_MAX,
        vsram_words: int = VSRAM_WORDS,
        tables: numerics.Tables | None = None,
    ) -> None:
        if wb <= 0 or wb % 8:
            raise ValueError(f"wb = {wb} must be a positive multiple of 8")
        if not 1 <= b_max <= isa.FIELD_BY_NAME["row_mask"].width:
            raise ValueError("b_max must lie in [1, 8]: row_mask addresses at most eight rows")
        self.mem = bytearray(image)
        self.wb = wb
        self.b_max = b_max
        self.vsram_words = vsram_words
        self.vsram = np.zeros((b_max, vsram_words * isa.VSRAM_WORD_ELEMS), dtype=np.int64)
        self.sreg: list[list[SregValue]] = [[SFLOAT_ZERO] * isa.SREG_COUNT for _ in range(b_max)]
        self.csr: dict[str, int] = {c.name: 0 for c in isa.CSRS}
        self.csr["ISA_VERSION"] = isa.ISA_VERSION
        self.csr["ROW_EN"] = 1
        self.perf = np.zeros(isa.PERF_COUNT, dtype=np.int64)
        self.stats_req = Stats()
        self.stats_vpu = Stats()
        self.err_bounds = 0
        self.tables = numerics.load_tables() if tables is None else tables
        self.argmax_out: np.ndarray | None = None
        self.log: list[Write] | None = None

    @classmethod
    def from_file(cls, image_path: str | Path, **kw: Any) -> Machine:
        """A machine over ``image.bin``, read straight into the image buffer."""
        path = Path(image_path)
        buf = bytearray(path.stat().st_size)
        with open(path, "rb") as f:
            f.readinto(buf)
        m = cls(b"", **kw)
        m.mem = buf
        return m

    # ---- counters and status

    def stats(self) -> Stats:
        """Requant plus vector-unit counters: the golden model's ``Stats`` for the same run."""
        return self.stats_req + self.stats_vpu

    def start(self) -> None:
        """``CTRL.START``: clear the PERF, SAT and ERR counters and DONE / STEP_HALTED / ERR."""
        self.perf[:] = 0
        self.stats_req = Stats()
        self.stats_vpu = Stats()
        self.err_bounds = 0
        self.argmax_out = None
        self._set_status(False, "DONE", "STEP_HALTED", "ERR")
        self._clear_fault()
        self.update_counter_csrs()
        for i in range(isa.PERF_COUNT):
            self.csr[f"PERF{i}_LO"] = 0
            self.csr[f"PERF{i}_HI"] = 0

    def _set_status(self, value: bool, *names: str) -> None:
        for name in names:
            bit = 1 << isa.STATUS_BITS[name]
            self.csr["STATUS"] = (
                (self.csr["STATUS"] | bit) if value else (self.csr["STATUS"] & ~bit)
            )

    def status(self, name: str) -> bool:
        return bool(self.csr["STATUS"] >> isa.STATUS_BITS[name] & 1)

    def _clear_fault(self) -> None:
        for lsb, width in isa.STATUS_FIELDS.values():
            self.csr["STATUS"] &= ~(((1 << width) - 1) << lsb) & MASK32

    def fault(self, code: Fault, opcode: int) -> None:
        """Stop with ``STATUS.ERR``: the fault code and the faulting opcode byte in ``STATUS``.

        ``PC`` keeps the address of the descriptor that faulted, as the
        sequencer leaves it (``docs/RTL.md`` 3.5).
        """
        self._clear_fault()
        self.csr["STATUS"] |= isa.status_word(fault=code, fault_op=opcode)
        self._set_status(True, "ERR", "DONE")

    def fault_state(self) -> tuple[Fault, int]:
        """``(fault, opcode byte)`` of the current ``STATUS`` word."""
        return isa.status_fault(self.csr["STATUS"])

    def update_counter_csrs(self) -> None:
        self.csr["SAT_REQ"] = self.stats_req.sat & MASK32
        self.csr["SAT_VPU"] = self.stats_vpu.sat & MASK32
        self.csr["ERR_SHIFT"] = (self.stats_req.err_shift + self.stats_vpu.err_shift) & MASK32
        self.csr["ERR_BOUNDS"] = self.err_bounds & MASK32

    def snapshot_perf(self) -> None:
        """HALT: copy the PERF counters into their CSR halves."""
        for i in range(isa.PERF_COUNT):
            v = int(self.perf[i])
            self.csr[f"PERF{i}_LO"] = v & MASK32
            self.csr[f"PERF{i}_HI"] = (v >> 32) & MASK32

    def perf_value(self, name: str) -> int:
        return int(self.perf[isa.PERF_INDEX[name]])

    def csr_words(self) -> list[int]:
        """The 64-word CSR window as the host reads it."""
        words = [0] * isa.CSR_WORDS
        for c in isa.CSRS:
            words[c.word] = self.csr[c.name] & MASK32
        return words

    def argmax_val(self) -> int:
        v = self.csr["ARGMAX_VAL"]
        return v - (1 << 32) if v >= 1 << 31 else v

    def write_csr(self, name: str, value: int) -> None:
        self.csr[name] = int(value) & MASK32
        if self.log is not None:
            self.log.append(Write("csr", name=name))

    # ---- rows

    def rows(self, d: Descriptor) -> list[tuple[int, int, int]]:
        """``(r, source row, destination row)`` for every participating row, ascending.

        Row ``r`` participates when ``row_mask[r] & ROW_EN[r]`` is set and
        ``r < b_max``; it reads VSRAM / SREG row ``src_row + r`` and writes row
        ``dst_row + r`` (both below ``b_max``).  An empty list retires the
        descriptor as a NOP.
        """
        out = []
        row_en = self.csr["ROW_EN"]
        for r in range(self.b_max):
            if (d.row_mask >> r) & 1 and (row_en >> r) & 1:
                src, dst = d.src_row + r, d.dst_row + r
                if src >= self.b_max or dst >= self.b_max:
                    raise ValueError(
                        f"row {r}: src_row/dst_row {src}/{dst} outside b_max {self.b_max}"
                    )
                out.append((r, src, dst))
        return out

    # ---- memory

    def view(self, addr: int, dtype: Any, count: int) -> np.ndarray:
        """A typed view of ``count`` items at byte address ``addr``, inside the image."""
        nbytes = count * np.dtype(dtype).itemsize
        if addr < 0 or addr + nbytes > len(self.mem):
            raise ValueError(
                f"read of {nbytes} bytes at 0x{addr:08x} leaves the image ({len(self.mem)} bytes)"
            )
        return np.frombuffer(self.mem, dtype=dtype, count=count, offset=addr)

    def read_i8(self, addr: int, count: int) -> np.ndarray:
        return self.view(addr, np.int8, count).astype(np.int64)

    def read_i16(self, addr: int, count: int) -> np.ndarray:
        return self.view(addr, "<i2", count).astype(np.int64)

    def read_i32(self, addr: int, count: int) -> np.ndarray:
        return self.view(addr, "<i4", count).astype(np.int64)

    def read_meta(self, addr: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(bias_q, m, e)`` int64 arrays of ``count`` 8-byte meta entries at ``addr``."""
        v = self.view(addr, META_DTYPE, count)
        return v["bias"].astype(np.int64), v["m"].astype(np.int64), v["e"].astype(np.int64)

    def write_bytes(self, addr: int, data: bytes) -> None:
        if addr < 0 or addr + len(data) > len(self.mem):
            raise ValueError(
                f"write of {len(data)} bytes at 0x{addr:08x} leaves the image "
                f"({len(self.mem)} bytes)"
            )
        self.mem[addr : addr + len(data)] = data
        if self.log is not None:
            self.log.append(Write("mem", start=addr, count=len(data)))

    def write_byte_scatter(self, addr: int, stride: int, values: np.ndarray) -> None:
        """``values[i]`` (int8) to byte ``addr + i * stride``: the K^T single-byte-strobe writes."""
        n = int(values.size)
        last = addr + (n - 1) * stride
        if addr < 0 or last >= len(self.mem):
            raise ValueError(f"scatter to 0x{addr:08x}..0x{last:08x} leaves the image")
        u8 = np.frombuffer(self.mem, dtype=np.uint8)
        u8[addr : last + 1 : stride] = values.astype(np.int8).view(np.uint8)
        del u8
        if self.log is not None:
            self.log.extend(Write("mem", start=addr + i * stride, count=1) for i in range(n))

    # ---- VSRAM and SREG

    def vs_read(self, row: int, start: int, count: int) -> np.ndarray:
        """``count`` elements from ``start``; those past the end read 0 and count in ERR_BOUNDS."""
        size = self.vsram.shape[1]
        if start + count > size:
            self.err_bounds += 1
            out = np.zeros(count, dtype=np.int64)
            out[: max(0, size - start)] = self.vsram[row, start:size]
            return out
        return self.vsram[row, start : start + count].copy()

    def vs_write(self, row: int, start: int, values: np.ndarray) -> None:
        """Write ``values`` from ``start``; elements past the end are dropped and counted."""
        size = self.vsram.shape[1]
        count = int(values.size)
        if start + count > size:
            self.err_bounds += 1
            count = max(0, size - start)
        self.vsram[row, start : start + count] = values[:count]
        if self.log is not None and count:
            self.log.append(Write("vsram", row=row, start=start, count=count))

    def act_i16(self, row: int, start: int, count: int) -> np.ndarray:
        """GEMV activations: the low 16 bits of each element, sign-extended."""
        v = self.vs_read(row, start, count)
        return ((v & 0xFFFF) ^ 0x8000) - 0x8000

    def sreg_scale(self, row: int, index: int) -> SFloat:
        """The sfloat in ``SREG[index]``; an index past the end reads the zero scale and counts."""
        if index >= isa.SREG_COUNT:
            self.err_bounds += 1
            return SFLOAT_ZERO
        v = self.sreg[row][index]
        if not isinstance(v, SFloat):
            raise ValueError(f"SREG[{row}][{index}] holds a tracked absmax, not a scale")
        return v

    def sreg_absmax(self, row: int, index: int) -> int:
        """The tracked absmax in ``SREG[index]``; an index past the end reads 0 and counts."""
        if index >= isa.SREG_COUNT:
            self.err_bounds += 1
            return 0
        v = self.sreg[row][index]
        if isinstance(v, SFloat):
            raise ValueError(f"SREG[{row}][{index}] holds a scale, not a tracked absmax")
        return int(v)

    def sreg_set(self, row: int, index: int, value: SregValue) -> None:
        """Write ``SREG[index]``; an index past the end drops the write and counts."""
        if not isinstance(value, SFloat) and int(value) < 0:
            raise ValueError("a tracked absmax is non-negative")
        if index >= isa.SREG_COUNT:
            self.err_bounds += 1
            return
        self.sreg[row][index] = value
        if self.log is not None:
            self.log.append(Write("sreg", row=row, start=index))


def sreg_json(v: SregValue) -> dict[str, int]:
    """JSON form of an SREG entry: ``{"m", "e"}`` for a scale, ``{"absmax"}`` for an absmax."""
    if isinstance(v, SFloat):
        return {"m": v.m, "e": v.e}
    return {"absmax": int(v)}


# --------------------------------------------------------------------------- POS-derived fields


def _round_up(x: int, a: int) -> int:
    return -(-x // a) * a


def gemv_dims(d: Descriptor, pos: int, wb: int) -> tuple[int, int, int]:
    """``(N, K, bounds_errors)`` of a GEMV at position ``pos``.

    ``n_from_pos``: ``N = POS + 1`` rounded up to the tile and capped at the
    capacity ``n``; ``k_from_pos``: ``K = POS + 1`` capped at ``k``.  A derived
    ``POS + 1`` above its capacity field counts one bounds error.
    """
    n, k, errors = d.n, d.k, 0
    if d.n_from_pos:
        want = pos + 1
        if want > n:
            errors += 1
            want = n
        n = min(_round_up(want, wb), n)
    if d.k_from_pos:
        want = pos + 1
        if want > k:
            errors += 1
            want = k
        k = want
    return n, k, errors


def softmax_len(d: Descriptor, pos: int) -> tuple[int, int]:
    """``(len, bounds_errors)`` of a VSOFTMAX: ``POS + 1`` or ``imm32``, clamped into ``[1, n]``."""
    want = pos + 1 if d.len_from_pos else d.imm32
    if want > d.n:
        return d.n, 1
    if want < 1:
        return 1, 1
    return want, 0


def shift_field(m: Machine, value: int, count: int) -> int:
    """A shift amount from a descriptor field, clamped into ``[0, 63]``.

    A value outside the window counts once per output element in
    ``ERR_SHIFT`` (``stats_vpu``), like a requant stage-2 shift out of range.
    """
    if 0 <= value <= program.SHIFT_MAX:
        return value
    m.stats_vpu.err_shift += count
    return min(max(value, 0), program.SHIFT_MAX)


# --------------------------------------------------------------------------- GEMV / EMBED


def _stream_gemv(
    m: Machine, addr: int, tiles: int, k: int, k_cap: int, a: np.ndarray
) -> np.ndarray:
    """Accumulators of ``tiles`` weight tiles at ``addr`` against the activation ``a[K]``.

    Tile ``t`` starts at ``addr + t * k_cap * WB`` (the ``k`` field is the tile
    stride, the capacity when ``k_from_pos`` derives ``K``) and its first ``k``
    beats are streamed.  Exact int64 through float64 BLAS (partial sums below
    ``2**53``, asserted), in blocks of at most :data:`GEMV_CHUNK_ELEMS` weight
    elements; raises when a value leaves the 40-bit accumulator.
    """
    wb = m.wb
    bound = k * numerics.absmax(a) * W_ABS_MAX
    if bound >= 1 << EXACT_BITS:
        raise ValueError(f"gemv: partial-sum bound 2^{bound.bit_length()} not exact in float64")
    af = a.astype(np.float64)
    acc = np.empty(tiles * wb, dtype=np.int64)
    per = max(1, GEMV_CHUNK_ELEMS // max(1, k * wb))
    for t0 in range(0, tiles, per):
        t1 = min(tiles, t0 + per)
        if k_cap == k:
            raw = m.view(addr + t0 * k * wb, np.int8, (t1 - t0) * k * wb).reshape(t1 - t0, k, wb)
        else:
            raw = np.stack(
                [
                    m.view(addr + t * k_cap * wb, np.int8, k * wb).reshape(k, wb)
                    for t in range(t0, t1)
                ]
            )
        blk = np.tensordot(af, raw.astype(np.float64), axes=([0], [1]))
        acc[t0 * wb : t1 * wb] = blk.reshape(-1).astype(np.int64)
    if numerics.absmax(acc) >= ACC_LIMIT:
        raise ValueError(f"gemv: accumulator exceeds {program.ACC_W} bits")
    return acc


def _requant(m: Machine, d: Descriptor, dst: int, acc, sw_m, sw_e, sx: SFloat, bias) -> np.ndarray:
    old = m.vs_read(dst, d.vs_dst, acc.size) if d.accumulate else None
    s1 = d.sh0
    if s1 > program.SHIFT_MAX:
        m.stats_req.err_shift += int(acc.size)
        s1 = program.SHIFT_MAX
    y = numerics.requant_rows(
        acc[None, :],
        sw_m,
        sw_e,
        np.array([sx.m], dtype=np.int64),
        np.array([sx.e], dtype=np.int64),
        s1,
        d.sh1,
        bias_q=bias,
        old=None if old is None else old[None, :],
        stats=m.stats_req,
    )
    return y[0]


def dump_address(d: Descriptor, r: int, n: int) -> int:
    """Where row ``r`` of a DUMP lands: ``addr_c + r * 4 * N``, ``addr_c`` a whole beat."""
    if d.imm32 == 0 or d.imm32 % isa.DUMP_ALIGN:
        raise ValueError(f"dump: addr_c 0x{d.imm32:08x} must be a non-zero multiple of 64")
    return d.imm32 + r * 4 * n


def _output(m: Machine, d: Descriptor, r: int, dst: int, y: np.ndarray) -> None:
    """GEMV / EMBED output of row ``r``: VSRAM, the ARGMAX CSRs, the DUMP, the tracked absmax.

    Rows execute in ascending order, so the ARGMAX CSRs hold the highest
    participating row's result; a DUMP writes row ``r`` at
    ``addr_c + r * 4 * N`` as little-endian int32.
    """
    mode = d.out_mode
    if mode in (OutMode.VSRAM, OutMode.VSRAM_DUMP):
        m.vs_write(dst, d.vs_dst, y)
    if mode in (OutMode.ARGMAX, OutMode.ARGMAX_DUMP) and y.size:
        idx = numerics.argmax(y)
        m.write_csr("ARGMAX_TOK", idx)
        m.write_csr("ARGMAX_VAL", int(y[idx]))
        m.argmax_out = y.astype(np.int32)
    if mode in (OutMode.ARGMAX_DUMP, OutMode.VSRAM_DUMP):
        m.write_bytes(dump_address(d, r, y.size), np.ascontiguousarray(y, dtype="<i4").tobytes())
    if d.track_absmax:
        m.sreg_set(dst, d.sreg_dst, numerics.absmax(y))


def _exec_gemv(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    pos = m.csr["POS"]
    n, k, errors = gemv_dims(d, pos, m.wb)
    m.err_bounds += errors
    # Zero work: N == 0 or K == 0 retires the descriptor with the bounds events
    # counted and nothing else touched (docs/RTL.md 2.2).
    if n == 0 or k == 0:
        return
    tiles = -(-n // m.wb)
    a = m.act_i16(src, d.vs_src, k)
    acc = _stream_gemv(m, d.addr_a, tiles, k, d.k, a)[:n]
    if d.unit_meta:
        sw_m = np.full(n, SFLOAT_ONE.m, dtype=np.int64)
        sw_e = np.full(n, SFLOAT_ONE.e, dtype=np.int64)
        bias = None
    else:
        bias, sw_m, sw_e = m.read_meta(d.addr_m, n)
    y = _requant(m, d, dst, acc, sw_m, sw_e, m.sreg_scale(src, d.sreg_src), bias)
    _output(m, d, r, dst, y)
    m.perf[isa.PERF_INDEX["MACS"]] += tiles * m.wb * k
    if not (d.n_from_pos or d.k_from_pos):
        m.perf[isa.PERF_INDEX["WT_BYTES"]] += tiles * k * m.wb + (
            0 if d.unit_meta else tiles * m.wb * isa.META_BYTES
        )


def _exec_embed(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    k, tok, wb = d.k, m.csr["TOK"], m.wb
    if k == 0:
        return
    tile, j = divmod(tok, wb)
    base = d.addr_a + tile * k * wb + j
    q = m.view(base, np.int8, (k - 1) * wb + 1)[::wb].astype(np.int64)
    _, sm, se = m.read_meta(d.addr_m + tok * isa.META_BYTES, 1)
    acc = q << program.EMBED_PRE_SHIFT
    y = _requant(m, d, dst, acc, np.full(k, sm[0]), np.full(k, se[0]), SFLOAT_ONE, None)
    _output(m, d, r, dst, y)
    m.perf[isa.PERF_INDEX["WT_BYTES"]] += k + isa.META_BYTES


# --------------------------------------------------------------------------- vector ops


def _exec_vrmsnorm(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    x = m.vs_read(src, d.vs_src, d.n)
    gamma = m.read_i16(d.addr_a, d.n)
    sqrt_d = isa.sfloat_from_imm(d.addr_m & 0xFFFFFF)
    g = shift_field(m, d.sh1, d.n)
    y = numerics.rmsnorm(x, gamma, -g, d.imm32, sqrt_d, d.sh0, m.tables, m.stats_vpu)
    m.vs_write(dst, d.vs_dst, y)
    if d.track_absmax:
        m.sreg_set(dst, d.sreg_dst, numerics.absmax(y))


def _exec_vquant(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    flags = VquantFlag(d.flags)
    width = 8 if flags & VquantFlag.W8 else 16
    scale_mul = isa.sfloat_from_imm(d.imm32) if flags & VquantFlag.SCALE_MUL else None
    x = m.vs_read(src, d.vs_src, d.n)
    if flags & VquantFlag.GROUP:
        if flags & VquantFlag.USE_TRACKED:
            raise ValueError(
                "vquant: GROUP and USE_TRACKED together have no tracked absmax per group"
            )
        q, scales = numerics.quant_groups(
            x, d.vs_aux, width, d.sh0, m.tables, scale_mul=scale_mul, stats=m.stats_vpu
        )
        for g, s in enumerate(scales):
            m.sreg_set(dst, d.sreg_dst + g, s)
    else:
        amax = m.sreg_absmax(src, d.sreg_src) if flags & VquantFlag.USE_TRACKED else None
        q, s = numerics.quant(
            x, width, d.sh0, m.tables, scale_mul=scale_mul, amax=amax, stats=m.stats_vpu
        )
        m.sreg_set(dst, d.sreg_dst, s)
    m.vs_write(dst, d.vs_dst, q)


def _exec_vrope(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    row = m.read_i16(d.addr_a + m.csr["POS"] * isa.ROPE_ROW_BYTES, HEAD_DIM)
    half = HEAD_DIM // 2
    x = m.vs_read(src, d.vs_src, d.n)
    y = numerics.rope(x, row[:half], row[half:], HEAD_DIM, m.stats_vpu)
    m.vs_write(src, d.vs_src, y)


def _exec_vsilumul(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    g = m.vs_read(src, d.vs_src, d.n)
    u = m.vs_read(src, d.vs_aux, d.n)
    sh_h = shift_field(m, d.sh1, d.n)
    y = numerics.silu_mul(g, u, d.sh0, 2 * d.sh0 - sh_h, m.tables, m.stats_vpu)
    m.vs_write(dst, d.vs_dst, y)
    if d.track_absmax:
        m.sreg_set(dst, d.sreg_dst, numerics.absmax(y))


def _exec_vsoftmax(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    length, errors = softmax_len(d, m.csr["POS"])
    m.err_bounds += errors
    s = m.vs_read(src, d.vs_src, length)
    _, vm, ve = m.read_meta(d.addr_a, length)
    w, sm, se = numerics.softmax_rows(
        s[None, :], np.array([length]), d.sh0, vm, ve, m.tables, stats=m.stats_vpu
    )
    out = np.zeros(d.n, dtype=np.int64)
    out[:length] = w[0]
    m.vs_write(dst, d.vs_dst, out)
    m.sreg_set(dst, d.sreg_dst, SFloat(int(sm[0]), int(se[0])))


def _exec_vsubc(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    x = m.vs_read(src, d.vs_src, d.n)
    c = m.read_i32(d.addr_a, d.n)
    m.vs_write(dst, d.vs_dst, numerics.subc(x, c, m.stats_vpu))


def kv_addresses(d: Descriptor, pos: int, wb: int) -> tuple[list[tuple[int, int]], int]:
    """The data writes ``(address, bytes)`` of a KVWRITE at ``pos`` and its meta address.

    ``TRANSPOSED``: 64 single bytes at ``addr_a + (POS/WB)*64*WB + d*WB +
    POS%WB``; otherwise one ``WB``-byte beat per V tile at
    ``addr_a + (t*k + POS)*WB`` for ``t`` below ``ceil(64/WB)`` (``k`` is the
    token capacity).  The meta record is at ``addr_m + POS*8``.
    """
    meta = d.addr_m + pos * isa.META_BYTES
    if d.flags & KvwriteFlag.TRANSPOSED:
        base = d.addr_a + (pos // wb) * HEAD_DIM * wb + pos % wb
        return [(base + i * wb, 1) for i in range(HEAD_DIM)], meta
    tiles = -(-HEAD_DIM // wb)
    return [(d.addr_a + (t * d.k + pos) * wb, wb) for t in range(tiles)], meta


def _exec_kvwrite(m: Machine, d: Descriptor, r: int, src: int, dst: int) -> None:
    if d.n != HEAD_DIM:
        raise ValueError(f"kvwrite: n must be {HEAD_DIM}")
    pos, wb = m.csr["POS"], m.wb
    if pos >= d.k:
        m.err_bounds += 1
        return
    v = m.vs_read(src, d.vs_src, HEAD_DIM)
    vals = (((v & 0xFF) ^ 0x80) - 0x80).astype(np.int8)
    writes, meta_addr = kv_addresses(d, pos, wb)
    if d.flags & KvwriteFlag.TRANSPOSED:
        m.write_byte_scatter(writes[0][0], wb, vals)
    else:
        for t, (addr, nbytes) in enumerate(writes):
            beat = bytearray(nbytes)
            chunk = vals[t * wb : (t + 1) * wb].tobytes()
            beat[: len(chunk)] = chunk
            m.write_bytes(addr, bytes(beat))
    s = m.sreg_scale(src, d.sreg_src)
    m.write_bytes(meta_addr, struct.pack("<iHbB", 0, s.m, s.e, 0))


_EXEC: dict[Opcode, Callable[[Machine, Descriptor, int, int, int], None]] = {
    Opcode.GEMV: _exec_gemv,
    Opcode.EMBED: _exec_embed,
    Opcode.VRMSNORM: _exec_vrmsnorm,
    Opcode.VQUANT: _exec_vquant,
    Opcode.VROPE: _exec_vrope,
    Opcode.VSILUMUL: _exec_vsilumul,
    Opcode.VSOFTMAX: _exec_vsoftmax,
    Opcode.VSUBC: _exec_vsubc,
    Opcode.KVWRITE: _exec_kvwrite,
}


def execute(m: Machine, d: Descriptor) -> None:
    """Execute one descriptor on every participating row; NOP, HALT and FENCE do nothing here."""
    fn = _EXEC.get(d.opcode)
    if fn is None:
        return
    for r, src, dst in m.rows(d):
        fn(m, d, r, src, dst)


# --------------------------------------------------------------------------- programs


def _descriptors(m: Machine, source: Source) -> Iterator[tuple[int, Descriptor | None, int]]:
    """``(index, descriptor, opcode byte)`` from a list, ``.prog`` bytes or the image at ``PC``.

    An undecodable descriptor (unknown opcode) yields ``None`` with the byte
    that could not be decoded.
    """
    if source is None or isinstance(source, bytes | bytearray):
        index = 0
        while True:
            if source is None:
                pc = m.csr["PC"]
                raw = bytes(m.mem[pc : pc + isa.DESC_BYTES])
            else:
                raw = bytes(source[index * isa.DESC_BYTES : (index + 1) * isa.DESC_BYTES])
            if len(raw) < isa.DESC_BYTES:
                raise ValueError("program runs past the end of its bytes without HALT")
            op = isa.opcode_of(raw)
            yield index, (isa.decode(raw) if isa.is_opcode(op) else None), op
            index += 1
    else:
        for index, d in enumerate(source):
            yield index, d, int(d.opcode)


def _retire(m: Machine, d: Descriptor) -> None:
    m.perf[isa.PERF_INDEX["DESCRIPTORS"]] += 1
    m.csr["PC"] = (m.csr["PC"] + isa.DESC_BYTES) & MASK32
    m.update_counter_csrs()


def run_program(
    m: Machine,
    source: Source,
    *,
    pc: int | None = None,
    start: bool = True,
    on_retire: RetireFn | None = None,
) -> int:
    """Run ``source`` to HALT as ``CTRL.START`` does; returns the number of descriptors retired.

    ``source`` is a descriptor sequence, the bytes of a ``.prog`` file, or
    ``None`` for the image at ``PC`` (``pc`` sets the CSR first).  ``start``
    clears the counters and status bits.  An unknown opcode stops the program
    with ``STATUS.ERR``, ``DONE``, ``FAULT = OPCODE`` and the undecodable byte
    in ``FAULT_OP``, leaving ``PC`` on the descriptor that faulted; a class
    field outside :data:`isa.CLASS_WINDOW` stops it the same way with
    ``FAULT = CLASS`` and its own opcode byte.
    ``on_retire(index, d)`` runs after every retired descriptor, HALT included.
    """
    if start:
        m.start()
    if pc is not None:
        m.csr["PC"] = pc & MASK32
    retired = 0
    for index, d, op in _descriptors(m, source):
        if d is None:
            m.fault(Fault.OPCODE, op)
            m.snapshot_perf()
            return retired
        if isa.class_fault(d):
            m.fault(Fault.CLASS, op)
            m.snapshot_perf()
            return retired
        execute(m, d)
        _retire(m, d)
        retired += 1
        if d.opcode == Opcode.HALT:
            m._set_status(True, "DONE")
            m.snapshot_perf()
        if on_retire is not None:
            on_retire(index, d)
        if d.opcode == Opcode.HALT:
            return retired
    raise ValueError("program ended without HALT")


def step(m: Machine, d: Descriptor) -> None:
    """``CTRL.STEP``: execute ``d`` at ``PC`` and set ``STEP_HALTED`` (``DONE`` for HALT).

    A class field outside :data:`isa.CLASS_WINDOW` faults instead, as it does
    inside a run: nothing executes and ``PC`` stays on the descriptor.
    """
    m._set_status(False, "DONE", "STEP_HALTED", "ERR")
    m._clear_fault()
    if isa.class_fault(d):
        m.fault(Fault.CLASS, int(d.opcode))
        m.snapshot_perf()
        return
    execute(m, d)
    _retire(m, d)
    if d.opcode == Opcode.HALT:
        m._set_status(True, "DONE")
        m.snapshot_perf()
    else:
        m._set_status(True, "STEP_HALTED")


def run_token(
    m: Machine,
    source: Source,
    tok: int,
    pos: int,
    *,
    row_en: int = 1,
    pc: int | None = None,
    on_retire: RetireFn | None = None,
) -> int | None:
    """One token: write ``TOK``, ``POS`` and ``ROW_EN``, START, run to HALT.

    Returns ``ARGMAX_TOK`` when the program executed an ARGMAX-mode GEMV
    (``decode.prog``) and ``None`` otherwise (``prefill.prog``).
    """
    m.csr["TOK"] = tok & MASK32
    m.csr["POS"] = pos & MASK32
    m.csr["ROW_EN"] = row_en & MASK32
    run_program(m, source, pc=pc, on_retire=on_retire)
    return None if m.argmax_out is None else m.csr["ARGMAX_TOK"]


def generate(
    m: Machine,
    decode: Source,
    prefill: Source,
    prompt_ids: Sequence[int],
    max_new: int,
    *,
    eos_ids: Sequence[int] = (),
    decode_pc: int | None = None,
    prefill_pc: int | None = None,
    on_token: Callable[[int, int, int], None] | None = None,
) -> list[int]:
    """Greedy continuation with the prefill/decode loop of :func:`quettos.golden.generate`.

    ::

        for i in 0..P-2:  prefill(TOK=prompt[i], POS=i)
        decode(TOK=prompt[P-1], POS=P-1)            -> gen[0]
        for j >= 1:       decode(TOK=gen[j-1], POS=P-1+j) -> gen[j]

    stopping after ``max_new`` tokens or once a generated id is in ``eos_ids``
    (that id is the last element).  ``on_token(j, pos, id)`` follows every
    decode step.
    """
    prompt = [int(t) for t in prompt_ids]
    p_len = len(prompt)
    if p_len < 1:
        raise ValueError("generate: empty prompt")
    if max_new < 0:
        raise ValueError("generate: max_new must be non-negative")
    for i in range(p_len - 1):
        run_token(m, prefill, prompt[i], i, pc=prefill_pc)
    gen: list[int] = []
    tok = prompt[p_len - 1]
    stop = {int(e) for e in eos_ids}
    for j in range(max_new):
        pos = p_len - 1 + j
        nxt = run_token(m, decode, tok, pos, pc=decode_pc)
        if nxt is None:
            raise ValueError("generate: the decode program has no ARGMAX GEMV")
        gen.append(nxt)
        if on_token is not None:
            on_token(j, pos, nxt)
        if nxt in stop:
            break
        tok = nxt
    return gen


# --------------------------------------------------------------------------- dump plan and records

PROGRAM_FILES: dict[str, str] = {
    "decode": "decode.prog",
    "prefill": "prefill.prog",
    "dump_plan": "dump_plan.json",
    "image": "image.bin",
}


@dataclass
class Programs:
    """The two programs of a compiled model with their dump-plan entries (``dump_plan.json``)."""

    decode: list[Descriptor]
    prefill: list[Descriptor]
    decode_plan: list[dict[str, Any]]
    prefill_plan: list[dict[str, Any]]

    @classmethod
    def from_dir(cls, out_dir: str | Path) -> Programs:
        """Read ``decode.prog``, ``prefill.prog`` and ``dump_plan.json`` from ``out_dir``."""
        d = Path(out_dir)
        plan = json.loads((d / PROGRAM_FILES["dump_plan"]).read_text(encoding="utf-8"))
        return cls(
            isa.parse((d / PROGRAM_FILES["decode"]).read_bytes()),
            isa.parse((d / PROGRAM_FILES["prefill"]).read_bytes()),
            plan["decode"],
            plan["prefill"],
        )

    @classmethod
    def from_compiled(cls, compiled: Any) -> Programs:
        """From a :class:`quettos.compiler.Compiled` (``decode``, ``prefill``, ``dump_plan``)."""
        return cls(
            list(compiled.decode),
            list(compiled.prefill),
            compiled.dump_plan["decode"],
            compiled.dump_plan["prefill"],
        )


def _participants(d: Descriptor, b_max: int, row_en: int) -> list[tuple[int, int, int]]:
    return [
        (r, d.src_row + r, d.dst_row + r)
        for r in range(b_max)
        if (d.row_mask >> r) & 1 and (row_en >> r) & 1
    ]


def written_ranges(
    d: Descriptor, pos: int, *, wb: int = WB, b_max: int = B_MAX, row_en: int = 1
) -> set[tuple]:
    """The :meth:`Write.key` set one descriptor produces at ``pos`` (what :func:`execute` logs).

    VSRAM keys are ``("vsram", row, start, count)``, scale registers
    ``("sreg", row, index)``, memory ``("mem", 0, addr, nbytes)`` (one key per
    byte of a transposed KVWRITE) and CSRs ``("csr", name)``.
    """
    keys: set[tuple] = set()
    op = d.opcode
    for r, src, dst in _participants(d, b_max, row_en):
        if op in (Opcode.GEMV, Opcode.EMBED):
            n, k = gemv_dims(d, pos, wb)[:2] if op == Opcode.GEMV else (d.k, d.k)
            if n == 0 or k == 0:
                continue
            if d.out_mode in (OutMode.VSRAM, OutMode.VSRAM_DUMP):
                keys.add(("vsram", dst, d.vs_dst, n))
            if d.out_mode in (OutMode.ARGMAX, OutMode.ARGMAX_DUMP):
                keys.update({("csr", "ARGMAX_TOK"), ("csr", "ARGMAX_VAL")})
            if d.out_mode in (OutMode.ARGMAX_DUMP, OutMode.VSRAM_DUMP):
                keys.add(("mem", 0, dump_address(d, r, n), 4 * n))
            if d.track_absmax:
                keys.add(("sreg", dst, d.sreg_dst))
        elif op in (Opcode.VRMSNORM, Opcode.VSILUMUL):
            keys.add(("vsram", dst, d.vs_dst, d.n))
            if d.track_absmax:
                keys.add(("sreg", dst, d.sreg_dst))
        elif op == Opcode.VQUANT:
            keys.add(("vsram", dst, d.vs_dst, d.n))
            scales = d.n // d.vs_aux if d.flags & VquantFlag.GROUP else 1
            keys.update(("sreg", dst, d.sreg_dst + g) for g in range(scales))
        elif op == Opcode.VROPE:
            keys.add(("vsram", src, d.vs_src, d.n))
        elif op == Opcode.VSOFTMAX:
            keys.update({("vsram", dst, d.vs_dst, d.n), ("sreg", dst, d.sreg_dst)})
        elif op == Opcode.VSUBC:
            keys.add(("vsram", dst, d.vs_dst, d.n))
        elif op == Opcode.KVWRITE and pos < d.k:
            writes, meta = kv_addresses(d, pos, wb)
            keys.update(("mem", 0, a, nbytes) for a, nbytes in writes)
            keys.add(("mem", 0, meta, isa.META_BYTES))
    return keys


@dataclass
class StepRecord:
    """State captured after one descriptor, as its dump-plan entry names it.

    ``vsram`` is the int32 copy of the entry's VSRAM range (row ``dst_row``, or
    ``src_row`` for the in-place VROPE), ``sreg`` the listed registers, ``mem``
    the listed regions by name, ``csr`` the listed registers; ``writes`` is the
    exact :meth:`Write.key` set the descriptor produced.
    """

    index: int
    descriptor: Descriptor
    entry: dict[str, Any]
    vsram: np.ndarray | None
    sreg: dict[int, SregValue]
    mem: dict[str, bytes]
    csr: dict[str, int]
    writes: set[tuple]

    def as_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "op": self.descriptor.opcode.name,
            "name": self.entry.get("name"),
            "vsram": None
            if self.vsram is None
            else {**self.entry["vsram"], "values": self.vsram.tolist()},
            "sreg": {str(i): sreg_json(v) for i, v in self.sreg.items()},
            "mem": {
                name: {"addr": r["addr"], "size": r["size"], "hex": self.mem[name].hex()}
                for r in self.entry.get("mem", [])
                for name in [r["name"]]
            },
            "csr": dict(self.csr),
        }


def _capture(m: Machine, index: int, d: Descriptor, entry: dict[str, Any]) -> StepRecord:
    if entry.get("op") != d.opcode.name:
        raise ValueError(f"descriptor {index} is {d.opcode.name}, dump plan says {entry.get('op')}")
    row = d.src_row if d.opcode == Opcode.VROPE else d.dst_row
    vs = entry.get("vsram")
    vsram = None
    if vs is not None:
        vsram = m.vsram[row, vs["start"] : vs["start"] + vs["count"]].astype(np.int32)
    sreg = {int(i): m.sreg[row][int(i)] for i in entry.get("sreg", [])}
    mem = {r["name"]: bytes(m.mem[r["addr"] : r["addr"] + r["size"]]) for r in entry.get("mem", [])}
    csr = {name: m.csr[name] for name in entry.get("csr", [])}
    writes = {w.key() for w in (m.log or [])}
    return StepRecord(index, d, entry, vsram, sreg, mem, csr, writes)


def record_program(
    m: Machine,
    source: Source,
    plan: Sequence[dict[str, Any]],
    tok: int,
    pos: int,
    *,
    row_en: int = 1,
    pc: int | None = None,
) -> list[StepRecord]:
    """Run one token in step mode and capture every descriptor's outputs per its dump-plan entry."""
    records: list[StepRecord] = []
    m.log = []

    def retire(index: int, d: Descriptor) -> None:
        if index >= len(plan):
            raise ValueError(f"descriptor {index} has no dump-plan entry")
        records.append(_capture(m, index, d, plan[index]))
        assert m.log is not None
        m.log.clear()

    try:
        run_token(m, source, tok, pos, row_en=row_en, pc=pc, on_retire=retire)
    finally:
        m.log = None
    return records


def check_plan(records: Sequence[StepRecord]) -> list[str]:
    """Problems where a descriptor's writes leave what its dump-plan entry declares.

    Every VSRAM write lies inside the entry's range (and equals it unless the
    entry is ``pos_dependent``), the scale registers and CSRs written are
    exactly the listed ones, and every memory write falls inside one listed
    region.  An empty list means the plan describes the writes.
    """
    problems: list[str] = []
    for rec in records:
        e = rec.entry
        tag = f"descriptor {rec.index} {rec.descriptor.opcode.name} {e.get('name')}"
        vs = e.get("vsram")
        vs_writes = [(k[2], k[3]) for k in rec.writes if k[0] == "vsram"]
        if vs is None and vs_writes:
            problems.append(f"{tag}: VSRAM written without a planned range")
        for start, count in vs_writes:
            if vs is None:
                break
            inside = vs["start"] <= start and start + count <= vs["start"] + vs["count"]
            exact = (start, count) == (vs["start"], vs["count"])
            if not inside or (not e.get("pos_dependent") and not exact):
                problems.append(f"{tag}: wrote VSRAM [{start}, +{count}) against plan {vs}")
        if vs is not None and not vs_writes and rec.descriptor.opcode != Opcode.HALT:
            problems.append(f"{tag}: planned VSRAM range {vs} not written")
        sregs = {k[2] for k in rec.writes if k[0] == "sreg"}
        if sregs != set(e.get("sreg", [])):
            problems.append(f"{tag}: SREG written {sorted(sregs)} != plan {e.get('sreg')}")
        csrs = {k[1] for k in rec.writes if k[0] == "csr"}
        if csrs != set(e.get("csr", [])):
            problems.append(f"{tag}: CSR written {sorted(csrs)} != plan {e.get('csr')}")
        regions = [(r["addr"], r["addr"] + r["size"]) for r in e.get("mem", [])]
        for k in rec.writes:
            if k[0] == "mem" and not any(lo <= k[2] and k[2] + k[3] <= hi for lo, hi in regions):
                problems.append(f"{tag}: wrote 0x{k[2]:08x} (+{k[3]}) outside the planned regions")
    return problems


# --------------------------------------------------------------------------- golden comparison


@dataclass
class Mismatch:
    """The first difference between the simulator and the golden model.

    ``index`` / ``opcode`` / ``name`` / ``listing`` identify the descriptor
    (``index = -1``: the per-token counters), ``what`` the compared state,
    ``element`` the first differing flat index (``None`` for a shape or type
    difference).
    """

    pos: int
    index: int
    opcode: str
    name: str
    listing: str
    what: str
    element: int | None
    expected: Any
    got: Any

    def __str__(self) -> str:
        where = f"pos {self.pos} descriptor {self.index} {self.opcode} {self.name}"
        at = "" if self.element is None else f" element {self.element}"
        return (
            f"{where}: {self.what}{at}: expected {self.expected}, got {self.got}\n  {self.listing}"
        )


@dataclass
class Comparison:
    """Result of a compared run: ``mismatch`` is ``None`` when every descriptor agreed."""

    mismatch: Mismatch | None
    argmax: list[int | None]
    gen: list[int]
    stats: Stats
    positions: int

    @property
    def ok(self) -> bool:
        return self.mismatch is None


class _Trace:
    def __init__(self) -> None:
        self.values: dict[tuple[str, int | None], np.ndarray] = {}

    def __call__(self, name: str, layer: int | None, value: np.ndarray) -> None:
        self.values[(name, layer)] = np.array(value, copy=True)


class _Stop(Exception):
    pass


_RANGE_OPS = frozenset(
    {
        "rmsnorm_in",
        "rmsnorm_post",
        "rmsnorm_final",
        "subc_k",
        "silu_mul",
        "gemv_qkv",
        "gemv_o",
        "gemv_gu",
        "gemv_down",
    }
)


def _sreg_pair(v: SregValue) -> np.ndarray:
    if isinstance(v, SFloat):
        return np.array([v.m, v.e], dtype=np.int64)
    return np.array([int(v)], dtype=np.int64)


def _expected(
    entry: dict[str, Any],
    d: Descriptor,
    m: Machine,
    tr: _Trace,
    cache: golden.KVCache,
    pos: int,
    qkv_start: int | None,
) -> list[tuple[str, Any, Any]]:
    """``(what, expected, got)`` checks for one retired descriptor from its dump-plan entry."""
    name, layer = entry["name"], entry.get("layer")
    head, kvh = entry.get("head"), entry.get("kv_head")
    row = d.src_row if d.opcode == Opcode.VROPE else d.dst_row

    def out(start: int, count: int) -> np.ndarray:
        return m.vsram[row, start : start + count]

    def tv(op: str) -> np.ndarray:
        return tr.values[(op, layer)]

    def sreg_checks(scales: np.ndarray, base: int) -> list[tuple[str, Any, Any]]:
        rows = scales.reshape(-1, 2)
        return [
            (f"SREG[{base + g}]", rows[g], _sreg_pair(m.sreg[row][base + g]))
            for g in range(rows.shape[0])
        ]

    if name == "embed":
        return [("vsram", tv("embed")[0], out(d.vs_dst, d.k))]
    if name in _RANGE_OPS:
        return [("vsram", tv(name)[0], out(d.vs_dst, d.n))]
    if name.startswith("quant_"):
        checks = [("vsram", tv(name)[0], out(d.vs_dst, d.n))]
        return checks + sreg_checks(tv(name + ".scale")[0], d.sreg_dst)
    if name == "rope":
        if qkv_start is None:
            raise ValueError("compare: rope before the QKV GEMV of its layer")
        exp = np.concatenate([tv("rope_q")[0], tv("rope_k")[0]])
        off = d.vs_src - qkv_start
        return [("vsram", exp[off : off + d.n], out(d.vs_src, d.n))]
    if name == "gemv_scores":
        return [("vsram", tv("gemv_scores")[0, head, : pos + 1], out(d.vs_dst, pos + 1))]
    if name == "softmax":
        exp = np.zeros(d.n, dtype=np.int64)
        exp[: pos + 1] = tv("softmax")[0, head]
        checks = [("vsram", exp, out(d.vs_dst, d.n))]
        return checks + sreg_checks(tv("softmax.sreg")[0, head], d.sreg_dst)
    if name == "gemv_pv":
        exp = tv("gemv_pv")[0, head * HEAD_DIM : (head + 1) * HEAD_DIM]
        return [("vsram", exp, out(d.vs_dst, HEAD_DIM))]
    if name == "gemv_lm_head":
        got = None if m.argmax_out is None else m.argmax_out.astype(np.int64)
        return [
            ("logits", tv("gemv_lm_head")[0], got),
            ("argmax", tv("argmax")[0:1], np.array([m.csr["ARGMAX_TOK"]])),
        ]
    if name in ("kvwrite_k", "kvwrite_v"):
        writes, meta_addr = kv_addresses(d, pos, m.wb)
        if name == "kvwrite_k":
            got = (
                np.array([m.mem[a] for a, _ in writes], dtype=np.uint8)
                .astype(np.int8)
                .astype(np.int64)
            )
            exp, sm, se = cache.k[layer, kvh, pos], cache.k_m, cache.k_e
        else:
            got = np.concatenate([m.read_i8(a, nbytes) for a, nbytes in writes])[:HEAD_DIM]
            exp, sm, se = cache.v[layer, kvh, pos], cache.v_m, cache.v_e
        bias, mm, ee = m.read_meta(meta_addr, 1)
        meta_exp = np.array([0, sm[layer, kvh, pos], se[layer, kvh, pos]], dtype=np.int64)
        return [
            ("kv bytes", exp.astype(np.int64), got),
            ("kv meta", meta_exp, np.array([bias[0], mm[0], ee[0]])),
        ]
    if name == "halt":
        return []
    raise ValueError(f"compare: unknown dump-plan name {name!r}")


def _first_difference(exp: Any, got: Any) -> tuple[bool, int | None]:
    if got is None:
        return True, None
    e, g = np.asarray(exp, dtype=np.int64), np.asarray(got, dtype=np.int64)
    if e.shape != g.shape:
        return True, None
    diff = np.nonzero(e.reshape(-1) != g.reshape(-1))[0]
    if diff.size == 0:
        return False, None
    return True, int(diff[0])


def compare_step(
    model: QuantModel,
    cache: golden.KVCache,
    m: Machine,
    source: Sequence[Descriptor],
    plan: Sequence[dict[str, Any]],
    tok: int,
    pos: int,
    *,
    lm_head: bool = True,
    a_bits: int = 16,
    prog: program.ProgramConstants | None = None,
) -> Mismatch | None:
    """Run ``golden.step`` and the simulator for one token; return the first differing descriptor.

    Every retired descriptor's outputs (VSRAM range, scales, KV bytes and meta,
    logits, argmax) are compared with the golden trace op its dump-plan entry
    names; the run stops at the first difference.  A difference in the
    per-token ``Stats`` after a fully matching run is reported at index -1.
    """
    tr = _Trace()
    gst = Stats()
    golden.step(
        model, cache, tok, pos, lm_head=lm_head, a_bits=a_bits, stats=gst, trace=tr, prog=prog
    )
    qkv_start: int | None = None
    found: list[Mismatch] = []

    def retire(index: int, d: Descriptor) -> None:
        nonlocal qkv_start
        entry = plan[index]
        if entry.get("op") != d.opcode.name:
            raise ValueError(
                f"descriptor {index} is {d.opcode.name}, dump plan says {entry.get('op')}"
            )
        if entry["name"] == "gemv_qkv":
            qkv_start = d.vs_dst
        for what, exp, got in _expected(entry, d, m, tr, cache, pos, qkv_start):
            differs, element = _first_difference(exp, got)
            if differs:
                e_val = exp if element is None else np.asarray(exp).reshape(-1)[element]
                g_val = (
                    got if element is None or got is None else np.asarray(got).reshape(-1)[element]
                )
                found.append(
                    Mismatch(
                        pos,
                        index,
                        d.opcode.name,
                        entry["name"],
                        isa.disassemble_one(d),
                        what,
                        element,
                        e_val.tolist() if isinstance(e_val, np.ndarray) else e_val,
                        g_val.tolist() if isinstance(g_val, np.ndarray) else g_val,
                    )
                )
                raise _Stop

    try:
        run_token(m, source, tok, pos, on_retire=retire)
    except _Stop:
        return found[0]
    if gst != m.stats():
        return Mismatch(pos, -1, "", "token", "", "stats", None, gst, m.stats())
    return None


def _compare_run(
    model: QuantModel,
    image: bytes | bytearray | str | Path,
    programs: Programs,
    steps: Iterator[tuple[int | None, int, bool]],
    *,
    a_bits: int,
    max_ctx: int,
    wb: int,
    vsram_words: int,
    eos_ids: Sequence[int],
    max_new: int,
) -> Comparison:
    if isinstance(image, str | Path):
        m = Machine.from_file(image, wb=wb, vsram_words=vsram_words)
    else:
        m = Machine(image, wb=wb, vsram_words=vsram_words)
    cache = golden.new_cache(model, max_ctx)
    prog = program.build(model, a_bits=a_bits)
    argmax: list[int | None] = []
    gen: list[int] = []
    total = Stats()
    stop = {int(e) for e in eos_ids}
    positions = 0
    tok_next: int | None = None
    for tok, pos, lm_head in steps:
        if tok is None:
            tok = tok_next
            assert tok is not None
        src, plan = (
            (programs.decode, programs.decode_plan)
            if lm_head
            else (programs.prefill, programs.prefill_plan)
        )
        mm = compare_step(
            model, cache, m, src, plan, tok, pos, lm_head=lm_head, a_bits=a_bits, prog=prog
        )
        positions += 1
        total = total + m.stats()
        if mm is not None:
            return Comparison(mm, argmax, gen, total, positions)
        out = None if m.argmax_out is None else m.csr["ARGMAX_TOK"]
        argmax.append(out)
        if lm_head and out is not None and len(gen) < max_new:
            gen.append(out)
            tok_next = out
            if out in stop:
                break
    return Comparison(None, argmax, gen, total, positions)


def compare_sequence(
    model: QuantModel,
    image: bytes | bytearray | str | Path,
    programs: Programs,
    ids: Sequence[int],
    *,
    a_bits: int = 16,
    max_ctx: int = program.MAX_CTX,
    wb: int = WB,
    vsram_words: int = VSRAM_WORDS,
    decode_every: bool = True,
) -> Comparison:
    """Teacher-forced comparison over ``ids``, each step checked against ``golden.step``.

    ``image`` is the image bytes or the path of ``image.bin`` (read straight
    into the machine).  ``decode.prog`` runs at every position; with
    ``decode_every=False`` the positions before the last run ``prefill.prog``.
    """
    last = len(ids) - 1
    steps = ((int(t), i, decode_every or i == last) for i, t in enumerate(ids))
    return _compare_run(
        model,
        image,
        programs,
        steps,
        a_bits=a_bits,
        max_ctx=max_ctx,
        wb=wb,
        vsram_words=vsram_words,
        eos_ids=(),
        max_new=0,
    )


def compare_generate(
    model: QuantModel,
    image: bytes | bytearray | str | Path,
    programs: Programs,
    prompt_ids: Sequence[int],
    max_new: int,
    *,
    eos_ids: Sequence[int] = (),
    a_bits: int = 16,
    max_ctx: int = program.MAX_CTX,
    wb: int = WB,
    vsram_words: int = VSRAM_WORDS,
) -> Comparison:
    """The prefill/decode loop of :func:`generate` with every step checked against ``golden.step``.

    ``Comparison.gen`` holds the generated ids up to the first mismatch.
    """
    prompt = [int(t) for t in prompt_ids]
    if not prompt:
        raise ValueError("compare_generate: empty prompt")
    p_len = len(prompt)

    def steps() -> Iterator[tuple[int | None, int, bool]]:
        for i in range(p_len - 1):
            yield prompt[i], i, False
        yield prompt[-1], p_len - 1, True
        for j in range(1, max_new):
            yield None, p_len - 1 + j, True

    return _compare_run(
        model,
        image,
        programs,
        steps(),
        a_bits=a_bits,
        max_ctx=max_ctx,
        wb=wb,
        vsram_words=vsram_words,
        eos_ids=eos_ids,
        max_new=max_new,
    )
