"""cocotb tests of qcore_gemv_wrap: GEMV and EMBED descriptors through the assembled path.

The arbiter, the stream controller, the rows with their VSRAMs and the requant
execute descriptors against the QMEM bus model; every output element (VSRAM
word, dump byte, ARGMAX CSR, tracked absmax) is compared with numerics.requant
over the exact integer matmul, and the saturation / shift-clamp counts with
numerics.Stats.  Inputs are driven and outputs sampled at falling edges.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cocotb
import numpy as np
import qc_numerics as qn
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from qc_qmem import QmemModel
from qc_stream import reset, value
from quettos import isa, isa_sim, program
from quettos.numerics import SFLOAT_ONE, SFLOAT_ZERO, SFloat, Stats

OP_GEMV, OP_EMBED = int(isa.Opcode.GEMV), int(isa.Opcode.EMBED)
OUT_VSRAM, OUT_ARGMAX, OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP = 0, 1, 2, 3
LAT = 32
WEIGHT_BASE, META_BASE, DUMP_BASE = 0x0001_0000, 0x0008_0000, 0x000C_0000
EMB_BASE, EMB_META = 0x0010_0000, 0x0020_0000
MASK32 = 0xFFFF_FFFF
ISSUE_GAP = 4  # cycles from done to the next issue
MAX_BURST = 64  # the MAX_BURST the pytest entry elaborates the wrapper with
TAG_FETCH, TAG_WEIGHT, TAG_META = 0, 1, 2  # qcore_pkg read tags

# The KV cache of one (layer, KV head) and the VSRAM slots an attention step uses.
HEAD_DIM = isa.HEAD_DIM
KT_BASE, V_BASE, KMETA_BASE = 0x0030_0000, 0x0060_0000, 0x00A0_0000
Q_AT, S_AT, W_AT, CTX_AT = 0, 2560, 8192, 12288


@dataclass
class Desc:
    """One GEMV / EMBED descriptor with the data placed in QMEM and VSRAM for it."""

    op: int = OP_GEMV
    n: int = 1
    k: int = 1
    k_stride: int = 1
    rows: int = 1
    vs_src: int = 0
    vs_dst: int = 1024
    out_mode: int = OUT_VSRAM
    accumulate: bool = False
    unit_meta: bool = False
    track_absmax: bool = False
    s1: int = 0
    sbias: int = 0
    sreg_dst: int = 0
    tok: int = 0
    addr_a: int = WEIGHT_BASE
    addr_m: int = META_BASE
    addr_c: int = DUMP_BASE
    sx: list[SFloat] = field(default_factory=list)  # per physical row
    meta: list[tuple[int, SFloat]] = field(default_factory=list)  # (bias_q, Sw) per channel
    w: np.ndarray | None = None  # GEMV: [tiles*WB][K] int8 (padded channels zero)
    q: np.ndarray | None = None  # EMBED: the K gathered int8 values
    emb: tuple[int, SFloat] = (0, SFLOAT_ONE)
    act: dict[int, list[int]] = field(default_factory=dict)  # row -> the K activation elements
    placed: bool = False  # the weights are already in QMEM under a layout of their own
    isa_d: isa.Descriptor | None = None  # the ISA descriptor this one executes, if any

    @property
    def embed(self) -> bool:
        return self.op == OP_EMBED

    @property
    def n_out(self) -> int:
        return self.k if self.embed else self.n

    def tiles(self, wb: int) -> int:
        return -(-self.n_out // wb)

    def part_rows(self, b_max: int) -> list[int]:
        return [r for r in range(b_max) if (self.rows >> r) & 1]

    def sw_bias(self, c: int) -> tuple[SFloat, int]:
        if self.unit_meta:
            return SFLOAT_ONE, 0
        if self.embed:
            return self.emb[1], 0
        bias, sw = self.meta[c]
        return sw, bias

    def weight_beats(self, wb: int) -> int:
        """Beats the rows accept: K per tile, or one pseudo-beat per EMBED tile."""
        return self.tiles(wb) if self.embed else self.tiles(wb) * self.k

    def rd_beats(self, wb: int) -> int:
        if self.embed:
            return self.k + 1
        return self.tiles(wb) * (self.k + (0 if self.unit_meta else 8))

    def requests(self, wb: int) -> list[tuple[int, int, int]]:
        """The bursts ``qcore_stream_ctrl`` issues for this descriptor, in order.

        The weight base of tile ``t`` walks the tile stride, ``addr_a + t*k_stride*WB``,
        and the tile's meta burst walks the token order, ``addr_m + t*WB*8``: the
        two are independent, which is what lets one address generator serve the
        transposed K region and the row-major V region (``docs/RTL.md`` 2.9).
        """
        out: list[tuple[int, int, int]] = []
        if self.n_out == 0 or self.k == 0:
            return out
        if self.embed:
            out.append((self.addr_m + self.tok * 8, 1, TAG_META))
        for t in range(1 if self.embed else self.tiles(wb)):
            base = (
                self.addr_a + (self.tok // wb) * self.k * wb
                if self.embed
                else self.addr_a + t * self.k_stride * wb
            )
            rem, addr = self.k, base
            while rem:
                length = min(MAX_BURST, rem)
                out.append((addr, length, TAG_WEIGHT))
                addr, rem = addr + length * wb, rem - length
            if not self.embed and not self.unit_meta:
                out.append((self.addr_m + t * wb * 8, 8, TAG_META))
        return out


def _sfloat(rng: random.Random, lo: int = -22, hi: int = -6) -> SFloat:
    return SFloat(rng.randrange(1 << 15, 1 << 16), rng.randrange(lo, hi + 1))


@dataclass
class KvHead:
    """The KV cache of one (layer, KV head) as ``docs/MEMORY_MAP.md`` lays it out."""

    max_ctx: int
    pos: int
    kt: int  # K^T base: token `t`, dimension `d` at `(t/WB)*64*WB + d*WB + t%WB`
    v: int  # V base: token `t`, dimension `d` at `((d/WB)*max_ctx + t)*WB + d%WB`
    k_meta: int  # 8 B per token at `k_meta + t*8`
    keys: np.ndarray  # [max_ctx][HEAD_DIM] int8, zero past `pos`
    values: np.ndarray  # [max_ctx][HEAD_DIM] int8, zero past `pos`
    k_scale: list[SFloat]  # per token; the canonical zero past `pos`

    @property
    def live(self) -> int:
        """Tokens the cache holds: the positions written up to and including ``pos``."""
        return min(self.pos + 1, self.max_ctx)


# --------------------------------------------------------------------------- bench


class Bench:
    def __init__(self, dut, seed: int, latency: int = LAT, bw_div: int = 1) -> None:
        self.dut = dut
        self.rng = random.Random(seed)
        self.wb = len(dut.wr_strb)
        self.b_max = len(dut.cmd_rows)
        self.words = 1 << len(dut.g_row[0].u_vsram.addr_a)
        self.model = QmemModel(dut, wb=self.wb, latency=latency, bw_div=bw_div)
        self.vs: dict[tuple[int, int], int] = {}  # (bank, word) -> 256-bit word
        self.contend_beats = 0  # TAG_FETCH beats the contender took off the bus
        self.stream_beats = 0  # TAG_WEIGHT and TAG_META beats routed to the stream controller
        self.sim_compared = 0  # elements compared with sw/quettos/isa_sim.py
        self.sim_nonzero = 0  # of those, the ones the simulator did not leave at zero
        for name in (
            "cmd_valid_gemv", "cmd_op", "cmd_out_mode", "cmd_accumulate", "cmd_unit_meta",
            "cmd_track_absmax", "cmd_addr_a", "cmd_addr_m", "cmd_imm32", "cmd_n", "cmd_k",
            "cmd_k_stride", "cmd_vs_src", "cmd_vs_dst", "cmd_sreg_dst", "cmd_src_row",
            "cmd_dst_row", "cmd_sh0", "cmd_sh1", "cmd_rows", "cmd_tok", "cmd_sx_m", "cmd_sx_e",
            "f_req_valid", "f_req_addr", "f_req_len", "f_req_tag", "dq_count", "v_req_valid",
            "v_req_addr", "v_req_len", "v_req_tag", "k_wr_valid", "k_wr_addr", "k_wr_data",
            "k_wr_strb", "sreg_rd_en", "sreg_rd_idx",
        ):  # fmt: skip
            getattr(dut, name).value = 0

    # ---- VSRAM contents (deposited into the RAM arrays and mirrored)

    def rows_of(self, rows: int) -> int:
        """A row mask the core has: bits at or above ``B_MAX`` name rows it does not carry.

        ``qcore_top`` drops those bits before the dispatcher sees them, so a bench case
        written for two rows runs on row 0 alone in a one-row configuration.
        """
        return (rows & ((1 << self.b_max) - 1)) or 1

    def vs_word(self, bank: int, w: int) -> int:
        return self.vs.get((bank, w), 0)

    def vs_set_word(self, bank: int, w: int, val: int) -> None:
        self.vs[(bank, w)] = val
        self.dut.g_row[bank].u_vsram.mem[w].value = val

    def vs_elem(self, bank: int, e: int) -> int:
        return (self.vs_word(bank, e // 8) >> (32 * (e % 8))) & MASK32

    def vs_set_elem(self, bank: int, e: int, val: int) -> None:
        w, slot = e // 8, e % 8
        word = self.vs_word(bank, w) & ~(MASK32 << (32 * slot))
        self.vs_set_word(bank, w, word | ((val & MASK32) << (32 * slot)))

    def vs_fill(self, bank: int, start: int, count: int) -> None:
        for w in range(start // 8, -(-(start + count) // 8)):
            self.vs_set_word(bank, w, self.rng.getrandbits(256))

    def vs_read_word(self, bank: int, w: int) -> int:
        return value(self.dut.g_row[bank].u_vsram.mem[w])

    def sreg_read(self, bank: int, idx: int) -> int:
        return value(self.dut.g_row[bank].u_row.sreg[idx])

    # ---- descriptor placement

    def place(self, d: Desc) -> None:
        wb, rng = self.wb, self.rng
        for r in d.part_rows(self.b_max):
            self.vs_fill(r, d.vs_src, max(d.k, 1))
            self.vs_fill(r, d.vs_dst, d.n_out)
            for i, v in enumerate(d.act.get(r, ())):
                self.vs_set_elem(r, d.vs_src + i, v)
        if d.placed:
            return
        if d.embed:
            base = d.addr_a + (d.tok // wb) * d.k * wb
            self.model.write_bytes(base, bytes(rng.getrandbits(8) for _ in range(d.k * wb)))
            for i in range(d.k):
                self.model.write_bytes(base + i * wb + d.tok % wb, bytes([int(d.q[i]) & 0xFF]))
            self.model.write_bytes(d.addr_m + d.tok * 8, qn.meta_bytes(d.emb[0], d.emb[1]))
            return
        for t in range(d.tiles(wb)):
            for kk in range(d.k):
                lanes = d.w[t * wb : (t + 1) * wb, kk]
                self.model.write_bytes(
                    d.addr_a + (t * d.k_stride + kk) * wb, lanes.astype(np.int8).tobytes()
                )
            for j in range(wb):
                bias, sw = d.meta[t * wb + j]
                self.model.write_bytes(d.addr_m + (t * wb + j) * 8, qn.meta_bytes(bias, sw))

    def gemv(self, n: int, k: int, rows: int = 1, **kw) -> Desc:
        wb, rng = self.wb, self.rng
        d = Desc(op=OP_GEMV, n=n, k=k, rows=self.rows_of(rows), **kw)
        d.k_stride = kw.get("k_stride", k + rng.choice([0, 0, 3]))
        d.vs_src = kw.get("vs_src", rng.randrange(0, 512))
        d.vs_dst = kw.get("vs_dst", rng.randrange(1024, 8192) & ~7)
        d.sx = [_sfloat(rng) for _ in range(self.b_max)]
        tiles = d.tiles(wb)
        d.w = np.zeros((tiles * wb, k), dtype=np.int64)
        d.w[:n, :] = rng.choice([1, 1, 4]) * np.array(
            [[rng.randrange(-32, 32) for _ in range(k)] for _ in range(n)]
        )
        d.w = np.clip(d.w, -128, 127)
        d.meta = []
        for c in range(tiles * wb):
            if c >= n:
                d.meta.append((0, SFLOAT_ZERO))
            elif rng.random() < 0.03:
                d.meta.append((rng.randrange(-1000, 1000), SFLOAT_ZERO))
            else:
                d.meta.append((rng.randrange(-(1 << 20), 1 << 20), _sfloat(rng)))
        d.s1 = kw.get("s1", rng.randrange(0, 12))
        d.sbias = kw.get("sbias", rng.randrange(-40, -10) + rng.randrange(0, 30))
        d.sreg_dst = rng.randrange(0, 32)
        d.addr_a = WEIGHT_BASE + rng.randrange(0, 8) * 0x8000
        d.addr_m = META_BASE + rng.randrange(0, 8) * 0x4000
        d.addr_c = DUMP_BASE + rng.randrange(0, 8) * 0x4000
        return d

    def embed(self, k: int, tok: int, rows: int = 1, **kw) -> Desc:
        rng = self.rng
        d = Desc(
            op=OP_EMBED,
            n=k,
            k=k,
            k_stride=0,
            rows=self.rows_of(rows),
            addr_a=EMB_BASE,
            addr_m=EMB_META,
            tok=tok,
            **kw,
        )
        d.vs_src = 0
        d.vs_dst = rng.randrange(1024, 8192) & ~7
        d.sx = [SFLOAT_ONE] * self.b_max
        d.q = np.array([rng.randrange(-128, 128) for _ in range(k)], dtype=np.int64)
        d.emb = (0, _sfloat(rng))
        d.s1 = rng.randrange(8, 25)
        d.sbias = -(rng.randrange(8, 16) + d.s1) + 24
        d.sreg_dst = rng.randrange(0, 32)
        return d

    # ---- the attention step

    def kv_head(self, max_ctx: int, pos: int, slot: int = 0) -> KvHead:
        """Write the K^T, V and K-meta sub-regions of one head and return what they hold.

        The bytes go down at the addresses ``docs/MEMORY_MAP.md`` gives, so the
        GEMV address generator has to walk the transposed K region and the
        row-major V region to read back the matrices this returns.  Positions
        past ``pos`` keep the zero data and the canonical-zero meta record the
        compiler writes over the whole capacity.
        """
        wb, rng = self.wb, self.rng
        assert max_ctx % wb == 0, "the KV capacity is a whole number of tiles"
        kt = KT_BASE + slot * 0x10_0000
        v = V_BASE + slot * 0x20_0000
        k_meta = KMETA_BASE + slot * 0x1_0000
        live = min(pos + 1, max_ctx)
        keys = np.zeros((max_ctx, HEAD_DIM), dtype=np.int64)
        values = np.zeros((max_ctx, HEAD_DIM), dtype=np.int64)
        if live:
            raw = np.frombuffer(rng.randbytes(2 * live * HEAD_DIM), dtype=np.int8).astype(np.int64)
            keys[:live] = raw[: live * HEAD_DIM].reshape(live, HEAD_DIM)
            values[:live] = raw[live * HEAD_DIM :].reshape(live, HEAD_DIM)
        scale = [_sfloat(rng) if t < live else SFLOAT_ZERO for t in range(max_ctx)]
        for t in range(max_ctx // wb):  # K^T tile t: dimension major, token minor
            self.model.write_bytes(
                kt + t * HEAD_DIM * wb, keys[t * wb : (t + 1) * wb].T.astype(np.int8).tobytes()
            )
        for t in range(-(-HEAD_DIM // wb)):  # V tile t: token major, dimension minor
            cols = values[:, t * wb : t * wb + wb]
            if cols.shape[1] < wb:
                cols = np.pad(cols, ((0, 0), (0, wb - cols.shape[1])))
            self.model.write_bytes(v + t * max_ctx * wb, cols.astype(np.int8).tobytes())
        self.model.write_bytes(k_meta, b"".join(qn.meta_bytes(0, s) for s in scale))
        return KvHead(max_ctx, pos, kt, v, k_meta, keys, values, scale)

    def attention(self, kv: KvHead, rows: int = 1) -> tuple[Desc, Desc]:
        """The scores and PV GEMVs of one head, as ``sw/quettos/compiler.py`` emits them.

        Both descriptors are built through ``quettos.isa`` and their executed
        extents are ``isa_sim.gemv_dims`` at ``kv.pos``, so the bench runs the
        shapes the dispatcher derives from POS rather than shapes of its own.
        """
        rng = self.rng
        rows = self.rows_of(rows)
        scores = isa.gemv(
            addr_a=kv.kt,
            addr_m=kv.k_meta,
            n=kv.max_ctx,
            k=HEAD_DIM,
            vs_src=Q_AT,
            vs_dst=S_AT,
            sreg_src=1,
            s1=rng.randrange(0, 12),
            sbias=rng.randrange(-40, -10) + rng.randrange(0, 30),
            n_from_pos=True,
            row_mask=rows,
        )
        pv = isa.gemv(
            addr_a=kv.v,
            n=HEAD_DIM,
            k=kv.max_ctx,
            vs_src=W_AT,
            vs_dst=CTX_AT,
            sreg_src=2,
            s1=rng.randrange(0, 12),
            sbias=rng.randrange(-40, -10) + rng.randrange(0, 30),
            unit_meta=True,
            k_from_pos=True,
            row_mask=rows,
        )
        return self._scores(scores, kv, rows), self._pv(pv, kv, rows)

    def _attn_base(self, d: isa.Descriptor, pos: int, rows: int) -> tuple[Desc, int]:
        """The bench descriptor of an attention GEMV, with the extents POS derives."""
        n, k, _ = isa_sim.gemv_dims(d, pos, self.wb)
        return Desc(
            op=OP_GEMV,
            n=n,
            k=k,
            k_stride=d.k,
            rows=rows,
            vs_src=d.vs_src,
            vs_dst=d.vs_dst,
            unit_meta=d.unit_meta,
            s1=d.sh0,
            sbias=d.sh1,
            addr_a=d.addr_a,
            addr_m=d.addr_m,
            placed=True,
            isa_d=d,
        ), k

    def _scores(self, d: isa.Descriptor, kv: KvHead, rows: int) -> Desc:
        """`q . K^T`: the tokens are the output channels, the head dimension is K."""
        out, _ = self._attn_base(d, kv.pos, rows)
        wide = out.tiles(self.wb) * self.wb
        out.w = np.zeros((wide, HEAD_DIM), dtype=np.int64)
        out.w[: out.n] = kv.keys[: out.n]
        out.meta = [(0, kv.k_scale[c]) for c in range(wide)]
        out.sx = [_sfloat(self.rng) for _ in range(self.b_max)]
        q = np.frombuffer(self.rng.randbytes(HEAD_DIM), dtype=np.int8).astype(np.int64)
        out.act = {r: [int(x) for x in q] for r in out.part_rows(self.b_max)}
        return out

    def _pv(self, d: isa.Descriptor, kv: KvHead, rows: int) -> Desc:
        """`w . V`: the head dimension is the output channels, the tokens are K."""
        rng = self.rng
        out, k = self._attn_base(d, kv.pos, rows)
        wide = out.tiles(self.wb) * self.wb
        out.w = np.zeros((wide, k), dtype=np.int64)
        out.w[:HEAD_DIM] = kv.values[:k].T
        out.meta = [(0, SFLOAT_ONE)] * wide  # unit_meta: the stream carries no record
        # The PV activation is a softmax row: int16 weights with one dominant token,
        # and the whole output scale rides on Sx as SREG_out = 2**(1 + e_max).
        weights = [
            rng.choice([0, 0, rng.randrange(0, 64), rng.randrange(0, 2048)]) for _ in range(k)
        ]
        weights[rng.randrange(k)] = rng.randrange(1 << 14, 1 << 15)
        out.act = {r: list(weights) for r in out.part_rows(self.b_max)}
        out.sx = [SFloat(1 << 15, rng.randrange(-24, -6)) for _ in range(self.b_max)]
        return out

    # ---- execution

    async def watch_beats(self) -> None:
        """Count returned beats by the sink the arbiter routed them to, one sample per cycle."""
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            self.stream_beats += value(dut.rdw_valid) + value(dut.rdm_valid)
            self.contend_beats += value(dut.rdf_valid)

    async def contend(self, addr: int = 0x0200_0000, length: int = 8) -> None:
        """Hold descriptor fetches on the arbiter so the weight stream shares the bus.

        The beats come back on ``TAG_FETCH`` and no sink takes them; what they cost
        the stream is the arbiter slot and the place in the memory's return order.
        """
        dut = self.dut
        dut.f_req_addr.value = addr
        dut.f_req_len.value = length
        dut.f_req_tag.value = TAG_FETCH
        while True:
            dut.f_req_valid.value = 1
            while True:
                await RisingEdge(dut.clk)
                if value(dut.f_req_ready):
                    break
            await FallingEdge(dut.clk)
            dut.f_req_valid.value = 0
            for _ in range(self.rng.randrange(4, 40)):
                await FallingEdge(dut.clk)

    def simulate(self, d: Desc, pos: int) -> dict[int, list[int]]:
        """The destination elements ``sw/quettos/isa_sim.py`` leaves for the same descriptor.

        The simulator runs over the bench's own QMEM pages, VSRAM banks and SREG
        scales at the position the descriptor was built for, so the comparison is
        against the reference machine and not only against ``numerics.requant``.
        """
        pages = self.model.pages
        image = bytearray(((max(pages) + 1) if pages else 1) * 4096)
        for page, data in pages.items():
            image[page * 4096 : (page + 1) * 4096] = data
        m = isa_sim.Machine(image, wb=self.wb, b_max=self.b_max, vsram_words=self.words)
        for (bank, w), word in self.vs.items():
            for slot in range(8):
                m.vsram[bank][w * 8 + slot] = qn.to_signed((word >> (32 * slot)) & MASK32, 32)
        for r, sx in enumerate(d.sx):
            m.sreg[r][d.isa_d.sreg_src] = sx
        m.csr["POS"], m.csr["TOK"] = pos, d.tok
        m.csr["ROW_EN"] = (1 << self.b_max) - 1
        isa_sim.execute(m, d.isa_d)
        return {
            r: [int(v) for v in m.vsram[r][d.vs_dst : d.vs_dst + d.n_out]]
            for r in d.part_rows(self.b_max)
        }

    async def run(
        self, d: Desc, tag: str, max_cycles: int = 40000, sim_pos: int | None = None
    ) -> None:
        dut, wb = self.dut, self.wb
        self.place(d)
        exp = self.expect(d)
        sim = self.simulate(d, sim_pos) if sim_pos is not None else None
        rd0 = self.stream_beats
        req0 = len(self.model.requests)
        sx_m = sum(s.m << (16 * r) for r, s in enumerate(d.sx))
        sx_e = sum(qn.from_signed(s.e, 8) << (8 * r) for r, s in enumerate(d.sx))
        await FallingEdge(dut.clk)
        dut.cmd_op.value = d.op
        dut.cmd_out_mode.value = d.out_mode
        dut.cmd_accumulate.value = int(d.accumulate)
        dut.cmd_unit_meta.value = int(d.unit_meta)
        dut.cmd_track_absmax.value = int(d.track_absmax)
        dut.cmd_addr_a.value = d.addr_a
        dut.cmd_addr_m.value = d.addr_m
        dut.cmd_imm32.value = d.addr_c
        dut.cmd_n.value = d.n_out
        dut.cmd_k.value = d.k
        dut.cmd_k_stride.value = d.k_stride
        dut.cmd_vs_src.value = d.vs_src
        dut.cmd_vs_dst.value = d.vs_dst
        dut.cmd_sreg_dst.value = d.sreg_dst
        dut.cmd_src_row.value = 0
        dut.cmd_dst_row.value = 0
        dut.cmd_sh0.value = d.s1
        dut.cmd_sh1.value = qn.from_signed(d.sbias, 8)
        dut.cmd_rows.value = d.rows
        dut.cmd_tok.value = d.tok
        dut.cmd_sx_m.value = sx_m
        dut.cmd_sx_e.value = sx_e
        dut.cmd_valid_gemv.value = 1
        await FallingEdge(dut.clk)
        dut.cmd_valid_gemv.value = 0
        sat = err_shift = err_bounds = beats = cycles = 0
        argmax = None
        while True:
            cycles += 1
            assert cycles < max_cycles, f"{tag}: no done within {max_cycles} cycles"
            sat += value(dut.sat_inc)
            err_shift += value(dut.err_shift_inc)
            err_bounds += value(dut.err_bounds_inc)
            beats += value(dut.gemv_beat)
            if value(dut.argmax_we):
                argmax = (value(dut.argmax_tok), value(dut.argmax_val))
            if value(dut.done_gemv):
                break
            await FallingEdge(dut.clk)
        for _ in range(ISSUE_GAP):
            await FallingEdge(dut.clk)
            beats += value(dut.gemv_beat)
        # The dispatcher issues only while the stream controller is idle: a
        # partial tile's padded meta beats may still be draining after done.
        idle_wait = 0
        while value(dut.stream_busy):
            await FallingEdge(dut.clk)
            idle_wait += 1
            assert idle_wait < 64, f"{tag}: stream busy long after done"
        assert value(dut.requant_busy) == 0, f"{tag}: requant busy after done"
        assert beats == d.weight_beats(wb), (
            f"{tag}: {beats} beats accepted, expected {d.weight_beats(wb)}"
        )
        beats_rd = self.stream_beats - rd0
        assert beats_rd == d.rd_beats(wb), f"{tag}: {beats_rd} read beats"
        got_req = [r for r in self.model.requests[req0:] if r[2] != TAG_FETCH]
        assert got_req == d.requests(wb), (
            f"{tag}: bursts {got_req[:6]}... != {d.requests(wb)[:6]}..."
        )
        assert err_bounds == 0, f"{tag}: {err_bounds} bounds errors"
        self.check(d, exp, tag, sat, err_shift, argmax)
        if sim is not None:
            assert sorted(sim) == d.part_rows(self.b_max) and all(
                len(v) == d.n_out for v in sim.values()
            ), f"{tag}: the simulator left no destination range to compare"
            for r, want in sim.items():
                got = [qn.to_signed(self.vs_elem(r, d.vs_dst + c), 32) for c in range(d.n_out)]
                assert got == want, (
                    f"{tag}: row {r} differs from isa_sim at element "
                    f"{next(c for c in range(d.n_out) if got[c] != want[c])}"
                )
                self.sim_compared += len(want)
                self.sim_nonzero += sum(1 for v in want if v)
        self.dut._log.info("%s: %d cycles, %d beats", tag, cycles, beats)

    # ---- reference

    def expect(self, d: Desc) -> dict:
        stats = Stats()
        out: dict[int, list[int]] = {}
        for r in d.part_rows(self.b_max):
            ys = []
            if d.embed:
                accs = [int(q) << 24 for q in d.q]
            else:
                a = np.array(
                    [
                        qn.to_signed(self.vs_elem(r, d.vs_src + kk) & 0xFFFF, 16)
                        for kk in range(d.k)
                    ],
                    dtype=np.int64,
                )
                accs = [int(np.dot(d.w[c], a)) for c in range(d.n)]
            for c, acc in enumerate(accs):
                sw, bias = d.sw_bias(c)
                old = qn.to_signed(self.vs_elem(r, d.vs_dst + c), 32) if d.accumulate else None
                ys.append(qn.requant(acc, sw, d.sx[r], d.s1, d.sbias, bias, old, stats))
            out[r] = ys
        return {"y": out, "sat": stats.sat, "err_shift": stats.err_shift}

    def check(self, d: Desc, exp: dict, tag: str, sat: int, err_shift: int, argmax) -> None:
        vsram = d.out_mode in (OUT_VSRAM, OUT_VSRAM_DUMP)
        dump = d.out_mode in (OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP)
        rows = d.part_rows(self.b_max)
        for r in rows:
            ys = exp["y"][r]
            # every word of the destination range: written elements, untouched others
            for w in range(d.vs_dst // 8, -(-(d.vs_dst + d.n_out) // 8)):
                want = self.vs_word(r, w)
                if vsram:
                    for slot in range(8):
                        c = w * 8 + slot - d.vs_dst
                        if 0 <= c < d.n_out:
                            want = (want & ~(MASK32 << (32 * slot))) | (
                                (ys[c] & MASK32) << (32 * slot)
                            )
                got = self.vs_read_word(r, w)
                assert got == want, (
                    f"{tag}: row {r} word {w}: got {got:#066x}, expected {want:#066x}"
                )
                self.vs[(r, w)] = got
            if dump:
                got = self.model.read_bytes(d.addr_c + r * 4 * d.n_out, 4 * d.n_out)
                want = b"".join((y & MASK32).to_bytes(4, "little") for y in ys)
                assert got == want, f"{tag}: row {r} dump differs"
            if d.track_absmax:
                amax = max((abs(y) for y in ys), default=0)
                got = self.sreg_read(r, d.sreg_dst)
                assert got == amax, f"{tag}: row {r} absmax {got} != {amax}"
        if d.out_mode in (OUT_ARGMAX, OUT_ARGMAX_DUMP):
            ys = exp["y"][rows[-1]]
            best = max(range(len(ys)), key=lambda c: (ys[c], -c))
            assert argmax == (best, ys[best] & MASK32), (
                f"{tag}: argmax {argmax} != {(best, ys[best] & MASK32)}"
            )
        assert sat == exp["sat"], f"{tag}: SAT {sat} != {exp['sat']}"
        assert err_shift == exp["err_shift"], f"{tag}: ERR_SHIFT {err_shift} != {exp['err_shift']}"


async def _start(dut, seed: int, latency: int = LAT, bw_div: int = 1) -> Bench:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    bench = Bench(dut, seed, latency=latency, bw_div=bw_div)
    await reset(dut, dut.rst)
    cocotb.start_soon(bench.model.run())
    cocotb.start_soon(bench.watch_beats())
    return bench


# --------------------------------------------------------------------------- tests


@cocotb.test()
async def test_directed(dut):
    """Full, partial and multi-burst tiles, two rows, accumulate, every output mode."""
    b = await _start(dut, 1)
    cases = [
        ("full tile", b.gemv(16, 8)),
        ("partial tile with meta", b.gemv(17, 3)),
        ("one channel", b.gemv(1, 1)),
        ("two bursts, accumulate, both rows", b.gemv(5, 70, rows=3, accumulate=True)),
        ("argmax and dump", b.gemv(33, 20, rows=2, out_mode=OUT_ARGMAX_DUMP)),
        (
            "three tiles, vsram and dump, absmax",
            b.gemv(40, 130, rows=3, out_mode=OUT_VSRAM_DUMP, track_absmax=True),
        ),
        ("unit meta", b.gemv(20, 9, unit_meta=True)),
        ("argmax only", b.gemv(31, 5, out_mode=OUT_ARGMAX, track_absmax=True)),
        ("embed 37", b.embed(37, 1234)),
        ("embed 16", b.embed(16, 15)),
        ("embed 1, both rows", b.embed(1, 0, rows=3)),
        ("embed 50, dump", b.embed(50, 3, out_mode=OUT_VSRAM_DUMP)),
    ]
    for name, d in cases:
        await b.run(d, name)


@cocotb.test()
async def test_random(dut):
    """Random GEMV and EMBED descriptors back to back."""
    b = await _start(dut, 2)
    await _random_sequence(b, 40)


@cocotb.test()
async def test_random_slow_memory(dut):
    """The same mix with one memory beat every three cycles."""
    b = await _start(dut, 3)
    b.model.bw_div = 3
    await _random_sequence(b, 24)


async def _random_sequence(b: Bench, count: int) -> None:
    for i in range(count):
        if b.rng.random() < 0.2:
            d = b.embed(
                b.rng.randrange(1, 60), b.rng.randrange(0, 2000), rows=b.rng.choice([1, 2, 3])
            )
        else:
            d = b.gemv(
                b.rng.randrange(1, 40),
                b.rng.randrange(1, 80),
                rows=b.rng.choice([1, 1, 2, 3]),
                out_mode=b.rng.choice(
                    [OUT_VSRAM, OUT_VSRAM, OUT_ARGMAX, OUT_ARGMAX_DUMP, OUT_VSRAM_DUMP]
                ),
                accumulate=b.rng.random() < 0.3,
                unit_meta=b.rng.random() < 0.15,
                track_absmax=b.rng.random() < 0.3,
            )
        await b.run(d, f"random {i}")


# --------------------------------------------------------------------------- attention


def _attention_shapes(b: Bench, scores: Desc, pv: Desc, kv: KvHead, where: str) -> None:
    """The two GEMVs run the extents and the tile strides an attention step gives them."""
    wb = b.wb
    assert scores.k == HEAD_DIM and scores.k_stride == HEAD_DIM, (
        f"{where}: the scores GEMV walks the head dimension, K {scores.k} stride {scores.k_stride}"
    )
    want_n = min(-(-kv.live // wb) * wb, kv.max_ctx)
    assert scores.n == want_n, f"{where}: scores N {scores.n}, expected {want_n}"
    assert pv.n == HEAD_DIM and pv.k == kv.live, f"{where}: PV shape {pv.n} x {pv.k}"
    assert pv.k_stride == kv.max_ctx and pv.unit_meta, (
        f"{where}: PV stride {pv.k_stride}, unit_meta {pv.unit_meta}"
    )


def _scores_past_pos_are_zero(b: Bench, scores: Desc, kv: KvHead, where: str) -> None:
    """A token the cache has not been written for carries the canonical-zero scale, so its
    score is zero: that is what lets the scores GEMV round N up to a whole tile."""
    for r in scores.part_rows(b.b_max):
        for c in range(kv.live, scores.n):
            got = b.vs_elem(r, scores.vs_dst + c)
            assert got == 0, f"{where}: row {r} scored {got:#x} for unwritten token {c}"


async def _attention_head(
    b: Bench, kv: KvHead, where: str, rows: int = 1, against_sim: bool = False
) -> None:
    """Run the scores GEMV and then the PV GEMV of one head over ``kv``."""
    scores, pv = b.attention(kv, rows=rows)
    _attention_shapes(b, scores, pv, kv, where)
    pos = kv.pos if against_sim else None
    await b.run(scores, f"{where} scores", max_cycles=400000, sim_pos=pos)
    _scores_past_pos_are_zero(b, scores, kv, where)
    await b.run(pv, f"{where} pv", max_cycles=400000, sim_pos=pos)


@cocotb.test()
async def test_attention_positions_latencies_and_backpressure(dut):
    """The scores and PV GEMVs over the KV layout at every position and every memory latency.

    The K^T and V sub-regions are written at the byte addresses of
    ``docs/MEMORY_MAP.md`` and the descriptors come from ``quettos.isa`` with the
    extents ``isa_sim.gemv_dims`` derives from POS, so the run exercises all four
    things an attention step asks of the matrix-vector path: the length and the
    depth from the position, the per-token meta stream at ``addr_m + t*WB*8``, the
    unit-scale form of the PV GEMV, and one address generator walking the
    transposed K region at stride 64 and the row-major V region at stride
    ``max_ctx``.  Every output element is ``numerics.requant`` of the exact
    integer matmul, every burst address is compared with the layout, and a
    descriptor fetch contends for the bus throughout.
    """
    b = await _start(dut, 11)
    cocotb.start_soon(b.contend())
    max_ctx = 256
    for pos in (0, 1, 63, 64, 65, max_ctx - 1):
        kv = b.kv_head(max_ctx, pos)
        for latency, bw_div in ((1, 1), (32, 3), (200, 1)):
            b.model.latency, b.model.bw_div = latency, bw_div
            await _attention_head(b, kv, f"POS={pos} LAT={latency} bw_div={bw_div}")
    assert b.contend_beats > 0, "the fetch contender never won the arbiter"


@cocotb.test()
async def test_attention_at_the_context_limit(dut):
    """One head at the largest position the ISA allows: MAX_CTX tokens of K^T and V.

    ``POS = MAX_CTX - 1`` makes the scores GEMV stream the whole capacity and the
    PV GEMV take ``K = MAX_CTX``, which is the deepest matrix-vector the core
    runs and the one whose tile stride is furthest from its depth.
    """
    b = await _start(dut, 12, latency=1)
    max_ctx = program.MAX_CTX
    kv = b.kv_head(max_ctx, max_ctx - 1)
    await _attention_head(b, kv, f"POS={max_ctx - 1}")


@cocotb.test()
async def test_attention_on_two_rows(dut):
    """Both activation rows share one attention head's weight stream."""
    b = await _start(dut, 13)
    rows = (1 << b.b_max) - 1
    for pos in (0, 63, 64):
        kv = b.kv_head(128, pos)
        await _attention_head(b, kv, f"rows {rows:#x} POS={pos}", rows=rows)


@cocotb.test()
async def test_meta_records_follow_the_token_order(dut):
    """The tile's meta burst is at ``addr_m + t*WB*8`` whatever the tile stride is.

    A GEMV whose tile stride is three times its depth still reads its records 8 B
    per channel in channel order, which is what makes the K meta region -- 8 B per
    token, one token per output channel -- readable by the same address generator.
    """
    b = await _start(dut, 14)
    for k, tiles in ((7, 4), (20, 3), (1, 5)):
        d = b.gemv(tiles * b.wb, k, k_stride=3 * k + 1)
        await b.run(d, f"stride {d.k_stride} K {k}")
        meta = [r for r in b.model.requests if r[2] == TAG_META][-tiles:]
        want = [(d.addr_m + t * b.wb * 8, 8, TAG_META) for t in range(tiles)]
        assert meta == want, f"meta bursts {meta} != {want}"


@cocotb.test()
async def test_attention_against_the_simulator(dut):
    """Every element of both attention GEMVs, against sw/quettos/isa_sim.py at the same POS.

    ``isa_sim`` runs the same ``quettos.isa`` descriptor over the bench's own QMEM
    pages, VSRAM banks and SREG scales, so the destination range is compared with
    the reference machine's, element by element, and not only with the arithmetic
    of ``numerics.requant``.
    """
    b = await _start(dut, 15)
    max_ctx = 128
    positions = (0, 1, 63, 64, 65, max_ctx - 1)
    for pos in positions:
        kv = b.kv_head(max_ctx, pos)
        await _attention_head(b, kv, f"vs isa_sim POS={pos}", against_sim=True)
    want = sum(min(-(-(p + 1) // b.wb) * b.wb, max_ctx) + HEAD_DIM for p in positions)
    assert b.sim_compared == want, f"{b.sim_compared} elements compared, expected {want}"
    assert b.sim_nonzero > 4 * len(positions), f"only {b.sim_nonzero} non-zero elements compared"
    dut._log.info(
        "isa_sim: %d elements compared, %d of them non-zero", b.sim_compared, b.sim_nonzero
    )
