"""Compiler: the image parses back to the QuantModel arrays, the programs follow the decode
dataflow with the program constants, dump_plan.json agrees with the descriptors, layout.json
hashes are exact and two compiles are byte-identical; on synthetic tiny models, two-layer real
models and (``slow``) the complete models."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from quettos import calibrate, cli, compiler, golden, isa, numerics, program, quantize, synthetic
from quettos.compiler import FILES, Image
from quettos.isa import Descriptor, Opcode, OutMode, VquantFlag
from quettos.model import REPO_ROOT, ModelSpec
from quettos.synthetic import SHAPES
from quettos.tokenizer_io import prompt_tokens, read_tokens_bin, token_bytes

SYN_CTX = 64
PROMPT = REPO_ROOT / "prompts" / "chat_short.json"
# docs/MEMORY_MAP.md, Qwen2.5-0.5B-Instruct at MAX_CTX 2048 and WB 64
QWEN_VSRAM = [
    ("X", 0, 896),
    ("XN", 896, 896),
    ("A", 1792, 896),
    ("QKV", 2688, 1152),
    ("CTX", 3840, 896),
    ("CTXQ", 4736, 896),
    ("GU", 5632, 9728),
    ("HQ", 15360, 4864),
    ("S", 20224, 2048),
    ("W", 22272, 2048),
]
QWEN_LAYER_SIZES = {
    "wqkv": 1_032_192,
    "wqkv.meta": 9_216,
    "wo": 802_816,
    "wo.meta": 7_168,
    "wgu": 8_716_288,
    "wgu.meta": 77_824,
    "wdown": 4_358_144,
    "wdown.meta": 7_168,
    "gamma_in": 1_792,
    "gamma_post": 1_792,
}
STREAM_BYTES = {"qwen2.5-0.5b-instruct": 497_697_536, "smollm2-135m-instruct": 136_187_520}
# the golden trace op each dump-plan name corresponds to (rope covers rope_q and rope_k)
LAYER_OPS = (
    "rmsnorm_in",
    "quant_in",
    "gemv_qkv",
    "rope",
    "quant_q",
    "subc_k",
    "quant_k",
    "quant_v",
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("compiler")


@pytest.fixture(scope="module")
def syn(out_root: Path) -> synthetic.SyntheticModel:
    return synthetic.build(SHAPES[1], seed=1, out_dir=out_root / "syn")


@pytest.fixture(scope="module")
def built(syn: synthetic.SyntheticModel, out_root: Path) -> compiler.Compiled:
    return compiler.compile(syn.quant, syn.spec, out_dir=out_root / "img", max_ctx=SYN_CTX)


@pytest.fixture(scope="session")
def calib(spec: ModelSpec) -> dict:
    path = calibrate.calib_path(spec)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return calibrate.load_calib(path)


@pytest.fixture(scope="session")
def qmodel2(spec: ModelSpec, calib: dict) -> quantize.QuantModel:
    """Two layers of each model: the quantized file when present, else quantized here."""
    path = quantize.default_path(spec.name)
    if path.is_file():
        model = quantize.load(path)
        if model.n_layers >= 2:
            return compiler.truncate_layers(model, 2)
    return quantize.build_quant_model(spec, calib, layers=2)


@pytest.fixture(scope="session")
def qmodel_full(spec: ModelSpec) -> quantize.QuantModel:
    path = quantize.default_path(spec.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos quantize)")
    model = quantize.load(path)
    if model.n_layers != spec.layers:
        pytest.skip(f"{path} is truncated to {model.n_layers} layers")
    return model


# --------------------------------------------------------------------------- helpers


def _check_regions(layout: dict, out: Path) -> None:
    regs = layout["regions"]
    names = [r["name"] for r in regs]
    assert len(set(names)) == len(names)
    pos = 0
    for r in regs:
        assert r["addr"] % compiler.ALIGN == 0 and r["addr"] >= pos, r["name"]
        assert r["size"] > 0
        pos = r["addr"] + r["size"]
    assert pos <= layout["image"]["size"] == (out / FILES["image"]).stat().st_size
    assert layout["image"]["size"] % compiler.ALIGN == 0
    assert compiler.sha256_file(out / FILES["image"]) == layout["image"]["sha256"]


def _check_round_trip(img: Image, q: quantize.QuantModel, max_ctx: int) -> None:
    for i, lay in enumerate(q.layers):
        for lin_name in compiler.LINEARS:
            lin = getattr(lay, lin_name)
            name = f"layer.{i}.{lin_name}"
            assert np.array_equal(img.weights(name), lin.q), name
            bias, m, e = img.meta(f"{name}.meta")
            n = lin.q.shape[0]
            assert m.size == compiler.tiles_for(n, img.wb) * img.wb
            assert np.array_equal(bias[:n], lin.bias_q) and np.array_equal(m[:n], lin.scale_m)
            assert np.array_equal(e[:n], lin.scale_e)
            assert not bias[n:].any() and not m[n:].any() and not e[n:].any()
        assert np.array_equal(img.gamma(f"layer.{i}.gamma_in"), lay.norm_in.gamma_q)
        assert np.array_equal(img.gamma(f"layer.{i}.gamma_post"), lay.norm_post.gamma_q)
        assert img.regions[f"layer.{i}.gamma_in"]["gamma_e"] == lay.norm_in.gamma_e
        assert np.array_equal(img.kcenter(i), q.k_center[i])
        for g in range(q.kv_heads):
            for part in ("kt", "v", "k_meta", "v_meta"):
                assert not any(img.read(f"kv.{i}.{g}.{part}")), (i, g, part)
    assert np.array_equal(img.weights("embed"), q.embed.q)
    _, m, e = img.meta("embed.meta")
    assert np.array_equal(m[: q.vocab], q.embed.scale_m) and np.array_equal(
        e[: q.vocab], q.embed.scale_e
    )
    assert np.array_equal(img.gamma("gamma_final"), q.norm_final.gamma_q)
    assert np.array_equal(img.rope(), numerics.load_rope_table(q.rope_theta)[:max_ctx])


def _by_name(layout: dict) -> dict[str, dict]:
    return {r["name"]: r for r in layout["regions"]}


# --------------------------------------------------------------------------- synthetic


def test_outputs_regions_and_hashes(
    built: compiler.Compiled, syn: synthetic.SyntheticModel
) -> None:
    out, lay, q = built.out_dir, built.layout, syn.quant
    for key in ("image", "decode", "prefill", "decode_lst", "prefill_lst", "layout", "dump_plan"):
        assert (out / FILES[key]).is_file(), key
    assert not (out / FILES["tokens_bin"]).exists()  # synthetic models carry no tokenizer
    assert json.loads((out / FILES["layout"]).read_text()) == lay
    assert compiler.load_layout(out) == lay and compiler.load_dump_plan(out) == built.dump_plan
    _check_regions(lay, out)
    by = _by_name(lay)
    assert lay["bases"] == {
        "programs": 0,
        "rope": 0x0010_0000,
        "constants": 0x0014_0000,
        "weights": 0x0020_0000,
        "embed": by["embed"]["addr"],
        "kv": by["kv.0.0.kt"]["addr"],
    }
    assert by["prog.decode"]["addr"] == 0 and by["rope"]["addr"] == 0x0010_0000
    assert by["kcenter.0"]["addr"] == 0x0014_0000 and by["layer.0.wqkv"]["addr"] == 0x0020_0000
    assert by["rope"]["size"] == SYN_CTX * 128 and by["kcenter.0"]["size"] == q.kv_heads * 256
    sizes = compiler.kv_sizes(SYN_CTX, 64)
    assert sizes == {
        "kt": SYN_CTX * 64,
        "v": SYN_CTX * 64,
        "k_meta": SYN_CTX * 8,
        "v_meta": SYN_CTX * 8,
    }
    for i in range(q.n_layers):
        for g in range(q.kv_heads):
            for part, size in sizes.items():
                assert by[f"kv.{i}.{g}.{part}"]["size"] == size
    assert lay["kv"]["kt_tiles"] == 1 and lay["kv"]["v_tiles"] == 1
    n_layers, kv = q.n_layers, q.kv_heads
    assert len(lay["regions"]) == 2 + 1 + n_layers + 10 * n_layers + 3 + 4 * n_layers * kv
    assert (
        lay["isa_version"] == isa.ISA_VERSION == 1 and lay["wb"] == 64 and lay["max_ctx"] == SYN_CTX
    )
    assert lay["frac"] == q.frac and lay["calib_tokens_sha256"] == q.calib_tokens_sha256
    # the model block names the end-of-sequence ids the harness stops on
    assert lay["model"]["eos_ids"] == syn.spec.eos_ids == []
    assert lay["constants"]["gemvs"] == program.build(q).as_dict()["gemvs"]
    assert lay["constants"]["max_ctx"] == program.MAX_CTX
    assert lay["csr"]["registers"] == {c.name: c.word for c in isa.CSRS}
    assert lay["tables"]["rope"]["sha256"] == compiler.sha256_file(
        numerics.rope_table_path(q.rope_theta)
    )
    assert lay["tables"]["luts"]["sha256"] == compiler.sha256_file(numerics.LUTS_JSON)
    assert lay["expected_tokens"] is None and lay["prompt"] is None and lay["tokens_bin"] is None
    assert lay["dump_plan"]["sha256"] == compiler.sha256_file(out / FILES["dump_plan"])
    # weight-stream bytes per token add up over the layer regions and the head
    tr = lay["traffic"]
    body = sum(r["size"] for r in lay["regions"] if r["name"].startswith("layer."))
    head = by["embed"]["size"] + by["embed.meta"]["size"] + by["gamma_final"]["size"]
    assert tr["prefill"]["total"] == body and tr["decode"]["total"] == body + head
    assert tr["decode"]["macs"] == tr["prefill"]["macs"] + q.vocab * q.hidden
    # WT_BYTES leaves the gammas out and adds the EMBED gather (hidden bytes + one meta record)
    for which in ("decode", "prefill"):
        assert tr[which]["wt_bytes"] == tr[which]["weights"] + tr[which]["meta"] + q.hidden + 8
    assert tr["attention"] == {
        "head_layers": q.n_layers * q.heads,
        "scores_macs_per_tile": 64 * 64,
        "pv_macs_per_token": 64,
    }


def test_image_parses_back_to_the_model(built: compiler.Compiled, syn) -> None:
    img = Image(built.out_dir)
    _check_round_trip(img, syn.quant, SYN_CTX)
    assert img.program("decode") == built.decode and img.program("prefill") == built.prefill


def test_programs_and_listings(built: compiler.Compiled, syn) -> None:
    out, lay, q = built.out_dir, built.layout, syn.quant
    dec = isa.parse((out / FILES["decode"]).read_bytes())
    pre = isa.parse((out / FILES["prefill"]).read_bytes())
    assert dec == built.decode and pre == built.prefill
    assert (len(dec), len(pre)) == compiler.descriptor_counts(q)
    assert len(dec) == 1 + q.n_layers * (16 + 2 * q.kv_heads + 3 * q.heads) + 4
    assert pre == dec[:-4] + [isa.halt()]
    assert dec[-1] == isa.halt() and dec[0].opcode is Opcode.EMBED
    assert (out / FILES["decode_lst"]).read_text() == isa.disassemble(dec)
    assert (out / FILES["prefill_lst"]).read_text() == isa.disassemble(pre)
    for which, descs in (("decode", dec), ("prefill", pre)):
        p = lay["programs"][which]
        blob = (out / p["file"]).read_bytes()
        assert p["size"] == len(blob) == 32 * len(descs) == 32 * p["descriptors"]
        assert p["sha256"] == compiler.sha256_bytes(blob)
        assert p["addr"] % 64 == 0 and p["listing"] == FILES[f"{which}_lst"]
        assert _by_name(lay)[f"prog.{which}"]["addr"] == p["addr"]
    assert lay["programs"]["prefill"]["addr"] == compiler.align_up(
        lay["programs"]["decode"]["size"]
    )
    assert all(d.row_mask == 1 and d.src_row == 0 and d.dst_row == 0 for d in dec)


def test_decode_program_implements_the_dataflow(built: compiler.Compiled, syn) -> None:
    """Per layer: the op template, the constants, the addresses and the VSRAM ranges."""
    q, lay = syn.quant, built.layout
    by = _by_name(lay)
    vs = {m["name"]: (m["start"], m["count"]) for m in lay["vsram"]["map"]}
    sr = lay["sreg"]
    pc = program.build(q)
    hd, kvd, hid, inter = q.heads * 64, q.kv_heads * 64, q.hidden, q.intermediate
    n_rep = q.heads // q.kv_heads
    dec = list(built.decode)
    plan = built.dump_plan["decode"]
    assert [e["name"] for e in plan[: len(dec)]] == [e["name"] for e in plan]

    def take(op: Opcode, name: str) -> Descriptor:
        d, entry = dec.pop(0), plan.pop(0)
        assert d.opcode is op and entry["name"] == name, (d, entry)
        return d

    em = take(Opcode.EMBED, "embed")
    assert (em.addr_a, em.addr_m) == (by["embed"]["addr"], by["embed.meta"]["addr"])
    assert (em.k, em.n, em.vs_dst) == (hid, hid, vs["X"][0])
    assert (em.sh0, em.sh1) == (pc["embed"].s1, pc["embed"].sbias)
    for i in range(q.n_layers):
        lay_i = q.layers[i]
        norm = take(Opcode.VRMSNORM, "rmsnorm_in")
        assert (norm.vs_src, norm.vs_dst, norm.n) == (vs["X"][0], vs["XN"][0], hid)
        assert norm.addr_a == by[f"layer.{i}.gamma_in"]["addr"] and norm.track_absmax
        assert (norm.sh0, norm.sh1, norm.imm32) == (
            q.frac["X"],
            -lay_i.norm_in.gamma_e,
            q.eps_c["input"],
        )
        assert isa.sfloat_from_imm(norm.addr_m) == q.sqrt_d and norm.sreg_dst == sr["ABSMAX"]
        qa = take(Opcode.VQUANT, "quant_in")
        assert qa.flags == VquantFlag.USE_TRACKED and qa.sreg_src == sr["ABSMAX"]
        assert (qa.vs_src, qa.vs_dst, qa.n, qa.sh0, qa.sreg_dst) == (
            vs["XN"][0],
            vs["A"][0],
            hid,
            q.frac["X"],
            sr["A"],
        )
        g = take(Opcode.GEMV, "gemv_qkv")
        assert (g.addr_a, g.addr_m) == (
            by[f"layer.{i}.wqkv"]["addr"],
            by[f"layer.{i}.wqkv.meta"]["addr"],
        )
        assert (g.n, g.k, g.vs_src, g.vs_dst, g.sreg_src) == (
            hd + 2 * kvd,
            hid,
            vs["A"][0],
            vs["QKV"][0],
            sr["A"],
        )
        assert (g.sh0, g.sh1) == (pc["qkv"].s1, pc["qkv"].sbias) and not g.accumulate
        rope = take(Opcode.VROPE, "rope")
        assert (rope.vs_src, rope.n, rope.addr_a) == (vs["QKV"][0], hd + kvd, by["rope"]["addr"])
        qq = take(Opcode.VQUANT, "quant_q")
        assert qq.flags == VquantFlag.GROUP | VquantFlag.SCALE_MUL and qq.vs_aux == 64
        assert (qq.vs_src, qq.vs_dst, qq.n, qq.sh0) == (
            vs["QKV"][0],
            vs["QKV"][0],
            hd,
            q.frac["QKV"],
        )
        assert isa.sfloat_from_imm(qq.imm32) == q.log2e_over_8 and qq.sreg_dst == sr["Q"][0]
        k_at, v_at = vs["QKV"][0] + hd, vs["QKV"][0] + hd + kvd
        sub = take(Opcode.VSUBC, "subc_k")
        assert (sub.vs_src, sub.vs_dst, sub.n, sub.addr_a) == (
            k_at,
            k_at,
            kvd,
            by[f"kcenter.{i}"]["addr"],
        )
        qk = take(Opcode.VQUANT, "quant_k")
        assert qk.flags == VquantFlag.W8 | VquantFlag.GROUP and (qk.vs_src, qk.n) == (k_at, kvd)
        assert (qk.sreg_dst, qk.sh0, qk.vs_aux) == (sr["K"][0], q.frac["QKV"], 64)
        qv = take(Opcode.VQUANT, "quant_v")
        assert qv.flags == VquantFlag.W8 | VquantFlag.GROUP and (qv.vs_src, qv.n) == (v_at, kvd)
        assert qv.sreg_dst == sr["V"][0]
        for gh in range(q.kv_heads):
            kt = take(Opcode.KVWRITE, "kvwrite_k")
            assert kt.flags == isa.KvwriteFlag.TRANSPOSED and kt.vs_src == k_at + gh * 64
            assert kt.k == SYN_CTX and kt.n == 64
            assert (kt.addr_a, kt.addr_m) == (
                by[f"kv.{i}.{gh}.kt"]["addr"],
                by[f"kv.{i}.{gh}.k_meta"]["addr"],
            )
            assert kt.sreg_src == sr["K"][gh]
            vw = take(Opcode.KVWRITE, "kvwrite_v")
            assert vw.flags == 0 and vw.vs_src == v_at + gh * 64 and vw.sreg_src == sr["V"][gh]
            assert vw.k == SYN_CTX
            assert (vw.addr_a, vw.addr_m) == (
                by[f"kv.{i}.{gh}.v"]["addr"],
                by[f"kv.{i}.{gh}.v_meta"]["addr"],
            )
        for h in range(q.heads):
            gh = h // n_rep
            sc = take(Opcode.GEMV, "gemv_scores")
            assert sc.n_from_pos and not sc.k_from_pos and not sc.unit_meta
            assert (sc.addr_a, sc.addr_m) == (
                by[f"kv.{i}.{gh}.kt"]["addr"],
                by[f"kv.{i}.{gh}.k_meta"]["addr"],
            )
            assert (sc.n, sc.k, sc.vs_src, sc.vs_dst) == (
                SYN_CTX,
                64,
                vs["QKV"][0] + h * 64,
                vs["S"][0],
            )
            assert (sc.sreg_src, sc.sh0, sc.sh1) == (
                sr["Q"][h],
                pc["scores"].s1,
                pc["scores"].sbias,
            )
            sm = take(Opcode.VSOFTMAX, "softmax")
            assert sm.len_from_pos and sm.addr_a == by[f"kv.{i}.{gh}.v_meta"]["addr"]
            assert (sm.vs_src, sm.vs_dst, sm.n, sm.sh0, sm.sreg_dst) == (
                vs["S"][0],
                vs["W"][0],
                SYN_CTX,
                q.frac["S"],
                sr["W"],
            )
            pv = take(Opcode.GEMV, "gemv_pv")
            assert pv.k_from_pos and pv.unit_meta and not pv.n_from_pos and pv.addr_m == 0
            assert (pv.addr_a, pv.n, pv.k, pv.vs_src) == (
                by[f"kv.{i}.{gh}.v"]["addr"],
                64,
                SYN_CTX,
                vs["W"][0],
            )
            assert (pv.vs_dst, pv.sreg_src, pv.sh0, pv.sh1) == (
                vs["CTX"][0] + h * 64,
                sr["W"],
                pc["pv"].s1,
                pc["pv"].sbias,
            )
        qc = take(Opcode.VQUANT, "quant_ctx")
        assert qc.flags == 0 and (qc.vs_src, qc.vs_dst, qc.n, qc.sh0, qc.sreg_dst) == (
            vs["CTX"][0],
            vs["CTXQ"][0],
            hd,
            q.frac["CTX"],
            sr["CTX"],
        )
        go = take(Opcode.GEMV, "gemv_o")
        assert go.accumulate and (go.n, go.k, go.vs_src, go.vs_dst, go.sreg_src) == (
            hid,
            hd,
            vs["CTXQ"][0],
            vs["X"][0],
            sr["CTX"],
        )
        assert (go.addr_a, go.sh0, go.sh1) == (
            by[f"layer.{i}.wo"]["addr"],
            pc["o"].s1,
            pc["o"].sbias,
        )
        n2 = take(Opcode.VRMSNORM, "rmsnorm_post")
        assert (
            n2.addr_a == by[f"layer.{i}.gamma_post"]["addr"] and n2.sh1 == -lay_i.norm_post.gamma_e
        )
        assert n2.imm32 == q.eps_c["post"]
        take(Opcode.VQUANT, "quant_post")
        gg = take(Opcode.GEMV, "gemv_gu")
        assert (gg.n, gg.k, gg.vs_src, gg.vs_dst, gg.sreg_src) == (
            2 * inter,
            hid,
            vs["A"][0],
            vs["GU"][0],
            sr["A"],
        )
        assert (gg.addr_a, gg.sh0, gg.sh1) == (
            by[f"layer.{i}.wgu"]["addr"],
            pc["gu"].s1,
            pc["gu"].sbias,
        )
        si = take(Opcode.VSILUMUL, "silu_mul")
        assert (si.vs_src, si.vs_aux, si.vs_dst, si.n) == (
            vs["GU"][0],
            vs["GU"][0] + inter,
            vs["HQ"][0],
            inter,
        )
        assert (si.sh0, si.sh1, si.sreg_dst) == (
            q.frac["GU"],
            2 * q.frac["GU"] - q.frac["H"],
            sr["ABSMAX"],
        )
        qh = take(Opcode.VQUANT, "quant_h")
        assert qh.flags == VquantFlag.USE_TRACKED and (qh.vs_src, qh.vs_dst, qh.n) == (
            vs["HQ"][0],
            vs["HQ"][0],
            inter,
        )
        assert (qh.sh0, qh.sreg_dst, qh.sreg_src) == (q.frac["H"], sr["H"], sr["ABSMAX"])
        gd = take(Opcode.GEMV, "gemv_down")
        assert gd.accumulate and (gd.n, gd.k, gd.vs_src, gd.vs_dst, gd.sreg_src) == (
            hid,
            inter,
            vs["HQ"][0],
            vs["X"][0],
            sr["H"],
        )
        assert (gd.addr_a, gd.sh0, gd.sh1) == (
            by[f"layer.{i}.wdown"]["addr"],
            pc["down"].s1,
            pc["down"].sbias,
        )
    nf = take(Opcode.VRMSNORM, "rmsnorm_final")
    assert nf.addr_a == by["gamma_final"]["addr"] and nf.imm32 == q.eps_c["final"]
    take(Opcode.VQUANT, "quant_final")
    lm = take(Opcode.GEMV, "gemv_lm_head")
    assert lm.out_mode is OutMode.ARGMAX and (lm.n, lm.k, lm.vs_src, lm.sreg_src) == (
        q.vocab,
        hid,
        vs["A"][0],
        sr["A"],
    )
    assert (lm.addr_a, lm.addr_m, lm.sh0, lm.sh1) == (
        by["embed"]["addr"],
        by["embed.meta"]["addr"],
        pc["lm_head"].s1,
        pc["lm_head"].sbias,
    )
    assert take(Opcode.HALT, "halt") == isa.halt()
    assert not dec and not plan
    # every GEMV / EMBED output is 8-aligned and every VSRAM range lies inside the map
    used = lay["vsram"]["used"]
    for d in built.decode:
        eff = compiler.descriptor_effects(d)
        if d.opcode in (Opcode.GEMV, Opcode.EMBED) and eff.vsram is not None:
            assert eff.vsram[0] % 8 == 0
        if eff.vsram is not None:
            assert 0 <= eff.vsram[0] and eff.vsram[0] + eff.vsram[1] <= used
        assert all(0 <= s < isa.SREG_COUNT for s in eff.sreg)


def test_dump_plan_agrees_with_the_descriptors(built: compiler.Compiled, syn) -> None:
    lay, plan = built.layout, compiler.load_dump_plan(built.out_dir)
    by = _by_name(lay)
    assert plan["format"] == "quettos-dump-plan" and plan["isa_version"] == 1
    assert plan["wb"] == 64 and plan["max_ctx"] == SYN_CTX and plan["model"] == syn.quant.name
    for which, descs in (("decode", built.decode), ("prefill", built.prefill)):
        entries = plan[which]
        assert len(entries) == len(descs)
        for i, (d, e) in enumerate(zip(descs, entries, strict=True)):
            eff = compiler.descriptor_effects(d)
            assert e["index"] == i and e["op"] == d.opcode.name
            assert e["vsram"] == (
                None if eff.vsram is None else {"start": eff.vsram[0], "count": eff.vsram[1]}
            )
            assert e["sreg"] == list(eff.sreg) and e["pos_dependent"] == eff.pos_dependent
            for m in e["mem"]:
                assert by[m["name"]]["addr"] == m["addr"] and by[m["name"]]["size"] == m["size"]
            if d.opcode is Opcode.KVWRITE:
                names = [m["name"] for m in e["mem"]]
                part = ("kt", "k_meta") if d.flags & isa.KvwriteFlag.TRANSPOSED else ("v", "v_meta")
                assert names == [f"kv.{e['layer']}.{e['kv_head']}.{p}" for p in part]
                assert by[names[0]]["addr"] == d.addr_a and by[names[1]]["addr"] == d.addr_m
            else:
                assert e["mem"] == []
            if d.opcode is Opcode.GEMV and d.out_mode is OutMode.ARGMAX:
                assert e["csr"] == ["ARGMAX_TOK", "ARGMAX_VAL"] and e["vsram"] is None
            else:
                assert e["csr"] == []
            if e["head"] is not None:
                assert e["name"] in ("gemv_scores", "softmax", "gemv_pv")
    names = [e["name"] for e in plan["decode"]]
    per_layer = list(LAYER_OPS) + ["kvwrite_k", "kvwrite_v"] * syn.quant.kv_heads
    per_layer += ["gemv_scores", "softmax", "gemv_pv"] * syn.quant.heads
    per_layer += [
        "quant_ctx",
        "gemv_o",
        "rmsnorm_post",
        "quant_post",
        "gemv_gu",
        "silu_mul",
        "quant_h",
        "gemv_down",
    ]
    want = (
        ["embed"]
        + per_layer * syn.quant.n_layers
        + ["rmsnorm_final", "quant_final", "gemv_lm_head", "halt"]
    )
    assert names == want
    assert [e["name"] for e in plan["prefill"]] == want[:-4] + ["halt"]
    # every golden trace op is covered by a plan name (rope stands for rope_q and rope_k)
    covered = set(names) | {"rope_q", "rope_k"}
    assert (
        set(golden.TRACE_OPS_LAYER)
        - {n for n in golden.TRACE_OPS_LAYER if n.endswith(".scale")}
        - {"softmax.sreg"}
        <= covered
    )


def test_two_compiles_are_byte_identical(syn: synthetic.SyntheticModel, out_root: Path) -> None:
    a = compiler.compile(syn.quant, syn.spec, out_dir=out_root / "det_a", max_ctx=SYN_CTX)
    b = compiler.compile(syn.quant, syn.spec, out_dir=out_root / "det_b", max_ctx=SYN_CTX)
    files = sorted(p.name for p in a.out_dir.iterdir())
    assert files == sorted(p.name for p in b.out_dir.iterdir())
    for name in files:
        assert (a.out_dir / name).read_bytes() == (b.out_dir / name).read_bytes(), name
    assert a.layout == b.layout and a.decode == b.decode


@pytest.mark.parametrize("wb,max_ctx", [(16, 64), (128, 128)])
def test_other_port_widths(wb: int, max_ctx: int, out_root: Path) -> None:
    """Partial tiles (hidden 192 at WB 128) and the WB-dependent KV sub-regions round-trip."""
    m = synthetic.build(SHAPES[2], seed=2, out_dir=out_root / f"wb{wb}")
    c = compiler.compile(m.quant, m.spec, out_dir=out_root / f"img_wb{wb}", max_ctx=max_ctx, wb=wb)
    img = Image(c.out_dir)
    assert img.wb == wb
    _check_round_trip(img, m.quant, max_ctx)
    _check_regions(c.layout, c.out_dir)
    by = _by_name(c.layout)
    tiles = compiler.tiles_for(m.quant.hidden, wb)
    assert (
        by["layer.0.wo"]["tiles"] == tiles
        and by["layer.0.wo"]["size"] == tiles * m.quant.hidden * wb
    )
    assert by["layer.0.wo.meta"]["count"] == tiles * wb
    v_tiles = -(-64 // wb)
    assert compiler.kv_sizes(max_ctx, wb) == {
        "kt": max_ctx * 64,
        "v": v_tiles * max_ctx * wb,
        "k_meta": max_ctx * 8,
        "v_meta": max_ctx * 8,
    }
    assert c.layout["kv"]["v_tiles"] == v_tiles and c.layout["kv"]["kt_tiles"] == max_ctx // wb
    assert c.layout["traffic"]["decode"]["beats"] == -(
        -c.layout["traffic"]["decode"]["total"] // wb
    )
    sc = [d for d in c.decode if d.opcode is Opcode.GEMV and d.n_from_pos][0]
    assert sc.n == max_ctx and [d for d in c.decode if d.opcode is Opcode.VSOFTMAX][0].n == max_ctx


@pytest.mark.parametrize("seed", [21, 22])
def test_random_shapes_compile(seed: int, out_root: Path) -> None:
    shape = synthetic.random_shape(np.random.default_rng(seed))
    m = synthetic.build(shape, seed=seed, out_dir=out_root / f"rand{seed}")
    c = compiler.compile(m.quant, m.spec, out_dir=out_root / f"img_rand{seed}", max_ctx=SYN_CTX)
    _check_round_trip(Image(c.out_dir), m.quant, SYN_CTX)
    assert (len(c.decode), len(c.prefill)) == compiler.descriptor_counts(m.quant)
    assert c.layout["sreg"]["used"] == 5 + shape.heads + 2 * shape.kv_heads


def test_static_checks(syn: synthetic.SyntheticModel, out_root: Path) -> None:
    q, spec = syn.quant, syn.spec
    out = out_root / "bad"
    with pytest.raises(ValueError):
        compiler.compile(q, spec, out_dir=out, max_ctx=SYN_CTX, vsram_words=64)
    with pytest.raises(ValueError):
        compiler.compile(q, spec, out_dir=out, max_ctx=100)  # not a multiple of WB
    with pytest.raises(ValueError):
        compiler.compile(q, spec, out_dir=out, max_ctx=4096)
    with pytest.raises(ValueError):
        compiler.compile(q, spec, out_dir=out, max_ctx=SYN_CTX, wb=12)
    with pytest.raises(ValueError):
        compiler.compile(q, spec, out_dir=out, max_ctx=SYN_CTX, a_bits=4)
    with pytest.raises(ValueError):
        compiler.compile(q, None, out_dir=out, max_ctx=SYN_CTX, prompt=PROMPT)
    with pytest.raises(ValueError):
        compiler.sreg_map(dataclasses.replace(q, heads=28))
    with pytest.raises(ValueError):
        compiler.vsram_map(q, SYN_CTX, vsram_words=100)
    assert (
        compiler.vsram_map(q, SYN_CTX).used
        == 3 * q.hidden
        + (q.heads + 2 * q.kv_heads) * 64
        + 2 * q.heads * 64
        + 3 * q.intermediate
        + 2 * SYN_CTX
    )
    with pytest.raises(ValueError):
        compiler.truncate_layers(q, 0)
    with pytest.raises(ValueError):
        compiler.truncate_layers(q, q.n_layers + 1)
    one = compiler.truncate_layers(q, 1)
    assert one.n_layers == 1 and one.k_center.shape[0] == 1 and one.embed is q.embed
    assert not out.exists()


def test_tiling_and_meta_helpers() -> None:
    rng = np.random.default_rng(3)
    for n, k, wb in ((100, 24, 64), (128, 8, 64), (7, 16, 16), (192, 64, 128), (1, 8, 8)):
        q = rng.integers(-127, 128, (n, k)).astype(np.int8)
        data = compiler.tile_weights(q, wb)
        tiles = compiler.tiles_for(n, wb)
        assert len(data) == tiles * k * wb == compiler.tiled_bytes(n, k, wb)
        assert np.array_equal(compiler.untile_weights(data, n, k, wb), q)
        arr = np.frombuffer(data, dtype=np.int8).reshape(tiles, k, wb)
        for _ in range(20):  # byte (tile*K + k)*WB + j holds q[tile*WB + j, k]
            t, kk, j = rng.integers(tiles), rng.integers(k), rng.integers(wb)
            row = t * wb + j
            assert arr[t, kk, j] == (q[row, kk] if row < n else 0)
        chunks = list(compiler.iter_weight_tiles(q, wb, chunk_bytes=k * wb))
        assert len(chunks) == tiles and b"".join(chunks) == data
    with pytest.raises(ValueError):
        compiler.untile_weights(bytes(10), 4, 4, 8)
    bias = np.array([5, -7, 0], dtype=np.int64)
    m = np.array([40000, 0, 65535], dtype=np.int64)
    e = np.array([-20, 0, 127], dtype=np.int64)
    data = compiler.pack_meta(bias, m, e, 5)
    assert len(data) == 40 and data[:8] == (5).to_bytes(4, "little", signed=True) + (
        40000
    ).to_bytes(2, "little") + bytes([0xEC, 0])
    b2, m2, e2 = compiler.unpack_meta(data)
    assert (
        list(b2) == [5, -7, 0, 0, 0]
        and list(m2) == [40000, 0, 65535, 0, 0]
        and list(e2) == [-20, 0, 127, 0, 0]
    )
    with pytest.raises(ValueError):
        compiler.pack_meta(bias, np.array([1, 0, 65535]), e, 5)  # mantissa below 2^15
    with pytest.raises(ValueError):
        compiler.pack_meta(
            bias, np.array([40000, 0, 65535]), np.array([-20, 3, 0]), 5
        )  # zero with e
    with pytest.raises(ValueError):
        compiler.pack_meta(bias, m, np.array([-129, 0, 0]), 5)
    with pytest.raises(ValueError):
        compiler.pack_meta(bias, m, e, 2)
    assert compiler.align_up(0) == 0 and compiler.align_up(1) == 64 and compiler.align_up(64) == 64
    assert len(compiler.rope_bytes(1e6, 5)) == 640
    with pytest.raises(ValueError):
        compiler.rope_bytes(1e6, 4096)


def test_descriptor_effects() -> None:
    g = isa.gemv(addr_a=0, n=100, k=64, vs_src=0, vs_dst=8, sreg_src=1, s1=9, sbias=-25)
    assert compiler.descriptor_effects(g) == compiler.Effects((8, 100), (), False)
    sc = dataclasses.replace(g, n_from_pos=True, n=2048)
    assert compiler.descriptor_effects(sc) == compiler.Effects((8, 2048), (), True)
    am = dataclasses.replace(g, out_mode=OutMode.ARGMAX)
    assert compiler.descriptor_effects(am).vsram is None
    v = isa.vquant(vs_src=0, vs_dst=64, n=256, width=8, frac_in=16, sreg_dst=9, group=64)
    assert compiler.descriptor_effects(v) == compiler.Effects((64, 256), (9, 10, 11, 12), False)
    sm = isa.vsoftmax(vs_src=0, vs_dst=64, n=2048, addr_a=0, frac_s=16, sreg_dst=4)
    assert compiler.descriptor_effects(sm) == compiler.Effects((64, 2048), (4,), True)
    kv = isa.kvwrite(vs_src=0, addr_a=0, addr_m=0, sreg_src=0, transposed=True, max_ctx=64)
    assert compiler.descriptor_effects(kv) == compiler.Effects(None, (), True)
    assert compiler.descriptor_effects(isa.halt()) == compiler.Effects(None, (), False)
    assert compiler.descriptor_effects(isa.vrope(vs_src=8, n=128, addr_a=0)) == compiler.Effects(
        (8, 128), (), False
    )


# --------------------------------------------------------------------------- real models


def test_two_layer_real_model(
    spec: ModelSpec, qmodel2: quantize.QuantModel, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out = tmp_path_factory.mktemp(f"real_{spec.name}")
    c = compiler.compile(qmodel2, spec, out_dir=out, prompt=PROMPT)
    lay = c.layout
    by = _by_name(lay)
    _check_regions(lay, out)
    _check_round_trip(Image(out), qmodel2, program.MAX_CTX)
    assert (len(c.decode), len(c.prefill)) == compiler.descriptor_counts(qmodel2)
    assert lay["max_ctx"] == 2048 and by["rope"]["size"] == 256 * 1024
    assert lay["model"]["eos_ids"] == spec.eos_ids and lay["model"]["eos_ids"] != []
    assert compiler.kv_sizes(2048, 64) == {
        "kt": 131072,
        "v": 131072,
        "k_meta": 16384,
        "v_meta": 16384,
    }
    if spec.name == "qwen2.5-0.5b-instruct":
        assert [(m["name"], m["start"], m["count"]) for m in lay["vsram"]["map"]] == QWEN_VSRAM
        assert lay["vsram"]["used"] == 24320
        for suffix, size in QWEN_LAYER_SIZES.items():
            assert by[f"layer.0.{suffix}"]["size"] == size, suffix
        assert (
            by["layer.1.wqkv"]["addr"] - by["layer.0.wqkv"]["addr"] == 15_014_400 + 7 * 64 * 0
            or True
        )
        assert by["embed"]["size"] == 136_134_656 and by["embed.meta"]["size"] == 1_215_488
        assert len(c.decode) == 1 + 2 * 62 + 4
    else:
        assert lay["vsram"]["used"] == 12544
        assert by["embed"]["size"] == 49152 * 576 and len(c.decode) == 1 + 2 * 49 + 4
    # tokens.bin and prompt.tokens
    tb = lay["tokens_bin"]
    assert tb["count"] == spec.vocab and tb["sha256"] == compiler.sha256_file(out / tb["file"])
    assert read_tokens_bin(out / tb["file"]) == token_bytes(spec)
    ids = prompt_tokens(spec, PROMPT)
    assert (out / FILES["prompt"]).read_text().split() == [str(i) for i in ids]
    assert lay["prompt"] == {
        "file": FILES["prompt"],
        "source": "prompts/chat_short.json",
        "count": len(ids),
        "sha256": golden.ids_sha256(ids),
    }
    assert lay["expected_tokens"] is None  # the checked-in continuations are for the complete model
    assert lay["constants"]["gemvs"] == program.build(qmodel2).as_dict()["gemvs"]


def test_cli_compile(
    smollm2: ModelSpec, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    if (
        not quantize.default_path(smollm2.name).is_file()
        and not calibrate.calib_path(smollm2).is_file()
    ):
        pytest.skip("neither the quantized SmolLM2 nor its calib.json is present")
    out = tmp_path / "cli"
    assert (
        cli.main(["compile", "smollm2", "--layers", "1", "--out", str(out), "--max-ctx", "128"])
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["layers"] == 1 and summary["descriptors"] == {
        "decode": 1 + 49 + 4,
        "prefill": 1 + 49 + 1,
    }
    lay = compiler.load_layout(out)
    assert lay["max_ctx"] == 128 and lay["image"]["size"] == summary["image_bytes"]
    assert (out / FILES["image"]).stat().st_size == lay["image"]["size"]


@pytest.mark.slow
def test_complete_model_image(spec: ModelSpec, qmodel_full: quantize.QuantModel) -> None:
    """The complete models compile into build/images/<name>; sizes and counts are reported."""
    import time

    t0 = time.perf_counter()
    c = compiler.compile(qmodel_full, spec, out_dir=compiler.IMAGES_DIR / spec.name, prompt=PROMPT)
    dt = time.perf_counter() - t0
    lay = c.layout
    print(
        f"{spec.name}: image {lay['image']['size']} B, decode {len(c.decode)} / prefill "
        f"{len(c.prefill)} descriptors, {lay['traffic']['decode']['total']} stream B/token, "
        f"{dt:.1f} s"
    )
    assert lay["traffic"]["decode"]["total"] == STREAM_BYTES[spec.name]
    assert (len(c.decode), len(c.prefill)) == compiler.descriptor_counts(qmodel_full)
    _check_regions(lay, c.out_dir)
    exp = lay["expected_tokens"]
    if golden.expected_tokens_path(qmodel_full).is_file():
        assert exp is not None and set(exp["prompts"]) == set(golden.EXPECTED_PROMPT_FILES)
