"""ISA simulator: bit-exact against the golden model on compiled synthetic models (kv_heads 1 to 3,
with and without QKV biases, W8A16 and W8A8, WB 16 / 64 / 128), truncated real models and
(``slow``) the complete models over prefill plus decode; the prefill/decode loop, determinism,
step records against ``dump_plan.json``, the CSR, status, shift and bounds rules and the
``isa-sim`` command."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
from quettos import calibrate, cli, golden, isa, isa_sim, quantize, synthetic
from quettos.isa import Descriptor, Opcode, OutMode
from quettos.model import REPO_ROOT, ModelSpec
from quettos.numerics import SFLOAT_ONE, SFloat, Stats
from quettos.synthetic import SHAPES
from quettos.tokenizer_io import prompt_tokens

compiler = pytest.importorskip("quettos.compiler", reason="quettos.compiler is not present")

T = 10  # tokens per teacher-forced run
SYN_CTX = 64
TINY_WB, TINY_B_MAX, TINY_VSRAM_WORDS = 16, 2, 2048  # the tiny configuration (rtl/cfg/README.md)
# (SHAPES index, a_bits, wb): kv_heads 1 / 2 / 3, with and without QKV biases, the port widths
CASES: tuple[tuple[int, int, int], ...] = (
    (0, 16, 64),
    (1, 16, 64),
    (2, 16, 64),
    (4, 8, 64),
    (2, 16, TINY_WB),
)
PROMPT = REPO_ROOT / "prompts" / "chat_short.json"
DECODE_TOKENS = {"smollm2-135m-instruct": 4, "qwen2.5-0.5b-instruct": 2}  # complete-model runs


class Built:
    """A compiled synthetic model with its programs, image bytes and test ids."""

    def __init__(self, index: int, a_bits: int, wb: int, root: Path) -> None:
        self.shape = SHAPES[index]
        self.syn = synthetic.build(self.shape, seed=index, out_dir=root / f"syn{index}")
        self.max_ctx = max(SYN_CTX, wb)
        self.vsram_words = TINY_VSRAM_WORDS if wb == TINY_WB else isa_sim.VSRAM_WORDS
        self.compiled = compiler.compile(
            self.syn.quant,
            self.syn.spec,
            out_dir=root / f"img{index}-a{a_bits}-w{wb}",
            max_ctx=self.max_ctx,
            wb=wb,
            a_bits=a_bits,
            vsram_words=self.vsram_words,
        )
        self.programs = isa_sim.Programs.from_compiled(self.compiled)
        self.image = self.compiled.image_path.read_bytes()
        self.a_bits, self.wb = a_bits, wb
        self.ids = np.random.default_rng(index + 100).integers(0, self.shape.vocab, T).tolist()

    @property
    def model(self) -> quantize.QuantModel:
        return self.syn.quant

    def machine(self, b_max: int = 1) -> isa_sim.Machine:
        return isa_sim.Machine(self.image, wb=self.wb, b_max=b_max, vsram_words=self.vsram_words)

    def compare(self, ids: list[int]) -> isa_sim.Comparison:
        return isa_sim.compare_sequence(
            self.model,
            self.image,
            self.programs,
            ids,
            a_bits=self.a_bits,
            max_ctx=self.max_ctx,
            wb=self.wb,
            vsram_words=self.vsram_words,
        )


@pytest.fixture(scope="module")
def out_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("isa_sim")


_BUILT: dict[tuple[int, int, int], Built] = {}


@pytest.fixture(scope="module")
def built(out_root: Path):
    def get(index: int, a_bits: int = 16, wb: int = 64) -> Built:
        key = (index, a_bits, wb)
        if key not in _BUILT:
            _BUILT[key] = Built(index, a_bits, wb, out_root)
        return _BUILT[key]

    return get


def _run_all(b: Built, ids: list[int]) -> tuple[isa_sim.Machine, list[int | None], Stats]:
    m = b.machine()
    outs: list[int | None] = []
    total = Stats()
    for pos, tok in enumerate(ids):
        outs.append(isa_sim.run_token(m, b.programs.decode, tok, pos))
        total = total + m.stats()
    return m, outs, total


# --------------------------------------------------------------------------- golden equivalence


@pytest.mark.parametrize(("index", "a_bits", "wb"), CASES)
def test_every_position_matches_golden(built, index: int, a_bits: int, wb: int) -> None:
    """decode.prog at every position: every descriptor, logits, argmax, KV bytes and counters."""
    b = built(index, a_bits, wb)
    cmp = b.compare(b.ids)
    assert cmp.ok, str(cmp.mismatch)
    assert cmp.positions == T
    st = Stats()
    ref = golden.forward_tokens(b.model, b.ids, a_bits=a_bits, stats=st, max_ctx=b.max_ctx)
    assert cmp.argmax == ref.argmax.tolist()
    assert cmp.stats == st and st.sat == 0 and st.err_shift == 0
    m, outs, total = _run_all(b, b.ids)
    assert outs == ref.argmax.tolist() and total == st
    assert np.array_equal(m.argmax_out, ref.logits[-1])
    assert m.argmax_val() == int(ref.logits[-1][ref.argmax[-1]])
    assert m.csr["ERR_BOUNDS"] == 0 and m.csr["ERR_SHIFT"] == 0 and m.csr["SAT_REQ"] == 0


@pytest.mark.parametrize("index", [0, 2])
def test_generate_loop_matches_golden(built, index: int) -> None:
    b = built(index)
    prompt = b.ids[:5]
    seen: list[tuple[int, int, int]] = []
    gen = isa_sim.generate(
        b.machine(),
        b.programs.decode,
        b.programs.prefill,
        prompt,
        6,
        on_token=lambda *a: seen.append(a),
    )
    want = golden.generate(b.model, prompt, 6, max_ctx=b.max_ctx)
    assert gen == want and len(gen) == 6
    assert seen == [(j, len(prompt) - 1 + j, t) for j, t in enumerate(gen)]
    cmp = isa_sim.compare_generate(b.model, b.image, b.programs, prompt, 6, max_ctx=b.max_ctx)
    assert cmp.ok, str(cmp.mismatch)
    assert cmp.gen == want and cmp.positions == len(prompt) - 1 + 6
    assert cmp.argmax[: len(prompt) - 1] == [None] * (len(prompt) - 1)
    # an end-of-sequence id stops both loops after it is produced
    eos = [gen[2]]
    got = isa_sim.generate(
        b.machine(), b.programs.decode, b.programs.prefill, prompt, 6, eos_ids=eos
    )
    assert got == golden.generate(b.model, prompt, 6, eos_ids=eos, max_ctx=b.max_ctx)
    assert got[-1] == eos[0] and len(got) <= 3


@pytest.mark.parametrize("wb", [128, TINY_WB])
def test_other_port_widths_produce_identical_tokens(built, wb: int) -> None:
    """The same model tiled for a 128- or 16-byte port (partial tiles, four V tiles) agrees."""
    a, b = built(2, 16, 64), built(2, 16, wb)
    assert a.compiled.layout["image"]["sha256"] != b.compiled.layout["image"]["sha256"]
    assert b.compiled.layout["kv"]["v_tiles"] == -(-64 // wb)
    _, outs_a, st_a = _run_all(a, a.ids)
    _, outs_b, st_b = _run_all(b, a.ids)
    assert outs_a == outs_b and st_a == st_b
    cmp = b.compare(a.ids)
    assert cmp.ok, str(cmp.mismatch)
    if wb == TINY_WB:  # the tiny configuration: two rows, row 0 driven
        m2 = b.machine(b_max=TINY_B_MAX)
        outs2 = [
            isa_sim.run_token(m2, b.programs.decode, tok, pos) for pos, tok in enumerate(a.ids)
        ]
        assert outs2 == outs_a and not np.any(m2.vsram[1])


def test_runs_are_deterministic(built) -> None:
    b = built(1)
    m1, outs1, _ = _run_all(b, b.ids)
    m2, outs2, _ = _run_all(b, b.ids)
    assert outs1 == outs2
    assert m1.mem == m2.mem and np.array_equal(m1.vsram, m2.vsram)
    assert m1.sreg == m2.sreg and m1.csr_words() == m2.csr_words()
    assert np.array_equal(m1.perf, m2.perf)


def test_truncated_real_model_matches_golden(spec: ModelSpec, out_root: Path) -> None:
    """Two layers of a downloaded model compiled at MAX_CTX 64: every descriptor matches."""
    path = calibrate.calib_path(spec)
    if not path.is_file():
        pytest.skip(f"{path} not present")
    model = quantize.build_quant_model(spec, calibrate.load_calib(path), layers=2)
    c = compiler.compile(model, spec, out_dir=out_root / f"real-{spec.name}", max_ctx=SYN_CTX)
    programs = isa_sim.Programs.from_dir(c.out_dir)
    ids = prompt_tokens(spec, PROMPT)[:8]
    cmp = isa_sim.compare_sequence(model, c.image_path, programs, ids, max_ctx=SYN_CTX)
    assert cmp.ok, str(cmp.mismatch)
    ref = golden.forward_tokens(model, ids, max_ctx=SYN_CTX)
    assert cmp.argmax == ref.argmax.tolist()
    # the prefill/decode loop over the same prompt: prefill positions, then two decode steps
    cmp2 = isa_sim.compare_generate(
        model, c.image_path, programs, ids, 2, eos_ids=spec.eos_ids, max_ctx=SYN_CTX
    )
    assert cmp2.ok, str(cmp2.mismatch)
    assert cmp2.gen == golden.generate(model, ids, 2, eos_ids=spec.eos_ids, max_ctx=SYN_CTX)


@pytest.mark.slow
def test_complete_model_prefill_and_decode_matches_golden(spec: ModelSpec) -> None:
    """The complete model on prompts/chat_short.json: prefill, then decode tokens, every descriptor,
    every KV byte and the per-token counters equal to the golden model; the ids equal the stored
    continuation."""
    path = quantize.default_path(spec.name)
    if not path.is_file():
        pytest.skip(f"{path} not present (run: uv run quettos quantize)")
    model = quantize.load(path)
    if model.n_layers != spec.layers:
        pytest.skip(f"{path} is truncated to {model.n_layers} layers")
    t0 = time.perf_counter()
    c = compiler.compile(model, spec, out_dir=compiler.IMAGES_DIR / spec.name, prompt=PROMPT)
    t_compile = time.perf_counter() - t0
    programs = isa_sim.Programs.from_compiled(c)
    ids = prompt_tokens(spec, PROMPT)
    max_new = DECODE_TOKENS[spec.name]
    t1 = time.perf_counter()
    cmp = isa_sim.compare_generate(
        model, c.image_path, programs, ids, max_new, eos_ids=spec.eos_ids
    )
    t_compare = time.perf_counter() - t1
    print(
        f"{spec.name}: image {c.layout['image']['size']} B compiled in {t_compile:.1f} s; "
        f"{len(ids)} prompt + {max_new} decode positions compared in {t_compare:.1f} s; "
        f"stats {cmp.stats}"
    )
    assert cmp.ok, str(cmp.mismatch)
    assert cmp.positions == len(ids) - 1 + max_new and len(cmp.gen) == max_new
    assert cmp.stats.sat == 0 and cmp.stats.err_shift == 0
    stored = json.loads(golden.expected_tokens_path(model).read_text(encoding="utf-8"))
    want = stored["prompts"][golden.prompt_key(PROMPT)]["generated_ids"]
    assert cmp.gen == want[:max_new]


def test_mismatch_names_the_first_differing_descriptor(built) -> None:
    b = built(0)
    regions = compiler.Image(b.compiled.out_dir).regions
    # a flipped bit in the gamma of layer 0 shows up at the first VRMSNORM, nowhere earlier
    image = bytearray(b.image)
    image[regions["layer.0.gamma_in"]["addr"]] ^= 0x01
    cmp = isa_sim.compare_sequence(b.model, image, b.programs, b.ids[:2], max_ctx=b.max_ctx)
    assert not cmp.ok
    mm = cmp.mismatch
    assert (mm.pos, mm.index, mm.opcode, mm.name, mm.what) == (
        0,
        1,
        "VRMSNORM",
        "rmsnorm_in",
        "vsram",
    )
    assert mm.element is not None and mm.expected != mm.got
    assert "VRMSNORM" in str(mm) and "rmsnorm_in" in str(mm)
    # a flipped bit in the K^T cache of layer 0 shows up at that head's scores GEMV
    image = bytearray(b.image)
    kt = regions["kv.0.0.kt"]
    m = b.machine()
    isa_sim.run_token(m, b.programs.decode, b.ids[0], 0)
    image[kt["addr"]] = m.mem[kt["addr"]] ^ 0x7F  # token 0, d = 0
    cmp = isa_sim.compare_sequence(b.model, image, b.programs, b.ids[:1], max_ctx=b.max_ctx)
    assert cmp.ok  # KVWRITE at position 0 overwrites the byte before the scores GEMV reads it
    cmp = isa_sim.compare_sequence(b.model, image, b.programs, b.ids[:1], max_ctx=b.max_ctx, wb=64)
    assert cmp.ok


# --------------------------------------------------------------------------- step records and plan


@pytest.mark.parametrize("wb", [64, TINY_WB])
def test_step_records_follow_the_dump_plan(built, wb: int) -> None:
    b = built(2, 16, wb)
    P = b.programs
    m = b.machine()
    for pos in range(5):
        isa_sim.run_token(m, P.decode, b.ids[pos], pos)
    recs = isa_sim.record_program(m, P.decode, P.decode_plan, b.ids[5], 5)
    assert len(recs) == len(P.decode) == len(P.decode_plan)
    assert isa_sim.check_plan(recs) == []
    assert m.log is None
    for r in recs:
        assert r.writes == isa_sim.written_ranges(r.descriptor, 5, wb=b.wb), r.index
        e = r.entry
        assert e["index"] == r.index and e["op"] == r.descriptor.opcode.name
        if e["vsram"] is not None:
            row = (
                r.descriptor.src_row
                if r.descriptor.opcode == Opcode.VROPE
                else r.descriptor.dst_row
            )
            assert r.vsram.dtype == np.int32 and r.vsram.shape == (e["vsram"]["count"],)
            got = m.vsram[row, e["vsram"]["start"] : e["vsram"]["start"] + e["vsram"]["count"]]
            assert np.array_equal(r.vsram, got) or r.index < len(recs) - 1  # later ops rewrite
        assert set(r.sreg) == set(e["sreg"]) and set(r.csr) == set(e["csr"])
        assert set(r.mem) == {x["name"] for x in e["mem"]}
    by_name: dict[str, isa_sim.StepRecord] = {}
    for r in recs:
        by_name.setdefault(r.entry["name"], r)  # the first descriptor of each name
    soft = by_name["softmax"]
    assert np.all(soft.vsram[6:] == 0) and soft.vsram.shape == (b.max_ctx,)
    assert isinstance(soft.sreg[soft.descriptor.sreg_dst], SFloat)
    head = by_name["gemv_lm_head"]
    assert head.vsram is None and set(head.csr) == {"ARGMAX_TOK", "ARGMAX_VAL"}
    assert head.csr["ARGMAX_TOK"] == m.csr["ARGMAX_TOK"]
    kv = by_name["kvwrite_k"]
    assert set(kv.mem) == {"kv.0.0.kt", "kv.0.0.k_meta"}
    assert len(kv.mem["kv.0.0.kt"]) == kv.entry["mem"][0]["size"]
    vw = by_name["kvwrite_v"]
    assert set(vw.mem) == {"kv.0.0.v", "kv.0.0.v_meta"}
    assert len([k for k in vw.writes if k[0] == "mem" and k[3] == b.wb]) == -(-64 // b.wb)
    js = json.loads(json.dumps(recs[1].as_json()))
    assert js["op"] == "VRMSNORM" and js["name"] == "rmsnorm_in"
    assert js["vsram"]["values"] == recs[1].vsram.tolist() and js["sreg"] == {
        "0": {"absmax": recs[1].sreg[0]}
    }
    # the prefill plan describes the prefill program the same way
    m2 = b.machine()
    recs2 = isa_sim.record_program(m2, P.prefill, P.prefill_plan, b.ids[0], 0)
    assert len(recs2) == len(P.prefill) and isa_sim.check_plan(recs2) == []
    assert recs2[-1].descriptor.opcode == Opcode.HALT and recs2[-1].writes == set()


def test_plan_check_reports_a_wrong_entry(built) -> None:
    b = built(0)
    m = b.machine()
    plan = [dict(e) for e in b.programs.decode_plan]
    plan[1] = {**plan[1], "sreg": [7]}
    plan[2] = {**plan[2], "vsram": {"start": plan[2]["vsram"]["start"] + 8, "count": 8}}
    recs = isa_sim.record_program(m, b.programs.decode, plan, b.ids[0], 0)
    problems = isa_sim.check_plan(recs)
    assert len(problems) == 2 and "descriptor 1" in problems[0] and "descriptor 2" in problems[1]
    bad = [{**plan[0], "op": "GEMV"}] + plan[1:]
    with pytest.raises(ValueError):
        isa_sim.record_program(b.machine(), b.programs.decode, bad, b.ids[0], 0)


# --------------------------------------------------------------------------- programs, CSRs, status


def test_program_bytes_and_image_resident_program_agree(built) -> None:
    b = built(1)
    lay = b.compiled.layout
    prog_bytes = (b.compiled.out_dir / "decode.prog").read_bytes()
    m_list, m_bytes, m_pc = b.machine(), b.machine(), b.machine()
    out_list = isa_sim.run_token(m_list, b.programs.decode, b.ids[0], 0)
    out_bytes = isa_sim.run_token(m_bytes, prog_bytes, b.ids[0], 0)
    out_pc = isa_sim.run_token(m_pc, None, b.ids[0], 0, pc=lay["programs"]["decode"]["addr"])
    assert out_list == out_bytes == out_pc
    assert np.array_equal(m_list.vsram, m_bytes.vsram) and np.array_equal(m_list.vsram, m_pc.vsram)
    assert m_pc.csr["PC"] == lay["programs"]["decode"]["addr"] + lay["programs"]["decode"]["size"]
    assert m_list.status("DONE") and not m_list.status("ERR") and not m_list.status("BUSY")
    assert m_list.perf_value("DESCRIPTORS") == len(b.programs.decode)
    lo, hi = isa.perf_words(isa.PERF_INDEX["DESCRIPTORS"])
    words = m_list.csr_words()
    assert len(words) == isa.CSR_WORDS and words[lo] == len(b.programs.decode) and words[hi] == 0
    assert words[isa.CSR_BY_NAME["ISA_VERSION"].word] == isa.ISA_VERSION
    # the prefill program sits at its image address too and produces no argmax
    m_pf = b.machine()
    assert isa_sim.run_token(m_pf, None, b.ids[0], 0, pc=lay["programs"]["prefill"]["addr"]) is None
    assert m_pf.argmax_out is None


def test_perf_counters_follow_the_descriptors(built) -> None:
    b = built(4, 8, 64)
    m, _, _ = _run_all(b, b.ids[:3])
    pos = 2
    macs = wt = 0
    for d in b.programs.decode:
        if d.opcode == Opcode.GEMV:
            n, k, _ = isa_sim.gemv_dims(d, pos, b.wb)
            tiles = -(-n // b.wb)
            macs += tiles * b.wb * k
            if not (d.n_from_pos or d.k_from_pos):
                wt += tiles * k * b.wb + (0 if d.unit_meta else n * isa.META_BYTES)
        elif d.opcode == Opcode.EMBED:
            wt += d.k + isa.META_BYTES
    assert m.perf_value("MACS") == macs and m.perf_value("WT_BYTES") == wt
    traffic = b.compiled.layout["traffic"]
    assert wt == traffic["decode"]["wt_bytes"]
    assert wt == traffic["decode"]["weights"] + traffic["decode"]["meta"] + b.model.hidden + 8
    att = traffic["attention"]
    tiles = -(-(pos + 1) // b.wb)
    assert macs == traffic["decode"]["macs"] + att["head_layers"] * (
        tiles * att["scores_macs_per_tile"] + (pos + 1) * att["pv_macs_per_token"]
    )
    assert m.perf_value("DESCRIPTORS") == len(b.programs.decode)
    for name in isa.PERF_INDEX:
        if name not in isa_sim.PERF_COUNTED:
            assert m.perf_value(name) == 0


def test_unknown_opcode_stops_with_err(built) -> None:
    b = built(0)
    prog = isa.assemble(b.programs.decode[:3]) + bytes([0x77]) + bytes(31)
    m = b.machine()
    assert isa_sim.run_program(m, prog) == 3
    assert m.status("ERR") and m.status("DONE") and m.perf_value("DESCRIPTORS") == 3
    m.start()
    assert not m.status("ERR") and not m.status("DONE") and m.perf_value("DESCRIPTORS") == 0
    with pytest.raises(ValueError):
        isa_sim.run_program(b.machine(), b.programs.decode[:5])  # no HALT
    with pytest.raises(ValueError):
        isa_sim.run_program(b.machine(), isa.assemble(b.programs.decode[:5]))


def test_softmax_class_outside_the_window_stops_with_err(built) -> None:
    """A VSOFTMAX class outside ``[16, 30]`` faults at decode, the way an unknown opcode does."""
    b = built(0)
    base = next(d for d in b.programs.decode if d.opcode == Opcode.VSOFTMAX)
    lo, hi = isa.CLASS_WINDOW[Opcode.VSOFTMAX]
    for frac_s in (lo - 1, hi + 1):
        m = b.machine()
        pc0 = m.csr["PC"]
        prog = [
            Descriptor(opcode=Opcode.NOP),
            dataclasses.replace(base, sh0=frac_s),
            Descriptor(opcode=Opcode.HALT),
        ]
        assert isa_sim.run_program(m, prog) == 1, f"frac_s {frac_s} executed past the fault"
        assert m.fault_state() == (isa.Fault.CLASS, int(Opcode.VSOFTMAX))
        assert m.status("ERR") and m.status("DONE")
        assert m.perf_value("DESCRIPTORS") == 1 and not np.any(m.vsram)
        assert m.csr["PC"] == pc0 + isa.DESC_BYTES, "PC left the descriptor that faulted"
        # a stepped descriptor faults the same way and executes nothing
        m2 = b.machine()
        m2.start()
        isa_sim.step(m2, dataclasses.replace(base, sh0=frac_s))
        assert m2.fault_state() == (isa.Fault.CLASS, int(Opcode.VSOFTMAX))
        assert m2.status("ERR") and not m2.status("STEP_HALTED")
        assert m2.perf_value("DESCRIPTORS") == 0 and not np.any(m2.vsram)
        # the fault describes the descriptor and not its work
        for bad in (
            dataclasses.replace(base, sh0=frac_s, row_mask=0),
            dataclasses.replace(base, sh0=frac_s, n=0),
        ):
            m3 = b.machine()
            assert isa_sim.run_program(m3, [bad, Descriptor(opcode=Opcode.HALT)]) == 0
            assert m3.fault_state() == (isa.Fault.CLASS, int(Opcode.VSOFTMAX))
    # both edges of the window and one class between them run the whole token
    for frac_s in (lo, (lo + hi) // 2, hi):
        m = b.machine()
        prog = [
            dataclasses.replace(d, sh0=frac_s) if d.opcode == Opcode.VSOFTMAX else d
            for d in b.programs.decode
        ]
        assert isa_sim.run_token(m, prog, b.ids[0], 0) is not None
        assert m.fault_state() == (isa.Fault.NONE, 0) and not m.status("ERR")
        assert m.status("DONE") and np.any(m.vsram)


def test_step_mode_status_bits(built) -> None:
    b = built(0)
    m = b.machine()
    m.start()
    m.csr["TOK"], m.csr["POS"] = b.ids[0], 0
    for d in b.programs.decode[:-1]:
        isa_sim.step(m, d)
        assert m.status("STEP_HALTED") and not m.status("DONE")
    isa_sim.step(m, b.programs.decode[-1])
    assert m.status("DONE") and not m.status("STEP_HALTED")
    ref = b.machine()
    isa_sim.run_token(ref, b.programs.decode, b.ids[0], 0)
    assert np.array_equal(m.vsram, ref.vsram) and m.csr["ARGMAX_TOK"] == ref.csr["ARGMAX_TOK"]


def test_row_enable_and_row_mask(built) -> None:
    b = built(0)
    m = b.machine()
    assert isa_sim.run_token(m, b.programs.decode, b.ids[0], 0, row_en=0) is None
    assert not np.any(m.vsram) and m.status("DONE")
    # two rows: a VSUBC with row_mask 0b11 runs on both, with row_mask 1 on row 0 only
    m2 = isa_sim.Machine(b.image, b_max=2)
    const = compiler.Image(b.compiled.out_dir).regions["kcenter.0"]
    m2.vsram[0, :64] = 1000
    m2.vsram[1, :64] = 2000
    c = m2.read_i32(const["addr"], 64)
    both = isa.vsubc(vs_src=0, vs_dst=64, n=64, addr_a=const["addr"], row_mask=0b11)
    m2.csr["ROW_EN"] = 0b11
    isa_sim.execute(m2, both)
    assert np.array_equal(m2.vsram[0, 64:128], 1000 - c)
    assert np.array_equal(m2.vsram[1, 64:128], 2000 - c)
    m2.vsram[:, 64:128] = 0
    isa_sim.execute(m2, isa.vsubc(vs_src=0, vs_dst=64, n=64, addr_a=const["addr"], row_mask=1))
    assert np.array_equal(m2.vsram[0, 64:128], 1000 - c) and not np.any(m2.vsram[1, 64:128])
    with pytest.raises(ValueError):
        isa_sim.execute(m2, Descriptor(opcode=Opcode.VSUBC, row_mask=0b11, n=64, dst_row=1))


def test_bounds_counter_and_pos_derived_fields(built) -> None:
    b = built(0)
    d = next(x for x in b.programs.decode if x.opcode == Opcode.GEMV and x.n_from_pos)
    assert isa_sim.gemv_dims(d, 0, 64) == (64, 64, 0)
    assert isa_sim.gemv_dims(d, 63, 64) == (64, 64, 0)
    assert isa_sim.gemv_dims(d, 64, 64) == (64, 64, 1)
    pv = next(x for x in b.programs.decode if x.opcode == Opcode.GEMV and x.k_from_pos)
    assert isa_sim.gemv_dims(pv, 5, 64) == (64, 6, 0)
    assert isa_sim.gemv_dims(pv, 70, 64) == (64, 64, 1)
    sm = next(x for x in b.programs.decode if x.opcode == Opcode.VSOFTMAX)
    assert isa_sim.softmax_len(sm, 3) == (4, 0) and isa_sim.softmax_len(sm, 64) == (64, 1)
    # a token at the KV capacity: every KVWRITE is dropped, the scores GEMV, the softmax and
    # the PV GEMV clamp their derived extents; each event counts once
    m = b.machine()
    for pos in range(3):
        isa_sim.run_token(m, b.programs.decode, b.ids[pos], pos)
    before = bytes(m.mem)
    isa_sim.run_token(m, b.programs.decode, b.ids[3], b.max_ctx)
    layers, heads, kv_heads = b.shape.layers, b.shape.heads, b.shape.kv_heads
    assert m.csr["ERR_BOUNDS"] == layers * (3 * heads + 2 * kv_heads)
    assert bytes(m.mem) == before
    # VSRAM ranges past the end are counted and clipped
    m3 = b.machine()
    m3.vs_write(0, isa_sim.VSRAM_ELEMS - 4, np.arange(8, dtype=np.int64))
    assert m3.err_bounds == 1 and m3.vsram[0, -4:].tolist() == [0, 1, 2, 3]
    assert m3.vs_read(0, isa_sim.VSRAM_ELEMS - 2, 4).tolist() == [2, 3, 0, 0] and m3.err_bounds == 2
    # a softmax length of 0 is clamped to 1 and counted; a KVWRITE at the capacity writes nothing
    fixed = Descriptor(**{**sm.__dict__, "len_from_pos": False, "imm32": 0})
    assert isa_sim.softmax_len(fixed, 0) == (1, 1)
    kv = next(x for x in b.programs.decode if x.opcode == Opcode.KVWRITE)
    assert kv.k == b.max_ctx
    assert isa_sim.written_ranges(kv, b.max_ctx) == set()
    assert len(isa_sim.written_ranges(kv, 0)) == 64 + 1


def test_gemv_with_no_inputs_is_zero_work(built) -> None:
    """K == 0 with N > 0 retires with the bounds events counted and nothing else touched."""
    b = built(0)
    m = b.machine()
    isa_sim.run_token(m, b.programs.decode, b.ids[0], 0)
    gemv = next(
        x
        for x in b.programs.decode
        if x.opcode == Opcode.GEMV
        and x.out_mode == OutMode.VSRAM
        and not x.n_from_pos
        and not x.k_from_pos
    )
    zero = Descriptor(**{**gemv.__dict__, "k": 0})
    before = m.vsram.copy()
    counters = {name: m.perf_value(name) for name in isa_sim.PERF_COUNTED}
    bounds = m.err_bounds
    m.log = []
    isa_sim.step(m, zero)
    assert m.log == [], "a zero-work descriptor writes nothing"
    assert np.array_equal(m.vsram, before)
    assert m.err_bounds == bounds
    assert m.perf_value("MACS") == counters["MACS"]
    assert m.perf_value("WT_BYTES") == counters["WT_BYTES"]
    assert m.perf_value("DESCRIPTORS") == counters["DESCRIPTORS"] + 1
    assert isa_sim.written_ranges(zero, 0) == set()
    assert isa_sim.written_ranges(gemv, 0) == {("vsram", 0, gemv.vs_dst, gemv.n)}


def test_sreg_holds_scales_or_tracked_absmax(built) -> None:
    b = built(0)
    m = b.machine()
    m.sreg_set(0, 3, 12345)
    assert m.sreg_absmax(0, 3) == 12345
    with pytest.raises(ValueError):
        m.sreg_scale(0, 3)
    m.sreg_set(0, 4, SFLOAT_ONE)
    assert m.sreg_scale(0, 4) == SFLOAT_ONE
    with pytest.raises(ValueError):
        m.sreg_absmax(0, 4)
    # an index past the 32 registers drops the write, reads zero and counts in ERR_BOUNDS
    m.sreg_set(0, 32, 1)
    assert m.err_bounds == 1 and m.sreg_scale(0, 40) == SFloat(0, 0) and m.sreg_absmax(0, 33) == 0
    assert m.err_bounds == 3 and len(m.sreg[0]) == isa.SREG_COUNT
    assert isa_sim.sreg_json(SFLOAT_ONE) == {"m": 1 << 15, "e": -15}
    assert isa_sim.sreg_json(7) == {"absmax": 7}
    # a GEMV reading a tracked absmax as its scale is a program error
    gemv = next(x for x in b.programs.decode if x.opcode == Opcode.GEMV)
    m.sreg_set(0, gemv.sreg_src, 5)
    with pytest.raises(ValueError):
        isa_sim.execute(m, gemv)


def test_dump_out_modes_write_memory(built) -> None:
    b = built(0)
    m = b.machine()
    isa_sim.run_token(m, b.programs.decode, b.ids[0], 0)
    lm = b.programs.decode[-2]
    assert lm.opcode == Opcode.GEMV and lm.out_mode == OutMode.ARGMAX
    scratch = len(m.mem)
    m.mem.extend(bytes(4 * lm.n))
    dump = Descriptor(**{**lm.__dict__, "out_mode": OutMode.ARGMAX_DUMP, "imm32": scratch})
    m.log = []
    isa_sim.execute(m, dump)
    logits = np.frombuffer(bytes(m.mem[scratch : scratch + 4 * lm.n]), dtype="<i4")
    assert np.array_equal(logits, m.argmax_out)
    assert ("mem", 0, scratch, 4 * lm.n) in {w.key() for w in m.log}
    assert isa_sim.written_ranges(dump, 0) == {
        ("csr", "ARGMAX_TOK"),
        ("csr", "ARGMAX_VAL"),
        ("mem", 0, scratch, 4 * lm.n),
    }
    with pytest.raises(ValueError):  # a dump past the image is rejected
        isa_sim.execute(m, Descriptor(**{**dump.__dict__, "imm32": len(m.mem) - 4}))
    with pytest.raises(ValueError):  # and so is one that is not a whole beat
        isa_sim.execute(m, Descriptor(**{**dump.__dict__, "imm32": scratch + 4}))
    # two rows: row r dumps at addr_c + r * 4 * N and the CSRs hold the highest row's argmax
    m2 = isa_sim.Machine(b.image, b_max=2)
    m2.mem.extend(bytes(8 * lm.n))
    m2.csr["ROW_EN"] = 0b11
    isa_sim.run_token(m2, b.programs.decode, b.ids[0], 0, row_en=0b11)
    m2.vsram[1] = 0  # row 1 sees zero activations, so its logits are the zero-input outputs
    both = Descriptor(**{**dump.__dict__, "row_mask": 0b11})
    isa_sim.execute(m2, both)
    row0 = np.frombuffer(bytes(m2.mem[scratch : scratch + 4 * lm.n]), dtype="<i4")
    row1 = np.frombuffer(bytes(m2.mem[scratch + 4 * lm.n : scratch + 8 * lm.n]), dtype="<i4")
    assert np.array_equal(row0, logits) and np.array_equal(row1, m2.argmax_out)
    assert m2.csr["ARGMAX_TOK"] == int(np.argmax(row1))
    assert isa_sim.written_ranges(both, 0, b_max=2, row_en=0b11) >= {
        ("mem", 0, scratch, 4 * lm.n),
        ("mem", 0, scratch + 4 * lm.n, 4 * lm.n),
    }


def test_shift_fields_outside_the_window_are_clamped_and_counted(built) -> None:
    """sh0 of a GEMV, sh1 of VRMSNORM and VSILUMUL: values outside [0, 63] clamp and count."""
    b = built(0)
    ref = b.machine()
    isa_sim.run_token(ref, b.programs.decode, b.ids[0], 0)
    norm = next(x for x in b.programs.decode if x.opcode == Opcode.VRMSNORM)
    m = b.machine()
    m.csr["TOK"], m.csr["POS"] = b.ids[0], 0
    isa_sim.execute(m, b.programs.decode[0])  # EMBED
    isa_sim.execute(m, Descriptor(**{**norm.__dict__, "sh1": 100}))
    assert m.stats_vpu.err_shift == norm.n
    m63 = b.machine()
    m63.csr["TOK"], m63.csr["POS"] = b.ids[0], 0
    isa_sim.execute(m63, b.programs.decode[0])
    isa_sim.execute(m63, Descriptor(**{**norm.__dict__, "sh1": 63}))
    assert m63.stats_vpu.err_shift == 0
    assert np.array_equal(
        m.vsram[0, norm.vs_dst : norm.vs_dst + norm.n], m63.vsram[0, norm.vs_dst :][: norm.n]
    )
    m.stats_vpu = Stats()
    isa_sim.execute(m, Descriptor(**{**norm.__dict__, "sh1": -3}))
    assert m.stats_vpu.err_shift == norm.n
    gemv = next(x for x in b.programs.decode if x.opcode == Opcode.GEMV)
    m.stats_req = Stats()
    isa_sim.execute(m, Descriptor(**{**gemv.__dict__, "sh0": 200}))
    assert m.stats_req.err_shift == gemv.n and m.csr["ERR_SHIFT"] == 0  # the CSR updates at retire
    isa_sim.step(m, Descriptor(**{**gemv.__dict__, "sh0": 200}))
    assert m.csr["ERR_SHIFT"] == 2 * gemv.n + norm.n
    silu = next(x for x in b.programs.decode if x.opcode == Opcode.VSILUMUL)
    m.stats_vpu = Stats()
    isa_sim.execute(m, Descriptor(**{**silu.__dict__, "sh1": 64}))
    assert m.stats_vpu.err_shift == silu.n


# --------------------------------------------------------------------------- command line


def test_isa_sim_cli(smollm2: ModelSpec, capsys: pytest.CaptureFixture[str]) -> None:
    """compile then isa-sim --compare on SmolLM2 through the command line (build/images/<name>)."""
    if not quantize.default_path(smollm2.name).is_file():
        pytest.skip("build/quant/smollm2-135m-instruct.npz not present")
    assert cli.main(["compile", "smollm2", "--prompt", str(PROMPT)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["out"] == str(compiler.IMAGES_DIR / smollm2.name)
    args = ["isa-sim", "smollm2", "--max-new", "3", "--prompt", str(PROMPT), "--compare"]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "prompts/chat_short.json: step   2" in text
    assert "golden: every descriptor matches" in text
    parsed = cli.build_parser().parse_args(["isa-sim", "qwen", "--dir", "x", "--max-new", "4"])
    assert parsed.dir == "x" and parsed.max_new == 4 and not parsed.compare
