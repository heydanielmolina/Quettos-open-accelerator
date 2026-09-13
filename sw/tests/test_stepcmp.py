"""Per-descriptor comparison of a whole compiled program: ``qcore_top`` against the ISA simulator.

``quettos.stepcmp`` runs ``prefill.prog`` and ``decode.prog`` one ``CTRL.STEP``
per descriptor on ``qcore_top`` through the Verilator harness and on
``quettos.isa_sim``, dumps what each descriptor's ``dump_plan.json`` entry
names, and compares the two.  The tests here check the pieces that hold that
statement up -- the hash the harness and the comparison share, the capture
being the simulator's own, the plan naming every write, and a difference being
reported at the element it is in -- and then run the comparison itself over a
random tiny model and, when the quantized checkpoints are there, over truncated
real ones.  The RTL tests skip when Verilator is not on ``PATH``.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from quettos import compare, compiler, isa_sim, numerics, stepcmp, synthetic
from quettos.isa import Opcode

TINY = compare.CONFIGS[16]

#: Two layers, four query heads over two KV heads, a QKV bias and both norms.
SHAPE = synthetic.SHAPES[1]

#: The prompt and decode lengths of the comparison: with a two-tile cache the
#: run covers every position the KV region holds, the last position of the
#: first weight-port tile and the one that opens the second among them.
PROMPT_LEN = TINY.wb
MAX_NEW = TINY.wb + 1


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory) -> Path:
    """A random tiny model compiled for the CI width, two weight-port tiles of context deep."""
    return compare.compile_shape(
        SHAPE, 1, TINY, tmp_path_factory.mktemp("stepcmp-w16"), max_ctx=2 * TINY.wb
    )


@pytest.fixture(scope="module")
def prompt(tiny_image) -> list[int]:
    vocab = int(compiler.load_layout(tiny_image)["model"]["vocab"])
    return stepcmp.token_ids(PROMPT_LEN, vocab, seed=3)


# --------------------------------------------------------------------------- the shared pieces


def test_fnv1a64_is_the_hash_the_harness_writes() -> None:
    """The published FNV-1a 64 vectors: the harness hashes a region with the same constants."""
    assert stepcmp.fnv1a64(b"") == 0xCBF29CE484222325
    assert stepcmp.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert stepcmp.fnv1a64(b"foobar") == 0x85944171F73967E8
    assert stepcmp.fnv1a64(bytes([0xFF]) * 8) < 1 << 64


def test_distinct_shapes_differ_in_hidden_vocab_and_heads() -> None:
    """The sweep's five shapes are five different architectures, not one drawn five times."""
    shapes = stepcmp.distinct_shapes(5, seed=0)
    assert len(shapes) == 5
    keys = {(s.hidden, s.vocab, s.heads, s.kv_heads) for s in shapes}
    assert len(keys) == 5
    assert len({s.hidden for s in shapes}) > 1
    assert len({s.vocab for s in shapes}) > 1
    assert len({s.heads for s in shapes}) > 1
    assert any(s.heads > s.kv_heads for s in shapes), "a grouped-query shape"
    assert stepcmp.distinct_shapes(5, seed=0) == shapes, "the draw is a function of the seed"


def test_capture_is_the_simulators_own(tiny_image, prompt) -> None:
    """``record_token`` captures exactly what ``isa_sim.record_program`` does, plus the counters."""
    layout = compiler.load_layout(tiny_image)
    programs = isa_sim.Programs.from_dir(tiny_image)
    addr = int(layout["programs"]["decode"]["addr"])
    image = tiny_image / layout["image"]["file"]

    mine = isa_sim.Machine.from_file(image, **TINY.widths)
    records, effects = stepcmp.record_token(
        mine, programs.decode, programs.decode_plan, prompt[0], 0, pc=addr
    )
    theirs = isa_sim.Machine.from_file(image, **TINY.widths)
    reference = isa_sim.record_program(
        theirs, programs.decode, programs.decode_plan, prompt[0], 0, pc=addr
    )

    assert len(records) == len(reference) == len(programs.decode) == len(effects)
    for a, b in zip(records, reference, strict=True):
        assert (a.index, a.descriptor, a.entry) == (b.index, b.descriptor, b.entry)
        assert (a.vsram is None) == (b.vsram is None)
        if a.vsram is not None and b.vsram is not None:
            assert np.array_equal(a.vsram, b.vsram)
        assert a.sreg == b.sreg and a.mem == b.mem and a.csr == b.csr and a.writes == b.writes
    assert isa_sim.check_plan(records) == [], "the plan names every write the descriptors made"


def test_plan_names_every_write_of_both_programs(tiny_image, prompt) -> None:
    """The premise of the layer: comparing the planned state is comparing all of it."""
    layout = compiler.load_layout(tiny_image)
    programs = isa_sim.Programs.from_dir(tiny_image)
    m = isa_sim.Machine.from_file(tiny_image / layout["image"]["file"], **TINY.widths)
    for which, source, plan in (
        ("prefill", programs.prefill, programs.prefill_plan),
        ("decode", programs.decode, programs.decode_plan),
    ):
        addr = int(layout["programs"][which]["addr"])
        records, _ = stepcmp.record_token(m, source, plan, prompt[0], 0, pc=addr)
        assert isa_sim.check_plan(records) == [], which


def test_effects_carry_the_counters_and_the_pc(tiny_image, prompt) -> None:
    """Each descriptor's effects hold the state its entry names and the counters since START."""
    layout = compiler.load_layout(tiny_image)
    programs = isa_sim.Programs.from_dir(tiny_image)
    addr = int(layout["programs"]["decode"]["addr"])
    m = isa_sim.Machine.from_file(tiny_image / layout["image"]["file"], **TINY.widths)
    records, effects = stepcmp.record_token(
        m, programs.decode, programs.decode_plan, prompt[0], 0, pc=addr
    )
    for i, (rec, eff) in enumerate(zip(records, effects, strict=True)):
        assert eff.index == i and eff.op == rec.descriptor.opcode.name
        assert eff.pc == addr + 32 * (i + 1), "PC advances by one descriptor per retire"
        assert eff.perf["DESCRIPTORS"] == i + 1
        assert set(eff.sreg) == set(rec.sreg) and set(eff.mem) == set(rec.mem)
        assert set(eff.perf) == set(stepcmp.PERF_COMPARED)
        assert set(eff.events) == set(stepcmp.EVENTS)
    assert [e.perf["MACS"] for e in effects] == sorted(e.perf["MACS"] for e in effects)
    assert effects[-1].perf["MACS"] == m.perf_value("MACS")
    assert effects[-1].perf["WT_BYTES"] == m.perf_value("WT_BYTES")
    head = next(e for e in effects if e.name == "gemv_lm_head")
    assert set(head.csr) == {"ARGMAX_TOK", "ARGMAX_VAL"} and head.vsram is None
    kv = next(e for e in effects if e.name == "kvwrite_k")
    assert set(kv.mem) == {"kv.0.0.kt", "kv.0.0.k_meta"}
    assert all(r.data is not None for r in kv.mem.values())


def _as_rtl(e: stepcmp.Effects) -> dict[str, Any]:
    """One ``Effects`` in the JSON shape ``sim/verilator/main.cpp`` writes for a descriptor."""
    return {
        "index": e.index,
        "name": e.name,
        "op": e.op,
        "pc": e.pc,
        "vsram": ""
        if e.vsram is None
        else {"start": e.vsram_start, "count": len(e.vsram), "values": list(e.vsram)},
        "sreg": {str(i): v for i, v in e.sreg.items()},
        "mem": {
            name: {
                "addr": r.addr,
                "size": r.size,
                "fnv1a64": r.digest,
                **({} if r.data is None else {"hex": r.data.hex()}),
            }
            for name, r in e.mem.items()
        },
        "csr": dict(e.csr),
        "perf": dict(e.perf),
        "events": dict(e.events),
    }


def test_two_sides_parse_into_the_same_effects(tiny_image, prompt) -> None:
    """A record written the way the harness writes it reads back as the effects it came from."""
    layout = compiler.load_layout(tiny_image)
    programs = isa_sim.Programs.from_dir(tiny_image)
    m = isa_sim.Machine.from_file(tiny_image / layout["image"]["file"], **TINY.widths)
    _, effects = stepcmp.record_token(
        m,
        programs.decode,
        programs.decode_plan,
        prompt[0],
        0,
        pc=int(layout["programs"]["decode"]["addr"]),
    )
    for e in effects:
        assert stepcmp.Effects.from_rtl(_as_rtl(e)) == e
        assert list(stepcmp.differences(e, stepcmp.Effects.from_rtl(_as_rtl(e)))) == []


def _effects(**kw) -> stepcmp.Effects:
    base = {
        "index": 4,
        "op": "GEMV",
        "name": "gemv_qkv",
        "pc": 160,
        "vsram_start": 64,
        "vsram": [1, 2, 3, 4],
        "sreg": {2: 0x1234},
        "mem": {"kv.0.0.kt": stepcmp.Region(0x1000, 4, stepcmp.fnv1a64(b"abcd"), b"abcd")},
        "csr": {"ARGMAX_TOK": 7},
        "perf": {"DESCRIPTORS": 5, "MACS": 64, "WT_BYTES": 128},
        "events": {"SAT_REQ": 0, "SAT_VPU": 1, "ERR_SHIFT": 0, "ERR_BOUNDS": 0},
    }
    return stepcmp.Effects(**{**base, **kw})


@pytest.mark.parametrize(
    ("change", "what", "element"),
    [
        ({"vsram": [1, 2, 9, 4]}, "VSRAM element 66", 66),
        ({"vsram": None}, "VSRAM range", None),
        ({"sreg": {2: 0x1235}}, "SREG[2]", 2),
        ({"sreg": {3: 0x1234}}, "SREG indices", None),
        ({"csr": {"ARGMAX_TOK": 8}}, "ARGMAX_TOK", None),
        ({"pc": 192}, "PC", None),
        ({"op": "EMBED"}, "op", None),
        ({"perf": {"DESCRIPTORS": 5, "MACS": 128, "WT_BYTES": 128}}, "PERF MACS", None),
        (
            {"events": {"SAT_REQ": 0, "SAT_VPU": 2, "ERR_SHIFT": 0, "ERR_BOUNDS": 0}},
            "SAT_VPU",
            None,
        ),
    ],
)
def test_difference_is_reported_where_it_is(
    change: dict[str, Any], what: str, element: int | None
) -> None:
    """Every compared field names itself, and an element difference names its index."""
    exp = _effects()
    got = _effects(**change)
    found = list(stepcmp.differences(exp, got))
    assert [f[0] for f in found] == [what]
    assert found[0][1] == element
    assert list(stepcmp.differences(exp, _effects())) == []


def test_memory_region_reports_its_byte_or_its_hash() -> None:
    """The bytes name the byte; a region too large to carry names the region and its hash."""
    exp = _effects()
    got = _effects(mem={"kv.0.0.kt": stepcmp.Region(0x1000, 4, stepcmp.fnv1a64(b"abXd"), b"abXd")})
    (what, element, e, g) = next(iter(stepcmp.differences(exp, got)))
    assert what == "kv.0.0.kt byte at 0x00001002" and element == 2
    assert (e, g) == (ord("c"), ord("X"))
    hashed_exp = _effects(mem={"kv.0.0.kt": stepcmp.Region(0x1000, 4, 111, None)})
    hashed_got = _effects(mem={"kv.0.0.kt": stepcmp.Region(0x1000, 4, 222, None)})
    (what, element, _, _) = next(iter(stepcmp.differences(hashed_exp, hashed_got)))
    assert what == "kv.0.0.kt hash over 4 bytes" and element is None
    assert list(stepcmp.differences(hashed_exp, hashed_exp)) == []


def test_mismatch_names_the_descriptor_and_the_listing() -> None:
    """The report line carries the program, position, index, opcode, name, element and listing."""
    m = stepcmp.Mismatch(
        "decode",
        17,
        42,
        "GEMV",
        "gemv_scores",
        "42 GEMV n_from_pos k=64",
        "VSRAM element 8",
        8,
        -3,
        5,
    )
    text = str(m)
    assert "decode POS 17 descriptor 42 GEMV gemv_scores" in text
    assert "VSRAM element 8 element 8" in text and "isa_sim -3, RTL 5" in text
    assert "42 GEMV n_from_pos k=64" in text
    plan = stepcmp.Mismatch(
        "decode", 0, -1, "", "", "", "the dump plan misses a write", None, "x", ""
    )
    assert str(plan) == "decode POS 0: the dump plan misses a write: x"


# --------------------------------------------------------------------------- the RTL


def test_every_descriptor_of_both_programs_matches_the_rtl(tiny_image, prompt) -> None:
    """The comparison itself: every descriptor of prefill.prog and decode.prog, every position."""
    needs_verilator()
    r = stepcmp.check_image(tiny_image, TINY, prompt=prompt, max_new=MAX_NEW)
    assert r.mismatches == [], "\n".join(str(m) for m in r.mismatches[:4])
    assert r.ok and r.reference_ids == r.rtl_ids and len(r.rtl_ids) == MAX_NEW
    layout = compiler.load_layout(tiny_image)
    counts = {k: int(v["descriptors"]) for k, v in layout["programs"].items()}
    prefill_steps = PROMPT_LEN - 1
    assert r.tokens == prefill_steps + MAX_NEW
    assert r.descriptors == prefill_steps * counts["prefill"] + MAX_NEW * counts["decode"]
    # The last position of the first weight-port tile, the one that opens the
    # second, and the last position the two-tile cache holds are all in the run.
    assert r.tokens == 2 * TINY.wb == int(layout["max_ctx"])


def test_flipped_weight_byte_names_its_descriptor(tiny_image, prompt, tmp_path) -> None:
    """A defect the layer exists to localize: one byte of a gamma row, named at its descriptor."""
    needs_verilator()
    layout = compiler.load_layout(tiny_image)
    hurt = tmp_path / "hurt"
    shutil.copytree(tiny_image, hurt)
    image = hurt / layout["image"]["file"]
    regions = {r["name"]: r for r in layout["regions"]}
    buf = bytearray(image.read_bytes())
    buf[regions["layer.0.gamma_in"]["addr"]] ^= 0x01
    image.write_bytes(bytes(buf))

    runs, _ = stepcmp.reference(hurt, TINY, prompt[:2], 1)
    dumps, _, _ = stepcmp.run_rtl(tiny_image, TINY, prompt[:2], 1, out_dir=tmp_path / "steps")
    found = [m for tr in runs for m in stepcmp.compare_token(tr, dumps.get((tr.program, tr.pos)))]
    assert found, "the flipped byte has to show up"
    first = found[0]
    assert (first.program, first.pos, first.opcode, first.name) == (
        "prefill",
        0,
        "VRMSNORM",
        "rmsnorm_in",
    )
    assert first.what.startswith("VSRAM element") and first.element is not None
    assert first.expected != first.got and "VRMSNORM" in first.listing


def test_rtl_dump_follows_the_plan(tiny_image, prompt) -> None:
    """What the harness writes per descriptor is what that descriptor's plan entry names."""
    needs_verilator()
    plan = compiler.load_dump_plan(tiny_image)["decode"]
    dumps, _, _ = stepcmp.run_rtl(tiny_image, TINY, prompt[:2], 1)
    got = dumps[("decode", 1)]
    assert len(got) == len(plan)
    for entry, eff in zip(plan, got, strict=True):
        assert (eff.index, eff.op, eff.name) == (entry["index"], entry["op"], entry["name"])
        assert set(eff.sreg) == set(entry["sreg"]) and set(eff.csr) == set(entry["csr"])
        assert set(eff.mem) == {r["name"] for r in entry["mem"]}
        if entry["vsram"] is None:
            assert eff.vsram is None
        else:
            assert eff.vsram is not None
            assert eff.vsram_start == entry["vsram"]["start"]
            assert len(eff.vsram) == entry["vsram"]["count"]


def test_sweep_runs_five_shapes(tmp_path) -> None:
    """``make stepcmp``: five random tiny models, both programs, every position of the cache."""
    needs_verilator()
    results = stepcmp.sweep(
        5, TINY, seed=0, prompt_len=PROMPT_LEN, max_new=MAX_NEW, out_dir=tmp_path
    )
    assert len(results) == 5
    for r in results:
        assert r.ok, "\n".join(str(m) for m in r.mismatches[:4])
        assert r.tokens == 2 * TINY.wb
    assert sum(r.descriptors for r in results) > 5_000
    assert stepcmp.report(results) == 0


@pytest.mark.slow
@pytest.mark.parametrize("alias", ["qwen", "smollm2"])
@pytest.mark.parametrize("layers", [1, 2])
def test_truncated_real_model_matches_the_rtl(alias: str, layers: int, tmp_path) -> None:
    """One and two layers of each real model, compiled and compared descriptor by descriptor."""
    needs_verilator()
    cfg = compare.CONFIGS[64]
    try:
        results = stepcmp.models(
            [alias], [layers], cfg, prompt_len=3, max_new=1, max_ctx=cfg.wb, out_dir=tmp_path
        )
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{alias}: {exc}")
    (r,) = results
    assert r.ok, "\n".join(str(m) for m in r.mismatches[:4])
    layout = compiler.load_layout(r.image)
    counts = {k: int(v["descriptors"]) for k, v in layout["programs"].items()}
    assert int(layout["model"]["layers"]) == layers
    assert r.descriptors == 2 * counts["prefill"] + counts["decode"]
    assert r.events == dict.fromkeys(stepcmp.EVENTS, 0)


def test_layers_list_names_numbers_and_the_whole_model() -> None:
    """``--layers 1,2,all``: two truncated compiles and the whole checkpoint."""
    assert stepcmp.layer_counts("1,2") == [1, 2]
    assert stepcmp.layer_counts(stepcmp.ALL_LAYERS) == [None]
    assert stepcmp.layer_counts("2,all") == [2, None]
    assert stepcmp.layer_counts("") == []


@pytest.mark.slow
@pytest.mark.parametrize("alias", ["qwen", "smollm2"])
def test_complete_real_model_matches_the_rtl(alias: str, tmp_path) -> None:
    """``make stepcmp-model``: every descriptor of the whole checkpoint, on both machines.

    Two prefill positions and two decode steps, so the state after every
    descriptor of every decoder layer, the final norm and the LM head is
    compared, and the KV cache the first decode step writes is what the second
    reads.
    """
    needs_verilator()
    cfg = compare.CONFIGS[64]
    try:
        results = stepcmp.models(
            [alias], [None], cfg, prompt_len=3, max_new=2, max_ctx=128, out_dir=tmp_path
        )
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{alias}: {exc}")
    (r,) = results
    assert r.ok, "\n".join(str(m) for m in r.mismatches[:4])
    layout = compiler.load_layout(r.image)
    counts = {k: int(v["descriptors"]) for k, v in layout["programs"].items()}
    spec_layers = int(compiler.load_layout(r.image)["model"]["layers"])
    assert spec_layers >= 24, "the complete checkpoint, not a truncated compile"
    assert r.tokens == 4 and r.descriptors == 2 * counts["prefill"] + 2 * counts["decode"]
    assert r.reference_ids == r.rtl_ids and len(r.rtl_ids) == 2
    assert r.events == dict.fromkeys(stepcmp.EVENTS, 0)
    ops = {d.opcode for d in isa_sim.Programs.from_dir(r.image).decode}
    assert {
        Opcode.VRMSNORM,
        Opcode.VQUANT,
        Opcode.VROPE,
        Opcode.VSILUMUL,
        Opcode.VSOFTMAX,
        Opcode.VSUBC,
        Opcode.KVWRITE,
    } <= ops


# --------------------------------------------------------------------------- the vector opcodes


def test_compared_programs_run_every_vector_opcode(tiny_image) -> None:
    """The programs the comparison walks issue all six vector opcodes and both KVWRITE forms."""
    programs = isa_sim.Programs.from_dir(tiny_image)
    ops = {d.opcode for d in programs.decode}
    assert {
        Opcode.VRMSNORM,
        Opcode.VQUANT,
        Opcode.VROPE,
        Opcode.VSILUMUL,
        Opcode.VSOFTMAX,
        Opcode.VSUBC,
        Opcode.KVWRITE,
        Opcode.GEMV,
        Opcode.EMBED,
        Opcode.HALT,
    } <= ops
    assert {d.opcode for d in programs.prefill} <= ops, "prefill is decode without the head"


def test_sreg_word_is_the_word_the_bank_holds() -> None:
    """A scale travels as its sfloat pair and a tracked absmax as its int32."""
    assert stepcmp.sreg_word(numerics.SFloat(0x8000, -15)) == stepcmp.sreg_word(
        numerics.SFloat(0x8000, -15)
    )
    assert stepcmp.sreg_word(1234) == 1234
    assert stepcmp.sreg_word(-1) == 0xFFFFFFFF
    packed = stepcmp.sreg_word(numerics.SFloat(0xABCD, -7))
    assert packed & 0xFFFF == 0xABCD and (packed >> 16) & 0xFF == 0xF9


def test_dataclasses_are_comparable() -> None:
    """``Effects`` and ``Region`` compare by value, which is what the comparison relies on."""
    a = _effects()
    assert a == _effects() and a != _effects(pc=0)
    assert dataclasses.replace(a, pc=0) == _effects(pc=0)
