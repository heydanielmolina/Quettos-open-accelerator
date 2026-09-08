"""The four comparison programs: their shape, their self-check on the ISA simulator, and the RTL.

``quettos.compare.bringup`` assembles the four descriptors ``docs/ISA.md``
defines -- ``EMBED``, ``VQUANT``, the tied LM head as a ``GEMV`` in ARGMAX mode
and ``HALT`` -- and ``quettos.compare.vector_ops`` a directed program that runs
the vector unit's four elementwise opcodes back to back.
``quettos.compare.attention`` and ``quettos.compare.layer`` are prefixes of the
compiled ``decode.prog`` with a ``HALT`` after them: the attention step, and the
whole decoder layer.  ``quettos.compare`` runs each on ``quettos.isa_sim`` and
on ``qcore_top`` through the Verilator harness and compares every VSRAM
element, SREG word, KV byte, CSR and PERF counter.  The RTL tests skip when
Verilator is not on ``PATH``.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest
from quettos import cli, compare, isa, isa_sim, synthetic
from quettos.isa import Opcode, OutMode, VquantFlag

TINY = compare.CONFIGS[16]
DEMO = compare.CONFIGS[64]

#: The six vector opcodes, every one of which qcore_vpu_top executes.
VECTOR_OPS = (
    Opcode.VRMSNORM,
    Opcode.VQUANT,
    Opcode.VROPE,
    Opcode.VSILUMUL,
    Opcode.VSOFTMAX,
    Opcode.VSUBC,
)

#: Opcode bytes the ISA does not define: one beside each defined group, and the
#: two ends of the byte. A descriptor carrying one is what the hardware cannot
#: execute, and what ``FAULT = OPCODE`` means.
UNDEFINED_OPCODES = (0x02, 0x0F, 0x12, 0x26, 0x32, 0x7F, 0xFF)


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


@pytest.fixture(scope="module")
def demo_image(tmp_path_factory) -> object:
    """A one-layer synthetic model compiled for the demo width, two tiles of context deep."""
    return compare.compile_shape(
        synthetic.SHAPES[0], 0, DEMO, tmp_path_factory.mktemp("img-w64"), max_ctx=2 * DEMO.wb
    )


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory) -> object:
    """The same model compiled for the tiny width, the configuration CI runs."""
    return compare.compile_shape(
        synthetic.SHAPES[0], 0, TINY, tmp_path_factory.mktemp("img-w16"), max_ctx=2 * TINY.wb
    )


@pytest.fixture(scope="module")
def gqa_image(tmp_path_factory) -> object:
    """Six query heads over three KV heads: the grouping the head loop walks."""
    return compare.compile_shape(
        synthetic.SHAPES[4], 4, TINY, tmp_path_factory.mktemp("img-gqa"), max_ctx=2 * TINY.wb
    )


# --------------------------------------------------------------------------- the programs


def test_the_program_is_embed_vquant_gemv_halt(demo_image) -> None:
    """Four descriptors, 128 bytes, loaded behind prefill.prog, with nothing loaded by the host."""
    c = compare.bringup(demo_image, DEMO, tok=3)
    assert [d.opcode for d in c.program] == [
        Opcode.EMBED,
        Opcode.VQUANT,
        Opcode.GEMV,
        Opcode.HALT,
    ]
    assert len(c.blob) == 4 * isa.DESC_BYTES
    assert isa.parse(c.blob) == list(c.program)
    assert c.addr % isa.PROGRAM_ALIGN == 0
    assert c.sreg == (), "the program writes its own activation scale"
    assert c.passes == ((3, 0),)
    embed, quant, gemv, _ = c.program
    assert embed.vs_dst == quant.vs_src, "the VQUANT reads the slot the EMBED wrote"
    assert quant.vs_dst == gemv.vs_src, "the GEMV reads the slot the VQUANT wrote"
    assert quant.n == embed.k and not quant.flags & VquantFlag.W8
    assert gemv.out_mode is OutMode.ARGMAX_DUMP and gemv.imm32 == c.mem[0][0]
    assert c.mem == ((gemv.imm32, 4 * gemv.n),)


def test_the_vquant_writes_the_scale_the_gemv_reads(demo_image) -> None:
    """No SREG is loaded from the host: descriptor 1 leaves the scale descriptor 2 reads."""
    c = compare.bringup(demo_image, DEMO, tok=4)
    _, quant, gemv, _ = c.program
    assert quant.sreg_dst == gemv.sreg_src
    records, _ = compare.reference(demo_image, c, DEMO)
    assert records[0].sreg[0][quant.sreg_dst] == 0, "the bank holds nothing before the VQUANT"
    assert records[1].sreg[0][quant.sreg_dst] != 0, "the VQUANT wrote the activation scale"
    assert records[2].sreg[0][gemv.sreg_src] == records[1].sreg[0][quant.sreg_dst]


def test_the_vector_program_runs_the_elementwise_opcodes(demo_image) -> None:
    """VRMSNORM, VQUANT with USE_TRACKED, VSUBC, VSILUMUL and a GROUP VQUANT, back to back."""
    c = compare.vector_ops(demo_image, DEMO, tok=6)
    ops = [d.opcode for d in c.program]
    assert ops == [
        Opcode.EMBED,
        Opcode.VRMSNORM,
        Opcode.VQUANT,
        Opcode.VSUBC,
        Opcode.VSILUMUL,
        Opcode.VQUANT,
        Opcode.VSILUMUL,
        Opcode.HALT,
    ]
    tracked, grouped = c.program[2], c.program[5]
    assert tracked.flags & VquantFlag.USE_TRACKED and tracked.sreg_src == c.program[1].sreg_dst
    assert grouped.flags & (VquantFlag.GROUP | VquantFlag.W8)
    assert c.program[6].sh1 == 0, "the last VSILUMUL leaves its product unshifted"
    assert c.mem == (), "the vector program compares no memory region"


def test_the_vector_program_carries_a_value_through_the_sreg_bank(demo_image) -> None:
    """The tracked absmax the VRMSNORM writes is what the VQUANT after it quantizes with."""
    c = compare.vector_ops(demo_image, DEMO, tok=6)
    rms, tracked = c.program[1], c.program[2]
    records, _ = compare.reference(demo_image, c, DEMO)
    absmax = records[1].sreg[0][rms.sreg_dst]
    assert absmax > 0, "the VRMSNORM tracked an absmax"
    assert records[2].sreg[0][tracked.sreg_dst] != 0, "the VQUANT wrote a scale from it"
    written = records[4].vsram[0][c.program[4].vs_dst : c.program[4].vs_dst + rms.n]
    assert any(v < 0 for v in written) and any(v > 0 for v in written)


def test_the_vector_program_puts_a_value_in_every_counter_it_can_reach(demo_image) -> None:
    """The last VSILUMUL saturates, so SAT_VPU is a number both models have to agree on."""
    c = compare.vector_ops(demo_image, DEMO, tok=6)
    records, _ = compare.reference(demo_image, c, DEMO)
    assert records[5].events["SAT_VPU"] == 0, "nothing saturates before the last VSILUMUL"
    saturated = records[6].events["SAT_VPU"]
    assert 0 < saturated < c.program[6].n, f"{saturated} of {c.program[6].n} elements saturated"
    assert "vector" in compare.EVENTFUL, "so the harness reports the events instead of failing"


def test_argmax_is_the_input_token(demo_image) -> None:
    """A tied embedding makes the largest logit of row TOK its own row, for every token."""
    vocab = synthetic.SHAPES[0].vocab
    for tok in range(vocab):
        c = compare.bringup(demo_image, DEMO, tok=tok)
        records, _ = compare.reference(demo_image, c, DEMO)
        last = records[-1]
        assert last.argmax_tok == tok, f"token {tok} -> {last.argmax_tok}"
        assert last.events == dict.fromkeys(compare.EVENTS, 0)
        assert last.perf["DESCRIPTORS"] == 4


def test_the_simulator_counts_what_the_isa_defines(demo_image) -> None:
    """DESCRIPTORS, MACS and WT_BYTES of the four descriptors, from the shapes alone."""
    shape = synthetic.SHAPES[0]
    c = compare.bringup(demo_image, DEMO, tok=5)
    records, _ = compare.reference(demo_image, c, DEMO)
    tiles = -(-shape.vocab // DEMO.wb)
    assert records[-1].perf["MACS"] == tiles * DEMO.wb * shape.hidden
    assert records[-1].perf["WT_BYTES"] == (
        shape.hidden
        + isa.META_BYTES
        + tiles * shape.hidden * DEMO.wb
        + tiles * DEMO.wb * isa.META_BYTES
    )


def test_compare_names_the_first_differing_element(demo_image) -> None:
    """A single changed logit is reported with its descriptor, its field and its index."""
    c = compare.bringup(demo_image, DEMO, tok=9)
    records, mem = compare.reference(demo_image, c, DEMO)
    doctored = [[list(vals) for vals in one_pass] for one_pass in mem]
    doctored[0][0][7] += 1
    bad = compare.compare(records, records, c, mem, doctored)
    assert len(bad) == 1
    assert bad[0].element == 7 and bad[0].opcode == "GEMV" and bad[0].pos == 0
    assert str(bad[0]).endswith(f"isa_sim {mem[0][0][7]}, RTL {mem[0][0][7] + 1}")
    assert compare.compare(records, records, c, mem, mem) == []


# --------------------------------------------------------------------------- the attention step


def test_the_attention_program_is_the_compilers_own_sequence(tiny_image) -> None:
    """A prefix of decode.prog with a HALT: the rotation, the scores, the softmax, the values."""
    c = compare.attention(tiny_image, TINY, tok=3)
    descs = isa.parse((tiny_image / compare.compiler.FILES["decode"]).read_bytes())
    assert list(c.program[:-1]) == descs[: len(c.program) - 1], "the compiler's own descriptors"
    assert c.program[-1].opcode is Opcode.HALT
    ops = [d.opcode for d in c.program]
    assert ops[0] is Opcode.EMBED and Opcode.VROPE in ops and Opcode.VSOFTMAX in ops
    assert ops.count(Opcode.KVWRITE) == 2, "one K write and one V write per KV head"
    scores = next(d for d in c.program if d.opcode is Opcode.GEMV and d.n_from_pos)
    pv = next(d for d in c.program if d.opcode is Opcode.GEMV and d.k_from_pos)
    softmax = next(d for d in c.program if d.opcode is Opcode.VSOFTMAX)
    assert scores.vs_dst == softmax.vs_src and softmax.vs_dst == pv.vs_src
    assert softmax.len_from_pos and pv.unit_meta
    assert c.program[-2].opcode is Opcode.GEMV, "the step ends on the value GEMV"


def test_the_attention_positions_step_the_derived_extents(tiny_image) -> None:
    """The first position, one inside the first tile, the boundary, one past it, and the last."""
    layout = compare.compiler.load_layout(tiny_image)
    max_ctx = int(layout["max_ctx"])
    got = compare.positions(max_ctx, TINY.wb)
    assert got == (0, 1, TINY.wb - 1, TINY.wb, TINY.wb + 1, max_ctx - 1)
    c = compare.attention(tiny_image, TINY, tok=3)
    assert [p for _, p in c.passes] == list(got)
    assert len({t for t, _ in c.passes}) == len(got), "each position embeds its own token"
    scores = next(d for d in c.program if d.opcode is Opcode.GEMV and d.n_from_pos)
    pv = next(d for d in c.program if d.opcode is Opcode.GEMV and d.k_from_pos)
    n = [isa_sim.gemv_dims(scores, p, TINY.wb)[0] for p in got]
    k = [isa_sim.gemv_dims(pv, p, TINY.wb)[1] for p in got]
    assert n == [TINY.wb, TINY.wb, TINY.wb, 2 * TINY.wb, 2 * TINY.wb, max_ctx]
    assert k == [p + 1 for p in got], "the value GEMV reads one more token at every position"
    assert len(set(zip(n, k, strict=True))) == len(got), "no two positions run the same shapes"


def test_the_attention_step_matches_the_simulator_on_the_tiny_configuration(tiny_image) -> None:
    """THE MILESTONE at WB=16: every element, register, KV byte and counter, at six positions."""
    needs_verilator()
    result = compare.check_image(
        tiny_image, TINY, 3, program="attention", timing=((32, 1), (1, 1), (200, 2))
    )
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []
    assert result.passes == 6


def test_the_attention_step_matches_the_simulator_at_the_demo_width(demo_image) -> None:
    """The synthesized configuration, WB=64: the same six positions a tile apart."""
    needs_verilator()
    result = compare.check_image(
        demo_image, DEMO, 11, program="attention", timing=((32, 1), (200, 2))
    )
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []


def test_the_attention_step_matches_the_simulator_across_a_kv_head_group(gqa_image) -> None:
    """Six query heads over three KV heads: every head reads the group its q vector belongs to."""
    needs_verilator()
    c = compare.attention(gqa_image, TINY, tok=7)
    assert [d.opcode for d in c.program].count(Opcode.VSOFTMAX) == synthetic.SHAPES[4].heads
    assert [d.opcode for d in c.program].count(Opcode.KVWRITE) == 2 * synthetic.SHAPES[4].kv_heads
    result = compare.check_image(gqa_image, TINY, 7, program="attention")
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)


def test_the_attention_step_matches_at_every_position_the_cache_holds(tiny_image) -> None:
    """Every position from the first to the last, so the softmax reduces over a full row."""
    needs_verilator()
    layout = compare.compiler.load_layout(tiny_image)
    max_ctx, vocab = int(layout["max_ctx"]), int(layout["model"]["vocab"])
    base = compare.attention(tiny_image, TINY, tok=1)
    c = dataclasses.replace(base, passes=tuple(((1 + p) % vocab, p) for p in range(max_ctx)))
    ref, ref_mem = compare.reference(tiny_image, c, TINY)
    got, got_mem, _ = compare.run_rtl(tiny_image, c, TINY, name="attention")
    assert compare.compare(ref, got, c, ref_mem, got_mem) == []
    softmax = next(i for i, d in enumerate(c.program) if d.opcode is Opcode.VSOFTMAX)
    weights = c.program[softmax]
    last = ref[(max_ctx - 1) * len(c.program) + softmax]
    row = last.vsram[0][weights.vs_dst : weights.vs_dst + max_ctx]
    assert sum(1 for v in row if v != 0) == max_ctx, "every token of the last row carries a weight"


def test_the_whole_layer_matches_the_simulator(tiny_image) -> None:
    """The decoder layer end to end: attention, the output projection, the norm and the MLP."""
    needs_verilator()
    c = compare.layer(tiny_image, TINY, tok=5)
    ops = [d.opcode for d in c.program]
    assert ops[-4:] == [Opcode.VSILUMUL, Opcode.VQUANT, Opcode.GEMV, Opcode.HALT]
    attention = compare.attention(tiny_image, TINY, tok=5)
    assert list(c.program[: len(attention.program) - 1]) == list(attention.program[:-1])
    result = compare.check_image(tiny_image, TINY, 5, program="layer", timing=((32, 1), (1, 1)))
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []


def test_the_whole_layer_matches_the_simulator_at_the_demo_width(demo_image) -> None:
    """The same layer on the synthesized configuration."""
    needs_verilator()
    result = compare.check_image(demo_image, DEMO, 13, program="layer")
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)


def test_the_kv_cache_the_hardware_writes_is_the_one_it_reads(tiny_image) -> None:
    """The KV region is compared as bytes at every position, and it grows as the passes run."""
    needs_verilator()
    c = compare.attention(tiny_image, TINY, tok=3)
    assert len(c.mem) == 1, "one region: the whole KV cache"
    ref, ref_mem = compare.reference(tiny_image, c, TINY)
    got, got_mem, _ = compare.run_rtl(tiny_image, c, TINY, name="attention")
    assert compare.compare(ref, got, c, ref_mem, got_mem) == []
    written = [sum(1 for v in one_pass[0] if v != 0) for one_pass in got_mem]
    assert written[0] > 0, "the first position wrote its own K and V"
    assert written == sorted(written) and written[-1] > written[0], f"{written}"


# --------------------------------------------------------------------------- the RTL


@pytest.mark.parametrize("byte", UNDEFINED_OPCODES)
def test_an_undefined_opcode_stops_the_run_with_the_fault_in_status(demo_image, byte) -> None:
    """The fault the ISA gives an opcode the hardware cannot execute, on both models.

    Every opcode the ISA defines now issues to a unit, so ``FAULT = OPCODE``
    means one thing: the opcode byte is none of the twelve.  The run stops
    before the descriptor is executed, ``FAULT_OP`` carries the byte, ``PC``
    names the descriptor, and nothing is counted for it.
    """
    needs_verilator()
    assert not isa.is_opcode(byte)
    good = compare.bringup(demo_image, DEMO, tok=2)
    blob = (
        isa.encode(good.program[0])
        + bytes([byte])
        + bytes(isa.DESC_BYTES - 1)
        + isa.encode(isa.halt())
    )
    c = dataclasses.replace(good, program=(good.program[0], isa.halt()), blob=blob, mem=())
    records, _, _ = compare.run_rtl(demo_image, c, DEMO, allow_error=True, name="undefined")
    last = records[-1]
    assert isa.status_fault(last.status) == (isa.Fault.OPCODE, byte)
    assert last.pc == c.addr + isa.DESC_BYTES, "PC names the descriptor that faulted"
    assert last.perf["DESCRIPTORS"] == 1, "nothing is counted for the undefined descriptor"

    layout = compare.compiler.load_layout(demo_image)
    m = isa_sim.Machine.from_file(demo_image / layout["image"]["file"], **DEMO.widths)
    m.mem[c.addr : c.addr + len(blob)] = blob
    m.csr["TOK"], m.csr["POS"], m.csr["ROW_EN"] = 2, 0, 1
    isa_sim.run_program(m, blob, pc=c.addr)
    assert m.fault_state() == (isa.Fault.OPCODE, byte)
    assert m.csr["PC"] == last.pc and m.perf_value("DESCRIPTORS") == last.perf["DESCRIPTORS"]


def test_every_vector_opcode_reaches_the_unit(tiny_image) -> None:
    """The counterpart: a program carrying all six V opcodes runs to DONE with no fault."""
    needs_verilator()
    c = compare.layer(tiny_image, TINY, tok=2)
    ops = {d.opcode for d in c.program}
    assert set(VECTOR_OPS) <= ops, "the layer program carries every vector opcode"
    ref, _ = compare.reference(tiny_image, c, TINY)
    got, _, _ = compare.run_rtl(tiny_image, c, TINY, name="layer")
    for r in got:
        assert isa.status_fault(r.status) == (isa.Fault.NONE, 0)
    last = got[-1]
    assert last.perf["DESCRIPTORS"] == len(c.program) * len(c.passes)
    assert last.events == ref[-1].events, "the RTL counts what the simulator counts"


@pytest.mark.parametrize("wb", [16, 64])
def test_the_generated_ids_match_the_simulator(tiny_image, demo_image, wb) -> None:
    """The whole compiled model: the prefill/decode loop on qcore_top emits isa_sim's ids.

    This is the oracle chain closed from the top -- `isa_sim` reproduces the
    integer golden model descriptor by descriptor, so ids that match it match
    the golden model too.
    """
    needs_verilator()
    cfg = compare.CONFIGS[wb]
    image = tiny_image if wb == 16 else demo_image
    g = compare.generate(image, cfg, max_new=4, prompt=[1, 2, 3])
    assert g.rtl == g.reference, f"first difference at generated id {g.first_difference}"
    assert len(g.rtl) == 4 and len(set(g.rtl)) > 1, f"{g.rtl} is not a generation"


def cross_row_program(image, cfg, tok: int):
    """The vector program's descriptors with ``src_row = 0`` and ``dst_row = 1``.

    Only a two-row build can run this, and only the top level can get it wrong:
    the vector unit reads bank ``src_row + cur_row`` on both ports and writes
    bank ``dst_row + cur_row`` on port B and in the SREG bank, so a crossbar
    that ignores either base sends the writes to the bank the reads came from.
    """
    c = compare.vector_ops(image, cfg, tok)
    embed, _, _, subc, silu, grouped = c.program[:6]
    across = {"src_row": 0, "dst_row": 1}
    program = (
        embed,
        dataclasses.replace(subc, **across),
        dataclasses.replace(silu, vs_src=embed.vs_dst, vs_aux=embed.vs_dst, **across),
        dataclasses.replace(grouped, vs_src=embed.vs_dst, **across),
        isa.halt(),
    )
    return dataclasses.replace(c, program=program, blob=isa.assemble(list(program)))


def test_the_crossbar_routes_a_descriptor_across_rows(tiny_image) -> None:
    """Reads of bank 0 and writes of bank 1, in the one configuration that has two."""
    needs_verilator()
    assert TINY.b_max == 2
    c = cross_row_program(tiny_image, TINY, tok=5)
    ref, ref_mem = compare.reference(tiny_image, c, TINY)
    got, got_mem, _ = compare.run_rtl(tiny_image, c, TINY, name="cross")
    assert compare.compare(ref, got, c, ref_mem, got_mem) == []
    last = ref[-1]
    assert any(v != 0 for v in last.vsram[1]), "row 1 was written"
    assert last.sreg[1][c.program[3].sreg_dst] != 0, "the scale landed in row 1's bank"
    assert last.sreg[0] == [0] * len(last.sreg[0]), "row 0's bank was only read"


def test_the_vector_unit_runs_both_rows_of_a_descriptor(tiny_image) -> None:
    """ROW_EN = 3: the unit walks the participating rows ascending, each to completion.

    The ``EMBED`` stays on row 0, so row 1 starts from an empty bank and the two
    banks hold different values all the way down: a crossbar that read or wrote
    the wrong bank for ``cur_row`` would make them agree.
    """
    needs_verilator()
    assert TINY.b_max == 2
    base = compare.vector_ops(tiny_image, TINY, tok=5)
    stays = (Opcode.EMBED, Opcode.HALT)
    program = tuple(
        d if d.opcode in stays else dataclasses.replace(d, row_mask=3) for d in base.program
    )
    c = dataclasses.replace(base, program=program, blob=isa.assemble(list(program)))
    ref, ref_mem = compare.reference(tiny_image, c, TINY, row_en=3)
    got, got_mem, _ = compare.run_rtl(tiny_image, c, TINY, name="vector", row_en=3)
    assert compare.compare(ref, got, c, ref_mem, got_mem) == []
    last = got[-1]
    assert any(v != 0 for v in last.vsram[0]) and any(v != 0 for v in last.vsram[1])
    assert last.vsram[0] != last.vsram[1], "the two rows started from different banks"
    assert last.events["SAT_VPU"] == ref[-1].events["SAT_VPU"] > 0


def test_a_gemv_with_no_inputs_matches_the_hardware(demo_image) -> None:
    """A GEMV with K == 0 is zero work in both models: nothing written, no MACs, no weight bytes."""
    needs_verilator()
    c = compare.bringup(demo_image, DEMO, tok=6)
    embed, quant, lm, halt = c.program
    program = (embed, quant, dataclasses.replace(lm, k=0), halt)
    zero = dataclasses.replace(c, program=program, blob=isa.assemble(list(program)))
    ref, ref_mem = compare.reference(demo_image, zero, DEMO)
    got, got_mem, _ = compare.run_rtl(demo_image, zero, DEMO, name="zero")
    assert compare.compare(ref, got, zero, ref_mem, got_mem) == []
    last = ref[-1]
    assert last.perf["DESCRIPTORS"] == 4 and last.perf["MACS"] == 0
    assert last.perf["WT_BYTES"] == embed.k + isa.META_BYTES  # the EMBED gather alone
    assert last.argmax_tok == 0, "the GEMV writes no logits, so ARGMAX_TOK stays as it was"


def test_the_compare_command_runs_the_comparison(tiny_image, capsys) -> None:
    """``quettos compare`` takes the options of ``python -m quettos.compare`` and runs it."""
    parsed = cli.build_parser().parse_args(
        ["compare", "--sweep", "--shapes", "2", "--widths", "16"]
    )
    assert parsed.sweep and parsed.shapes == 2 and parsed.widths == "16"
    assert parsed.programs == "bringup,vector,attention,layer"
    needs_verilator()
    assert (
        cli.main(
            [
                "compare",
                "--image",
                str(tiny_image),
                "--wb",
                "16",
                "--tok",
                "3",
                "--programs",
                "bringup,attention",
            ]
        )
        == 0
    )
    assert "2/2 runs match sw/quettos/isa_sim.py" in capsys.readouterr().out


@pytest.mark.parametrize("program", ["bringup", "vector"])
@pytest.mark.parametrize("tok", [0, 42])
def test_rtl_matches_the_simulator_on_the_tiny_configuration(tiny_image, program, tok) -> None:
    """The CI configuration: WB=16, B_MAX=2, at three latencies and a halved bandwidth."""
    needs_verilator()
    result = compare.check_image(
        tiny_image, TINY, tok, program=program, timing=((32, 1), (1, 1), (200, 1), (32, 2))
    )
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []
    if program == "bringup":
        assert result.argmax_tok == tok


@pytest.mark.parametrize("program", ["bringup", "vector"])
def test_rtl_matches_the_simulator_at_the_demo_width(demo_image, program) -> None:
    """The synthesized configuration: WB=64, B_MAX=1."""
    needs_verilator()
    result = compare.check_image(demo_image, DEMO, 11, program=program, timing=((32, 1), (200, 2)))
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []
    if program == "bringup":
        assert result.argmax_tok == 11


@pytest.mark.slow
def test_rtl_matches_the_simulator_over_random_shapes() -> None:
    """Five random tiny shapes at WB 64 and WB 128, each at LAT 1 / 32 / 200 and --bw-div 2."""
    needs_verilator()
    results = compare.sweep(5, [64, 128], seed=0, tokens=1)
    assert len(results) == 40
    bad = [(r.image.name, str(m)) for r in results for m in r.mismatches]
    assert bad == []
    assert all(r.determinism == [] for r in results)
    assert all(r.argmax_tok == r.tok for r in results if r.program == "bringup")
