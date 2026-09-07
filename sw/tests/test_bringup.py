"""The two comparison programs: their shape, their self-check on the ISA simulator, and the RTL.

``quettos.compare.bringup`` assembles the four descriptors ``docs/ISA.md``
defines -- ``EMBED``, ``VQUANT``, the tied LM head as a ``GEMV`` in ARGMAX mode
and ``HALT`` -- and ``quettos.compare.vector_ops`` a directed program that runs
the vector unit's four opcodes back to back.  ``quettos.compare`` runs each on
``quettos.isa_sim`` and on ``qcore_top`` through the Verilator harness and
compares every VSRAM element, SREG word, CSR and PERF counter.  The RTL tests
skip when Verilator is not on ``PATH``.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest
from quettos import cli, compare, isa, synthetic
from quettos.isa import Opcode, OutMode, VquantFlag

TINY = compare.CONFIGS[16]
DEMO = compare.CONFIGS[64]

#: The opcodes qcore_vpu_top executes, and the two it refuses.
EXECUTED = (Opcode.VRMSNORM, Opcode.VQUANT, Opcode.VSILUMUL, Opcode.VSUBC)
REFUSED = (Opcode.VROPE, Opcode.VSOFTMAX)


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


@pytest.fixture(scope="module")
def demo_image(tmp_path_factory) -> object:
    """A one-layer synthetic model compiled for the demo width."""
    return compare.compile_shape(synthetic.SHAPES[0], 0, DEMO, tmp_path_factory.mktemp("img-w64"))


@pytest.fixture(scope="module")
def tiny_image(tmp_path_factory) -> object:
    """The same model compiled for the tiny width, the configuration CI runs."""
    return compare.compile_shape(synthetic.SHAPES[0], 0, TINY, tmp_path_factory.mktemp("img-w16"))


# --------------------------------------------------------------------------- the programs


def test_the_program_is_embed_vquant_gemv_halt(demo_image) -> None:
    """Four descriptors, 128 bytes, loaded behind prefill.prog, with nothing loaded by the host."""
    b = compare.bringup(demo_image, DEMO, tok=3)
    assert [d.opcode for d in b.program] == [
        Opcode.EMBED,
        Opcode.VQUANT,
        Opcode.GEMV,
        Opcode.HALT,
    ]
    assert len(b.blob) == 4 * isa.DESC_BYTES
    assert isa.parse(b.blob) == list(b.program)
    assert b.addr % isa.PROGRAM_ALIGN == 0
    assert b.sreg == (), "the program writes its own activation scale"
    embed, quant, gemv, _ = b.program
    assert embed.vs_dst == quant.vs_src, "the VQUANT reads the slot the EMBED wrote"
    assert quant.vs_dst == gemv.vs_src, "the GEMV reads the slot the VQUANT wrote"
    assert quant.n == embed.k and not quant.flags & VquantFlag.W8
    assert gemv.out_mode is OutMode.ARGMAX_DUMP and gemv.imm32 == b.dump[0]
    assert b.dump == (gemv.imm32, 4 * gemv.n)


def test_the_vquant_writes_the_scale_the_gemv_reads(demo_image) -> None:
    """No SREG is loaded from the host: descriptor 1 leaves the scale descriptor 2 reads."""
    b = compare.bringup(demo_image, DEMO, tok=4)
    _, quant, gemv, _ = b.program
    assert quant.sreg_dst == gemv.sreg_src
    records, _ = compare.reference(demo_image, b, DEMO)
    assert records[0].sreg[0][quant.sreg_dst] == 0, "the bank holds nothing before the VQUANT"
    assert records[1].sreg[0][quant.sreg_dst] != 0, "the VQUANT wrote the activation scale"
    assert records[2].sreg[0][gemv.sreg_src] == records[1].sreg[0][quant.sreg_dst]


def test_the_vector_program_runs_the_four_executed_opcodes(demo_image) -> None:
    """VRMSNORM, VQUANT with USE_TRACKED, VSUBC, VSILUMUL and a GROUP VQUANT, back to back."""
    b = compare.vector_ops(demo_image, DEMO, tok=6)
    ops = [d.opcode for d in b.program]
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
    assert set(EXECUTED) <= set(ops), "every opcode the vector unit executes appears"
    assert not any(op in REFUSED for op in ops)
    tracked, grouped = b.program[2], b.program[5]
    assert tracked.flags & VquantFlag.USE_TRACKED and tracked.sreg_src == b.program[1].sreg_dst
    assert grouped.flags & (VquantFlag.GROUP | VquantFlag.W8)
    assert b.program[6].sh1 == 0, "the last VSILUMUL leaves its product unshifted"
    assert b.dump[1] == 0, "the vector program dumps no memory region"


def test_the_vector_program_carries_a_value_through_the_sreg_bank(demo_image) -> None:
    """The tracked absmax the VRMSNORM writes is what the VQUANT after it quantizes with."""
    b = compare.vector_ops(demo_image, DEMO, tok=6)
    rms, tracked = b.program[1], b.program[2]
    records, _ = compare.reference(demo_image, b, DEMO)
    absmax = records[1].sreg[0][rms.sreg_dst]
    assert absmax > 0, "the VRMSNORM tracked an absmax"
    assert records[2].sreg[0][tracked.sreg_dst] != 0, "the VQUANT wrote a scale from it"
    written = records[4].vsram[0][b.program[4].vs_dst : b.program[4].vs_dst + rms.n]
    assert any(v < 0 for v in written) and any(v > 0 for v in written)


def test_the_vector_program_puts_a_value_in_every_counter_it_can_reach(demo_image) -> None:
    """The last VSILUMUL saturates, so SAT_VPU is a number both models have to agree on."""
    b = compare.vector_ops(demo_image, DEMO, tok=6)
    records, _ = compare.reference(demo_image, b, DEMO)
    assert records[5].events["SAT_VPU"] == 0, "nothing saturates before the last VSILUMUL"
    saturated = records[6].events["SAT_VPU"]
    assert 0 < saturated < b.program[6].n, f"{saturated} of {b.program[6].n} elements saturated"
    assert "vector" in compare.EVENTFUL, "so the harness reports the events instead of failing"


def test_argmax_is_the_input_token(demo_image) -> None:
    """A tied embedding makes the largest logit of row TOK its own row, for every token."""
    vocab = synthetic.SHAPES[0].vocab
    for tok in range(vocab):
        b = compare.bringup(demo_image, DEMO, tok=tok)
        records, _ = compare.reference(demo_image, b, DEMO)
        last = records[-1]
        assert last.argmax_tok == tok, f"token {tok} -> {last.argmax_tok}"
        assert last.events == dict.fromkeys(compare.EVENTS, 0)
        assert last.perf["DESCRIPTORS"] == 4


def test_the_simulator_counts_what_the_isa_defines(demo_image) -> None:
    """DESCRIPTORS, MACS and WT_BYTES of the four descriptors, from the shapes alone."""
    shape = synthetic.SHAPES[0]
    b = compare.bringup(demo_image, DEMO, tok=5)
    records, _ = compare.reference(demo_image, b, DEMO)
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
    b = compare.bringup(demo_image, DEMO, tok=9)
    records, mem = compare.reference(demo_image, b, DEMO)
    doctored = list(mem)
    doctored[7] += 1
    bad = compare.compare(records, records, b, mem, doctored)
    assert len(bad) == 1
    assert bad[0].element == 7 and bad[0].opcode == "GEMV"
    assert str(bad[0]).endswith(f"isa_sim {mem[7]}, RTL {mem[7] + 1}")
    assert compare.compare(records, records, b, mem, mem) == []


# --------------------------------------------------------------------------- the RTL


@pytest.mark.parametrize("opcode", REFUSED)
def test_an_opcode_with_no_unit_stops_the_run_with_the_fault_in_status(demo_image, opcode) -> None:
    """VROPE and VSOFTMAX name passes qcore_vpu_top has none, so they halt with FAULT = OPCODE."""
    needs_verilator()
    b = compare.bringup(demo_image, DEMO, tok=2)
    refused = isa.decode(bytes([int(opcode)]) + bytes(isa.DESC_BYTES - 1))
    program = (b.program[0], refused, isa.halt())
    bad = dataclasses.replace(
        b, program=program, blob=isa.assemble(list(program)), dump=(b.dump[0], 0)
    )
    records, _, _ = compare.run_rtl(demo_image, bad, DEMO, allow_error=True, name="refused")
    last = records[-1]
    fault, byte = isa.status_fault(last.status)
    assert fault is isa.Fault.OPCODE and byte == int(opcode)
    assert last.pc == bad.addr + isa.DESC_BYTES, "PC names the descriptor that faulted"
    assert last.perf["DESCRIPTORS"] == 1, "nothing is counted for the refused descriptor"


def test_the_executed_vector_opcodes_reach_the_unit(demo_image) -> None:
    """The counterpart: a program of the four executed opcodes runs to DONE with no fault."""
    needs_verilator()
    b = compare.vector_ops(demo_image, DEMO, tok=2)
    ref, _ = compare.reference(demo_image, b, DEMO)
    got, _, _ = compare.run_rtl(demo_image, b, DEMO, name="vector")
    last = got[-1]
    assert isa.status_fault(last.status) == (isa.Fault.NONE, 0)
    assert last.perf["DESCRIPTORS"] == len(b.program)
    assert last.events == ref[-1].events, "the RTL counts what the simulator counts"
    assert last.events["SAT_VPU"] > 0, "the last VSILUMUL is what makes that a real check"


def cross_row_program(image, cfg, tok: int):
    """The vector program's descriptors with ``src_row = 0`` and ``dst_row = 1``.

    Only a two-row build can run this, and only the top level can get it wrong:
    the vector unit reads bank ``src_row + cur_row`` on both ports and writes
    bank ``dst_row + cur_row`` on port B and in the SREG bank, so a crossbar
    that ignores either base sends the writes to the bank the reads came from.
    """
    b = compare.vector_ops(image, cfg, tok)
    embed, _, _, subc, silu, grouped = b.program[:6]
    across = {"src_row": 0, "dst_row": 1}
    program = (
        embed,
        dataclasses.replace(subc, **across),
        dataclasses.replace(silu, vs_src=embed.vs_dst, vs_aux=embed.vs_dst, **across),
        dataclasses.replace(grouped, vs_src=embed.vs_dst, **across),
        isa.halt(),
    )
    return dataclasses.replace(b, program=program, blob=isa.assemble(list(program)))


def test_the_crossbar_routes_a_descriptor_across_rows(tiny_image) -> None:
    """Reads of bank 0 and writes of bank 1, in the one configuration that has two."""
    needs_verilator()
    assert TINY.b_max == 2
    b = cross_row_program(tiny_image, TINY, tok=5)
    ref, ref_mem = compare.reference(tiny_image, b, TINY)
    got, got_mem, _ = compare.run_rtl(tiny_image, b, TINY, name="cross")
    assert compare.compare(ref, got, b, ref_mem, got_mem) == []
    last = ref[-1]
    assert any(v != 0 for v in last.vsram[1]), "row 1 was written"
    assert last.sreg[1][b.program[3].sreg_dst] != 0, "the scale landed in row 1's bank"
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
    b = dataclasses.replace(base, program=program, blob=isa.assemble(list(program)))
    ref, ref_mem = compare.reference(tiny_image, b, TINY, row_en=3)
    got, got_mem, _ = compare.run_rtl(tiny_image, b, TINY, name="vector", row_en=3)
    assert compare.compare(ref, got, b, ref_mem, got_mem) == []
    last = got[-1]
    assert any(v != 0 for v in last.vsram[0]) and any(v != 0 for v in last.vsram[1])
    assert last.vsram[0] != last.vsram[1], "the two rows started from different banks"
    assert last.events["SAT_VPU"] == ref[-1].events["SAT_VPU"] > 0


def test_a_gemv_with_no_inputs_matches_the_hardware(demo_image) -> None:
    """A GEMV with K == 0 is zero work in both models: nothing written, no MACs, no weight bytes."""
    needs_verilator()
    b = compare.bringup(demo_image, DEMO, tok=6)
    embed, quant, lm, halt = b.program
    program = (embed, quant, dataclasses.replace(lm, k=0), halt)
    zero = dataclasses.replace(b, program=program, blob=isa.assemble(list(program)))
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
    assert parsed.programs == "bringup,vector"
    needs_verilator()
    assert cli.main(["compare", "--image", str(tiny_image), "--wb", "16", "--tok", "3"]) == 0
    assert "2/2 runs match sw/quettos/isa_sim.py" in capsys.readouterr().out


@pytest.mark.parametrize("program", list(compare.PROGRAMS))
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


@pytest.mark.parametrize("program", list(compare.PROGRAMS))
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
    assert len(results) == 20
    bad = [(r.image.name, str(m)) for r in results for m in r.mismatches]
    assert bad == []
    assert all(r.determinism == [] for r in results)
    assert all(r.argmax_tok == r.tok for r in results if r.program == "bringup")
