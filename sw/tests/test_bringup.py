"""The bring-up program: its shape, its self-check on the ISA simulator, and the RTL comparison.

``quettos.compiler.build_bringup`` assembles ``EMBED`` / ``GEMV`` in ARGMAX mode
/ ``HALT`` over a random tiny model; ``quettos.compare`` runs it on
``quettos.isa_sim`` and on ``qcore_top`` through the Verilator harness and
compares every VSRAM element, SREG word, CSR and PERF counter.  The RTL tests
skip when Verilator is not on ``PATH``.
"""

from __future__ import annotations

import dataclasses
import shutil

import pytest
from quettos import cli, compare, compiler, isa, synthetic
from quettos.isa import Opcode, OutMode

TINY = compare.CONFIGS[16]
DEMO = compare.CONFIGS[64]


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


def test_the_program_is_embed_gemv_halt(demo_image) -> None:
    """Three descriptors, 96 bytes, loaded behind prefill.prog with one host-supplied scale."""
    b = compiler.build_bringup(demo_image, tok=3, **DEMO.widths)
    assert [d.opcode for d in b.program] == [Opcode.EMBED, Opcode.GEMV, Opcode.HALT]
    assert len(b.blob) == 3 * isa.DESC_BYTES
    assert isa.parse(b.blob) == list(b.program)
    assert b.addr % isa.PROGRAM_ALIGN == 0
    embed, gemv, _ = b.program
    assert embed.vs_dst == gemv.vs_src, "the EMBED writes the slot the GEMV reads"
    assert gemv.out_mode is OutMode.ARGMAX_DUMP and gemv.imm32 == b.dump[0]
    assert b.dump == (gemv.imm32, 4 * gemv.n)
    assert len(b.sreg) == 1 and b.sreg[0][:2] == (0, gemv.sreg_src)
    assert isa.sfloat_from_imm(b.sreg[0][2]).m == 1 << 15


def test_the_embed_output_fits_the_activation_window(demo_image) -> None:
    """The shift the program raises keeps the gathered row inside the int16 a GEMV reads."""
    for tok in (0, 1, 17, 127):
        b = compiler.build_bringup(demo_image, tok=tok, **DEMO.widths)
        records, _ = compare.reference(demo_image, b, DEMO)
        embed = b.program[0]
        written = records[0].vsram[0][embed.vs_dst : embed.vs_dst + embed.k]
        assert max(abs(v) for v in written) <= 32767, f"token {tok} needs a larger shift"


def test_argmax_is_the_input_token(demo_image) -> None:
    """A tied embedding makes the largest logit of row TOK its own row, for every token."""
    vocab = synthetic.SHAPES[0].vocab
    for tok in range(vocab):
        b = compiler.build_bringup(demo_image, tok=tok, **DEMO.widths)
        records, _ = compare.reference(demo_image, b, DEMO)
        last = records[-1]
        assert last.argmax_tok == tok, f"token {tok} -> {last.argmax_tok}"
        assert last.events == dict.fromkeys(compare.EVENTS, 0)
        assert last.perf["DESCRIPTORS"] == 3


def test_the_simulator_counts_what_the_isa_defines(demo_image) -> None:
    """DESCRIPTORS, MACS and WT_BYTES of the three descriptors, from the shapes alone."""
    shape = synthetic.SHAPES[0]
    b = compiler.build_bringup(demo_image, tok=5, **DEMO.widths)
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
    b = compiler.build_bringup(demo_image, tok=9, **DEMO.widths)
    records, mem = compare.reference(demo_image, b, DEMO)
    doctored = list(mem)
    doctored[7] += 1
    bad = compare.compare(records, records, b, mem, doctored)
    assert len(bad) == 1
    assert bad[0].element == 7 and bad[0].opcode == "GEMV"
    assert str(bad[0]).endswith(f"isa_sim {mem[7]}, RTL {mem[7] + 1}")
    assert compare.compare(records, records, b, mem, mem) == []


def test_a_vector_opcode_stops_the_run_with_the_fault_in_status(demo_image) -> None:
    """qcore_top carries no vector unit, so a vector descriptor halts with FAULT = OPCODE on it."""
    needs_verilator()
    b = compiler.build_bringup(demo_image, tok=2, **DEMO.widths)
    vector = isa.decode(bytes([int(Opcode.VRMSNORM)]) + bytes(isa.DESC_BYTES - 1))
    program = (b.program[0], vector, isa.halt())
    bad = dataclasses.replace(b, program=program, blob=isa.assemble(list(program)))
    records, _, _ = compare.run_rtl(demo_image, bad, DEMO, allow_error=True)
    last = records[-1]
    fault, opcode = isa.status_fault(last.status)
    assert fault is isa.Fault.OPCODE and opcode == int(Opcode.VRMSNORM)
    assert last.pc == bad.addr + isa.DESC_BYTES, "PC names the descriptor that faulted"
    assert last.perf["DESCRIPTORS"] == 1, "nothing is counted for the refused descriptor"


def test_a_gemv_with_no_inputs_matches_the_hardware(demo_image) -> None:
    """A GEMV with K == 0 is zero work in both models: nothing written, no MACs, no weight bytes."""
    needs_verilator()
    b = compiler.build_bringup(demo_image, tok=6, **DEMO.widths)
    embed, lm, halt = b.program
    program = (embed, dataclasses.replace(lm, k=0), halt)
    zero = dataclasses.replace(b, program=program, blob=isa.assemble(list(program)))
    ref, ref_mem = compare.reference(demo_image, zero, DEMO)
    got, got_mem, _ = compare.run_rtl(demo_image, zero, DEMO)
    assert compare.compare(ref, got, zero, ref_mem, got_mem) == []
    last = ref[-1]
    assert last.perf["DESCRIPTORS"] == 3 and last.perf["MACS"] == 0
    assert last.perf["WT_BYTES"] == embed.k + isa.META_BYTES  # the EMBED gather alone
    assert last.argmax_tok == 0, "the GEMV writes no logits, so ARGMAX_TOK stays as it was"


def test_the_compare_command_runs_the_comparison(tiny_image, capsys) -> None:
    """``quettos compare`` takes the options of ``python -m quettos.compare`` and runs it."""
    parsed = cli.build_parser().parse_args(
        ["compare", "--sweep", "--shapes", "2", "--widths", "16"]
    )
    assert parsed.sweep and parsed.shapes == 2 and parsed.widths == "16"
    needs_verilator()
    assert cli.main(["compare", "--image", str(tiny_image), "--wb", "16", "--tok", "3"]) == 0
    assert "1/1 runs match sw/quettos/isa_sim.py" in capsys.readouterr().out


@pytest.mark.parametrize("tok", [0, 42])
def test_rtl_matches_the_simulator_on_the_tiny_configuration(tiny_image, tok) -> None:
    """The CI configuration: WB=16, B_MAX=2, at three latencies and a halved bandwidth."""
    needs_verilator()
    result = compare.check_image(tiny_image, TINY, tok, timing=((32, 1), (1, 1), (200, 1), (32, 2)))
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []
    assert result.argmax_tok == tok


def test_rtl_matches_the_simulator_at_the_demo_width(demo_image) -> None:
    """The synthesized configuration: WB=64, B_MAX=1."""
    needs_verilator()
    result = compare.check_image(demo_image, DEMO, 11, timing=((32, 1), (200, 2)))
    assert result.mismatches == [], "\n".join(str(m) for m in result.mismatches)
    assert result.determinism == []
    assert result.argmax_tok == 11


@pytest.mark.slow
def test_rtl_matches_the_simulator_over_random_shapes() -> None:
    """Five random tiny shapes at WB 64 and WB 128, each at LAT 1 / 32 / 200 and --bw-div 2."""
    needs_verilator()
    results = compare.sweep(5, [64, 128], seed=0, tokens=1)
    assert len(results) == 10
    bad = [(r.image.name, str(m)) for r in results for m in r.mismatches]
    assert bad == []
    assert all(r.determinism == [] for r in results)
    assert all(r.argmax_tok == r.tok for r in results)
