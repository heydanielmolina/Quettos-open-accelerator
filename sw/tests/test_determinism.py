"""The settings a run can take, what a difference between two runs looks like, and the RTL.

``quettos.determinism`` runs one compiled program on ``qcore_top`` under every
setting that changes the timing or the starting state and nothing else, and
compares what came back with the run at the configuration every measurement is
taken at.  The first half of this file checks the bookkeeping -- how a setting
reaches the build and the simulation kernel, and how a moved value is reported --
on records made here; the second half runs the RTL and skips when Verilator is
not on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from quettos import compare, compiler, determinism
from quettos.compare import Case, Record
from quettos.determinism import Difference, Report, Row, Run, Setting

DEMO = compare.CONFIGS[64]
WIDE = compare.CONFIGS[128]


def needs_verilator() -> None:
    if shutil.which("verilator") is None:
        pytest.skip("verilator is not on PATH")


def record(index: int, vsram: list[int], sreg: list[int] | None = None) -> Record:
    """One descriptor's worth of state, with everything but the banks held fixed."""
    return Record(
        pass_index=0,
        pos=0,
        index=index,
        pc=0x1000,
        status=0,
        argmax_tok=7,
        argmax_val=11,
        events=dict.fromkeys(compare.EVENTS, 0),
        perf=dict.fromkeys(compare.PERF_COMPARED, 1),
        vsram=[list(vsram)],
        sreg=[list(sreg if sreg is not None else [0] * 32)],
    )


def case(descriptors: int = 1, count: int = 4) -> Case:
    """A case whose only purpose is to name the descriptors and the dumped range."""
    from quettos import isa

    program = tuple(isa.halt() for _ in range(descriptors))
    return Case(
        program=program,
        addr=0x1000,
        blob=isa.assemble(list(program)),
        ranges=((0, 0, count),),
    )


def run_of(setting: Setting, records: list[Record], ids: list[int] | None = None) -> Run:
    return Run(setting=setting, cycles=1000, records=records, mem=[], ids=list(ids or []))


@pytest.fixture
def tiny_case() -> tuple[Case, Run, Run]:
    """A one-descriptor case and the two baseline runs of it, made here rather than on the RTL."""
    c = case(descriptors=1, count=4)
    return (
        c,
        run_of(determinism.BASELINE, [record(0, [1, 2, 3, 4])]),
        run_of(determinism.BASELINE, [], ids=[8]),
    )


# --------------------------------------------------------------------------- the settings


def test_the_baseline_is_the_configuration_every_measurement_is_taken_at() -> None:
    """`--lat 32 --bw-div 1`, one thread, and the zero start every measured run builds with."""
    assert (determinism.BASELINE.lat, determinism.BASELINE.bw_div) == (32, 1)
    assert determinism.BASELINE.threads == 1
    assert determinism.BASELINE.make_args == ["THREADS=1", "XINIT=fast"]
    assert determinism.BASELINE.plusargs == [], "a fast build takes no runtime argument"


def test_the_timing_settings_move_the_memory_model_and_the_thread_count() -> None:
    """Latency 1 and 200, one beat every two cycles, and four threads -- nothing else."""
    by_name = {s.name: s for s in determinism.TIMING}
    assert set(by_name) == {"lat 1", "lat 200", "bw-div 2", "threads 4"}
    assert by_name["lat 1"].lat == 1 and by_name["lat 200"].lat == 200
    assert by_name["bw-div 2"].bw_div == 2 and by_name["bw-div 2"].lat == 32
    assert by_name["threads 4"].make_args == ["THREADS=4", "XINIT=fast"]
    assert all(s.x_initial == "fast" for s in determinism.TIMING)


def test_an_undefined_start_builds_unique_and_names_its_value() -> None:
    """`XINIT=unique` plus the two plusargs that say what a variable starts at."""
    s = Setting("x", x_initial="unique", rand_reset=2, seed=7)
    assert s.make_args == ["THREADS=1", "XINIT=unique"]
    assert s.plusargs == ["+verilator+rand+reset+2", "+verilator+seed+7"]


def test_the_undefined_starts_are_zeros_ones_and_one_per_seed() -> None:
    """Every value Verilator's `unique` initialization can take, and a seed for the random one."""
    settings = determinism.x_initial((4, 9))
    assert [s.rand_reset for s in settings] == [0, 1, 2, 2]
    assert [s.seed for s in settings[2:]] == [4, 9]
    assert all(s.x_initial == "unique" for s in settings)
    assert len(determinism.x_initial()) == 2 + len(determinism.X_SEEDS)


def test_a_setting_names_a_file_the_shell_takes() -> None:
    assert determinism._tag(Setting("x-init seed 12")) == "x-init-seed-12"
    assert determinism._tag(Setting("bw-div 2")) == "bw-div-2"


def test_a_stopped_run_carries_the_line_it_stopped_on_without_the_repository_path() -> None:
    """What the simulator printed is the report, so a stop names the file and the line."""
    line = f"%Error: {determinism.REPO}/rtl/qcore_row.sv:254: Assertion failed"
    proc = subprocess.CompletedProcess(["qcore_sim"], 1, stdout=f"{line}\n", stderr="")
    assert determinism._stopped(proc) == "%Error: rtl/qcore_row.sv:254: Assertion failed"
    quiet = subprocess.CompletedProcess(["qcore_sim"], 3, stdout="", stderr="")
    assert determinism._stopped(quiet) == "qcore_sim exited 3"


# --------------------------------------------------------------------------- the comparison


def test_the_first_id_that_moved_is_the_one_reported() -> None:
    assert determinism.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert determinism.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert determinism.first_difference([1, 2], [1, 2, 3]) == 2


def test_a_moved_id_is_reported_with_its_index_and_both_values() -> None:
    base = run_of(determinism.BASELINE, [], ids=[4, 5, 6])
    got = run_of(Setting("lat 1", lat=1), [], ids=[4, 7, 6])
    (d,) = determinism.differences(base, got)
    assert (d.setting, d.what, d.element) == ("lat 1", "generated id", 1)
    assert (d.baseline, d.got) == (5, 7)
    assert "5 at the baseline, 7 here" in str(d)


def test_a_moved_vsram_element_is_reported_with_the_descriptor_it_is_in() -> None:
    """The comparison is compare.compare, so a difference names the descriptor and the element."""
    c = case(descriptors=2, count=4)
    base = run_of(determinism.BASELINE, [record(0, [1, 2, 3, 4]), record(1, [1, 2, 3, 4])])
    got = run_of(Setting("lat 200", lat=200), [record(0, [1, 2, 3, 4]), record(1, [1, 2, 9, 4])])
    (d,) = determinism.differences(base, got, c)
    assert d.setting == "lat 200" and d.index == 1 and d.element == 2
    assert (d.baseline, d.got) == (3, 9)
    assert "VSRAM[0] element 2" in str(d)


def test_two_runs_that_agree_report_nothing() -> None:
    c = case(descriptors=1, count=4)
    base = run_of(determinism.BASELINE, [record(0, [1, 2, 3, 4])], ids=[8])
    got = run_of(Setting("threads 4", threads=4), [record(0, [1, 2, 3, 4])], ids=[8])
    assert determinism.differences(base, got, c) == []


def test_an_undefined_start_is_not_held_to_storage_the_program_did_not_write() -> None:
    """The vector SRAM and the scale registers hold whatever the start put in them."""
    c = case(descriptors=1, count=4)
    s = Setting("x-init seed 1", x_initial="unique", rand_reset=2)
    base_case = run_of(determinism.BASELINE, [record(0, [1, 2, 3, 4], sreg=[0] * 32)])
    base_gen = run_of(determinism.BASELINE, [], ids=[8])
    banked = run_of(s, [record(0, [1, 2, 9, 4], sreg=[5] + [0] * 31)])
    gen = run_of(s, [], ids=[8])
    assert determinism._row(base_case, base_gen, banked, gen, c, banks=True).differences
    assert determinism._row(base_case, base_gen, banked, gen, c, banks=False).ok

    moved = run_of(s, [], ids=[9])
    row = determinism._row(base_case, base_gen, banked, moved, c, banks=False)
    assert [d.what for d in row.differences] == ["generated id"], "an id is still compared"


def test_what_an_undefined_start_is_still_held_to_is_measured(tiny_case) -> None:
    """The per-descriptor fields the bank filter leaves, from the baseline's own records.

    The generated ids are compared too, by the generation run rather than the
    stepped one, which is why they are not in this set: a filter cannot strip
    them, so counting them here would make the guard unable to see a state
    comparison that has become a no-op.
    """
    c, base_case, base_gen = tiny_case
    kept = determinism.kept_fields(base_case, base_gen, c)
    assert {"PC", "STATUS", "ARGMAX_TOK", "ARGMAX_VAL"} <= kept
    assert {"SAT_REQ", "ERR_BOUNDS", "PERF MACS", "PERF DESCRIPTORS"} <= kept
    assert "generated id" not in kept, "the ids are the generation run's, not the filter's"
    assert not any(f.startswith(determinism.BANK_FIELDS) for f in kept), "the banks are filtered"
    ids = determinism.differences(base_gen, determinism._moved(base_gen))
    assert [d.what for d in ids] == ["generated id"], "and the ids are compared regardless"


def test_a_bank_filter_that_covers_the_comparison_leaves_nothing(tiny_case, monkeypatch) -> None:
    """The no-op the guard exists for: a filter that swallows every field it is given."""
    c, base_case, base_gen = tiny_case
    monkeypatch.setattr(determinism, "BANK_FIELDS", ("",))
    assert determinism.kept_fields(base_case, base_gen, c) == frozenset()


def test_a_row_says_what_happened_and_a_report_is_only_ok_when_every_row_is() -> None:
    ok = Row("lat 1", 10, [3])
    stopped = Row("x-init ones", 0, stopped="%Error: rtl/qcore_row.sv:254: Assertion failed")
    moved = Row(
        "lat 200", 20, [4], differences=[Difference("lat 200", 0, 0, "GEMV", "PC", None, 1, 2)]
    )
    assert (ok.result, stopped.result, moved.result) == ("match", "STOPPED", "DIFFERS")
    assert Report("timing", "m", ok, [ok]).ok
    assert not Report("timing", "m", ok, [ok, stopped]).ok
    assert not Report("timing", "m", ok, [ok, moved]).ok
    assert not Report("timing", "m", stopped, []).ok


def test_the_width_check_leaves_out_only_the_padded_traffic() -> None:
    """A partial last tile is padded to the width, so only the two traffic counters are its own."""
    assert determinism.WIDTH_EXEMPT == frozenset({"PERF MACS", "PERF WT_BYTES"})
    assert not {"PC", "STATUS", "ARGMAX_TOK", "PERF DESCRIPTORS"} & determinism.WIDTH_EXEMPT
    assert not any(f.startswith(("VSRAM", "SREG")) for f in determinism.WIDTH_EXEMPT)


# --------------------------------------------------------------------------- the images


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict[int, Path]:
    """One random tiny model compiled at WB 64 and WB 128 from the same weights."""
    return determinism.tiny_images(tmp_path_factory.mktemp("det"), seed=0)


def test_the_two_widths_are_one_model(tiny) -> None:
    """Same weights, same descriptors, same context, different tiling."""
    layouts = {wb: compiler.load_layout(tiny[wb]) for wb in tiny}
    assert set(layouts) == {64, 128}
    assert layouts[64]["wb"] == 64 and layouts[128]["wb"] == 128
    assert layouts[64]["model"] == layouts[128]["model"]
    assert layouts[64]["frac"] == layouts[128]["frac"]
    assert layouts[64]["max_ctx"] == layouts[128]["max_ctx"] == 2 * WIDE.wb
    assert layouts[64]["vsram"]["map"] == layouts[128]["vsram"]["map"], (
        "one context is one VSRAM map, so an element means the same thing at both widths"
    )


def test_the_width_cases_dump_the_same_elements_at_both_widths(tiny) -> None:
    """One int32 per vocabulary entry, and the VSRAM window the shorter compile has."""
    cases = determinism.width_cases(tiny, tok=0)
    counts = {c.ranges[0][2] for c in cases.values()}
    sizes = {c.mem[0][1] for c in cases.values()}
    assert len(counts) == 1 and len(sizes) == 1
    vocab = compiler.load_layout(tiny[64])["model"]["vocab"]
    assert sizes == {4 * vocab}
    assert all(len(c.program) == 4 for c in cases.values())


def test_the_layer_width_case_runs_the_positions_both_widths_name(tiny) -> None:
    """A whole decoder layer at the shared positions, and no region the tiling moves."""
    cases = determinism.width_cases(tiny, tok=0, program="layer")
    passes = {c.passes for c in cases.values()}
    assert len(passes) == 1, "both widths embed the same tokens at the same positions"
    (shared,) = passes
    at = [p for _, p in shared]
    assert at == sorted(set(at)) and len(at) >= 3
    assert set(at) <= set(compare.positions(2 * WIDE.wb, 64))
    assert set(at) <= set(compare.positions(2 * WIDE.wb, 128))
    assert 0 in at and 2 * WIDE.wb - 1 in at, "the first position and the last the cache holds"
    assert all(c.mem == () for c in cases.values()), (
        "the KV cache is stored in the port width's own tiling, so its bytes are not compared"
    )
    assert len({len(c.program) for c in cases.values()}) == 1
    assert all(len(c.program) > 4 for c in cases.values())


# --------------------------------------------------------------------------- the checks


def stub_rtl(
    monkeypatch,
    case_cycles: dict[int, int],
    gen_cycles: dict[int, int],
    ids: tuple[int, ...] = (5, 6),
) -> None:
    """The bookkeeping of a check without the RTL under it.

    Both runs come back with the records the case asks for and the cycle count
    the width is given, so what a report carries is a function of what it was
    handed rather than of a simulation.
    """

    def records_for(c: Case) -> list[Record]:
        return [
            Record(
                pass_index=p,
                pos=pos,
                index=i,
                pc=0x1000 + 32 * i,
                status=0,
                argmax_tok=3,
                argmax_val=4,
                events=dict.fromkeys(compare.EVENTS, 0),
                perf=dict.fromkeys(compare.PERF_COMPARED, 7),
                vsram=[[0] * count for _, _, count in c.ranges],
                sreg=[[0] * 32],
            )
            for p, (_, pos) in enumerate(c.passes)
            for i in range(len(c.program))
        ]

    def run_case(image_dir, c, cfg, setting, **kw) -> Run:
        return Run(
            setting=setting,
            cycles=case_cycles[cfg.wb],
            records=records_for(c),
            mem=[[[0] * 4 for _ in c.mem] for _ in c.passes],
        )

    def run_generate(image_dir, cfg, setting, **kw) -> Run:
        return Run(setting=setting, cycles=gen_cycles[cfg.wb], ids=list(ids))

    monkeypatch.setattr(determinism, "run_case", run_case)
    monkeypatch.setattr(determinism, "run_generate", run_generate)


def test_the_width_report_prints_the_cycles_of_the_program_its_row_names(tiny, monkeypatch) -> None:
    """A row names a program, so the cycles beside it are that program's own run."""
    stub_rtl(monkeypatch, case_cycles={64: 1111, 128: 2222}, gen_cycles={64: 3333, 128: 4444})
    r = determinism.check_width(tiny, program="bringup", max_new=2)
    assert r.ok, "\n".join(str(d) for row in r.all_rows for d in row.differences)
    assert [row.setting for row in r.all_rows] == ["WB 64", "WB 128"]
    assert [row.cycles for row in r.all_rows] == [1111, 2222], "the bring-up program's own runs"
    assert [row.ids for row in r.all_rows] == [[5, 6], [5, 6]]


def test_a_check_whose_filter_leaves_nothing_to_compare_is_refused(tiny, monkeypatch) -> None:
    """The guard: a bank-free comparison that judges a run on nothing does not run at all.

    It is the counterpart of the width check's, where two widths that share no
    position of a program raise rather than comparing an empty set of passes.
    """
    stub_rtl(monkeypatch, case_cycles={64: 10, 128: 20}, gen_cycles={64: 30, 128: 40})
    settings = determinism.x_initial((1,))
    kw = {"name": "x-initial", "program": "bringup", "banks": False, "max_new": 2}
    ok = determinism.check(tiny[64], DEMO, settings, **kw)
    assert ok.ok and len(ok.rows) == len(settings)
    monkeypatch.setattr(determinism, "BANK_FIELDS", ("",))
    with pytest.raises(ValueError, match="hold a run to nothing"):
        determinism.check(tiny[64], DEMO, settings, **kw)


def test_a_filter_that_leaves_only_the_ids_is_refused(tiny, monkeypatch) -> None:
    """A filter that strips the per-descriptor state, but not the ids, is still a no-op.

    The stepped state run is what a bank-free comparison judges; the generation
    run compares ids alone, so measuring it too would keep the surviving set
    non-empty under any filter short of the empty prefix and hide exactly the
    comparison this guard exists to refuse.
    """
    stub_rtl(monkeypatch, case_cycles={64: 10, 128: 20}, gen_cycles={64: 30, 128: 40})
    settings = determinism.x_initial((1,))
    kw = {"name": "x-initial", "program": "bringup", "banks": False, "max_new": 2}
    every_state_field = (
        "VSRAM",
        "SREG",
        "PC",
        "STATUS",
        "ARGMAX",
        "SAT_",
        "ERR_",
        "PERF",
        "mem",
        "memory word",
    )
    monkeypatch.setattr(determinism, "BANK_FIELDS", every_state_field)
    with pytest.raises(ValueError, match="hold a run to nothing"):
        determinism.check(tiny[64], DEMO, settings, **kw)


# --------------------------------------------------------------------------- the RTL


def test_the_timing_does_not_move_a_value(tiny) -> None:
    """Latency 1 / 32 / 200, half bandwidth and four threads, over a whole decoder layer."""
    needs_verilator()
    r = determinism.check_timing(tiny[64], DEMO, program="layer", max_new=2)
    assert r.ok, "\n".join(str(d) for row in r.all_rows for d in row.differences)
    assert [row.result for row in r.all_rows] == ["match"] * 5
    cycles = {row.setting: row.cycles for row in r.all_rows}
    assert cycles["lat 1"] < cycles["lat 32"] < cycles["lat 200"]
    assert cycles["bw-div 2"] > cycles["lat 32"], "half the bandwidth costs cycles"
    assert cycles["threads 4"] == cycles["lat 32"], "threads partition the eval, not the design"
    assert len(set(r.baseline.ids)) >= 1 and r.baseline.ids == r.rows[0].ids


def test_the_two_port_widths_produce_the_same_values(tiny) -> None:
    """WB 64 against WB 128 on one model: the ids and every dumped logit, at different cycles."""
    needs_verilator()
    r = determinism.check_width(tiny, max_new=2)
    assert r.ok, "\n".join(str(d) for row in r.all_rows for d in row.differences)
    assert [row.setting for row in r.all_rows] == ["WB 64", "WB 128"]
    assert r.baseline.ids == r.rows[0].ids and r.baseline.ids
    assert r.baseline.cycles != r.rows[0].cycles, "the widths read the weights in different beats"


def test_the_two_port_widths_agree_over_a_whole_decoder_layer(tiny) -> None:
    """The layer program at both widths: every VSRAM element and scale register, per descriptor."""
    needs_verilator()
    r = determinism.check_width(tiny, program="layer", max_new=2)
    assert r.ok, "\n".join(str(d) for row in r.all_rows for d in row.differences)
    assert r.subject.endswith("layer")
    assert [row.setting for row in r.all_rows] == ["WB 64", "WB 128"]
    assert r.baseline.ids == r.rows[0].ids and r.baseline.ids


def test_the_padded_traffic_is_the_only_thing_the_two_widths_disagree_on(tiny) -> None:
    """What the exemption covers is measured, not assumed: it is the whole of the difference."""
    needs_verilator()
    cases = determinism.width_cases(tiny, 0, program="layer")
    runs = {
        wb: determinism.run_case(
            tiny[wb], cases[wb], compare.CONFIGS[wb], determinism.BASELINE, name="layer"
        )
        for wb in (64, 128)
    }
    found = {d.what for d in determinism.differences(runs[64], runs[128], cases[64], label="w")}
    assert found <= determinism.WIDTH_EXEMPT, sorted(found - determinism.WIDTH_EXEMPT)
    assert found, "a partial tile has to make the traffic counters differ somewhere"


def _flipped(image_dir: Path, region: str, where: Path) -> Path:
    """A copy of ``image_dir`` with one byte of ``region`` inverted."""
    layout = compiler.load_layout(image_dir)
    shutil.copytree(image_dir, where)
    image = where / layout["image"]["file"]
    at = {r["name"]: r for r in layout["regions"]}[region]["addr"]
    buf = bytearray(image.read_bytes())
    buf[at] ^= 0x01
    image.write_bytes(bytes(buf))
    return where


def test_the_width_check_reports_a_flipped_embedding_byte(tiny, tmp_path) -> None:
    """The counterpart for the bring-up program: a byte of the row the token gathers."""
    needs_verilator()
    hurt = _flipped(tiny[128], "embed", tmp_path / "hurt-embed")
    r = determinism.check_width({64: tiny[64], 128: hurt}, max_new=2)
    assert not r.ok
    assert any(d.what.startswith("memory word") for d in r.rows[0].differences), (
        "the dumped logits are what the bring-up width comparison rests on"
    )


def test_the_layer_width_check_reports_a_flipped_weight_byte(tiny, tmp_path) -> None:
    """The counterpart for the layer program: one byte of a layer-0 gamma row, at its element."""
    needs_verilator()
    hurt = _flipped(tiny[128], "layer.0.gamma_in", tmp_path / "hurt-gamma")
    r = determinism.check_width({64: tiny[64], 128: hurt}, program="layer", max_new=2)
    assert not r.ok
    moved = [d for d in r.rows[0].differences if d.what.startswith("VSRAM")]
    assert moved, "\n".join(str(d) for d in r.rows[0].differences[:4])
    assert moved[0].element is not None and moved[0].baseline != moved[0].got
