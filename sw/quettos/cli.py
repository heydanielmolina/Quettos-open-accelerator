"""Command-line entry point: ``quettos <command>`` (installed by ``pyproject.toml``).

Commands: ``download``, ``tokens``, ``export-tokens-bin``, ``calibrate``,
``quantize``, ``compile``, ``golden``, ``isa-sim``, ``compare``, ``check``,
``csr-defs``, ``demo-report``.
Each command is a thin wrapper over the module of the same name; the file
formats they read and write are described in ``docs/``.  ``quantize`` and
``check`` take ``--no-qk-smoothing``, which builds and scores the
K-centering-only variant of the model into the ``-nosmooth`` quality rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quettos.model import load_spec
from quettos.tokenizer_io import prompt_tokens, render_prompt, write_tokens_bin


def _cmd_calibrate(args: argparse.Namespace) -> int:
    import time

    from quettos import calibrate

    spec = load_spec(args.model)
    t0 = time.perf_counter()
    path, report = calibrate.write_calib(spec, args.out)
    elapsed = time.perf_counter() - t0
    summary = {
        "model": spec.repo_id,
        "out": str(path),
        "bytes": path.stat().st_size,
        "tokens": report["tokens"]["count"],
        "tokens_sha256": report["tokens"]["sha256"],
        "absmax": {k: float(f"{v:.6g}") for k, v in report["absmax"].items()},
        "frac": report["frac"],
        "k_centering_gate": {
            k: float(f"{v:.6g}")
            for k, v in report["k_centering_gate"].items()
            if isinstance(v, float)
        },
        "v_scale_spread": {k: report["v_scale_spread"][k] for k in ("p50", "p99", "max")},
        "seconds": round(elapsed, 2),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_quantize(args: argparse.Namespace) -> int:
    import time

    from quettos import calibrate, quantize

    spec = load_spec(args.model)
    calib_path = calibrate.calib_path(spec) if args.calib is None else args.calib
    t0 = time.perf_counter()
    model = quantize.build_quant_model(
        spec, calib_path, layers=args.layers, smoothing=args.smoothing
    )
    t1 = time.perf_counter()
    path = quantize.save(model, args.out)
    t2 = time.perf_counter()
    summary = {
        "model": spec.repo_id,
        "calib": str(calib_path),
        "out": str(path),
        "bytes": path.stat().st_size,
        "layers": model.n_layers,
        "qk_smoothing": model.extra["qk_smoothing"],
        "weight_bytes": quantize.weight_bytes(model),
        "frac": model.frac,
        "eps_c": model.eps_c,
        "sqrt_d": {"m": model.sqrt_d.m, "e": model.sqrt_d.e},
        "log2e_over_8": {"m": model.log2e_over_8.m, "e": model.log2e_over_8.e},
        "build_seconds": round(t1 - t0, 2),
        "save_seconds": round(t2 - t1, 2),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from quettos import calibrate, compiler, quantize

    spec = load_spec(args.model)
    qpath = Path(args.quant) if args.quant else quantize.default_path(spec.name)
    t0 = time.perf_counter()
    if qpath.is_file():
        model = quantize.load(qpath)
        if args.layers is not None:
            if args.layers > model.n_layers:
                print(
                    f"{qpath} holds {model.n_layers} layers, fewer than {args.layers}",
                    file=sys.stderr,
                )
                return 1
            model = compiler.truncate_layers(model, args.layers)
    elif args.layers is not None:
        model = quantize.build_quant_model(spec, calibrate.calib_path(spec), layers=args.layers)
    else:
        print(f"{qpath} not found; run: uv run quettos quantize {args.model}", file=sys.stderr)
        return 1
    suffix = "" if model.n_layers == spec.layers else f"-l{model.n_layers}"
    out = Path(args.out) if args.out else compiler.IMAGES_DIR / f"{spec.name}{suffix}"
    t1 = time.perf_counter()
    result = compiler.compile(
        model,
        spec,
        out_dir=out,
        max_ctx=args.max_ctx,
        wb=args.wb,
        a_bits=args.a_bits,
        prompt=args.prompt,
    )
    t2 = time.perf_counter()
    lay = result.layout
    summary = {
        "model": spec.repo_id,
        "layers": model.n_layers,
        "out": str(out),
        "image_bytes": lay["image"]["size"],
        "image_sha256": lay["image"]["sha256"],
        "descriptors": {k: v["descriptors"] for k, v in lay["programs"].items()},
        "vsram_elements_used": lay["vsram"]["used"],
        "stream_bytes_per_token": {k: lay["traffic"][k]["total"] for k in ("decode", "prefill")},
        "load_seconds": round(t1 - t0, 2),
        "compile_seconds": round(t2 - t1, 2),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_golden(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from quettos import golden, quantize
    from quettos.model import REPO_ROOT
    from quettos.numerics import Stats
    from quettos.tokenizer_io import token_bytes

    spec = load_spec(args.model)
    qpath = Path(args.quant) if args.quant else quantize.default_path(spec.name)
    if not qpath.is_file():
        print(f"{qpath} not found; run: uv run quettos quantize {args.model}", file=sys.stderr)
        return 1
    model = quantize.load(qpath)
    table = token_bytes(spec)
    prompts = args.prompt or [str(REPO_ROOT / f) for f in golden.EXPECTED_PROMPT_FILES]
    stats = Stats()
    clock = [time.perf_counter()]

    def on_token(key: str, j: int, pos: int, tok: int) -> None:
        now = time.perf_counter()
        ms = 1000.0 * (now - clock[0])
        clock[0] = now
        note = "prefill + decode" if j == 0 else "decode"
        text = table[tok].decode("utf-8", errors="replace")
        print(f"{key}: step {j:3d} pos {pos:4d} argmax {tok:6d} {text!r} ({ms:.0f} ms {note})")

    t0 = time.perf_counter()
    report = golden.expected_tokens(
        model,
        spec,
        prompts,
        max_new=args.max_new,
        a_bits=args.a_bits,
        stats=stats,
        on_token=on_token,
    )
    elapsed = time.perf_counter() - t0
    for key, r in report["prompts"].items():
        n = len(r["generated_ids"])
        print(f"{key}: {r['prompt_tokens']} prompt tokens -> {n} generated: {r['text']!r}")
        print(f"  ids {r['generated_ids']} sha256 {r['sha256']}")
    print(f"stats: sat={stats.sat} err_shift={stats.err_shift} clip={stats.clip}")
    print(
        f"model: {model.repo_id} layers={model.n_layers} a_bits={args.a_bits} seconds={elapsed:.1f}"
    )
    if args.write_expected:
        if stats.sat or stats.err_shift:
            print(
                "refusing to write: saturation or shift-error counters are non-zero",
                file=sys.stderr,
            )
            return 1
        path = golden.write_expected_tokens(report, golden.expected_tokens_path(model))
        print(f"wrote {path}")
    return 0


def _isa_sim_prompt(args, spec, layout, programs, image_path, model, table, key, ids) -> int:
    """Generate for one prompt on a fresh machine; returns 1 on any mismatch."""
    import time

    from quettos import golden, isa_sim
    from quettos.model import REPO_ROOT
    from quettos.numerics import Stats

    wb, max_ctx, a_bits = int(layout["wb"]), int(layout["max_ctx"]), int(layout["a_bits"])
    m = isa_sim.Machine.from_file(image_path, wb=wb)
    clock = [time.perf_counter()]
    total = Stats()

    def on_token(j: int, pos: int, tok: int) -> None:
        now = time.perf_counter()
        ms = 1000.0 * (now - clock[0])
        clock[0] = now
        st = m.stats()
        total.sat, total.err_shift, total.clip = (
            total.sat + st.sat,
            total.err_shift + st.err_shift,
            total.clip + st.clip,
        )
        text = table[tok].decode("utf-8", errors="replace")
        print(f"{key}: step {j:3d} pos {pos:4d} argmax {tok:6d} {text!r} ({ms:.0f} ms)")

    t0 = time.perf_counter()
    gen = isa_sim.generate(
        m,
        programs.decode,
        programs.prefill,
        ids,
        args.max_new,
        eos_ids=spec.eos_ids,
        on_token=on_token,
    )
    elapsed = time.perf_counter() - t0
    text = b"".join(table[i] for i in gen).decode("utf-8", errors="replace")
    print(f"{key}: {len(ids)} prompt tokens -> {len(gen)} generated: {text!r}")
    print(f"  ids {gen} sha256 {golden.ids_sha256(gen)}")
    print(
        f"  decode steps: sat={total.sat} err_shift={total.err_shift} clip={total.clip} "
        f"ERR_BOUNDS={m.csr['ERR_BOUNDS']} MACS={m.perf_value('MACS')} "
        f"WT_BYTES={m.perf_value('WT_BYTES')} seconds={elapsed:.1f}"
    )
    status = 0
    stored = REPO_ROOT / "models" / spec.name / "expected_tokens.json"
    if stored.is_file() and int(layout["model"]["layers"]) == spec.layers:
        expected = json.loads(stored.read_text(encoding="utf-8"))
        if key in expected.get("prompts", {}) and expected.get("max_new") == args.max_new:
            want = expected["prompts"][key]["generated_ids"]
            print(f"  expected_tokens.json: {'match' if want == gen else 'MISMATCH ' + str(want)}")
            status |= int(want != gen)
    if model is not None:
        t1 = time.perf_counter()
        cmp = isa_sim.compare_generate(
            model,
            image_path,
            programs,
            ids,
            args.max_new,
            eos_ids=spec.eos_ids,
            a_bits=a_bits,
            max_ctx=max_ctx,
            wb=wb,
        )
        secs = time.perf_counter() - t1
        if cmp.ok:
            print(
                f"  golden: every descriptor matches over {cmp.positions} positions ({secs:.1f} s)"
            )
        else:
            print(f"  golden: MISMATCH {cmp.mismatch}")
            status |= 1
    return status


def _cmd_isa_sim(args: argparse.Namespace) -> int:
    from pathlib import Path

    from quettos import compiler, golden, isa_sim, quantize
    from quettos.model import REPO_ROOT
    from quettos.tokenizer_io import token_bytes

    spec = load_spec(args.model)
    out_dir = Path(args.dir) if args.dir else compiler.IMAGES_DIR / spec.name
    layout_path = out_dir / compiler.FILES["layout"]
    if not layout_path.is_file():
        print(f"{layout_path} not found; run: uv run quettos compile {args.model}", file=sys.stderr)
        return 1
    layout = compiler.load_layout(out_dir)
    layers = int(layout["model"]["layers"])
    programs = isa_sim.Programs.from_dir(out_dir)
    image_path = out_dir / layout["image"]["file"]
    model = None
    if args.compare:
        qpath = Path(args.quant) if args.quant else quantize.default_path(spec.name)
        if not qpath.is_file():
            print(f"{qpath} not found; run: uv run quettos quantize {args.model}", file=sys.stderr)
            return 1
        model = quantize.load(qpath)
        if model.n_layers < layers:
            print(f"{qpath} holds {model.n_layers} layers, the image {layers}", file=sys.stderr)
            return 1
        if model.n_layers > layers:
            model = compiler.truncate_layers(model, layers)
    table = token_bytes(spec)
    prompts = args.prompt or [str(REPO_ROOT / f) for f in golden.EXPECTED_PROMPT_FILES]
    status = 0
    for f in prompts:
        ids = prompt_tokens(spec, f)
        key = golden.prompt_key(f)
        status |= _isa_sim_prompt(args, spec, layout, programs, image_path, model, table, key, ids)
    return status


def _cmd_compare(args: argparse.Namespace) -> int:
    from quettos import compare

    return compare.run(args)


def _cmd_check(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from quettos import calibrate, quality, quantize

    spec = load_spec(args.model)
    if args.quant:
        qpath = Path(args.quant)
    else:
        qpath = quantize.default_path(spec.name, smoothing=args.smoothing)
    if not qpath.is_file():
        flag = "" if args.smoothing else " --no-qk-smoothing"
        print(
            f"{qpath} not found; run: uv run quettos quantize {args.model}{flag}", file=sys.stderr
        )
        return 1
    model = quantize.load(qpath)
    if model.n_layers != spec.layers:
        print(f"{qpath} holds {model.n_layers} of {spec.layers} layers", file=sys.stderr)
        return 1
    seqs = calibrate.calibration_sequences(spec)
    widths = list(quality.CONFIGS) if args.a_bits is None else [args.a_bits]
    t0 = time.perf_counter()
    ref = quality.reference_logits(spec, seqs)
    t1 = time.perf_counter()
    n_tok = sum(len(s) for s in seqs)
    print(f"reference: {n_tok} tokens in {len(seqs)} sequences, {t1 - t0:.1f} s")
    rows = []
    for a_bits in widths:
        t2 = time.perf_counter()
        row = quality.evaluate(spec, model, seqs, a_bits=a_bits, ref=ref)
        rows.append(row)
        st = row["stats"]
        print(
            f"{row['config']}: tokens {row['tokens']} top-1 {row['top1_percent']:.2f}% "
            f"KL {row['kl_mean']:.4g} nats delta-NLL {row['delta_nll']:+.4g} +/- "
            f"{row['delta_nll_se']:.3g} PPL {row['ppl_fp32']:.3f} -> {row['ppl_int']:.3f} "
            f"sat={st['sat']} err_shift={st['err_shift']} clip={st['clip']} "
            f"({time.perf_counter() - t2:.1f} s)"
        )
    rep = quality.report(model, seqs, rows)
    if any(r["stats"]["sat"] or r["stats"]["err_shift"] for r in rows):
        print("refusing to write: saturation or shift-error counters are non-zero", file=sys.stderr)
        return 1
    if args.a_bits is None:
        out = Path(args.out) if args.out else quality.quality_path(model.name)
        path = quality.write_quality(quality.merge_quality(rep, out), out)
        print(f"wrote {path}")
    else:
        print(quality.quality_json_text(rep), end="")
    return 0


def _cmd_csr_defs(args: argparse.Namespace) -> int:
    from quettos import csrgen

    return csrgen.main(["--check"] if args.check else [])


def _cmd_download(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    print(spec.to_json())
    return 0


def _cmd_tokens(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    ids = prompt_tokens(spec, args.prompt)
    out = {"model": spec.repo_id, "prompt": args.prompt, "count": len(ids), "ids": ids}
    if args.show_text:
        out["text"] = render_prompt(spec, args.prompt)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def _cmd_export_tokens_bin(args: argparse.Namespace) -> int:
    spec = load_spec(args.model)
    nbytes = write_tokens_bin(spec, args.out)
    print(f"wrote {args.out}: {spec.vocab} entries, {nbytes} bytes")
    return 0


# --------------------------------------------------------------------------- the demo


#: The counters a clean run leaves at zero (``docs/ISA.md``, the PERF table).
DEMO_ZERO_COUNTERS = ("SAT_REQ", "SAT_VPU", "ERR_SHIFT", "ERR_BOUNDS")

#: The six exclusive busy buckets, which sum to ``BUSY``.
DEMO_BUCKETS = ("MAC_ACTIVE", "STALL_MEM", "STALL_VPU", "STALL_KV", "STALL_SEQ", "STALL_DRAIN")


class DemoInput(Exception):
    """A file the report reads is not there, or is not what it has to be.

    The report's job is a verdict, so a file that cannot be read ends it the way
    a failed check does -- named, with the reason -- rather than in a traceback.
    """


def _demo_read_json(path: Path | str, what: str) -> Any:
    """One of the report's input files, parsed; :class:`DemoInput` on anything else."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise DemoInput(f"{path}, {what}, cannot be read: {exc.strerror}") from exc
    except ValueError as exc:
        raise DemoInput(f"{path}, {what}, is not JSON: {exc}") from exc


def _demo_stages(path: str | None) -> list[tuple[str, float, str]]:
    """The stage timings ``scripts/demo.sh`` appends, one JSON object per line."""
    if path is None or not Path(path).is_file():
        return []
    rows: list[tuple[str, float, str]] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            rows.append((str(d["name"]), float(d["seconds"]), str(d.get("note", ""))))
        except (ValueError, KeyError, TypeError) as exc:
            raise DemoInput(f"{path} line {n} is not a stage timing: {exc}") from exc
    return rows


def _demo_sum(tokens: list[dict[str, Any]]) -> dict[str, int]:
    """Cycles, MAC-active cycles, read bytes and descriptors over a set of token records."""
    keys = ("cycles", "mac_active", "rd_bytes", "descriptors")
    return {k: sum(int(t[k]) for t in tokens) for k in keys}


def _demo_decode(token: dict[str, Any]) -> bool:
    """Whether a token record ran ``decode.prog``, as the harness recorded it.

    The record names its own program (``sim/verilator/main.cpp``), which is what
    a token that faulted has no generated id to say: its ``out`` is -1 like a
    prefill token's, and it is a decode token all the same.
    """
    return str(token["pass"]) == "decode"


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, read a megabyte at a time."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The layout entries whose recorded SHA-256 is over what the file says rather
#: than over its bytes: ``prompt.tokens`` is one id per line and the compiler
#: hashes the ids it carries (``golden.ids_sha256``).
DEMO_ID_FILES = ("prompt",)


def _layout_hashed(layout: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every entry of ``layout.json`` that names a file and a SHA-256 of it.

    The compiler records one for the image, both descriptor programs, the dump
    plan, the token table, the prompt, the rope table, the lookup tables and the
    golden model's recorded continuation (``sw/quettos/compiler.py``), each as an
    object carrying ``file`` and ``sha256``.  Walking for that pair is what keeps
    this list and the compiler's from drifting apart: an entry the compiler adds
    is hashed here without a change.
    """
    out: list[tuple[str, dict[str, Any]]] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if isinstance(node.get("file"), str) and isinstance(node.get("sha256"), str):
            out.append((path, node))
            return
        for key, value in node.items():
            walk(value, f"{path}.{key}" if path else key)

    walk(layout, "")
    return out


def _layout_file(image: Path, name: str) -> Path:
    """Where a file ``layout.json`` names lives.

    A bare name is beside ``layout.json`` in the compiled directory; a name with
    a directory in it is a repository path, which is how the compiler records
    the tables and the golden model's continuation.
    """
    from quettos.model import REPO_ROOT

    return REPO_ROOT / name if "/" in name else image / name


def _demo_digest(name: str, path: Path) -> str:
    """The digest the compiler took of this file, taken again from the file."""
    if name in DEMO_ID_FILES:
        from quettos.golden import ids_sha256

        try:
            ids = [int(v) for v in path.read_text(encoding="utf-8").split()]
        except (OSError, ValueError) as exc:
            raise DemoInput(f"{path} is not one token id per line: {exc}") from exc
        return ids_sha256(ids)
    return _sha256_file(path)


def _demo_files(
    image: Path, layout: dict[str, Any], run: dict[str, Any]
) -> tuple[str, int, int, list[str]]:
    """Every file ``layout.json`` carries a hash for, hashed and held to it.

    The token table is as load-bearing as the weights -- it is what turns the
    ids the hardware produced into the sentence the summary prints -- and the
    programs, the dump plan, the prompt, the tables and the recorded
    continuation are all read by something that reports on the run, so each of
    them is hashed here rather than the image alone.  Returns the image's own
    SHA-256 and size, how many files matched, and the list of the things that do
    not: a file the layout names and is not there, one whose bytes have moved,
    the image's size, and the file the run says it mapped with the bytes it
    mapped of it.  The whole set is 0.19 s on the 513,950,464 B Qwen image, paid
    once at the end of a demo; the harness checks the image size at the start of
    every run, which costs nothing.
    """
    entries = _layout_hashed(layout)
    bad: list[str] = []
    digest, size, matched = "", 0, 0
    for name, entry in entries:
        path = _layout_file(image, str(entry["file"]))
        if not path.is_file():
            bad.append(f"{path}, the {name} of layout.json, is not there")
            continue
        got = _demo_digest(name, path)
        if name == "image":
            digest, size = got, path.stat().st_size
            if size != int(entry["size"]):
                bad.append(f"{path} is {size:,} B; layout.json describes {int(entry['size']):,}")
        if got != str(entry["sha256"]):
            bad.append(f"{path} hashes to {got}; layout.json describes {entry['sha256']}")
        else:
            matched += 1
    if not size:
        return digest, size, matched, bad
    ran, ran_bytes = run.get("image"), run.get("image_bytes")
    path = _layout_file(image, str(layout["image"]["file"]))
    if ran is None or ran_bytes is None:
        bad.append("the run record does not name the image the hardware executed")
    elif Path(str(ran)).resolve() != path.resolve():
        bad.append(f"the hardware executed {ran}, not {path}")
    elif int(ran_bytes) != size:
        bad.append(f"the hardware mapped {int(ran_bytes):,} B of {path}, which is {size:,} B")
    return digest, size, matched, bad


def _demo_prompt(
    perf: dict[str, Any], layout: dict[str, Any]
) -> tuple[list[int], str | None, list[str]]:
    """The ids the run was given, the prompt file they are, and what does not match.

    ``layout.json`` names the prompt the image was compiled with and the run
    records the ids it actually prefilled.  The source comes back only when the
    two are the same ids, so a run is neither labelled with nor held to the
    record of a prompt it was not given.
    """
    from quettos.golden import ids_sha256

    given = perf["run"].get("prompt")
    entry = layout.get("prompt")
    if given is None or "ids" not in given:
        return [], None, ["the run record does not carry the prompt the hardware was given"]
    ids = [int(i) for i in given["ids"]]
    bad: list[str] = []
    prefilled = sum(1 for t in perf["tokens"] if not _demo_decode(t)) + 1
    if prefilled != len(ids):
        bad.append(f"the run prefilled {prefilled} ids of its {len(ids)}-id prompt")
    if entry is None:
        return ids, None, bad
    if len(ids) != int(entry["count"]) or ids_sha256(ids) != str(entry["sha256"]):
        bad.append(
            f"the run was given {len(ids)} ids from {given.get('from', 'its own record')}, not the "
            f"{int(entry['count'])} of {entry['file']}, the ids of {entry['source']}"
        )
        return ids, None, bad
    return ids, str(entry["source"]), bad


def _demo_checks(perf: dict[str, Any], layout: dict[str, Any]) -> list[str]:
    """What has to hold of a finished run's counters, as the list of the things that do not.

    Everything here is read back out of the file the harness wrote: the counters
    against each other, against the memory model's own counts and against the
    compiled program, and the four counters a clean run leaves at zero.  The
    image and the prompt the same file records are :func:`_demo_image` and
    :func:`_demo_prompt`.
    """
    bad: list[str] = []
    c = perf["counters"]
    buckets = sum(int(c[k]) for k in DEMO_BUCKETS)
    if buckets != int(c["BUSY"]):
        bad.append(f"BUSY {int(c['BUSY']):,} is not the sum of the six buckets ({buckets:,})")
    wb = int(perf["build"]["wb"])
    if int(c["RD_BYTES"]) != int(c["RD_BEATS"]) * wb:
        bad.append(f"RD_BYTES {int(c['RD_BYTES']):,} is not RD_BEATS * {wb}")
    mem = perf["memory"]
    if int(c["WR_BEATS"]) != int(mem["wr_beats"]) or int(c["WR_BYTES"]) != int(mem["wr_bytes"]):
        bad.append(
            f"the core counted {int(c['WR_BEATS']):,} write beats / {int(c['WR_BYTES']):,} bytes, "
            f"the memory model {int(mem['wr_beats']):,}/{int(mem['wr_bytes']):,}"
        )
    tokens = perf["tokens"]
    per_token = sum(int(t["cycles"]) for t in tokens)
    if per_token != int(c["CYCLES"]):
        bad.append(
            f"CYCLES {int(c['CYCLES']):,} is not the sum of the tokens' own counts ({per_token:,})"
        )
    want = {
        False: int(layout["programs"]["prefill"]["descriptors"]),
        True: int(layout["programs"]["decode"]["descriptors"]),
    }
    for t in tokens:
        n = want[_demo_decode(t)]
        if int(t["descriptors"]) != n:
            bad.append(
                f"the token at position {int(t['pos'])} retired {int(t['descriptors']):,} "
                f"descriptors, not the program's {n:,}"
            )
            break
    run = perf["run"]
    if run.get("status") != "ok":
        bad.append(f"the run stopped: {run.get('stop_reason', run.get('status'))}")
    for name in DEMO_ZERO_COUNTERS:
        if int(perf["events"][name]) != 0:
            bad.append(f"{name} is {int(perf['events'][name]):,}, not zero")
    return bad


def _demo_pass_table(perf: dict[str, Any]) -> list[str]:
    """The cycle and utilization table: prefill, decode and the whole run."""
    tokens = perf["tokens"]
    groups = [
        ("prefill", [t for t in tokens if not _demo_decode(t)]),
        ("decode", [t for t in tokens if _demo_decode(t)]),
    ]
    head = f"  {'pass':<9}{'tokens':>7}{'cycles':>16}{'cycles/token':>15}"
    head += f"{'MAC_ACTIVE':>13}{'read B/cycle':>14}"
    lines = [head]
    for name, group in groups:
        if not group:
            continue
        s = _demo_sum(group)
        per = s["cycles"] // len(group)
        mac = 100.0 * s["mac_active"] / s["cycles"] if s["cycles"] else 0.0
        bpc = s["rd_bytes"] / s["cycles"] if s["cycles"] else 0.0
        lines.append(
            f"  {name:<9}{len(group):>7}{s['cycles']:>16,}{per:>15,}{mac:>12.1f}%{bpc:>14.2f}"
        )
    c = perf["counters"]
    cycles, busy = int(c["CYCLES"]), int(c["BUSY"])
    mac = 100.0 * int(c["MAC_ACTIVE"]) / busy if busy else 0.0
    bpc = int(c["RD_BYTES"]) / cycles if cycles else 0.0
    lines.append(f"  {'run':<9}{len(tokens):>7}{cycles:>16,}{'':>15}{mac:>12.1f}%{bpc:>14.2f}")
    return lines


def _demo_bucket_lines(perf: dict[str, Any]) -> list[str]:
    """Where the run's cycles went, as a share of ``BUSY``, three buckets to a line."""
    c = perf["counters"]
    busy = int(c["BUSY"]) or 1
    cells = [f"{k} {100.0 * int(c[k]) / busy:.2f}%" for k in DEMO_BUCKETS]
    return ["  ".join(f"{cell:<22}" for cell in cells[i : i + 3]).rstrip() for i in (0, 3)]


def _demo_report(args: argparse.Namespace) -> int:
    """Print the summary of a demo run; returns 1 on anything that is not right.

    Reads the ``perf.json`` the harness wrote, the compiled ``layout.json`` and
    the golden model's recorded continuation, and computes no model value of its
    own: the ids are the ones ``qcore_top`` produced and the reference is the
    checked-in ``models/<name>/expected_tokens.json``.  Every file
    ``layout.json`` carries a hash for is held to it, the prompt the run was
    given to the ids the image was compiled with, and a run that generated
    nothing fails rather than matching an empty record.
    """
    from quettos import compare, compiler
    from quettos.tokenizer_io import detokenize, read_tokens_bin

    image = Path(args.image)
    layout = _demo_read_json(image / compiler.FILES["layout"], "the compiled layout")
    perf = _demo_read_json(args.perf, "the record of the run")
    build, run, counters = perf["build"], perf["run"], perf["counters"]
    model = layout["model"]
    ids = [int(t["out"]) for t in perf["tokens"] if int(t["out"]) >= 0]
    digest, size, matched, failures = _demo_files(image, layout, run)
    hashed = len(_layout_hashed(layout))
    given, source, prompt_bad = _demo_prompt(perf, layout)
    failures.extend(prompt_bad)

    print("\n" + "=" * 78)
    print(f"demo summary: {model['name']} on {build['top']}")
    print("=" * 78)
    print(
        f"  model      {model['repo_id']}, {model['layers']} layers, "
        f"hidden {model['hidden']}, vocab {model['vocab']:,}"
    )
    stamp = f"{size:,} B, sha256 {digest[:16]}" if digest else "not found"
    print(f"  image      {image}/{layout['image']['file']}, {stamp}, max_ctx {layout['max_ctx']}")
    print(f"  files      {matched} of the {hashed} files layout.json records a SHA-256 for match")
    print(
        f"  core       {build['top']} WB={build['wb']} B_MAX={build['b_max']} VL={build['vl']} "
        f"VSRAM_WORDS={build['vsram_words']} ACC_W={build['acc_w']}, "
        f"Verilator --threads {build['threads']}"
    )
    print(
        f"  memory     read latency {run['lat']} cycles, one returned beat every "
        f"{run['bw_div']} cycle(s)"
    )
    print(
        f"  programs   decode {int(layout['programs']['decode']['descriptors']):,} descriptors, "
        f"prefill {int(layout['programs']['prefill']['descriptors']):,}"
    )
    where = "" if source is None else f" from {source}"
    print(f"  prompt     {len(given)} ids the run was given{where}")

    # --- the ids the hardware produced, against the record the golden model wrote
    print(f"\n  {len(ids)} ids generated on {build['top']}, one decode program each")
    print(f"    ids   {ids}")
    table_path = image / "tokens.bin"
    if table_path.is_file():
        try:
            print(f"    text  {detokenize(read_tokens_bin(table_path), ids)!r}")
        except (ValueError, struct.error) as exc:
            raise DemoInput(f"{table_path} is not a token table this reads: {exc}") from exc
    reference: list[str] = []
    try:
        recorded = compare.recorded_continuation(image, source)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        recorded = None
        reference.append(str(exc))
    expected = None if recorded is None else recorded[: int(run["max_new"])]
    entry = layout.get("expected_tokens")
    if not ids:
        failures.append(f"{build['top']} generated no ids: there is nothing to hold to a record")
    if expected is None:
        print(f"    reference  none recorded for {source}")
        if not args.no_reference and not reference:
            reference.append(f"no recorded continuation of {source} to hold the ids to")
    elif not ids:
        print(f"    reference  nothing generated to hold to {entry['file']}")
    elif ids == expected:
        print(f"    reference  {len(ids)}/{len(expected)} identical to {entry['file']},")
        print(f"               the integer golden model's own continuation of {source}")
    else:
        i = next(
            (k for k, (a, b) in enumerate(zip(expected, ids, strict=False)) if a != b),
            min(len(expected), len(ids)),
        )
        reference.append(
            f"generated id {i} is {ids[i] if i < len(ids) else 'missing'}, "
            f"{entry['file']} records {expected[i] if i < len(expected) else 'nothing'}"
        )
        print(f"    reference  MISMATCH against {entry['file']} at generated id {i}")
        print(f"               recorded {expected}")

    # --- the counters, and everything the run has to add up to
    print("\n  counters that a clean run leaves at zero")
    print("    " + "  ".join(f"{k} {int(perf['events'][k]):,}" for k in DEMO_ZERO_COUNTERS))
    failures.extend(reference)
    failures.extend(_demo_checks(perf, layout))

    print("\n  cycles and utilization, from the core's own PERF counters")
    for line in _demo_pass_table(perf):
        print(line)
    head, tail = _demo_bucket_lines(perf)
    print(f"\n  where the cycles go   {head}")
    print(f"                        {tail}")
    print(
        f"  traffic               WT_BYTES {int(counters['WT_BYTES']):,}, "
        f"MACS {int(counters['MACS']):,}, DESCRIPTORS {int(counters['DESCRIPTORS']):,}, "
        f"WR_BYTES {int(counters['WR_BYTES']):,}"
    )

    # --- the wall clock: every stage of the run, as it was measured
    stages = _demo_stages(args.stages)
    clock = (
        f"the harness clock loop is {float(run['wall_seconds']):.2f} s, "
        f"{float(run['mcycles_per_s']):.3f} Mcycles/s over "
        f"{int(run['clock_cycles']):,} clock cycles"
    )
    print("\n  wall clock")
    for name, secs, note in stages:
        print(f"    {name:<12}{secs:>9.2f} s   {note}")
    if stages:
        print("    " + "-" * 23)
        total = sum(s for _, s, _ in stages)
        print(f"    {'end to end':<12}{total:>9.2f} s   of which {clock}")
    else:
        print(f"    {'RTL run':<12}{float(run['wall_seconds']):>9.2f} s   {clock}")

    if failures:
        print("\ndemo: FAILED")
        for f in failures:
            print(f"  {f}")
        return 1
    held = "" if expected is None else f"{len(ids)} ids matching the recorded reference, "
    print(
        f"\ndemo: OK -- {model['name']} on {build['top']}, "
        f"{int(counters['CYCLES']):,} cycles, "
        f"{100.0 * int(counters['MAC_ACTIVE']) / (int(counters['BUSY']) or 1):.1f}% MAC-active, "
        f"{held}no saturation or range events"
    )
    return 0


def _cmd_demo_report(args: argparse.Namespace) -> int:
    """:func:`_demo_report`, with a file it cannot read reported as a verdict."""
    try:
        return _demo_report(args)
    except DemoInput as exc:
        print(f"\ndemo: FAILED\n  {exc}")
        return 1
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        print(
            f"\ndemo: FAILED\n  {args.perf} and {args.image}/layout.json are not the record the "
            f"harness writes and the layout the compiler writes: {exc!r}"
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the ``quettos`` CLI."""
    parser = argparse.ArgumentParser(prog="quettos", description="Quettos Core host tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download a model and print its spec as JSON")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("tokens", help="render a prompt file through the chat template and tokenize")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument("prompt", help="path to a prompts/*.json file")
    p.add_argument("--show-text", action="store_true", help="also print the rendered prompt text")
    p.set_defaults(func=_cmd_tokens)

    p = sub.add_parser("export-tokens-bin", help="write the id -> raw bytes table (tokens.bin)")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument("out", help="output path")
    p.set_defaults(func=_cmd_export_tokens_bin)

    p = sub.add_parser("calibrate", help="measure activation ranges and write calib.json")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument("--out", default=None, help="output path (default models/<name>/calib.json)")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("quantize", help="quantize weights to int8 and write build/quant/<name>.npz")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument("--calib", default=None, help="calib.json to use (default models/<name>/)")
    p.add_argument("--out", default=None, help="output .npz path (default build/quant/<name>.npz)")
    p.add_argument("--layers", type=int, default=None, help="keep only the first N layers")
    p.add_argument(
        "--no-qk-smoothing",
        dest="smoothing",
        action="store_false",
        help="force every Q/K smoothing factor to 1 (K-centering alone); "
        "writes build/quant/<name>-nosmooth.npz",
    )
    p.set_defaults(func=_cmd_quantize)

    p = sub.add_parser("compile", help="lay out image.bin, the descriptor programs and layout.json")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument("--layers", type=int, default=None, help="compile only the first N layers")
    p.add_argument("--out", default=None, help="output directory (default build/images/<name>)")
    p.add_argument("--max-ctx", type=int, default=2048, help="KV positions in the image")
    p.add_argument("--wb", type=int, default=64, help="weight-port width in bytes (tile width)")
    p.add_argument("--a-bits", type=int, default=16, choices=(8, 16), help="activation width")
    p.add_argument("--quant", default=None, help="quantized model .npz (default build/quant/)")
    p.add_argument("--prompt", default=None, help="prompt file whose ids go to prompt.tokens")
    p.set_defaults(func=_cmd_compile)

    p = sub.add_parser("golden", help="greedy generation with the bit-exact integer golden model")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="prompts/*.json file (repeatable; default: the two expected_tokens prompts)",
    )
    p.add_argument("--max-new", type=int, default=20, help="tokens to generate (default 20)")
    p.add_argument("--a-bits", type=int, default=16, choices=(8, 16), help="activation width")
    p.add_argument("--quant", default=None, help="quantized model .npz (default build/quant/)")
    p.add_argument(
        "--write-expected",
        action="store_true",
        help="write models/<name>/expected_tokens.json from this run",
    )
    p.set_defaults(func=_cmd_golden)

    p = sub.add_parser("isa-sim", help="run the compiled programs on the ISA simulator")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument(
        "--dir", default=None, help="compiler output directory (default build/images/<name>)"
    )
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="prompts/*.json file (repeatable; default: the two expected_tokens prompts)",
    )
    p.add_argument("--max-new", type=int, default=20, help="tokens to generate (default 20)")
    p.add_argument(
        "--compare", action="store_true", help="check every descriptor against the golden model"
    )
    p.add_argument("--quant", default=None, help="quantized model .npz for --compare")
    p.set_defaults(func=_cmd_isa_sim)

    from quettos import compare

    p = sub.add_parser("compare", help="RTL against the ISA simulator on the bring-up program")
    compare.add_arguments(p)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("check", help="quality of the integer golden model against fp32")
    p.add_argument("model", help="alias (qwen, smollm2) or Hugging Face repo id")
    p.add_argument(
        "--a-bits",
        type=int,
        default=None,
        choices=(8, 16),
        help="evaluate one activation width and print the row (default: both, written to file)",
    )
    p.add_argument("--quant", default=None, help="quantized model .npz (default build/quant/)")
    p.add_argument(
        "--no-qk-smoothing",
        dest="smoothing",
        action="store_false",
        help="score the build of `quantize --no-qk-smoothing` "
        "(default build/quant/<name>-nosmooth.npz) into the -nosmooth rows",
    )
    p.add_argument("--out", default=None, help="output path (default models/<name>/quality.json)")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("demo-report", help="the summary of a demo run (scripts/demo.sh)")
    p.add_argument("--image", required=True, help="the compiled model directory the run used")
    p.add_argument("--perf", required=True, help="the perf.json the harness wrote")
    p.add_argument("--stages", default=None, help="stage timings from scripts/demo.sh")
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="accept an image whose prompt has no recorded golden continuation",
    )
    p.set_defaults(func=_cmd_demo_report)

    p = sub.add_parser("csr-defs", help="write the ISA/CSR headers for the RTL and the harness")
    p.add_argument(
        "--check", action="store_true", help="compare against the checked-in files; write nothing"
    )
    p.set_defaults(func=_cmd_csr_defs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
