"""Command-line entry point: ``quettos <command>`` (installed by ``pyproject.toml``).

Commands: ``download``, ``tokens``, ``export-tokens-bin``, ``calibrate``,
``quantize``, ``compile``, ``golden``, ``isa-sim``, ``compare``, ``check``,
``csr-defs``.
Each command is a thin wrapper over the module of the same name; the file
formats they read and write are described in ``docs/``.  ``quantize`` and
``check`` take ``--no-qk-smoothing``, which builds and scores the
K-centering-only variant of the model into the ``-nosmooth`` quality rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

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
