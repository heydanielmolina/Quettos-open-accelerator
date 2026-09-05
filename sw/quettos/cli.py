"""Command-line entry point: ``quettos <command>``.

Commands: ``download``, ``tokens``, ``export-tokens-bin``, ``calibrate``,
``quantize``, ``golden``, ``check``.
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
    model = quantize.build_quant_model(spec, calib_path, layers=args.layers)
    t1 = time.perf_counter()
    path = quantize.save(model, args.out)
    t2 = time.perf_counter()
    summary = {
        "model": spec.repo_id,
        "calib": str(calib_path),
        "out": str(path),
        "bytes": path.stat().st_size,
        "layers": model.n_layers,
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


def _cmd_check(args: argparse.Namespace) -> int:
    import time
    from pathlib import Path

    from quettos import calibrate, quality, quantize

    spec = load_spec(args.model)
    qpath = Path(args.quant) if args.quant else quantize.default_path(spec.name)
    if not qpath.is_file():
        print(f"{qpath} not found; run: uv run quettos quantize {args.model}", file=sys.stderr)
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
        path = quality.write_quality(rep, args.out)
        print(f"wrote {path}")
    else:
        print(quality.quality_json_text(rep), end="")
    return 0


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
    p.set_defaults(func=_cmd_quantize)

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
    p.add_argument("--out", default=None, help="output path (default models/<name>/quality.json)")
    p.set_defaults(func=_cmd_check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
