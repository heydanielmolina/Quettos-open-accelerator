"""Command-line entry point: ``quettos <command>``.

Commands: ``download``, ``tokens``, ``export-tokens-bin``, ``calibrate``,
``quantize``.  The remaining commands of the stack register here as they land
(see ``docs/ROADMAP.md``).
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
