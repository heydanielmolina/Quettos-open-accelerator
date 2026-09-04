"""``quettos`` command-line entry point.

Subcommands::

    quettos download <alias|repo>              fetch a model, print its ModelSpec as JSON
    quettos tokens <alias|repo> <prompt.json>  render the chat template, print ids and count
    quettos export-tokens-bin <alias|repo> <out>   write the id -> raw bytes table

Later days add ``quantize``, ``compile``, ``golden``, ``run``, ``check``,
``perf``, ``synth-report`` and ``demo`` here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from quettos.model import load_spec
from quettos.tokenizer_io import prompt_tokens, render_prompt, write_tokens_bin


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
