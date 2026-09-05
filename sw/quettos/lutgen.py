"""Generate the lookup tables and RoPE tables shared by the hardware and the golden model.

Writes ``sw/quettos/tables/luts.json``, ``rtl/gen/{exp2,sigmoid,rsqrt,recip}.hex``
(``{v[15:0], dv[15:0]}`` per line for ``$readmemh``) and the int16 Q1.14 RoPE
tables ``rope_theta{1e6,1e5}_2048.npy``.  Everything is evaluated with mpmath at
128 bits and rounded half up, so the files regenerate identically on any
platform; ``--check`` compares against the checked-in files without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mpmath
import numpy as np

from quettos import numerics

REPO_ROOT = Path(__file__).resolve().parents[2]
HEX_DIR = REPO_ROOT / "rtl" / "gen"
HEAD_DIM = 64
ROPE_THETAS = (1e6, 1e5)
ROPE_MAX_POS = 2048

mpmath.mp.prec = 128


def _q15(value: mpmath.mpf) -> int:
    """Round half up to Q1.15 (``1.0 -> 32768``)."""
    return int(mpmath.floor(value * (1 << 15) + mpmath.mpf(1) / 2))


def _table(name: str) -> tuple[list[int], list[int]]:
    """Values ``v_i`` and forward differences ``dv_i`` for one table (see numerics.TABLE_SPECS)."""
    n = numerics.TABLE_SPECS[name]["entries"]
    one = mpmath.mpf(1)
    if name == "exp2":
        xs = [mpmath.mpf(i) / 256 for i in range(n)]
        f = lambda x: mpmath.power(2, x)  # noqa: E731
        right = f(one)
    elif name == "sigmoid":
        xs = [mpmath.mpf(i) / 32 for i in range(n)]
        f = lambda x: one / (one + mpmath.exp(-x))  # noqa: E731
        right = f(mpmath.mpf(16))
    elif name == "rsqrt":
        xs = [one + mpmath.mpf(i) / 256 for i in range(256)]
        xs += [2 + 2 * mpmath.mpf(i) / 256 for i in range(256)]
        f = lambda x: one / mpmath.sqrt(x)  # noqa: E731
        right = f(mpmath.mpf(4))
    elif name == "recip":
        xs = [one + mpmath.mpf(i) / 256 for i in range(n)]
        f = lambda x: one / x  # noqa: E731
        right = f(mpmath.mpf(2))
    else:
        raise ValueError(name)
    v = [_q15(f(x)) for x in xs]
    v_end = _q15(right)
    dv = [v[i + 1] - v[i] for i in range(n - 1)] + [v_end - v[-1]]
    return v, dv


def build_luts() -> dict:
    tables = {
        name: dict(zip(("v", "dv"), _table(name), strict=True)) for name in numerics.TABLE_SPECS
    }
    payload = json.dumps(tables, sort_keys=True, separators=(",", ":")).encode()
    tables["meta"] = {
        "generator": "quettos.lutgen",
        "mp_prec_bits": mpmath.mp.prec,
        "format": "v: Q1.15 value, dv: v[i+1]-v[i]; interp = v + ((dv*frac8 + 128)>>8)",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return tables


def hex_lines(v: list[int], dv: list[int]) -> str:
    """One ``{v, dv}`` 32-bit word per line, dv as 16-bit two's complement."""
    return "".join(f"{(vv << 16) | (d & 0xFFFF):08x}\n" for vv, d in zip(v, dv, strict=True))


def build_rope(theta: float, max_pos: int = ROPE_MAX_POS, head_dim: int = HEAD_DIM) -> np.ndarray:
    """int16 ``[max_pos, 2, head_dim/2]`` cos/sin table in Q1.14 with exact angle reduction.

    ``inv_freq_i = theta ** (-2i/head_dim)`` and ``angle = pos * inv_freq_i``
    (the Hugging Face rotate_half convention).
    """
    half = head_dim // 2
    th = mpmath.mpf(theta)
    inv_freq = [mpmath.power(th, -mpmath.mpf(2 * i) / head_dim) for i in range(half)]
    out = np.empty((max_pos, 2, half), dtype=np.int16)
    scale = 1 << 14
    for pos in range(max_pos):
        for i in range(half):
            ang = pos * inv_freq[i]
            c = int(mpmath.floor(mpmath.cos(ang) * scale + mpmath.mpf(1) / 2))
            s = int(mpmath.floor(mpmath.sin(ang) * scale + mpmath.mpf(1) / 2))
            out[pos, 0, i] = c
            out[pos, 1, i] = s
    return out


def json_text(luts: dict) -> str:
    return json.dumps(luts, indent=0, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check", action="store_true", help="compare against checked-in files; write nothing"
    )
    ap.add_argument("--no-rope", action="store_true", help="skip the RoPE tables (slow part)")
    args = ap.parse_args(argv)

    luts = build_luts()
    outputs: dict[Path, bytes] = {numerics.LUTS_JSON: json_text(luts).encode()}
    for name in numerics.TABLE_SPECS:
        outputs[HEX_DIR / f"{name}.hex"] = hex_lines(luts[name]["v"], luts[name]["dv"]).encode()
    if not args.no_rope:
        for theta in ROPE_THETAS:
            table = build_rope(theta)
            path = numerics.rope_table_path(theta, ROPE_MAX_POS)
            buf = _npy_bytes(table)
            outputs[path] = buf

    if args.check:
        bad = 0
        for path, data in outputs.items():
            current = path.read_bytes() if path.exists() else None
            status = "ok" if current == data else ("missing" if current is None else "DIFFERS")
            bad += status != "ok"
            print(f"{status:8s} {path.relative_to(REPO_ROOT)}")
        return 1 if bad else 0

    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"wrote {path.relative_to(REPO_ROOT)} ({len(data)} bytes)")
    return 0


def _npy_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


if __name__ == "__main__":
    sys.exit(main())
