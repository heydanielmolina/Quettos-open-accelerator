"""Quality of the integer golden model against the float32 reference.

``uv run quettos check <alias>`` runs the calibration sequences teacher-forced
through :func:`quettos.golden.forward_tokens` and
:func:`quettos.reference_np.forward` and writes ``models/<name>/quality.json``:
top-1 agreement, mean KL, paired delta-NLL with its standard error and
perplexity for the W8A16 and W8A8 configurations, each with the golden
counters of its run.  Entry point: :func:`evaluate`; :func:`report` assembles
the file.  The measured table and the protocol: ``docs/NUMERICS.md`` (Quality).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quettos import golden, program, reference_np
from quettos.calibrate import FRAC_LOGITS, MODELS_OUT_DIR, canonical_json_text, token_ids_sha256
from quettos.model import ModelSpec
from quettos.numerics import Stats
from quettos.quantize import QuantModel

CONFIGS: dict[int, str] = {16: "W8A16", 8: "W8A8"}  # activation width -> row name
LOGITS_FRAC = FRAC_LOGITS  # fixed-point class of the golden logits
ROW_CHUNK = 64  # logit rows converted to float64 at a time (bounds the temporaries)

PROTOCOL = {
    "text": (
        "calibration set of quettos.calibrate: the three prompt files under prompts/ and "
        "three prose passages, rendered through the model's chat template"
    ),
    "reference": "quettos.reference_np.forward (float32) over the same token ids",
    "scoring": (
        "teacher-forced; every position with a next token is scored (T - 1 per sequence): "
        "top-1 is argmax(int) == argmax(fp32); KL is sum_v p_fp32 log(p_fp32 / p_int) in nats "
        "from float64 log-softmax; NLL is -log p(next token) under each model; delta-NLL is the "
        "mean of the per-token nll_int - nll_fp32 with its standard error "
        "std(ddof=1) / sqrt(tokens); PPL is exp(mean NLL)"
    ),
}


# --------------------------------------------------------------------------- metrics


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise ``z - m - log(sum(exp(z - m)))`` with ``m = max(z)``, in float64."""
    z = np.asarray(z, dtype=np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    return z - m - np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))


@dataclass
class TokenMetrics:
    """Per-position values over the scored positions (those that have a next token).

    ``match[t]`` is ``argmax(int) == argmax(fp32)``; ``kl[t]`` is
    ``sum_v p_fp32 (log p_fp32 - log p_int)`` in nats; ``nll_fp32[t]`` and
    ``nll_int[t]`` are ``-log p(next token)`` under each model.
    """

    match: np.ndarray
    kl: np.ndarray
    nll_fp32: np.ndarray
    nll_int: np.ndarray

    @staticmethod
    def concat(parts: Sequence[TokenMetrics]) -> TokenMetrics:
        return TokenMetrics(
            *(np.concatenate([getattr(p, f) for p in parts]) for f in _TOKEN_FIELDS)
        )

    def summary(self) -> dict[str, Any]:
        """Aggregate over the positions.

        ``top1_percent = 100 * matches / tokens``; ``kl_mean``, ``nll_fp32``,
        ``nll_int`` are means; ``delta_nll = mean(nll_int - nll_fp32)`` with
        ``delta_nll_se = std(nll_int - nll_fp32, ddof=1) / sqrt(tokens)``
        (0 for a single token); ``ppl_* = exp(nll_*)``.
        """
        n = int(self.kl.size)
        if n < 1:
            raise ValueError("TokenMetrics.summary: no scored positions")
        matches = int(np.count_nonzero(self.match))
        delta = self.nll_int - self.nll_fp32
        nll_fp32 = float(np.mean(self.nll_fp32))
        nll_int = float(np.mean(self.nll_int))
        return {
            "tokens": n,
            "top1_matches": matches,
            "top1_percent": 100.0 * matches / n,
            "kl_mean": float(np.mean(self.kl)),
            "nll_fp32": nll_fp32,
            "nll_int": nll_int,
            "delta_nll": float(np.mean(delta)),
            "delta_nll_se": float(np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "ppl_fp32": math.exp(nll_fp32),
            "ppl_int": math.exp(nll_int),
        }


_TOKEN_FIELDS = ("match", "kl", "nll_fp32", "nll_int")


def score_sequence(
    ref_logits: np.ndarray,
    int_logits: np.ndarray,
    ids: Sequence[int],
    *,
    frac: int = LOGITS_FRAC,
) -> TokenMetrics:
    """Score positions ``0 .. T-2`` of one sequence against the next token ``ids[t+1]``.

    ``ref_logits`` is the float32 reference ``[T, V]``; ``int_logits`` the
    golden int32 ``[T, V]`` of class ``frac`` (real value ``v * 2**-frac``).
    Both argmaxes take the first maximal index, matching :func:`numerics.argmax`.
    """
    t = len(ids)
    if ref_logits.shape != int_logits.shape or ref_logits.shape[0] != t:
        raise ValueError(
            f"score_sequence: shapes {ref_logits.shape} / {int_logits.shape} for {t} ids"
        )
    if t < 2:
        raise ValueError("score_sequence: a sequence needs at least two tokens")
    n = t - 1
    targets = np.asarray(list(ids[1:]), dtype=np.int64)
    match = np.empty(n, dtype=bool)
    kl = np.empty(n, dtype=np.float64)
    nll_fp32 = np.empty(n, dtype=np.float64)
    nll_int = np.empty(n, dtype=np.float64)
    for r0 in range(0, n, ROW_CHUNK):
        r1 = min(n, r0 + ROW_CHUNK)
        ref = ref_logits[r0:r1]
        got = int_logits[r0:r1]
        lr = log_softmax(ref)
        li = log_softmax(got.astype(np.float64) * 2.0**-frac)
        rows = np.arange(r1 - r0)
        tgt = targets[r0:r1]
        match[r0:r1] = np.argmax(ref, axis=-1) == np.argmax(got, axis=-1)
        kl[r0:r1] = np.sum(np.exp(lr) * (lr - li), axis=-1)
        nll_fp32[r0:r1] = -lr[rows, tgt]
        nll_int[r0:r1] = -li[rows, tgt]
    return TokenMetrics(match, kl, nll_fp32, nll_int)


# --------------------------------------------------------------------------- evaluation


def reference_logits(
    spec: ModelSpec, seqs: Sequence[Sequence[int]], *, layers: int | None = None
) -> list[np.ndarray]:
    """Float32 reference logits ``[T, V]`` per sequence (shared by the W8A16 and W8A8 rows).

    ``layers`` truncates the reference stack like :func:`reference_np.forward`,
    for a :class:`QuantModel` that was quantized with fewer layers.
    """
    return [reference_np.forward(spec, ids, layers=layers) for ids in seqs]


def evaluate(
    spec: ModelSpec,
    qmodel: QuantModel,
    seqs: Sequence[Sequence[int]],
    *,
    a_bits: int = 16,
    ref: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """One row of the quality table: the golden model at ``a_bits`` against fp32 over ``seqs``.

    Every sequence runs teacher-forced through :func:`golden.forward_tokens`
    and :func:`reference_np.forward` over the same number of layers (``ref``
    supplies precomputed reference logits); the positions are scored by
    :func:`score_sequence` and pooled.
    Returns :meth:`TokenMetrics.summary` plus ``a_bits``, ``config``,
    ``stats`` (the golden ``sat`` / ``err_shift`` / ``clip`` counters of the
    run) and ``sequences`` (per-sequence ``tokens``, ``top1_percent``,
    ``kl_mean``, ``delta_nll``).
    """
    if a_bits not in CONFIGS:
        raise ValueError(f"evaluate: a_bits {a_bits} not in {sorted(CONFIGS)}")
    if not seqs:
        raise ValueError("evaluate: no sequences")
    if ref is None:
        ref = reference_logits(spec, seqs, layers=qmodel.n_layers)
    if len(ref) != len(seqs):
        raise ValueError("evaluate: one reference logit block per sequence is required")
    prog = program.build(qmodel, a_bits=a_bits)
    stats = Stats()
    parts: list[TokenMetrics] = []
    for ids, r in zip(seqs, ref, strict=True):
        out = golden.forward_tokens(qmodel, ids, a_bits=a_bits, stats=stats, prog=prog)
        parts.append(score_sequence(r, out.logits, ids, frac=qmodel.frac["LOGITS"]))
    row = TokenMetrics.concat(parts).summary()
    row["a_bits"] = a_bits
    row["config"] = CONFIGS[a_bits]
    row["stats"] = {"sat": stats.sat, "err_shift": stats.err_shift, "clip": stats.clip}
    row["sequences"] = [
        {k: p.summary()[k] for k in ("tokens", "top1_percent", "kl_mean", "delta_nll")}
        for p in parts
    ]
    return row


# --------------------------------------------------------------------------- report file


def report(
    qmodel: QuantModel, seqs: Sequence[Sequence[int]], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """The ``quality.json`` document: model identity, protocol and the rows keyed by config."""
    lengths = [len(s) for s in seqs]
    return {
        "format": "quettos-quality",
        "numerics": qmodel.numerics_version,
        "model": {"repo_id": qmodel.repo_id, "name": qmodel.name, "layers": qmodel.n_layers},
        "calib_tokens_sha256": qmodel.calib_tokens_sha256,
        "protocol": {
            **PROTOCOL,
            "sequences": lengths,
            "tokens": sum(n - 1 for n in lengths),
            "ids_sha256": token_ids_sha256(seqs),
            "logits_frac": qmodel.frac["LOGITS"],
        },
        "rows": {row["config"]: row for row in rows},
    }


def quality_json_text(rep: dict[str, Any]) -> str:
    """Canonical text of a report (sorted keys, six significant digits, trailing newline)."""
    return canonical_json_text(rep)


def quality_path(name: str) -> Path:
    return MODELS_OUT_DIR / name / "quality.json"


def load_quality(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_quality(rep: dict[str, Any], path: Path | str | None = None) -> Path:
    path = quality_path(rep["model"]["name"]) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(quality_json_text(rep), encoding="utf-8")
    return path


def close(a: Any, b: Any, rel: float = 1e-4) -> bool:
    """Structural equality with floats compared by ``math.isclose(rel_tol=rel, abs_tol=1e-9)``.

    Dicts need the same keys, lists the same length; ints, strings and
    booleans compare exactly.  Used to check a stored report against a fresh
    evaluation, whose floats differ only by the six-digit rounding of the file.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return (
            isinstance(a, int | float)
            and isinstance(b, int | float)
            and math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rel) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b, strict=True))
    return a == b
