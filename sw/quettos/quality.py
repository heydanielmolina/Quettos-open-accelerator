"""Quality of the integer golden model against the float32 reference.

``uv run quettos check <alias>`` runs sequences teacher-forced through
:func:`quettos.golden.forward_tokens` and :func:`quettos.reference_np.forward`
and writes ``models/<name>/quality.json``: top-1 agreement, mean KL, paired
delta-NLL with its standard error and perplexity for the W8A16 and W8A8
configurations, each with the golden counters of its run.  A model built
without the Q/K smoothing fold (``quantize.build_quant_model(...,
smoothing=False)``) is scored by the same protocol and lands as a
``-nosmooth`` row beside them.  Two sets are scored and kept apart in the
file: the calibration set, under ``rows``, and the held-out WikiText-2 windows
of :mod:`quettos.corpus` (``check --heldout``), under ``heldout``.  Entry
points: :func:`evaluate` and :func:`evaluate_multi`, which streams one
sequence at a time so a corpus-scale set fits in memory; :func:`report`
assembles the file and :func:`merge_quality` keeps the rows of earlier runs.
:func:`provenance` reads a written file back the cheap way, rebuilding the ids
of each set from this clone and holding them to the SHA-256 stored beside the
rows, so which text a published row was scored on is a second's work rather
than a rescoring run (``uv run quettos provenance <alias>``, ``make
provenance``).  The measured tables and the protocol: ``docs/NUMERICS.md``
(Quality).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quettos import corpus, golden, program, reference_np
from quettos.calibrate import (
    FRAC_LOGITS,
    MODELS_OUT_DIR,
    calibration_sequences,
    canonical_json_text,
    token_ids_sha256,
)
from quettos.model import REPO_ROOT, ModelSpec
from quettos.numerics import Stats
from quettos.quantize import NOSMOOTH_SUFFIX, QuantModel, smoothing_enabled

ProgressFn = Callable[[int, int, float], None]  # (sequences done, total, seconds so far)

CONFIGS: dict[int, str] = {16: "W8A16", 8: "W8A8"}  # activation width -> row name
# Every row name a complete ``quality.json`` carries: each config for the
# shipped model and for the smoothing-free ablation build.
ROW_NAMES: tuple[str, ...] = tuple(CONFIGS.values()) + tuple(
    name + NOSMOOTH_SUFFIX for name in CONFIGS.values()
)
# Fields a stored report shares with a fresh one of the same model and set.
IDENTITY_KEYS = ("format", "numerics", "model", "calib_tokens_sha256", "protocol")
HELDOUT_KEY = "heldout"  # the held-out block of the report: its own protocol and rows
HELDOUT_SET = "held-out"  # what :func:`provenance` calls that block, and the other one
CALIB_SET = "calibration"
SEQ_KEYS = ("tokens", "top1_percent", "kl_mean", "delta_nll")  # per-sequence row detail
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


def config_name(qmodel: QuantModel, a_bits: int) -> str:
    """Row name of ``qmodel`` at ``a_bits``: the config, plus the ablation suffix.

    A model built with ``smoothing=False`` carries ``-nosmooth``, so a row is
    labelled by the build it came from and cannot be attributed to the other.
    """
    if a_bits not in CONFIGS:
        raise ValueError(f"config_name: a_bits {a_bits} not in {sorted(CONFIGS)}")
    return CONFIGS[a_bits] + ("" if smoothing_enabled(qmodel) else NOSMOOTH_SUFFIX)


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
    Returns :meth:`TokenMetrics.summary` plus ``a_bits``, ``config``
    (:func:`config_name`, which names the build ``qmodel`` came from),
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
    return _row(qmodel, a_bits, stats, parts)


def _row(
    qmodel: QuantModel, a_bits: int, stats: Stats, parts: Sequence[TokenMetrics]
) -> dict[str, Any]:
    """One table row: the pooled summary, the build it came from and its per-sequence detail."""
    row = TokenMetrics.concat(parts).summary()
    row["a_bits"] = a_bits
    row["config"] = config_name(qmodel, a_bits)
    row["stats"] = {"sat": stats.sat, "err_shift": stats.err_shift, "clip": stats.clip}
    row["sequences"] = [{k: v for k, v in p.summary().items() if k in SEQ_KEYS} for p in parts]
    return row


def evaluate_multi(
    spec: ModelSpec,
    builds: Sequence[tuple[QuantModel, int]],
    seqs: Sequence[Sequence[int]],
    *,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """One row per ``(qmodel, a_bits)`` build, over ``seqs``, a sequence at a time.

    The float32 reference of a sequence is computed once, shared by every build
    and then dropped, so peak memory is one sequence's ``[T, V]`` logits rather
    than the whole set's: this is what scores a corpus-scale set.  Every build
    must carry the same number of layers, since one reference serves them all,
    and no two may name the same row.  ``progress(done, total, seconds)`` is
    called after each sequence.
    """
    if not builds:
        raise ValueError("evaluate_multi: no builds")
    if not seqs:
        raise ValueError("evaluate_multi: no sequences")
    names = [config_name(m, b) for m, b in builds]
    if len(set(names)) != len(names):
        raise ValueError(f"evaluate_multi: repeated row name in {names}")
    n_layers = builds[0][0].n_layers
    if any(m.n_layers != n_layers for m, _ in builds):
        raise ValueError("evaluate_multi: every build must carry the same number of layers")
    progs = [program.build(m, a_bits=b) for m, b in builds]
    stats = [Stats() for _ in builds]
    parts: list[list[TokenMetrics]] = [[] for _ in builds]
    t0 = time.perf_counter()
    for done, ids in enumerate(seqs, start=1):
        ref = reference_np.forward(spec, ids, layers=n_layers)
        for j, ((qmodel, a_bits), prog) in enumerate(zip(builds, progs, strict=True)):
            out = golden.forward_tokens(qmodel, ids, a_bits=a_bits, stats=stats[j], prog=prog)
            parts[j].append(score_sequence(ref, out.logits, ids, frac=qmodel.frac["LOGITS"]))
            del out
        del ref
        if progress is not None:
            progress(done, len(seqs), time.perf_counter() - t0)
    return [
        _row(qmodel, a_bits, st, pt)
        for (qmodel, a_bits), st, pt in zip(builds, stats, parts, strict=True)
    ]


# --------------------------------------------------------------------------- report file


def report(
    qmodel: QuantModel,
    seqs: Sequence[Sequence[int]],
    rows: Sequence[dict[str, Any]],
    *,
    heldout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``quality.json`` document: model identity, protocol and the rows keyed by config.

    ``seqs`` and ``rows`` are the calibration set and its rows; ``heldout`` is
    the block :func:`heldout_block` builds, carried under :data:`HELDOUT_KEY`
    beside them and never mixed in.
    """
    lengths = [len(s) for s in seqs]
    doc = {
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
    if heldout is not None:
        doc[HELDOUT_KEY] = heldout
    return doc


def heldout_block(
    qmodel: QuantModel,
    seqs: Sequence[Sequence[int]],
    rows: Sequence[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """The held-out half of the document: the text it was scored on, then the rows.

    ``source`` is :func:`quettos.corpus.source_record` -- the archive, its
    SHA-256, the member and the cut -- so the windows behind these rows can be
    rebuilt byte for byte.
    """
    return {
        "protocol": {
            "text": (
                f"{source['name']}, the {source['split']} split ({source['member']}), "
                f"in {source['windows']} non-overlapping windows of "
                f"{source['window_tokens']} tokens, tokenized with no chat template; "
                "held out from the calibration set and from the quantizer"
            ),
            "reference": PROTOCOL["reference"],
            "scoring": PROTOCOL["scoring"],
            "source": source,
            "tokens": sum(len(s) - 1 for s in seqs),
            "ids_sha256": token_ids_sha256(seqs),
            "logits_frac": qmodel.frac["LOGITS"],
        },
        "rows": {row["config"]: row for row in rows},
    }


def merge_quality(rep: dict[str, Any], path: Path | str) -> dict[str, Any]:
    """``rep`` carrying forward the rows of the report at ``path`` it does not itself measure.

    A row measured by an earlier run -- another activation width, the
    ``-nosmooth`` ablation, or the other set -- survives only when the stored
    file describes the same model, numerics and protocol
    (:data:`IDENTITY_KEYS`), so rows under one key are always the same tokens
    scored the same way against the same reference; a report of anything else
    is replaced outright, rows and all.  The held-out block merges on its own
    protocol, which carries the corpus hash and the cut, so a run over
    different windows replaces that block and leaves the calibration rows
    alone.
    """
    path = Path(path)
    if not path.is_file():
        return rep
    stored = load_quality(path)
    if any(stored.get(k) != rep[k] for k in IDENTITY_KEYS):
        return rep
    out = {**rep, "rows": {**stored["rows"], **rep["rows"]}}
    heldout = _merge_heldout(stored.get(HELDOUT_KEY), rep.get(HELDOUT_KEY))
    if heldout is not None:
        out[HELDOUT_KEY] = heldout
    return out


def _merge_heldout(
    stored: dict[str, Any] | None, fresh: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The held-out block to keep: fresh rows over stored ones when both name the same windows."""
    if fresh is None:
        return stored
    if stored is None or stored.get("protocol") != fresh["protocol"]:
        return fresh
    return {**fresh, "rows": {**stored["rows"], **fresh["rows"]}}


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


# --------------------------------------------------------------------------- provenance


@dataclass(frozen=True)
class Check:
    """One provenance line: what was checked, what this clone found, and whether it agrees."""

    label: str
    detail: str
    ok: bool


@dataclass(frozen=True)
class SetProvenance:
    """One scored set of a stored report, beside the text this clone rebuilds for it."""

    name: str
    text: str
    tokens: int
    rows: tuple[str, ...]
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def scored_sets(rep: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The report's scored sets as ``(name, block)``, held-out first.

    A block is a ``protocol`` and the ``rows`` measured under it: the held-out
    one is stored that way, and the calibration set is the top-level pair,
    which :func:`report` writes without a block of its own.
    """
    sets = [(HELDOUT_SET, rep[HELDOUT_KEY])] if HELDOUT_KEY in rep else []
    sets.append((CALIB_SET, {"protocol": rep["protocol"], "rows": rep["rows"]}))
    return sets


def provenance(
    spec: ModelSpec, rep: dict[str, Any], *, download: bool = True
) -> list[SetProvenance]:
    """Every published row of ``rep`` against the text its own record names.

    Rebuilds the ids of each scored set from this clone -- the calibration set
    from ``prompts/`` and the passages of :mod:`quettos.calibrate`, the
    held-out windows from the WikiText-2 archive the record names, both through
    ``spec``'s tokenizer -- and holds their SHA-256 to the one stored beside
    the rows.  The held-out set is checked back to the archive as well (its
    bytes, the member taken out of it, and the cut), and its ids are compared
    with the calibration ids, which are the ids ``calib.json``'s ranges were
    measured on.  Nothing is scored and no model runs: a second's work against
    the sixteen minutes of ``uv run quettos check``.
    """
    calib_seqs = calibration_sequences(spec)
    calib_ids = token_ids_sha256(calib_seqs)
    out: list[SetProvenance] = []
    for name, block in scored_sets(rep):
        proto = block["protocol"]
        checks: list[Check] = []
        if name == HELDOUT_SET:
            seqs, checks = _heldout_checks(spec, proto["source"], download=download)
            origin = "the archive above, through the model's tokenizer"
        else:
            seqs = calib_seqs
            origin = "prompts/ and the passages of calibrate.py"
        ids = token_ids_sha256(seqs)
        checks.append(
            Check(
                "scored ids",
                f"sha256 {ids[:16]}, rebuilt from {origin}",
                ids == proto["ids_sha256"],
            )
        )
        scored = sum(len(s) - 1 for s in seqs)
        checks.append(Check("scored positions", str(scored), scored == proto["tokens"]))
        if name == HELDOUT_SET:
            # The suspicion this answers: the held-out rows are the calibration
            # text under another name. The record's own hash, against the hash
            # of the ids calibration measured on, settles it.
            checks.append(
                Check(
                    "not the calibration ids",
                    f"{proto['ids_sha256'][:16]} != {calib_ids[:16]}",
                    proto["ids_sha256"] != calib_ids,
                )
            )
        else:
            lengths = [len(s) for s in seqs]
            checks.append(
                Check("sequence lengths", str(lengths), lengths == list(proto["sequences"]))
            )
            checks.append(
                Check(
                    "the ids calibration saw",
                    f"calib_tokens_sha256 {rep['calib_tokens_sha256'][:16]}",
                    rep["calib_tokens_sha256"] == proto["ids_sha256"],
                )
            )
        out.append(
            SetProvenance(
                name=name,
                text=proto["text"],
                tokens=int(proto["tokens"]),
                rows=tuple(sorted(block["rows"])),
                checks=tuple(checks),
            )
        )
    return out


def _heldout_checks(
    spec: ModelSpec, source: dict[str, Any], *, download: bool
) -> tuple[list[list[int]], list[Check]]:
    """The held-out windows the ``source`` record names, and the checks that got there."""
    split, length, count = source["split"], source["window_tokens"], source["windows"]
    path = corpus.fetch_archive(download=download)
    size = path.stat().st_size
    digest = corpus.sha256_file(path)
    where = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    member = hashlib.sha256(corpus.split_bytes(split)).hexdigest()
    checks = [
        Check(
            "archive",
            f"{where}, {size} B, sha256 {digest[:16]}",
            digest == corpus.WIKITEXT2_SHA256 == source["archive_sha256"]
            and size == source["archive_bytes"],
        ),
        Check(
            "member", f"{source['member']}, sha256 {member[:16]}", member == source["member_sha256"]
        ),
        Check(
            "the cut",
            f"{count} non-overlapping windows of {length} tokens, {split} split",
            corpus.source_record(split=split, length=length, count=count) == source,
        ),
    ]
    return corpus.heldout_sequences(spec, split=split, length=length, count=count), checks


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
