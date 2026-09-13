"""Held-out text for the quality measurement: WikiText-2 raw, fetched and hashed.

``wikitext-2-raw-v1.zip`` is downloaded once into ``build/corpus/``, checked
against :data:`WIKITEXT2_SHA256` on every use, and read with :mod:`zipfile`;
:func:`heldout_sequences` tokenizes one split with the model's own tokenizer
and cuts the id stream into equal non-overlapping windows.  The text is plain
prose, never seen by ``quettos.calibrate``, and is used verbatim -- no chat
template, no cleaning -- so the archive hash pins every scored id.
:func:`source_record` is the provenance block ``models/<name>/quality.json``
carries.  The measured table: ``docs/NUMERICS.md`` (Quality).
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

from quettos.model import BUILD_DIR, ModelSpec
from quettos.tokenizer_io import encode

WIKITEXT2_URL = "https://wikitext.smerity.com/wikitext-2-raw-v1.zip"
WIKITEXT2_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
WIKITEXT2_BYTES = 4721645
WIKITEXT2_NAME = "wikitext-2-raw-v1"

# Split -> the member of the archive that holds it.
MEMBERS: dict[str, str] = {
    "train": "wikitext-2-raw/wiki.train.raw",
    "valid": "wikitext-2-raw/wiki.valid.raw",
    "test": "wikitext-2-raw/wiki.test.raw",
}

USER_AGENT = "quettos"  # the host rejects the urllib default
CORPUS_DIR = BUILD_DIR / "corpus"
DEFAULT_SPLIT = "test"
WINDOW_TOKENS = 512  # the length band the calibration sequences cover (36 .. 515)
WINDOWS = 64  # 64 * 512 = 32,768 tokens, 32,704 of them scored
DOWNLOAD_TIMEOUT_S = 300


def archive_path() -> Path:
    """Where the archive lives: ``build/corpus/wikitext-2-raw-v1.zip``."""
    return CORPUS_DIR / f"{WIKITEXT2_NAME}.zip"


def sha256_file(path: Path | str) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def fetch_archive(*, path: Path | None = None, download: bool = True) -> Path:
    """The archive at ``path``, downloaded from :data:`WIKITEXT2_URL` if absent.

    The SHA-256 is checked on every call, downloaded or cached; a file that
    does not hash to :data:`WIKITEXT2_SHA256` is an error naming both digests.
    ``download=False`` requires the file to be there already.
    """
    path = archive_path() if path is None else Path(path)
    if not path.is_file():
        if not download:
            raise FileNotFoundError(f"{path} not present (run: uv run quettos corpus)")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(WIKITEXT2_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as src:
            tmp.write_bytes(src.read())
        tmp.replace(path)
    got = sha256_file(path)
    if got != WIKITEXT2_SHA256:
        raise ValueError(f"{path}: sha256 {got}, expected {WIKITEXT2_SHA256}")
    return path


def split_bytes(split: str = DEFAULT_SPLIT, *, path: Path | None = None) -> bytes:
    """Raw bytes of one split, straight out of the verified archive."""
    if split not in MEMBERS:
        raise ValueError(f"corpus: split {split!r} not in {sorted(MEMBERS)}")
    with zipfile.ZipFile(fetch_archive(path=path)) as zf:
        return zf.read(MEMBERS[split])


def split_text(split: str = DEFAULT_SPLIT, *, path: Path | None = None) -> str:
    """UTF-8 text of one split."""
    return split_bytes(split, path=path).decode("utf-8")


def cut_windows(ids: Sequence[int], *, length: int, count: int) -> list[list[int]]:
    """The first ``count`` non-overlapping windows of ``length`` ids: ``ids[i*L : (i+1)*L]``."""
    if length < 2 or count < 1:
        raise ValueError(f"cut_windows: length {length}, count {count}")
    if len(ids) < length * count:
        raise ValueError(f"cut_windows: {len(ids)} ids, {length * count} needed")
    return [list(ids[i * length : (i + 1) * length]) for i in range(count)]


def heldout_sequences(
    spec: ModelSpec,
    *,
    split: str = DEFAULT_SPLIT,
    length: int = WINDOW_TOKENS,
    count: int = WINDOWS,
    path: Path | None = None,
) -> list[list[int]]:
    """Token id windows of one split, encoded with ``spec``'s tokenizer and no specials."""
    return cut_windows(encode(spec, split_text(split, path=path)), length=length, count=count)


def source_record(
    *,
    split: str = DEFAULT_SPLIT,
    length: int = WINDOW_TOKENS,
    count: int = WINDOWS,
    path: Path | None = None,
) -> dict[str, object]:
    """Provenance of the scored text: the archive, the member and the cut."""
    if split not in MEMBERS:
        raise ValueError(f"corpus: split {split!r} not in {sorted(MEMBERS)}")
    return {
        "name": WIKITEXT2_NAME,
        "url": WIKITEXT2_URL,
        "archive_bytes": WIKITEXT2_BYTES,
        "archive_sha256": WIKITEXT2_SHA256,
        "member": MEMBERS[split],
        "member_sha256": hashlib.sha256(split_bytes(split, path=path)).hexdigest(),
        "split": split,
        "windows": count,
        "window_tokens": length,
    }
