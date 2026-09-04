"""Chat-template rendering, tokenization and byte-level vocabulary export.

The RTL never sees text: the host feeds token ids in and prints the bytes of
the ids that come out. This module is the whole text boundary:

* :func:`render_chat` renders ``tokenizer_config.json``'s Jinja chat template
  with an environment that mirrors ``transformers.apply_chat_template``
  (``trim_blocks``/``lstrip_blocks`` on, ``loopcontrols`` extension,
  non-strict undefined, ``raise_exception`` global and a ``tojson`` filter
  with the same argument semantics, special-token names injected as globals).
* :func:`encode` tokenizes with the ``tokenizers`` library and
  ``add_special_tokens=False`` -- the template text already carries the
  ChatML specials, exactly as ``apply_chat_template(tokenize=True)`` does.
* :func:`token_bytes` / :func:`write_tokens_bin` export ``tokens.bin``: for
  every id the raw byte string of that token (both models use byte-level BPE,
  so vocab strings are translated back through the inverse of GPT-2's
  ``bytes_to_unicode`` map; added/special tokens map to their literal UTF-8).
  Format: ``u32 count`` then, per id, ``u16 length`` + ``bytes``
  (little-endian).
* :func:`detokenize` joins token bytes and decodes UTF-8 with replacement.
* :func:`prompt_tokens` turns a ``prompts/*.json`` file into the id list the
  harness prefills.
"""

from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from quettos.model import ModelSpec

# Keys of ``tokenizer_config.json`` that transformers exposes to templates via
# ``special_tokens_map`` (string values only; ``None`` entries are omitted).
SPECIAL_TOKEN_KEYS: tuple[str, ...] = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
)

TOKENS_BIN_MAGIC_COUNT_FMT = "<I"
TOKENS_BIN_LEN_FMT = "<H"


# --------------------------------------------------------------------------- template


def _raise_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(
    x: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    """``json.dumps`` with transformers' defaults (no HTML escaping, ``ensure_ascii=False``)."""
    return json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


def make_jinja_env() -> ImmutableSandboxedEnvironment:
    """The Jinja environment transformers uses for chat templates."""
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    return env


@lru_cache(maxsize=8)
def _compiled_template(source: str) -> jinja2.Template:
    return make_jinja_env().from_string(source)


def _special_token_string(value: Any) -> str | None:
    """A special token in ``tokenizer_config.json`` is a string or an AddedToken dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return str(value["content"])
    return str(value)


def load_tokenizer_config(spec: ModelSpec) -> dict[str, Any]:
    """Parsed ``tokenizer_config.json`` of the model."""
    return json.loads(spec.path("tokenizer_config.json").read_text(encoding="utf-8"))


def special_tokens_map(tokenizer_config: dict[str, Any]) -> dict[str, Any]:
    """Mirror of ``PreTrainedTokenizerBase.special_tokens_map`` built from the config file."""
    out: dict[str, Any] = {}
    for key in SPECIAL_TOKEN_KEYS:
        value = tokenizer_config.get(key)
        if value is None:
            continue
        if key == "additional_special_tokens":
            out[key] = [_special_token_string(v) for v in value]
        else:
            out[key] = _special_token_string(value)
    return out


def render_chat(
    spec: ModelSpec,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
    **template_kwargs: Any,
) -> str:
    """Render the model's chat template exactly like ``apply_chat_template(tokenize=False)``."""
    cfg = load_tokenizer_config(spec)
    template = _compiled_template(cfg["chat_template"])
    kwargs = {**special_tokens_map(cfg), **template_kwargs}
    return template.render(
        messages=messages,
        tools=tools,
        documents=None,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


# --------------------------------------------------------------------------- tokenizer


@lru_cache(maxsize=4)
def _tokenizer_from_path(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def load_tokenizer(spec: ModelSpec) -> Tokenizer:
    """The fast tokenizer built from the model's ``tokenizer.json``."""
    return _tokenizer_from_path(str(spec.path("tokenizer.json")))


def encode(spec: ModelSpec, text: str) -> list[int]:
    """Token ids of ``text`` without any implicit BOS/EOS (``add_special_tokens=False``)."""
    return load_tokenizer(spec).encode(text, add_special_tokens=False).ids


# --------------------------------------------------------------------------- byte-level vocab


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode-character map (used by byte-level BPE)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


_UNICODE_TO_BYTE: dict[str, int] = {c: b for b, c in bytes_to_unicode().items()}


def vocab_string_to_bytes(token: str) -> bytes:
    """Translate a byte-level BPE vocab string back to the raw bytes it stands for."""
    return bytes(_UNICODE_TO_BYTE[ch] for ch in token)


def token_bytes(spec: ModelSpec, count: int | None = None) -> list[bytes]:
    """Raw byte string for every id in ``range(count)`` (default: the model's ``vocab``).

    Ids that exist in the tokenizer as added/special tokens map to their
    literal UTF-8; regular BPE tokens go through the inverse byte-level map;
    ids past the tokenizer's vocabulary (Qwen pads its embedding to 151,936
    rows) map to ``b""``.
    """
    tok = load_tokenizer(spec)
    if count is None:
        count = spec.vocab
    added = {i: a.content for i, a in tok.get_added_tokens_decoder().items()}
    out: list[bytes] = []
    for i in range(count):
        if i in added:
            out.append(added[i].encode("utf-8"))
            continue
        s = tok.id_to_token(i)
        out.append(b"" if s is None else vocab_string_to_bytes(s))
    return out


def write_tokens_bin(spec: ModelSpec, out_path: str | Path, count: int | None = None) -> int:
    """Write ``tokens.bin`` (u32 count, then u16 length + bytes per id); return the byte count."""
    table = token_bytes(spec, count)
    chunks = [struct.pack(TOKENS_BIN_MAGIC_COUNT_FMT, len(table))]
    for b in table:
        if len(b) > 0xFFFF:
            raise ValueError("token longer than 65535 bytes")
        chunks.append(struct.pack(TOKENS_BIN_LEN_FMT, len(b)) + b)
    data = b"".join(chunks)
    Path(out_path).write_bytes(data)
    return len(data)


def read_tokens_bin(path: str | Path) -> list[bytes]:
    """Parse a ``tokens.bin`` written by :func:`write_tokens_bin`."""
    data = Path(path).read_bytes()
    (count,) = struct.unpack_from(TOKENS_BIN_MAGIC_COUNT_FMT, data, 0)
    pos = 4
    out: list[bytes] = []
    for _ in range(count):
        (n,) = struct.unpack_from(TOKENS_BIN_LEN_FMT, data, pos)
        pos += 2
        out.append(data[pos : pos + n])
        pos += n
    if pos != len(data):
        raise ValueError(f"tokens.bin has {len(data) - pos} trailing bytes")
    return out


def detokenize(table: list[bytes], ids: list[int]) -> str:
    """Concatenate the byte strings of ``ids`` and decode UTF-8 with replacement."""
    return b"".join(table[i] for i in ids).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- prompts


def load_prompt(prompt_json_path: str | Path) -> dict[str, Any]:
    """Load a ``prompts/*.json`` file: ``{"messages": [...], "tools": [...]?}``."""
    return json.loads(Path(prompt_json_path).read_text(encoding="utf-8"))


def render_prompt(spec: ModelSpec, prompt_json_path: str | Path) -> str:
    """Rendered prompt text for a prompt file (tools are passed only if present and non-empty)."""
    prompt = load_prompt(prompt_json_path)
    tools = prompt.get("tools") or None
    return render_chat(spec, prompt["messages"], tools=tools, add_generation_prompt=True)


def prompt_tokens(spec: ModelSpec, prompt_json_path: str | Path) -> list[int]:
    """Token ids the harness prefills for a prompt file (template + generation prompt)."""
    return encode(spec, render_prompt(spec, prompt_json_path))
