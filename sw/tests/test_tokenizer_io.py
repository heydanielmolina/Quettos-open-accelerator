"""Chat-template rendering and tokenization against transformers; tokens.bin byte fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from quettos.model import ModelSpec
from quettos.tokenizer_io import (
    bytes_to_unicode,
    detokenize,
    encode,
    load_prompt,
    prompt_tokens,
    read_tokens_bin,
    render_chat,
    render_prompt,
    token_bytes,
    vocab_string_to_bytes,
    write_tokens_bin,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS = [
    REPO_ROOT / "prompts" / n
    for n in ("chat_short.json", "tool_call_weather.json", "long_512.json")
]

# Rendered prompt token counts (Qwen template injects a default system prompt and the tools block).
EXPECTED_COUNTS = {
    ("qwen2", "chat_short.json"): 36,
    ("qwen2", "tool_call_weather.json"): 180,
    ("qwen2", "long_512.json"): 515,
    ("llama", "chat_short.json"): 37,
    ("llama", "tool_call_weather.json"): 39,
    ("llama", "long_512.json"): 520,
}

ROUNDTRIP_STRINGS = [
    "Hello, world!",
    "line one\nline two\n\n\tindented",
    "日本語のテキストと中文和한국어",
    "emoji: 🚀🔥🧠 and flags 🇫🇷",
    '{"name": "get_weather", "arguments": {"city": "Paris"}}',
    '<tool_call>\n{"a": [1, 2, 3]}\n</tool_call>',
    "  leading and trailing spaces  ",
    "mixed: café, naïve, Ærø, ß, ¿qué?",
]


@pytest.fixture(scope="module")
def hf_tokenizer_factory():
    transformers = pytest.importorskip("transformers")
    cache: dict[str, object] = {}

    def get(spec: ModelSpec):
        if spec.repo_id not in cache:
            cache[spec.repo_id] = transformers.AutoTokenizer.from_pretrained(str(spec.model_dir))
        return cache[spec.repo_id]

    return get


def _tools_for(spec: ModelSpec, prompt: dict) -> list | None:
    # Qwen's template renders tools; SmolLM2 has no tools template, so render without them.
    return (prompt.get("tools") or None) if spec.arch == "qwen2" else None


@pytest.mark.parametrize("prompt_path", PROMPTS, ids=lambda p: p.name)
def test_render_matches_transformers(
    spec: ModelSpec, prompt_path: Path, hf_tokenizer_factory
) -> None:
    hf = hf_tokenizer_factory(spec)
    prompt = load_prompt(prompt_path)
    tools = _tools_for(spec, prompt)
    ours = render_chat(spec, prompt["messages"], tools=tools, add_generation_prompt=True)
    ref = hf.apply_chat_template(
        prompt["messages"], tools=tools, tokenize=False, add_generation_prompt=True
    )
    assert ours == ref
    assert render_prompt(spec, prompt_path) == ref


@pytest.mark.parametrize("prompt_path", PROMPTS, ids=lambda p: p.name)
def test_ids_match_transformers(spec: ModelSpec, prompt_path: Path, hf_tokenizer_factory) -> None:
    hf = hf_tokenizer_factory(spec)
    prompt = load_prompt(prompt_path)
    tools = _tools_for(spec, prompt)
    ref = hf.apply_chat_template(
        prompt["messages"], tools=tools, tokenize=True, add_generation_prompt=True
    )
    ref_ids = ref if isinstance(ref, list) else list(ref["input_ids"])
    ours = prompt_tokens(spec, prompt_path)
    assert ours == ref_ids
    assert len(ours) == EXPECTED_COUNTS[(spec.arch, prompt_path.name)]


def test_qwen_tools_block(qwen: ModelSpec) -> None:
    prompt = load_prompt(PROMPTS[1])
    text = render_chat(qwen, prompt["messages"], tools=prompt["tools"])
    assert text.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
    assert "# Tools" in text and "<tools>\n" in text
    assert json.dumps(prompt["tools"][0], ensure_ascii=False) in text
    assert text.endswith("<|im_start|>assistant\n")


def test_bytes_to_unicode_is_a_bijection() -> None:
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256
    assert vocab_string_to_bytes("".join(m[b] for b in range(256))) == bytes(range(256))


def test_tokens_bin_roundtrip(spec: ModelSpec, tmp_path: Path) -> None:
    table = token_bytes(spec)
    assert len(table) == spec.vocab
    out = tmp_path / "tokens.bin"
    write_tokens_bin(spec, out)
    assert read_tokens_bin(out) == table
    for text in ROUNDTRIP_STRINGS:
        ids = encode(spec, text)
        assert b"".join(table[i] for i in ids) == text.encode("utf-8"), text
        assert detokenize(table, ids) == text
    # Special tokens carry their literal string.
    im_end = encode(spec, "<|im_end|>")
    assert len(im_end) == 1 and table[im_end[0]] == b"<|im_end|>"
    for eos in spec.eos_ids:
        assert table[eos].startswith(b"<|")
