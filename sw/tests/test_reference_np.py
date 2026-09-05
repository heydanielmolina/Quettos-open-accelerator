"""Float32 reference forward: structure, hooks, RoPE table agreement and the HF fp32 oracle."""

from __future__ import annotations

import numpy as np
import pytest
from quettos import numerics, reference_np
from quettos.model import REPO_ROOT, ModelSpec
from quettos.tokenizer_io import prompt_tokens

PROMPTS = ("chat_short.json", "tool_call_weather.json")
HF_MAX_ABS_LOGIT_DIFF = 2e-2


def _ids(spec: ModelSpec, prompt: str) -> list[int]:
    return prompt_tokens(spec, REPO_ROOT / "prompts" / prompt)


def test_forward_shapes_and_hooks(spec: ModelSpec) -> None:
    ids = _ids(spec, "chat_short.json")
    t = len(ids)
    cap = reference_np.Capture()
    logits = reference_np.forward(spec, ids, hooks=cap)
    assert logits.shape == (t, spec.vocab)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))
    assert np.array_equal(cap.get("logits"), logits)

    h, kv, d = spec.heads, spec.kv_heads, spec.head_dim
    want = {
        ("embed", None): (t, spec.hidden),
        ("x_norm_final", None): (t, spec.hidden),
        ("x_norm_attn", 0): (t, spec.hidden),
        ("q", 0): (t, h * d),
        ("k", 0): (t, kv * d),
        ("v", 0): (t, kv * d),
        ("v_raw", 0): (t, kv * d),
        ("q_rope", 0): (t, h * d),
        ("k_rope", 0): (t, kv * d),
        ("scores", 0): (h, t, t),
        ("ctx", 0): (t, h * d),
        ("x_attn", 0): (t, spec.hidden),
        ("x_norm_mlp", 0): (t, spec.hidden),
        ("gate", 0): (t, spec.intermediate),
        ("up", 0): (t, spec.intermediate),
        ("h", 0): (t, spec.intermediate),
        ("x", spec.layers - 1): (t, spec.hidden),
    }
    for key, shape in want.items():
        assert cap.values[key].shape == shape, key
    recorded = {name for name, _ in cap.values}
    assert recorded == set(reference_np.HOOK_NAMES)

    # The embedding rows come straight from the table.
    table = reference_np.load_embedding(spec)
    assert np.array_equal(cap.get("embed"), table[np.asarray(ids)])
    # Scores outside the causal window are zero; inside they are finite.
    scores = cap.get("scores", 0)
    upper = ~reference_np.causal_mask(t)
    assert np.all(scores[:, upper] == 0.0)
    assert np.all(np.isfinite(scores))
    # V without bias differs from V exactly by the bias on Qwen2 and is identical on Llama.
    if spec.has_qkv_bias:
        assert not np.array_equal(cap.get("v", 0), cap.get("v_raw", 0))
    else:
        assert np.array_equal(cap.get("v", 0), cap.get("v_raw", 0))


def test_causal_masking(spec: ModelSpec) -> None:
    ids = _ids(spec, "chat_short.json")
    base = reference_np.forward(spec, ids, layers=2)
    changed = list(ids)
    changed[-1] = (changed[-1] + 1) % spec.vocab
    other = reference_np.forward(spec, changed, layers=2)
    assert np.max(np.abs(base[:-1] - other[:-1])) <= 1e-4
    assert np.max(np.abs(base[-1] - other[-1])) > 1e-2


def test_forward_rejects_bad_input(spec: ModelSpec) -> None:
    with pytest.raises(ValueError):
        reference_np.forward(spec, [])
    with pytest.raises(ValueError):
        reference_np.forward(spec, [spec.vocab])


def test_rope_matches_checked_in_table(spec: ModelSpec) -> None:
    """The float32 cos/sin agree with the exact Q1.14 hardware table to 2 LSB at every position."""
    table = numerics.load_rope_table(spec.rope_theta)
    positions = np.arange(table.shape[0])
    cos, sin = reference_np.rope_cos_sin(positions, spec.rope_theta, spec.head_dim)
    half = spec.head_dim // 2
    cos_q = np.floor(cos[:, :half].astype(np.float64) * (1 << 14) + 0.5)
    sin_q = np.floor(sin[:, :half].astype(np.float64) * (1 << 14) + 0.5)
    assert np.max(np.abs(cos_q - table[:, 0])) <= 2
    assert np.max(np.abs(sin_q - table[:, 1])) <= 2


def test_gqa_and_softmax_helpers() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(reference_np.rotate_half(x), [[-3.0, -4.0, 1.0, 2.0]])
    p = reference_np.softmax_rows(np.array([[0.0, -np.inf, 0.0]], dtype=np.float32))
    assert np.allclose(p, [[0.5, 0.0, 0.5]])
    assert p[0, 1] == 0.0


@pytest.mark.parametrize("prompt", PROMPTS)
def test_matches_transformers_fp32(spec: ModelSpec, prompt: str) -> None:
    """Argmax agreement at every position and max |logit diff| below 2e-2 against HF fp32."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    ids = _ids(spec, prompt)
    ours = reference_np.forward(spec, ids)
    theirs = reference_np.hf_logits(spec, ids)
    result = reference_np.compare_logits(ours, theirs)
    print(f"{spec.name} {prompt} T={len(ids)}: {result}")
    assert result["argmax_agreement"] == 1.0
    assert result["max_abs_diff"] < HF_MAX_ABS_LOGIT_DIFF
