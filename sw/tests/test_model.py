"""ModelSpec fields for both models against the documented model table."""

from __future__ import annotations

import pytest
from quettos.model import (
    ALIASES,
    MODEL_FILES,
    ModelSpec,
    expected_tensor_shapes,
    resolve_repo_id,
    sanitize_name,
)

EXPECTED: dict[str, dict[str, object]] = {
    "qwen": {
        "name": "qwen2.5-0.5b-instruct",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "arch": "qwen2",
        "layers": 24,
        "hidden": 896,
        "heads": 14,
        "kv_heads": 2,
        "head_dim": 64,
        "intermediate": 4864,
        "vocab": 151936,
        "has_qkv_bias": True,
        "rope_theta": 1e6,
        "rms_norm_eps": 1e-6,
        "tied_embeddings": True,
        "max_position_embeddings": 32768,
        "eos_ids": [151645, 151643],
        "bos_id": 151643,
    },
    "smollm2": {
        "name": "smollm2-135m-instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "arch": "llama",
        "layers": 30,
        "hidden": 576,
        "heads": 9,
        "kv_heads": 3,
        "head_dim": 64,
        "intermediate": 1536,
        "vocab": 49152,
        "has_qkv_bias": False,
        "rope_theta": 1e5,
        "rms_norm_eps": 1e-5,
        "tied_embeddings": True,
        "max_position_embeddings": 8192,
        # generation_config.json lists only <|im_end|> (2); <|endoftext|> (0) is not an EOS there.
        "eos_ids": [2],
        "bos_id": 1,
    },
}


def test_aliases_and_names() -> None:
    assert resolve_repo_id("qwen") == ALIASES["qwen"]
    assert resolve_repo_id("SmolLM2") == ALIASES["smollm2"]
    assert resolve_repo_id("Org/Other-Model") == "Org/Other-Model"
    assert sanitize_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b-instruct"


@pytest.mark.parametrize("alias", ["qwen", "smollm2"])
def test_spec_matches_table(alias: str, request: pytest.FixtureRequest) -> None:
    spec: ModelSpec = request.getfixturevalue(alias)
    for field_name, want in EXPECTED[alias].items():
        got = getattr(spec, field_name)
        assert got == want, f"{alias}.{field_name}: got {got!r}, want {want!r}"
    assert spec.model_dir.is_dir()
    for f in MODEL_FILES:
        assert spec.path(f).is_file(), f
    assert "build/models" in str(spec.model_dir)


def test_dimensions_are_multiples_of_64(spec: ModelSpec) -> None:
    for n in (
        spec.hidden,
        spec.intermediate,
        spec.vocab,
        spec.heads * spec.head_dim,
        spec.kv_heads * spec.head_dim,
    ):
        assert n % 64 == 0


def test_expected_tensor_count(spec: ModelSpec) -> None:
    per_layer = 12 if spec.has_qkv_bias else 9
    assert len(expected_tensor_shapes(spec)) == 2 + per_layer * spec.layers
