"""Model specification and Hugging Face download for the Quettos Core targets.

Two ungated Apache-2.0 models are supported: ``Qwen/Qwen2.5-0.5B-Instruct``
(headline) and ``HuggingFaceTB/SmolLM2-135M-Instruct`` (CI). Both are
Llama-shaped decoders; Qwen2 adds biases on the Q/K/V projections.

Everything lands under ``build/models/<sanitized-name>/`` inside the repo and
the Hugging Face cache lives under ``build/hf_cache``; ``~/.cache`` is never
touched. No token is needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
MODELS_DIR = BUILD_DIR / "models"
HF_CACHE_DIR = BUILD_DIR / "hf_cache"

ALIASES: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-135M-Instruct",
}

# HF config.model_type -> Quettos architecture name.
ARCH_BY_MODEL_TYPE: dict[str, str] = {
    "llama": "llama",
    "qwen2": "qwen2",
}

MODEL_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
)


@dataclass(frozen=True)
class ModelSpec:
    """Architecture facts derived from the model's ``config.json``.

    ``eos_ids`` comes from ``generation_config.json`` (falling back to
    ``config.json``) and preserves the order the model card lists them in.
    ``model_dir`` is the local directory holding the downloaded files.
    """

    name: str
    repo_id: str
    arch: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    has_qkv_bias: bool
    rope_theta: float
    rms_norm_eps: float
    tied_embeddings: bool
    max_position_embeddings: int
    eos_ids: list[int] = field(default_factory=list)
    bos_id: int | None = None
    model_dir: Path = MODELS_DIR

    def path(self, filename: str) -> Path:
        """Absolute path of ``filename`` inside :attr:`model_dir`."""
        return self.model_dir / filename

    def to_json(self) -> str:
        """Serialize the spec (paths as strings) for the CLI."""
        d = asdict(self)
        d["model_dir"] = str(self.model_dir)
        return json.dumps(d, indent=2)


def resolve_repo_id(repo_id_or_alias: str) -> str:
    """Map a short alias (``qwen``, ``smollm2``) to its Hub repo id; pass repo ids through."""
    return ALIASES.get(repo_id_or_alias.lower(), repo_id_or_alias)


def sanitize_name(repo_id: str) -> str:
    """``Qwen/Qwen2.5-0.5B-Instruct`` -> ``qwen2.5-0.5b-instruct`` (directory-safe, lower case)."""
    return repo_id.rsplit("/", 1)[-1].lower().replace(" ", "-")


def _as_id_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(v) for v in value]
    raise TypeError(f"unexpected token-id value: {value!r}")


def _first_id(value: object) -> int | None:
    ids = _as_id_list(value)
    return ids[0] if ids else None


def download_model(repo_id_or_alias: str, *, files: tuple[str, ...] = MODEL_FILES) -> Path:
    """Download ``files`` of the model into ``build/models/<name>/`` and return that directory.

    Uses :func:`huggingface_hub.snapshot_download` with ``allow_patterns`` so
    only the needed files are fetched; repeated calls are cache hits.
    """
    repo_id = resolve_repo_id(repo_id_or_alias)
    target = MODELS_DIR / sanitize_name(repo_id)
    target.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=list(files),
        cache_dir=str(HF_CACHE_DIR),
        local_dir=str(target),
        token=False,
    )
    missing = [f for f in files if not (target / f).exists()]
    if missing:
        raise FileNotFoundError(f"{repo_id}: files not downloaded: {missing}")
    return target


def spec_from_dir(repo_id: str, model_dir: Path) -> ModelSpec:
    """Build a :class:`ModelSpec` from already-downloaded files in ``model_dir``."""
    cfg = json.loads((model_dir / "config.json").read_text())
    gen_path = model_dir / "generation_config.json"
    gen = json.loads(gen_path.read_text()) if gen_path.exists() else {}

    model_type = cfg["model_type"]
    if model_type not in ARCH_BY_MODEL_TYPE:
        raise ValueError(f"{repo_id}: unsupported model_type {model_type!r}")
    arch = ARCH_BY_MODEL_TYPE[model_type]

    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg.get("head_dim", hidden // heads))

    if arch == "qwen2":
        # Qwen2 always carries Q/K/V biases (no config switch in transformers' Qwen2Attention).
        has_qkv_bias = True
    else:
        has_qkv_bias = bool(cfg.get("attention_bias", False))

    eos_ids = _as_id_list(gen.get("eos_token_id", cfg.get("eos_token_id")))
    bos_id = _first_id(gen.get("bos_token_id", cfg.get("bos_token_id")))

    return ModelSpec(
        name=sanitize_name(repo_id),
        repo_id=repo_id,
        arch=arch,
        layers=int(cfg["num_hidden_layers"]),
        hidden=hidden,
        heads=heads,
        kv_heads=int(cfg.get("num_key_value_heads", heads)),
        head_dim=head_dim,
        intermediate=int(cfg["intermediate_size"]),
        vocab=int(cfg["vocab_size"]),
        has_qkv_bias=has_qkv_bias,
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        rms_norm_eps=float(cfg["rms_norm_eps"]),
        tied_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        max_position_embeddings=int(cfg["max_position_embeddings"]),
        eos_ids=eos_ids,
        bos_id=bos_id,
        model_dir=model_dir,
    )


def load_spec(repo_id_or_alias: str) -> ModelSpec:
    """Download (if needed) and return the :class:`ModelSpec` for an alias or Hub repo id."""
    repo_id = resolve_repo_id(repo_id_or_alias)
    model_dir = download_model(repo_id)
    return spec_from_dir(repo_id, model_dir)


def expected_tensor_shapes(spec: ModelSpec) -> dict[str, tuple[int, ...]]:
    """Every tensor name and shape a checkpoint of ``spec`` must contain (HF naming).

    Tied models have no ``lm_head.weight``. Qwen2 adds Q/K/V biases.
    """
    h, kv = spec.hidden, spec.kv_heads * spec.head_dim
    q = spec.heads * spec.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (spec.vocab, h),
        "model.norm.weight": (h,),
    }
    if not spec.tied_embeddings:
        shapes["lm_head.weight"] = (spec.vocab, h)
    for i in range(spec.layers):
        p = f"model.layers.{i}."
        shapes[p + "input_layernorm.weight"] = (h,)
        shapes[p + "post_attention_layernorm.weight"] = (h,)
        shapes[p + "self_attn.q_proj.weight"] = (q, h)
        shapes[p + "self_attn.k_proj.weight"] = (kv, h)
        shapes[p + "self_attn.v_proj.weight"] = (kv, h)
        shapes[p + "self_attn.o_proj.weight"] = (h, q)
        if spec.has_qkv_bias:
            shapes[p + "self_attn.q_proj.bias"] = (q,)
            shapes[p + "self_attn.k_proj.bias"] = (kv,)
            shapes[p + "self_attn.v_proj.bias"] = (kv,)
        shapes[p + "mlp.gate_proj.weight"] = (spec.intermediate, h)
        shapes[p + "mlp.up_proj.weight"] = (spec.intermediate, h)
        shapes[p + "mlp.down_proj.weight"] = (h, spec.intermediate)
    return shapes
