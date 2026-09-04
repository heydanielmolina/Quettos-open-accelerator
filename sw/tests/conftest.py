"""Shared fixtures: model specs (downloaded on demand, skipped when offline)."""

from __future__ import annotations

import pytest
from quettos.model import ALIASES, ModelSpec, load_spec


def _try_load(alias: str) -> ModelSpec:
    try:
        return load_spec(alias)
    except Exception as exc:  # network / hub errors -> skip, never fail
        pytest.skip(f"cannot download {ALIASES[alias]}: {exc!r}")


@pytest.fixture(scope="session")
def qwen() -> ModelSpec:
    return _try_load("qwen")


@pytest.fixture(scope="session")
def smollm2() -> ModelSpec:
    return _try_load("smollm2")


@pytest.fixture(scope="session", params=["qwen", "smollm2"])
def spec(request: pytest.FixtureRequest) -> ModelSpec:
    """Parametrized over both supported models."""
    return _try_load(request.param)
