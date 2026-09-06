"""Build one RTL top with Verilator through cocotb's runner and run its cocotb tests.

``run(top, sources, test_module, ...)`` compiles ``rtl/<source>.sv`` files
(or absolute paths, for the wrappers under ``sim/cocotb/wrappers/``) plus
``rtl/qcore_pkg.sv`` on the tiny configuration (``TINY``), runs the
``@cocotb.test`` coroutines of ``test_module`` (a ``tb_*.py`` file in this
directory) and raises on any failed test.  Build products live under
``build/cocotb/<top>/<key>/``, where ``key`` is a digest of the sources and the
parameters, so two configurations of one top never share object files.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cocotb_tools.runner import get_results, get_runner

REPO = Path(__file__).resolve().parents[2]
RTL = REPO / "rtl"
ROM_DIR = RTL / "gen"
BUILD = REPO / "build" / "cocotb"
HERE = Path(__file__).resolve().parent

# The unit-test configuration (rtl/cfg/README.md, tiny_w16).
TINY: dict[str, int] = {
    "WB": 16,
    "B_MAX": 2,
    "VL": 2,
    "VSRAM_WORDS": 2048,
    "FIFO_BEATS": 128,
    "ACC_W": 40,
}

# ROM image parameters (untyped, empty default) by name.
ROM_FILES: dict[str, Path] = {
    "ROM_FILE": ROM_DIR / "exp2.hex",
    "ROM_FILE_EXP2": ROM_DIR / "exp2.hex",
    "ROM_FILE_SIGMOID": ROM_DIR / "sigmoid.hex",
    "ROM_FILE_RSQRT": ROM_DIR / "rsqrt.hex",
    "ROM_FILE_RECIP": ROM_DIR / "recip.hex",
}
_ROM_PARAM = re.compile(r"^\s*parameter\s+(?:string\s+)?(ROM_FILE\w*)")

# Verilator flags added to cocotb's own (-cc --exe --vpi --public-flat-rw ...).
VERILATOR_ARGS: tuple[str, ...] = (
    "--timing",  # cocotb 2.x drives Verilator's timing-aware scheduler
    "-Wall",
    "-Wpedantic",
    "--x-assign",
    "fast",
    "--x-initial",
    "unique",
    "--assert",
)
TIMESCALE = ("1ns", "1ps")


def _rom_params(top_file: Path) -> dict[str, str]:
    """``ROM_FILE*`` parameters declared by a top, mapped to their ``rtl/gen`` images."""
    out: dict[str, str] = {}
    for line in top_file.read_text(encoding="utf-8").splitlines():
        m = _ROM_PARAM.match(line)
        if m and m.group(1) in ROM_FILES:
            out[m.group(1)] = f'"{ROM_FILES[m.group(1)]}"'
    return out


def build_key(top: str, paths: Sequence[Path], params: Mapping[str, object]) -> str:
    """Digest of one build: the top, its sources and the parameters it is elaborated with."""
    material = [top, *(str(p) for p in paths), *(f"{k}={v}" for k, v in sorted(params.items()))]
    return hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()[:12]


def build(
    top: str,
    sources: Sequence[str],
    *,
    parameters: Mapping[str, object] | None = None,
    include_pkg: bool = True,
    waves: bool = False,
) -> tuple[Any, Path]:
    """Verilate ``top`` from ``rtl/<name>.sv`` sources; returns the runner and build directory."""
    files = list(sources)
    if include_pkg and "qcore_pkg.sv" not in files and top != "qcore_pkg":
        files.insert(0, "qcore_pkg.sv")
    paths = [Path(f) if Path(f).is_absolute() else RTL / f for f in files]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    top_file = next((p for p in paths if p.stem == top), RTL / f"{top}.sv")
    params: dict[str, object] = dict(_rom_params(top_file))
    if parameters:
        params.update(parameters)
    build_dir = BUILD / top / build_key(top, paths, params)
    build_dir.mkdir(parents=True, exist_ok=True)
    runner = get_runner("verilator")
    runner.build(
        sources=paths,
        hdl_toplevel=top,
        includes=[RTL],
        parameters=params,
        build_args=list(VERILATOR_ARGS),
        build_dir=build_dir,
        timescale=TIMESCALE,
        waves=waves,
        always=True,
    )
    return runner, build_dir


def run(
    top: str,
    sources: Sequence[str],
    test_module: str,
    *,
    parameters: Mapping[str, object] | None = None,
    testcase: str | Sequence[str] | None = None,
    seed: int | None = None,
    waves: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Build ``top``, run ``test_module``, assert zero failures; returns ``(tests, failures)``."""
    runner, build_dir = build(top, sources, parameters=parameters, waves=waves)
    env = {"PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")])}
    if extra_env:
        env.update(extra_env)
    results = runner.test(
        hdl_toplevel=top,
        test_module=test_module,
        build_dir=build_dir,
        test_dir=build_dir,
        testcase=testcase,
        seed=seed,
        waves=waves,
        extra_env=env,
        results_xml=str(build_dir / f"{test_module}.xml"),
    )
    tests, failures = get_results(Path(results))
    assert tests > 0, f"{test_module}: no tests ran"
    assert failures == 0, f"{test_module}: {failures} of {tests} tests failed (see {results})"
    return tests, failures
