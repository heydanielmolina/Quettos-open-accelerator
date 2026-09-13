"""Quettos Core software stack.

Host-side tooling for the Quettos Core accelerator: model download and
specification (:mod:`quettos.model`), a numpy-only bf16 safetensors reader
(:mod:`quettos.safetensors_np`), chat-template rendering and tokenizer export
(:mod:`quettos.tokenizer_io`), the integer numerics the RTL mirrors bit for bit
(:mod:`quettos.numerics`) with their lookup and RoPE tables
(:mod:`quettos.lutgen`), the float32 reference forward
(:mod:`quettos.reference_np`), calibration (:mod:`quettos.calibrate`), the int8
quantizer (:mod:`quettos.quantize`), the per-GEMV requant constants
(:mod:`quettos.program`), the bit-exact golden model (:mod:`quettos.golden`),
its quality against fp32 (:mod:`quettos.quality`) over the calibration set and
the held-out corpus (:mod:`quettos.corpus`), and the CLI (:mod:`quettos.cli`).

The hardware side of the same stack: the descriptor ISA (:mod:`quettos.isa`)
and the headers generated from it (:mod:`quettos.csrgen`), the image and
program compiler (:mod:`quettos.compiler`) with the random models it is
exercised on (:mod:`quettos.synthetic`), the ISA simulator
(:mod:`quettos.isa_sim`), the three comparisons of ``qcore_top`` against it
(:mod:`quettos.compare`, :mod:`quettos.stepcmp`, :mod:`quettos.determinism`),
and the prefix save/restore that drives the tool-call demo
(:mod:`quettos.prefix`).
"""

__version__ = "0.1.0"
