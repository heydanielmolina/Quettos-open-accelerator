"""Quettos Core software stack.

This package holds the host-side tooling for the Quettos Core accelerator:
model download and specification (:mod:`quettos.model`), a numpy-only bf16
safetensors reader (:mod:`quettos.safetensors_np`), chat-template rendering
and tokenizer export (:mod:`quettos.tokenizer_io`), the integer numerics that
the RTL mirrors bit for bit (:mod:`quettos.numerics`) with their lookup and
RoPE tables (:mod:`quettos.lutgen`), the float32 reference forward
(:mod:`quettos.reference_np`), activation calibration
(:mod:`quettos.calibrate`), the int8 quantizer (:mod:`quettos.quantize`) and
the ``quettos`` command-line entry point (:mod:`quettos.cli`).
"""

__version__ = "0.1.0"
