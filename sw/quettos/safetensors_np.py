"""Numpy-only reader for ``.safetensors`` files, including bf16 tensors.

The official ``safetensors.numpy`` API cannot load ``BF16`` (numpy has no
bfloat16 dtype), so this module parses the container itself: an 8-byte
little-endian header length, a JSON header mapping tensor names to
``{dtype, shape, data_offsets}``, then the raw byte buffer. bf16 payloads are
read as ``uint16`` and shifted into the high half of a ``uint32`` which is
then viewed as ``float32`` -- an exact conversion, since bf16 is the top 16
bits of an IEEE float32.

Every tensor is returned as ``float32`` regardless of its stored dtype so the
quantizer downstream has one code path. The file is memory-mapped, so
iterating a 1 GB checkpoint does not load it all at once.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np

HEADER_LEN_BYTES = 8

# safetensors dtype tag -> (numpy dtype for the raw payload, itemsize).
_RAW_DTYPES: dict[str, np.dtype] = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "BOOL": np.dtype("?"),
}


class TensorInfo(TypedDict):
    """One entry of the safetensors JSON header."""

    dtype: str
    shape: list[int]
    data_offsets: list[int]


def read_header(path: str | Path) -> tuple[dict[str, TensorInfo], int]:
    """Return ``(header, data_start)`` where ``data_start`` is the byte offset of the buffer.

    The ``__metadata__`` entry, if present, is dropped from the returned header.
    """
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(HEADER_LEN_BYTES))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, HEADER_LEN_BYTES + header_len


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exactly widen an array of bf16 bit patterns (``uint16``) to ``float32``."""
    if bits.dtype != np.uint16:
        raise TypeError(f"expected uint16 bf16 bit patterns, got {bits.dtype}")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _decode(raw: np.ndarray, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == "BF16":
        out = bf16_bits_to_f32(np.ascontiguousarray(raw).view(np.uint16))
    else:
        out = raw.astype(np.float32)
    return out.reshape(shape)


def _tensor_from_mmap(mm: np.memmap, data_start: int, info: TensorInfo) -> np.ndarray:
    dtype = info["dtype"]
    if dtype not in _RAW_DTYPES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}")
    begin, end = info["data_offsets"]
    raw_dtype = _RAW_DTYPES[dtype]
    nbytes = end - begin
    if nbytes % raw_dtype.itemsize:
        raise ValueError(f"payload of {nbytes} bytes is not a multiple of {raw_dtype.itemsize}")
    raw = np.frombuffer(
        mm, dtype=raw_dtype, count=nbytes // raw_dtype.itemsize, offset=data_start + begin
    )
    return _decode(raw, dtype, info["shape"])


def iter_tensors(path: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, float32 array)`` for every tensor in header order (sorted by name)."""
    header, data_start = read_header(path)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    for name in sorted(header):
        yield name, _tensor_from_mmap(mm, data_start, header[name])


def load_tensor(path: str | Path, name: str) -> np.ndarray:
    """Load a single tensor by name as ``float32``."""
    header, data_start = read_header(path)
    if name not in header:
        raise KeyError(f"{name!r} not in {path}")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return _tensor_from_mmap(mm, data_start, header[name])


def tensor_shapes(path: str | Path) -> dict[str, tuple[int, ...]]:
    """Return ``{name: shape}`` for every tensor without reading any payload."""
    header, _ = read_header(path)
    return {name: tuple(info["shape"]) for name, info in header.items()}


def parameter_count(path: str | Path) -> int:
    """Total number of elements across all tensors (from the header only)."""
    header, _ = read_header(path)
    return sum(int(np.prod(info["shape"], dtype=np.int64)) for info in header.values())
