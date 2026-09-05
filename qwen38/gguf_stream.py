#!/usr/bin/env python3
"""Minimal streaming GGUF v3 directory parser for the Qwen3.8 K3 lab.

The parser deliberately does not mmap tensor data and does not materialize huge
metadata arrays (tokenizer tables are skipped unless explicitly requested).
It only turns the GGUF header/tensor directory into validated byte spans that a
separate K3 repacker/reader can consume.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import BinaryIO, Iterable

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_DIMS = 4

# gguf-py / llama.cpp pinned at 557614e0296ff4a5b6f649737a65ae2076eea2fd.
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor type ids used by ordinary dense GGUF quantization.  Keep these
# numerically pinned to the llama.cpp revision above; do not infer from file type.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_BF16 = 30

# (name, quant block elements, encoded bytes). Values mirror
# gguf-py/gguf/constants.py::GGML_QUANT_SIZES at the pinned llama.cpp commit.
GGML_TYPE_INFO: dict[int, tuple[str, int, int]] = {
    GGML_TYPE_F32: ("F32", 1, 4),
    GGML_TYPE_F16: ("F16", 1, 2),
    GGML_TYPE_Q4_0: ("Q4_0", 32, 18),
    GGML_TYPE_Q4_1: ("Q4_1", 32, 20),
    GGML_TYPE_Q5_0: ("Q5_0", 32, 22),
    GGML_TYPE_Q5_1: ("Q5_1", 32, 24),
    GGML_TYPE_Q8_0: ("Q8_0", 32, 34),
    GGML_TYPE_Q8_1: ("Q8_1", 32, 40),
    GGML_TYPE_Q2_K: ("Q2_K", 256, 84),
    GGML_TYPE_Q3_K: ("Q3_K", 256, 110),
    GGML_TYPE_Q4_K: ("Q4_K", 256, 144),
    GGML_TYPE_Q5_K: ("Q5_K", 256, 176),
    GGML_TYPE_Q6_K: ("Q6_K", 256, 210),
    GGML_TYPE_Q8_K: ("Q8_K", 256, 292),
    GGML_TYPE_I8: ("I8", 1, 1),
    GGML_TYPE_I16: ("I16", 1, 2),
    GGML_TYPE_I32: ("I32", 1, 4),
    GGML_TYPE_I64: ("I64", 1, 8),
    GGML_TYPE_F64: ("F64", 1, 8),
    GGML_TYPE_BF16: ("BF16", 1, 2),
}

# Exact encoded block layouts used by llama.cpp for the primary gold path.
Q6_K_LAYOUT = {
    "ql": (0, 128),
    "qh": (128, 192),
    "scales": (192, 208),
    "d": (208, 210),
}
Q8_0_LAYOUT = {
    "d": (0, 2),
    "qs": (2, 34),
}

_WANTED_DEFAULT = frozenset({
    "general.alignment",
    "general.architecture",
    "general.file_type",
    "qwen35.block_count",
    "qwen35.embedding_length",
    "qwen35.full_attention_interval",
})
_MAX_STRING_BYTES = 1 << 30
_MAX_CONTAINER_ITEMS = 10_000_000


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorSpan:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    type_name: str
    relative_offset: int
    data_offset: int
    nbytes: int
    block_elements: int
    block_bytes: int

    @property
    def end_offset(self) -> int:
        return self.data_offset + self.nbytes

    @property
    def n_elements(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


@dataclass(frozen=True)
class GGUFDirectory:
    path: Path
    version: int
    file_bytes: int
    tensor_count: int
    kv_count: int
    alignment: int
    data_offset: int
    metadata: dict[str, object]
    tensors: tuple[TensorSpan, ...]

    def by_name(self) -> dict[str, TensorSpan]:
        return {tensor.name: tensor for tensor in self.tensors}

    def tensors_with_prefix(self, prefix: str) -> tuple[TensorSpan, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.name.startswith(prefix))


class _Reader:
    def __init__(self, fp: BinaryIO, file_bytes: int):
        self.fp = fp
        self.file_bytes = file_bytes

    def tell(self) -> int:
        return self.fp.tell()

    def _need(self, n: int) -> None:
        if n < 0 or self.tell() + n > self.file_bytes:
            raise GGUFError(f"truncated GGUF at offset {self.tell()} requesting {n} bytes")

    def read_exact(self, n: int) -> bytes:
        self._need(n)
        data = self.fp.read(n)
        if len(data) != n:
            raise GGUFError(f"short read at offset {self.tell() - len(data)}")
        return data

    def unpack(self, fmt: str):
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read_exact(size))[0]

    def u32(self) -> int:
        return int(self.unpack("I"))

    def u64(self) -> int:
        return int(self.unpack("Q"))

    def string(self, *, materialize: bool = True) -> str | None:
        n = self.u64()
        if n > _MAX_STRING_BYTES:
            raise GGUFError(f"string length {n} exceeds parser safety cap")
        self._need(n)
        if not materialize:
            self.fp.seek(n, os.SEEK_CUR)
            return None
        raw = self.read_exact(n)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("invalid UTF-8 string in GGUF metadata/directory") from exc

    def skip(self, n: int) -> None:
        self._need(n)
        self.fp.seek(n, os.SEEK_CUR)


_SCALAR_FORMATS: dict[int, tuple[str, int]] = {
    GGUF_TYPE_UINT8: ("B", 1),
    GGUF_TYPE_INT8: ("b", 1),
    GGUF_TYPE_UINT16: ("H", 2),
    GGUF_TYPE_INT16: ("h", 2),
    GGUF_TYPE_UINT32: ("I", 4),
    GGUF_TYPE_INT32: ("i", 4),
    GGUF_TYPE_FLOAT32: ("f", 4),
    GGUF_TYPE_BOOL: ("?", 1),
    GGUF_TYPE_UINT64: ("Q", 8),
    GGUF_TYPE_INT64: ("q", 8),
    GGUF_TYPE_FLOAT64: ("d", 8),
}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _validate_alignment(value: object) -> int:
    if not isinstance(value, int):
        raise GGUFError(f"general.alignment must be an integer, got {type(value).__name__}")
    if value <= 0 or value & (value - 1):
        raise GGUFError(f"general.alignment must be a positive power of two, got {value}")
    if value > (1 << 20):
        raise GGUFError(f"general.alignment is implausibly large: {value}")
    return value


def _read_value(r: _Reader, value_type: int, *, materialize: bool) -> object | None:
    if value_type in _SCALAR_FORMATS:
        fmt, n = _SCALAR_FORMATS[value_type]
        if not materialize:
            r.skip(n)
            return None
        return r.unpack(fmt)

    if value_type == GGUF_TYPE_STRING:
        return r.string(materialize=materialize)

    if value_type == GGUF_TYPE_ARRAY:
        elem_type = r.u32()
        count = r.u64()
        if count > _MAX_CONTAINER_ITEMS:
            raise GGUFError(f"metadata array length {count} exceeds parser safety cap")
        if not materialize:
            if elem_type in _SCALAR_FORMATS:
                r.skip(_SCALAR_FORMATS[elem_type][1] * count)
                return None
            for _ in range(count):
                _read_value(r, elem_type, materialize=False)
            return None
        return [_read_value(r, elem_type, materialize=True) for _ in range(count)]

    raise GGUFError(f"unsupported GGUF metadata value type {value_type}")


def tensor_nbytes(shape: Iterable[int], ggml_type: int) -> tuple[int, str, int, int]:
    try:
        type_name, block_elements, block_bytes = GGML_TYPE_INFO[ggml_type]
    except KeyError as exc:
        raise GGUFError(f"unsupported GGML tensor type {ggml_type}") from exc

    dims = tuple(int(v) for v in shape)
    if not dims or len(dims) > GGML_MAX_DIMS:
        raise GGUFError(f"tensor dimension count must be in [1,{GGML_MAX_DIMS}], got {len(dims)}")
    for dim in dims:
        if dim <= 0:
            raise GGUFError(f"tensor dimensions must be positive, got {dims}")

    if dims[0] % block_elements:
        raise GGUFError(
            f"{type_name} tensor row width {dims[0]} is not divisible by block size {block_elements}"
        )
    rows = 1
    for dim in dims[1:]:
        rows *= dim
    nbytes = (dims[0] // block_elements) * block_bytes * rows
    return nbytes, type_name, block_elements, block_bytes


def parse_gguf(
    path: str | os.PathLike[str],
    *,
    wanted_metadata: Iterable[str] = _WANTED_DEFAULT,
    validate_spans: bool = True,
) -> GGUFDirectory:
    """Parse only the GGUF v3 directory and selected metadata.

    Tensor data itself is never mmap'd/read. Unselected metadata is skipped in
    place, including large tokenizer arrays. Tensor entries are all collected
    before type validation so one expensive real-file scan reports every
    unsupported GGML type id at once instead of failing on the first tensor.
    """
    path_obj = Path(path)
    file_bytes = path_obj.stat().st_size
    wanted = frozenset(wanted_metadata)

    with path_obj.open("rb") as fp:
        r = _Reader(fp, file_bytes)
        if r.read_exact(4) != GGUF_MAGIC:
            raise GGUFError("bad GGUF magic")
        version = r.u32()
        if version != GGUF_VERSION:
            raise GGUFError(f"only GGUF v{GGUF_VERSION} is supported, got v{version}")

        tensor_count = r.u64()
        kv_count = r.u64()
        if tensor_count > _MAX_CONTAINER_ITEMS or kv_count > _MAX_CONTAINER_ITEMS:
            raise GGUFError("GGUF header count exceeds parser safety cap")

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = r.string(materialize=True)
            assert key is not None
            value_type = r.u32()
            keep = key in wanted
            value = _read_value(r, value_type, materialize=keep)
            if keep:
                metadata[key] = value

        alignment = _validate_alignment(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        seen_names: set[str] = set()
        for _ in range(tensor_count):
            name = r.string(materialize=True)
            assert name is not None
            if name in seen_names:
                raise GGUFError(f"duplicate tensor name: {name}")
            seen_names.add(name)

            n_dims = r.u32()
            if not 1 <= n_dims <= GGML_MAX_DIMS:
                raise GGUFError(f"{name}: invalid dimension count {n_dims}")
            shape = tuple(r.u64() for _ in range(n_dims))
            ggml_type = r.u32()
            relative_offset = r.u64()
            raw_tensors.append((name, shape, ggml_type, relative_offset))

        data_offset = _align_up(r.tell(), alignment)
        if data_offset > file_bytes:
            raise GGUFError("tensor data offset is beyond end of file")

        unsupported = sorted({ggml_type for _, _, ggml_type, _ in raw_tensors if ggml_type not in GGML_TYPE_INFO})
        if unsupported:
            raise GGUFError(f"unsupported GGML tensor types {unsupported}")

        tensors_list: list[TensorSpan] = []
        for name, shape, ggml_type, relative_offset in raw_tensors:
            nbytes, type_name, block_elements, block_bytes = tensor_nbytes(shape, ggml_type)
            tensors_list.append(TensorSpan(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                type_name=type_name,
                relative_offset=relative_offset,
                data_offset=data_offset + relative_offset,
                nbytes=nbytes,
                block_elements=block_elements,
                block_bytes=block_bytes,
            ))
        tensors = tuple(tensors_list)

        if validate_spans:
            ordered = sorted(tensors, key=lambda t: (t.data_offset, t.end_offset, t.name))
            previous_end = data_offset
            for tensor in ordered:
                if tensor.relative_offset % alignment:
                    raise GGUFError(
                        f"{tensor.name}: relative tensor offset {tensor.relative_offset} "
                        f"is not aligned to {alignment}"
                    )
                if tensor.data_offset < data_offset:
                    raise GGUFError(f"{tensor.name}: tensor starts before data section")
                if tensor.end_offset > file_bytes:
                    raise GGUFError(
                        f"{tensor.name}: tensor span [{tensor.data_offset},{tensor.end_offset}) "
                        f"exceeds file size {file_bytes}"
                    )
                if tensor.data_offset < previous_end:
                    raise GGUFError(f"{tensor.name}: tensor span overlaps prior tensor")
                previous_end = tensor.end_offset

        return GGUFDirectory(
            path=path_obj,
            version=version,
            file_bytes=file_bytes,
            tensor_count=tensor_count,
            kv_count=kv_count,
            alignment=alignment,
            data_offset=data_offset,
            metadata=metadata,
            tensors=tensors,
        )
