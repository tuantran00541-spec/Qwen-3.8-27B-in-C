#!/usr/bin/env python3
"""Synthetic zero-model gate for gguf_stream.py."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile

from gguf_stream import (
    GGML_TYPE_F32,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGUFError,
    Q6_K_LAYOUT,
    Q8_0_LAYOUT,
    parse_gguf,
    tensor_nbytes,
)

TYPE_UINT32 = 4
TYPE_STRING = 8
TYPE_ARRAY = 9


def pack_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def pack_kv(key: str, value_type: int, payload: bytes) -> bytes:
    return pack_string(key) + struct.pack("<I", value_type) + payload


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def tensor_info(name: str, shape: tuple[int, ...], ggml_type: int, rel: int) -> bytes:
    out = bytearray(pack_string(name))
    out += struct.pack("<I", len(shape))
    for dim in shape:
        out += struct.pack("<Q", dim)
    out += struct.pack("<IQ", ggml_type, rel)
    return bytes(out)


def build_fixture(path: Path) -> dict[str, int]:
    alignment = 64
    token_payload = struct.pack("<IQ", TYPE_STRING, 5)
    for token in ("a", "bb", "ccc", "dddd", "eeeee"):
        token_payload += pack_string(token)

    kvs = [
        pack_kv("general.architecture", TYPE_STRING, pack_string("qwen35")),
        pack_kv("general.alignment", TYPE_UINT32, struct.pack("<I", alignment)),
        pack_kv("tokenizer.ggml.tokens", TYPE_ARRAY, token_payload),
    ]

    q6_bytes = tensor_nbytes((256, 2), GGML_TYPE_Q6_K)[0]
    q4_bytes = tensor_nbytes((32, 2), GGML_TYPE_Q4_0)[0]
    q8_bytes = tensor_nbytes((32, 3), GGML_TYPE_Q8_0)[0]
    f32_bytes = tensor_nbytes((4,), GGML_TYPE_F32)[0]

    rel_q6 = 0
    rel_q4 = align_up(rel_q6 + q6_bytes, alignment)
    rel_q8 = align_up(rel_q4 + q4_bytes, alignment)
    rel_f32 = align_up(rel_q8 + q8_bytes, alignment)

    tensors = [
        tensor_info("blk.0.ffn_gate.weight", (256, 2), GGML_TYPE_Q6_K, rel_q6),
        tensor_info("blk.0.ssm_conv1d.weight", (32, 2), GGML_TYPE_Q4_0, rel_q4),
        tensor_info("token_embd.weight", (32, 3), GGML_TYPE_Q8_0, rel_q8),
        tensor_info("output_norm.weight", (4,), GGML_TYPE_F32, rel_f32),
    ]

    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), len(kvs))
    header += b"".join(kvs)
    header += b"".join(tensors)
    data_offset = align_up(len(header), alignment)
    header += b"\x00" * (data_offset - len(header))

    total = data_offset + rel_f32 + f32_bytes
    blob = bytearray(total)
    blob[:len(header)] = header
    blob[data_offset + rel_q6:data_offset + rel_q6 + q6_bytes] = b"\x61" * q6_bytes
    blob[data_offset + rel_q4:data_offset + rel_q4 + q4_bytes] = b"\x42" * q4_bytes
    blob[data_offset + rel_q8:data_offset + rel_q8 + q8_bytes] = b"\x82" * q8_bytes
    blob[data_offset + rel_f32:data_offset + rel_f32 + f32_bytes] = b"\x43" * f32_bytes
    path.write_bytes(blob)
    return {
        "alignment": alignment,
        "data_offset": data_offset,
        "q6_bytes": q6_bytes,
        "q4_bytes": q4_bytes,
        "q8_bytes": q8_bytes,
        "f32_bytes": f32_bytes,
    }


def main() -> None:
    checks: list[str] = []
    with tempfile.TemporaryDirectory(prefix="qwen38-gguf-") as tmp:
        path = Path(tmp) / "synthetic.gguf"
        expected = build_fixture(path)
        directory = parse_gguf(path)

        assert directory.version == 3
        assert directory.alignment == expected["alignment"]
        assert directory.data_offset == expected["data_offset"]
        assert directory.metadata["general.architecture"] == "qwen35"
        assert "tokenizer.ggml.tokens" not in directory.metadata
        checks.append("streaming metadata skip")

        by_name = directory.by_name()
        q6 = by_name["blk.0.ffn_gate.weight"]
        q4 = by_name["blk.0.ssm_conv1d.weight"]
        q8 = by_name["token_embd.weight"]
        f32 = by_name["output_norm.weight"]
        assert q6.nbytes == expected["q6_bytes"] == 420
        assert q4.nbytes == expected["q4_bytes"] == 36
        assert q8.nbytes == expected["q8_bytes"] == 102
        assert f32.nbytes == expected["f32_bytes"] == 16
        checks.append("tensor block-size accounting incl Q4_0")

        raw = path.read_bytes()
        assert raw[q6.data_offset:q6.end_offset] == b"\x61" * q6.nbytes
        assert raw[q4.data_offset:q4.end_offset] == b"\x42" * q4.nbytes
        assert raw[q8.data_offset:q8.end_offset] == b"\x82" * q8.nbytes
        assert raw[f32.data_offset:f32.end_offset] == b"\x43" * f32.nbytes
        checks.append("absolute span addressing")

        expected_blocks = {
            GGML_TYPE_Q4_0: (32, 18),
            GGML_TYPE_Q4_1: (32, 20),
            GGML_TYPE_Q5_0: (32, 22),
            GGML_TYPE_Q5_1: (32, 24),
            GGML_TYPE_Q8_0: (32, 34),
            GGML_TYPE_Q8_1: (32, 40),
            GGML_TYPE_Q2_K: (256, 84),
            GGML_TYPE_Q3_K: (256, 110),
            GGML_TYPE_Q4_K: (256, 144),
            GGML_TYPE_Q5_K: (256, 176),
            GGML_TYPE_Q6_K: (256, 210),
            GGML_TYPE_Q8_K: (256, 292),
        }
        for ggml_type, (block_elements, block_bytes) in expected_blocks.items():
            nbytes, _, got_elements, got_bytes = tensor_nbytes((block_elements,), ggml_type)
            assert (got_elements, got_bytes, nbytes) == (block_elements, block_bytes, block_bytes)
        checks.append("pinned classic/K GGML quant-size table")

        assert Q6_K_LAYOUT == {
            "ql": (0, 128),
            "qh": (128, 192),
            "scales": (192, 208),
            "d": (208, 210),
        }
        assert Q8_0_LAYOUT == {"d": (0, 2), "qs": (2, 34)}
        checks.append("pinned Q6_K/Q8_0 block layout")

        for invalid_shape in ((255,), (1, 256)):
            try:
                tensor_nbytes(invalid_shape, GGML_TYPE_Q6_K)
            except GGUFError:
                pass
            else:
                raise AssertionError(f"Q6_K invalid row layout was accepted: {invalid_shape}")
        checks.append("invalid quant row layout rejected")

    result = {
        "schema": "qwen38-gguf-stream-sanity-v2",
        "status": "PASS",
        "checks": checks,
        "q4_0": {"elements_per_block": 32, "bytes_per_block": 18, "bpw": 4.5},
        "q6_k": {"elements_per_block": 256, "bytes_per_block": 210, "bpw": 6.5625},
        "q8_0": {"elements_per_block": 32, "bytes_per_block": 34, "bpw": 8.5},
        "model_weights_downloaded": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
