#!/usr/bin/env python3
"""Zero-model GGUF -> K3 packing/ring safety gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile

from gguf_stream import GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, parse_gguf, tensor_nbytes
from gguf_k3_layout import ManifestK3Trunk, pack_gguf_layers
from k3_stream import ALIGN, TENSOR_ALIGN, K3Trunk, plan_memory, verify_layer

TYPE_UINT32 = 4
TYPE_STRING = 8


def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, typ: int, payload: bytes) -> bytes:
    return s(key) + struct.pack("<I", typ) + payload


def up(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def ti(name: str, shape: tuple[int, ...], typ: int, rel: int) -> bytes:
    out = bytearray(s(name) + struct.pack("<I", len(shape)))
    for d in shape:
        out += struct.pack("<Q", d)
    return bytes(out + struct.pack("<IQ", typ, rel))


def fixture(path: Path) -> dict[str, bytes]:
    alignment = 32
    specs = [
        ("blk.0.a_f32", (8,), GGML_TYPE_F32, b"A"),
        ("blk.0.b_q6", (256,), GGML_TYPE_Q6_K, b"B"),
        ("blk.1.a_q4", (32, 2), GGML_TYPE_Q4_0, b"C"),
        ("blk.1.b_q8", (32,), GGML_TYPE_Q8_0, b"D"),
        ("token_embd.weight", (32, 2), GGML_TYPE_Q8_0, b"E"),
        ("output.weight", (32, 2), GGML_TYPE_Q8_0, b"F"),
        ("output_norm.weight", (8,), GGML_TYPE_F32, b"G"),
    ]
    rel = 0
    entries, payloads, expected = [], [], {}
    for name, shape, typ, byte in specs:
        nbytes = tensor_nbytes(shape, typ)[0]
        rel = up(rel, alignment)
        entries.append(ti(name, shape, typ, rel))
        data = byte * nbytes
        payloads.append((rel, data))
        expected[name] = data
        rel += nbytes
    kvs = [kv("general.architecture", TYPE_STRING, s("qwen35")), kv("general.alignment", TYPE_UINT32, struct.pack("<I", alignment))]
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(entries), len(kvs)) + b"".join(kvs) + b"".join(entries))
    data_offset = up(len(header), alignment)
    blob = bytearray(data_offset + rel)
    blob[:len(header)] = header
    for off, data in payloads:
        blob[data_offset + off:data_offset + off + len(data)] = data
    path.write_bytes(blob)
    return expected


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen38-gguf-k3-") as tmp:
        root = Path(tmp)
        src, trunk, index = root / "fixture.gguf", root / "layers.k3", root / "layers.json"
        expected = fixture(src)
        directory = parse_gguf(src)
        source_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        manifest = pack_gguf_layers(directory, trunk, index, model_id="synthetic/qwen", revision="fixture", source_sha256=source_sha, expected_layers=2, chunk_bytes=17)
        assert manifest["source_format"] == "gguf-v3" and manifest["manifest_version"] == 2
        assert manifest["max_copy_chunk_observed"] <= 17
        assert {g["name"] for g in manifest["globals"]} == {"token_embd.weight", "output.weight", "output_norm.weight"}
        assert set(manifest["tensor_index"]) == {"blk.0.a_f32", "blk.0.b_q6", "blk.1.a_q4", "blk.1.b_q8"}
        for layer in manifest["layers"]:
            assert layer["file_offset"] % ALIGN == 0 and layer["read_bytes"] % ALIGN == 0
            for tensor in layer["tensors"]:
                assert tensor["offset"] % TENSOR_ALIGN == 0
                assert tensor["ggml_type"] in (GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0)
                assert verify_layer(manifest, trunk, layer["layer"])["tensors_checked"] == layer["tensor_count"]

        layer_bytes = max(x["read_bytes"] for x in manifest["layers"])
        two = plan_memory(manifest, 2 * layer_bytes, want_ring=2, max_pinned=0)
        one = plan_memory(manifest, layer_bytes, want_ring=2, max_pinned=0)
        assert two.ring_slots == 2 and one.ring_slots == 1

        # Base K3Trunk must understand GGUF tensor names through manifest.tensor_index.
        with K3Trunk(trunk, index, budget_bytes=2 * layer_bytes, want_ring=2, max_pinned=0, prefer_direct_io=False) as reader:
            v0 = reader.bind(0)
            assert bytes(reader.tensor_view(v0, "blk.0.a_f32")) == expected["blk.0.a_f32"]
            assert reader.prefetch(1) is True
            v1 = reader.bind(1)
            assert bytes(reader.tensor_view(v1, "blk.1.a_q4")) == expected["blk.1.a_q4"]
            assert reader.report()["async_prefetch_enabled"] is True
            assert set(reader.tensor_index) == set(manifest["tensor_index"])
            del v0, v1

        with K3Trunk(trunk, index, budget_bytes=layer_bytes, want_ring=2, max_pinned=0, prefer_direct_io=False) as reader:
            v0 = reader.bind(0)
            snapshot = bytes(reader.tensor_view(v0, "blk.0.b_q6"))
            assert reader.prefetch(1) is False
            assert snapshot == expected["blk.0.b_q6"]
            assert reader.report()["async_prefetch_enabled"] is False
            del v0

        # Keep the old subclass as a compatibility surface until callers migrate.
        with ManifestK3Trunk(trunk, index, budget_bytes=layer_bytes, want_ring=1, max_pinned=0, prefer_direct_io=False) as reader:
            v0 = reader.bind(0)
            assert bytes(reader.tensor_view(v0, "blk.0.a_f32")) == expected["blk.0.a_f32"]
            del v0

        print(json.dumps({"schema":"qwen38-gguf-k3-layout-sanity-v2","status":"PASS","model_weights_downloaded":False,"layers":2,"globals":3,"max_copy_chunk":manifest["max_copy_chunk_observed"],"base_manifest_index":True,"compat_subclass":True,"two_slot_prefetch":True,"one_slot_prefetch":False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
