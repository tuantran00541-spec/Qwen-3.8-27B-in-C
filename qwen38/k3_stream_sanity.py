#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path

from k3_stream import ALIGN, K3Trunk, pack_layers, verify_layer


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    header = {}
    offset = 0
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padded = raw + b" " * ((8 - len(raw) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(padded)))
        f.write(padded)
        f.write(payload)


def blob(seed: int, n: int) -> bytes:
    return bytes(((i * 17 + seed) & 0xFF) for i in range(n))


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard = "model-00001-of-00001.safetensors"
        tensors = []
        weight_map = {}
        expected = {}
        for layer in range(3):
            for kind, size in (
                ("input_layernorm.weight", 128),
                ("mlp.gate_proj.weight", 1536),
                ("mlp.down_proj.weight", 1024),
            ):
                name = f"model.language_model.layers.{layer}.{kind}"
                data = blob(layer * 23 + len(kind), size)
                tensors.append((name, "U8", [size], data))
                weight_map[name] = shard
                expected[name] = hashlib.sha256(data).hexdigest()
        write_safetensors(root / shard, tensors)
        out_bin = root / "qwen38.trunk.bin"
        out_idx = root / "qwen38.trunk.json"
        manifest = pack_layers(
            root, weight_map, [0, 1, 2], out_bin, out_idx,
            model_id="synthetic/qwen38", revision="synthetic",
        )
        assert manifest["packed_file_bytes"] == 3 * ALIGN
        for layer in range(3):
            assert verify_layer(manifest, out_bin, layer)["tensors_checked"] == 3

        with K3Trunk(out_bin, out_idx, budget_bytes=2 * ALIGN, max_pinned=0, prefer_direct_io=False) as trunk:
            assert trunk.plan.ring_slots == 2
            v0 = trunk.bind(0)
            snap0 = hashlib.sha256(v0).hexdigest()
            assert trunk.prefetch(1) is True
            assert hashlib.sha256(v0).hexdigest() == snap0
            del v0
            v1 = trunk.bind(1)
            tensor = manifest["layers"][1]["tensors"][1]
            tv = trunk.tensor_view(v1, tensor["name"])
            assert hashlib.sha256(tv).hexdigest() == expected[tensor["name"]]
            del tv, v1
            report_two = trunk.report()
            assert report_two["async_prefetch_enabled"] is True

        with K3Trunk(out_bin, out_idx, budget_bytes=ALIGN, max_pinned=0, prefer_direct_io=False) as trunk:
            assert trunk.plan.ring_slots == 1
            v0 = trunk.bind(0)
            assert trunk.prefetch(1) is False
            del v0
            v1 = trunk.bind(1)
            del v1
            report_one = trunk.report()
            assert report_one["async_prefetch_enabled"] is False

        with K3Trunk(out_bin, out_idx, budget_bytes=3 * ALIGN, want_ring=2, max_pinned=1, prefer_direct_io=False) as trunk:
            assert trunk.plan.pinned_layers == (0,)
            assert trunk.plan.ring_slots == 2
            report_pinned = trunk.report()

        print(json.dumps({
            "schema": "qwen38-k3-stream-sanity-v1",
            "status": "PASS",
            "packed_file_bytes": manifest["packed_file_bytes"],
            "two_slot": report_two,
            "one_slot": report_one,
            "pinned_prefix": report_pinned,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
