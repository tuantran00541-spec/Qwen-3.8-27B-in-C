#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from k3_stream import ALIGN, pack_layers
from k3_stream_qd import K3QDTrunk
from k3_stream_sanity import blob, write_safetensors


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard = "model-00001-of-00001.safetensors"
        tensors = []
        weight_map = {}
        expected = {}
        for layer in range(5):
            name = f"model.language_model.layers.{layer}.payload"
            data = blob(layer * 31 + 7, ALIGN - 128)
            tensors.append((name, "U8", [len(data)], data))
            weight_map[name] = shard
            expected[name] = hashlib.sha256(data).hexdigest()
        write_safetensors(root / shard, tensors)

        out_bin = root / "qd.trunk.bin"
        out_idx = root / "qd.trunk.json"
        manifest = pack_layers(
            root,
            weight_map,
            range(5),
            out_bin,
            out_idx,
            model_id="synthetic/qwen38-qd",
            revision="synthetic",
        )
        assert manifest["packed_file_bytes"] == 5 * ALIGN

        with K3QDTrunk(
            out_bin,
            out_idx,
            budget_bytes=4 * ALIGN,
            max_pinned=0,
            prefer_direct_io=False,
            ring_slots=4,
            io_workers=3,
            lookahead=3,
        ) as trunk:
            v0 = trunk.bind(0)
            h0 = hashlib.sha256(v0).hexdigest()
            assert trunk.prefetch(1) is True
            # Prefetch of 1/2/3 must never overwrite the currently bound slot.
            assert hashlib.sha256(v0).hexdigest() == h0
            v0.release()

            for layer in range(1, 5):
                view = trunk.bind(layer)
                name = f"model.language_model.layers.{layer}.payload"
                tensor = trunk.tensor_view(view, name)
                assert hashlib.sha256(tensor).hexdigest() == expected[name]
                tensor.release()
                view.release()
                if layer + 1 < 5:
                    trunk.prefetch(layer + 1)

            report = trunk.report()
            assert report["experimental_qd"] is True
            assert report["ring_slots"] == 4
            assert report["io_workers"] == 3
            assert report["lookahead_layers"] == 3
            assert report["planned_bytes"] == 4 * ALIGN
            assert report["bytes_read"] == 5 * ALIGN
            assert report["io_bytes_completed"] == 5 * ALIGN
            assert report["max_pending"] >= 2
            assert report["io_span_seconds"] >= 0.0

        failed_budget = False
        try:
            K3QDTrunk(
                out_bin,
                out_idx,
                budget_bytes=3 * ALIGN,
                max_pinned=0,
                prefer_direct_io=False,
                ring_slots=4,
                io_workers=3,
                lookahead=3,
            )
        except MemoryError:
            failed_budget = True
        assert failed_budget

        print(json.dumps({
            "schema": "qwen38-k3-stream-qd-sanity-v1",
            "status": "PASS",
            "reader": report,
            "budget_guard": True,
            "active_buffer_preserved": True,
            "tensor_hashes_exact": True,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
