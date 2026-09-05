#!/usr/bin/env python3
"""Cached-trunk entry point for the Qwen3.8 K3 generator.

The proven generator intentionally repacks all 64 decoder layers on every
process invocation. That is useful while validating layout, but pathological
for an interactive/runtime lane: a valid ~20 GiB K3 trunk gets rewritten for
every prompt.

This wrapper changes only construction/storage reuse. Tokenization, decoder
math, recurrent state, KV cache, logits and greedy token selection are inherited
unchanged from qwen35_k3_generate.py. A cached trunk is accepted only when its
manifest is pinned to the same model/revision/source SHA and covers exactly the
64 decoder layers with in-bounds aligned ranges.

The multi-queue K3 reader is strictly opt-in through QWEN38_K3_IO_EXPERIMENT.
Without that environment variable the proven two-slot K3Trunk path is unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import qwen35_k3_generate as gen


def _reusable_manifest(trunk: Path, manifest_path: Path, model: Path) -> dict[str, Any] | None:
    if not trunk.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "qwen38-k3-trunk-v1":
            return None
        if int(manifest.get("manifest_version", -1)) != 2:
            return None
        if manifest.get("source_format") != "gguf-v3":
            return None
        if manifest.get("model_id") != gen.gdn.MODEL_ID:
            return None
        if manifest.get("revision") != gen.gdn.REVISION:
            return None
        source = manifest.get("source") or {}
        if source.get("sha256") != gen.gdn.SHA256:
            return None
        if int(source.get("file_bytes", -1)) != model.stat().st_size:
            return None

        layers = manifest.get("layers")
        if not isinstance(layers, list) or [int(x.get("layer", -1)) for x in layers] != list(range(gen.N_LAYER)):
            return None
        packed_size = trunk.stat().st_size
        if int(manifest.get("packed_file_bytes", -1)) != packed_size:
            return None
        if not isinstance(manifest.get("tensor_index"), dict) or not manifest["tensor_index"]:
            return None
        for layer in layers:
            offset = int(layer.get("file_offset", -1))
            read_bytes = int(layer.get("read_bytes", -1))
            data_bytes = int(layer.get("data_bytes", -1))
            if offset < 0 or read_bytes <= 0 or data_bytes < 0 or data_bytes > read_bytes:
                return None
            if offset % 4096 or read_bytes % 4096 or offset + read_bytes > packed_size:
                return None
        return manifest
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _make_reader(trunk: Path, manifest_path: Path, max_layer: int):
    if os.environ.get("QWEN38_K3_IO_EXPERIMENT") != "1":
        return gen.K3Trunk(
            trunk,
            manifest_path,
            budget_bytes=2 * max_layer,
            want_ring=2,
            max_pinned=0,
            prefer_direct_io=True,
        )

    from k3_stream_qd import K3QDTrunk

    ring_slots = int(os.environ.get("QWEN38_K3_RING_SLOTS", "2"))
    io_workers = int(os.environ.get("QWEN38_K3_IO_WORKERS", "1"))
    lookahead = int(os.environ.get("QWEN38_K3_LOOKAHEAD", "1"))
    return K3QDTrunk(
        trunk,
        manifest_path,
        budget_bytes=ring_slots * max_layer,
        want_ring=2,
        max_pinned=0,
        prefer_direct_io=True,
        ring_slots=ring_slots,
        io_workers=io_workers,
        lookahead=lookahead,
    )


class CachedStatefulK3Generator(gen.StatefulK3Generator):
    def __init__(self, model: Path, native_lib: Path, state_lib_path: Path,
                 inventory_json: Path, work_dir: Path):
        gen.exact.install()
        inv = json.loads(inventory_json.read_text(encoding="utf-8"))
        if inv.get("status") != "PASS" or inv.get("sha256") != gen.gdn.SHA256:
            raise RuntimeError("mixed decoder inventory is not PASS for the pinned GGUF")

        self.model = model
        self.directory = gen.parse_gguf(model)
        self.tensors = self.directory.by_name()
        self.runtime = gen.gdn.QuantRuntime(gen.gdn._load_native(native_lib))
        self.state_lib = gen.t2.load_state_lib(state_lib_path)
        self.states = {
            il: (gen.t2.ctypes.c_float * gen.t2.STATE_ELEMS)()
            for il in range(gen.N_LAYER)
            if il % 4 != 3
        }
        self.conv_history = {il: [] for il in range(gen.N_LAYER) if il % 4 != 3}
        self.caches = {il: {"k": [], "v": []} for il in range(gen.N_LAYER) if il % 4 == 3}
        self.position = 0

        work_dir.mkdir(parents=True, exist_ok=True)
        trunk = work_dir / "decoder64.k3.bin"
        manifest_path = work_dir / "decoder64.k3.json"
        manifest = _reusable_manifest(trunk, manifest_path, model)
        self.packed_trunk_reused = manifest is not None
        if manifest is None:
            manifest = gen.pack_gguf_layers(
                self.directory,
                trunk,
                manifest_path,
                layers=range(gen.N_LAYER),
                model_id=gen.gdn.MODEL_ID,
                revision=gen.gdn.REVISION,
                source_sha256=gen.gdn.SHA256,
                expected_layers=gen.N_LAYER,
            )
        self.manifest = manifest
        max_layer = max(int(x["read_bytes"]) for x in self.manifest["layers"])
        self.reader = _make_reader(trunk, manifest_path, max_layer)
        self.output_norm_w = gen.base._read_f32_tensor(model, self.tensors["output_norm.weight"])

    def state_report(self) -> dict[str, Any]:
        report = super().state_report()
        report["packed_trunk_reused"] = self.packed_trunk_reused
        return report


def main() -> None:
    # generate() resolves this module global at call time, so patching the class
    # preserves the original CLI and result schema while changing only storage
    # construction/reuse.
    gen.StatefulK3Generator = CachedStatefulK3Generator
    gen.main()


if __name__ == "__main__":
    main()
