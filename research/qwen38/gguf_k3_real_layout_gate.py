#!/usr/bin/env python3
"""Real Q6_K_L GGUF -> K3 storage/layout integration gate.

Downloads the pinned GGUF once, verifies it, inventories the 64 decoder layers
and any auxiliary GGUF blocks (Qwen3.8 carries its MTP head as blk.64), then
repacks representative decoder layers 0 and 3 and exercises the existing K3
ring reader. No inference or dequantization is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from gguf_stream import parse_gguf
from gguf_k3_layout import ManifestK3Trunk, pack_gguf_layers, partition_tensors
from k3_stream import plan_memory, verify_layer

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DECODER_LAYERS = 64
EXPECTED_AUX_BLOCKS = [64]  # one MTP layer in the pinned official config/GGUF


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = Path(hf_hub_download(REPO, filename=FILE, local_dir=str(args.work_dir)))
    digest = sha256(model)
    if digest != SHA256:
        raise RuntimeError(f"SHA256 mismatch: {digest}")

    directory = parse_gguf(model)
    grouped, auxiliary, globals_ = partition_tensors(directory, expected_layers=DECODER_LAYERS)
    aux_ids = sorted(auxiliary)
    if aux_ids != EXPECTED_AUX_BLOCKS:
        raise RuntimeError(f"unexpected auxiliary GGUF blocks: {aux_ids}")

    q4_names = sorted(t.name for t in directory.tensors if t.type_name == "Q4_0")
    layer_stats = []
    for layer in range(DECODER_LAYERS):
        tensors = grouped[layer]
        types = Counter(t.type_name for t in tensors)
        raw = sum(t.nbytes for t in tensors)
        layer_stats.append({
            "layer": layer,
            "tensor_count": len(tensors),
            "raw_bytes": raw,
            "type_counts": dict(sorted(types.items())),
        })
    auxiliary_stats = [{
        "block": block,
        "tensor_count": len(tensors),
        "raw_bytes": sum(t.nbytes for t in tensors),
        "type_counts": dict(sorted(Counter(t.type_name for t in tensors).items())),
        "tensor_names": [t.name for t in tensors],
    } for block, tensors in sorted(auxiliary.items())]

    trunk = args.work_dir / "representative-layers.k3.bin"
    index = args.work_dir / "representative-layers.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, index,
        layers=[0, 3], model_id=MODEL_ID, revision=REVISION,
        source_sha256=digest, expected_layers=DECODER_LAYERS,
    )

    max_read = max(int(x["read_bytes"]) for x in manifest["layers"])
    two_slot_budget = max_read * 2
    plan = plan_memory(manifest, two_slot_budget, want_ring=2, max_pinned=0)
    if plan.ring_slots != 2:
        raise RuntimeError(f"expected two safe slots, got {plan.ring_slots}")

    verified0 = verify_layer(manifest, trunk, 0)
    verified3 = verify_layer(manifest, trunk, 3)
    with ManifestK3Trunk(
        trunk, index, budget_bytes=two_slot_budget, want_ring=2,
        max_pinned=0, prefer_direct_io=True,
    ) as reader:
        v0 = reader.bind(0)
        sample0 = manifest["layers"][0]["tensors"][0]["name"]
        sample0_bytes = len(reader.tensor_view(v0, sample0))
        prefetch3 = reader.prefetch(3)
        v3 = reader.bind(3)
        sample3 = manifest["layers"][1]["tensors"][0]["name"]
        sample3_bytes = len(reader.tensor_view(v3, sample3))
        reader_report = reader.report()
        v0.release()
        v3.release()

    result = {
        "schema": "qwen38-gguf-k3-real-layout-v1",
        "status": "PASS",
        "repo": REPO,
        "file": FILE,
        "file_bytes": model.stat().st_size,
        "sha256": digest,
        "gguf_version": directory.version,
        "architecture": directory.metadata.get("general.architecture"),
        "tensor_count": directory.tensor_count,
        "decoder_layer_count": DECODER_LAYERS,
        "global_tensor_count": len(globals_),
        "auxiliary_block_ids": aux_ids,
        "auxiliary_stats": auxiliary_stats,
        "q4_0_count": len(q4_names),
        "q4_0_names": q4_names,
        "max_decoder_layer": max(grouped),
        "layer_stats": layer_stats,
        "packed_layers": [0, 3],
        "packed_file_bytes": manifest["packed_file_bytes"],
        "max_packed_read_bytes": max_read,
        "copy_chunk_bytes": manifest["copy_chunk_bytes"],
        "max_copy_chunk_observed": manifest["max_copy_chunk_observed"],
        "two_slot_budget_bytes": two_slot_budget,
        "slots": plan.ring_slots,
        "prefetch_3": prefetch3,
        "verified_0": verified0,
        "verified_3": verified3,
        "sample_0": sample0,
        "sample_0_bytes": sample0_bytes,
        "sample_3": sample3,
        "sample_3_bytes": sample3_bytes,
        "reader_report": reader_report,
        "manifest_auxiliary_blocks": manifest.get("auxiliary_blocks", []),
        "manifest_layers": manifest["layers"],
    }
    verification_ok = verified0.get("tensors_checked", 0) > 0 and verified3.get("tensors_checked", 0) > 0
    if len(q4_names) != 8 or not (verification_ok and prefetch3):
        result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
