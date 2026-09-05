#!/usr/bin/env python3
"""Real-weight storage/runtime gate for Qwen3.8 K3-style layer streaming.

No prompt is generated here. The gate downloads only the SafeTensors shards
needed for selected language layers, packs them into the aligned trunk format,
then proves exact tensor bytes through the pinned-prefix/ring reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from core import MODEL_ID, PINNED_REVISION, validate_official_metadata
from k3_stream import K3Trunk, needed_shards, pack_layers, verify_layer


def parse_layers(text: str) -> list[int]:
    layers = sorted({int(v.strip()) for v in text.split(",") if v.strip()})
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if any(v < 0 or v >= 64 for v in layers):
        raise argparse.ArgumentTypeError("layers must be in [0, 63]")
    return layers


def sha256_view(view: memoryview) -> str:
    digest = hashlib.sha256()
    chunk = 8 * 1024 * 1024
    for at in range(0, len(view), chunk):
        digest.update(view[at : at + chunk])
    return digest.hexdigest()


def download_metadata(root: Path) -> tuple[dict, dict, dict]:
    paths = {}
    for filename in ("config.json", "model.safetensors.index.json"):
        paths[filename] = Path(hf_hub_download(
            MODEL_ID, filename=filename, revision=PINNED_REVISION, local_dir=str(root)
        ))
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    index = json.loads(paths["model.safetensors.index.json"].read_text(encoding="utf-8"))
    return config, index, validate_official_metadata(config, index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-packed", action="store_true")
    args = parser.parse_args()

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    out_bin = trunk_dir / "qwen38-language.trunk.bin"
    out_idx = trunk_dir / "qwen38-language.trunk.json"

    started = time.monotonic()
    _, index, validated = download_metadata(model_dir)
    weight_map = index["weight_map"]
    shards = needed_shards(weight_map, args.layers)
    download_started = time.monotonic()
    for shard in shards:
        hf_hub_download(MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(model_dir))
    download_seconds = time.monotonic() - download_started

    pack_started = time.monotonic()
    manifest = pack_layers(
        model_dir, weight_map, args.layers, out_bin, out_idx,
        model_id=MODEL_ID, revision=PINNED_REVISION,
    )
    pack_seconds = time.monotonic() - pack_started
    exact = [verify_layer(manifest, out_bin, layer) for layer in args.layers]
    max_run = max(int(layer["read_bytes"]) for layer in manifest["layers"])

    runtime_started = time.monotonic()
    with K3Trunk(
        out_bin, out_idx, budget_bytes=2 * max_run,
        want_ring=2, max_pinned=0, prefer_direct_io=True,
    ) as trunk:
        if trunk.plan.ring_slots != 2:
            raise RuntimeError(f"expected two real ring slots, got {trunk.plan}")
        first = args.layers[0]
        view0 = trunk.bind(first)
        first_meta = next(x for x in manifest["layers"] if int(x["layer"]) == first)
        first_tensor = first_meta["tensors"][0]
        t0 = trunk.tensor_view(view0, first_tensor["name"])
        if sha256_view(t0) != first_tensor["sha256"]:
            raise RuntimeError("first bound tensor checksum mismatch")
        del t0
        active_before = sha256_view(view0)

        prefetch_proof = None
        if len(args.layers) >= 2:
            second = args.layers[1]
            if not trunk.prefetch(second):
                raise RuntimeError("two-slot runtime refused real async prefetch")
            active_after = sha256_view(view0)
            if active_after != active_before:
                raise RuntimeError("async prefetch overwrote the active layer slot")
            del view0
            view1 = trunk.bind(second)
            second_meta = next(x for x in manifest["layers"] if int(x["layer"]) == second)
            for tensor in second_meta["tensors"]:
                tv = trunk.tensor_view(view1, tensor["name"])
                if sha256_view(tv) != tensor["sha256"]:
                    raise RuntimeError(f"streamed checksum mismatch: {tensor['name']}")
                del tv
            del view1
            prefetch_proof = {"from_layer": first, "to_layer": second, "active_layer_stable": True}
        else:
            del view0
        runtime_report = trunk.report()
    runtime_seconds = time.monotonic() - runtime_started

    result = {
        "schema": "qwen38-k3-real-layer-smoke-v1",
        "status": "PASS",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "layers": args.layers,
        "official_metadata": validated,
        "source_shards": shards,
        "source_shard_count": len(shards),
        "download_seconds": download_seconds,
        "pack_seconds": pack_seconds,
        "runtime_seconds": runtime_seconds,
        "total_seconds": time.monotonic() - started,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "total_tensor_bytes": manifest["total_tensor_bytes"],
        "max_layer_read_bytes": max_run,
        "exact_pack_verification": exact,
        "prefetch_proof": prefetch_proof,
        "runtime": runtime_report,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "generation_attempted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.keep_packed:
        try:
            out_bin.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
