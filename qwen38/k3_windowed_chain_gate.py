#!/usr/bin/env python3
"""Windowed exact Qwen3.8 decoder-chain gate for bounded CI disk usage.

Each window downloads only the official shards needed for that layer range,
packs those layers into a temporary K3 trunk, advances an official Transformers
reference and an independent two-slot K3 candidate, checks exact equality, then
deletes both source shards and the temporary trunk before moving on.

The windowing is a CI/storage gate, not a claim that the final runtime should
split the model this way. It exists so all 64 decoder layers can be verified on
a runner without keeping the full BF16 source checkpoint and a second full
packed copy on disk simultaneously.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import time
from pathlib import Path

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from core import LAYERS as TOTAL_LAYERS, MODEL_ID, PINNED_REVISION, PREFIX
from k3_full_attention_gate import candidate_full_attention_layer, causal_mask
from k3_linear_layer_gate import (
    candidate_linear_layer,
    deterministic_hidden,
    download_metadata,
    metrics,
    source_tensor,
    streamed_tensor,
)
from k3_stream import K3Trunk, needed_shards, pack_layers


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def layer_names(weight_map: dict[str, str], layer: int) -> list[str]:
    prefix = f"{PREFIX}.{layer}."
    return sorted(name for name in weight_map if name.startswith(prefix))


def source_state(
    model_dir: Path,
    weight_map: dict[str, str],
    layer: int,
) -> dict[str, torch.Tensor]:
    prefix = f"{PREFIX}.{layer}."
    return {
        name[len(prefix):]: source_tensor(model_dir, weight_map, name)
        for name in layer_names(weight_map, layer)
    }


def streamed_state(
    layer_view: memoryview,
    layer_meta: dict,
    weight_map: dict[str, str],
    layer: int,
) -> tuple[dict[str, torch.Tensor], list[memoryview]]:
    prefix = f"{PREFIX}.{layer}."
    state: dict[str, torch.Tensor] = {}
    keepalive: list[memoryview] = []
    for full_name in layer_names(weight_map, layer):
        tensor, view = streamed_tensor(layer_view, layer_meta, full_name)
        state[full_name[len(prefix):]] = tensor
        keepalive.append(view)
    return state, keepalive


def advance_reference_layer(
    hidden: torch.Tensor,
    cfg: Qwen3_5TextConfig,
    model_dir: Path,
    weight_map: dict[str, str],
    layer: int,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    position_ids: torch.Tensor,
    full_mask: torch.Tensor,
) -> torch.Tensor:
    module = Qwen3_5DecoderLayer(cfg, layer).to(dtype=torch.bfloat16).eval()
    state = source_state(model_dir, weight_map, layer)
    expected = set(module.state_dict().keys())
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise RuntimeError(
            f"layer {layer} reference state mismatch missing={missing} unexpected={unexpected}"
        )
    module.load_state_dict(state, strict=True)
    del state
    layer_type = cfg.layer_types[layer]
    dummy = torch.empty(0, dtype=torch.float32)
    with torch.inference_mode():
        if layer_type == "linear_attention":
            out = module(
                hidden,
                position_embeddings=(dummy, dummy),
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
            )
        elif layer_type == "full_attention":
            out = module(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=None,
            )
        else:
            raise RuntimeError(f"unsupported layer type {layer_type!r}")
    del module
    gc.collect()
    return out


def remove_file(path: Path) -> int:
    if not path.is_file():
        return 0
    size = path.stat().st_size
    path.unlink()
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--end-layer", type=int, default=15)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 0 <= args.end_layer < TOTAL_LAYERS:
        raise SystemExit(f"--end-layer must be in [0,{TOTAL_LAYERS - 1}]")
    if args.window_size < 2:
        raise SystemExit("--window-size must be >= 2 so two-slot prefetch is exercised")
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be >= 2")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_config = config["text_config"]
    cfg = Qwen3_5TextConfig(**text_config)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]

    layers_all = tuple(range(args.end_layer + 1))
    layer_types = [text_config["layer_types"][layer] for layer in layers_all]
    unsupported = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unsupported:
        raise RuntimeError(f"unsupported layer types: {unsupported}")

    x = deterministic_hidden(args.seq_len, args.seed)
    reference = x.clone()
    candidate = x.clone()
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        position_embeddings = rotary(x, position_ids)
    full_mask = causal_mask(args.seq_len)

    windows: list[dict] = []
    total_stream_bytes = 0
    total_deleted_source_bytes = 0
    total_deleted_trunk_bytes = 0
    total_prefetch_accepted = 0
    total_prefetch_expected = 0
    all_exact = True

    for start in range(0, args.end_layer + 1, args.window_size):
        stop = min(args.end_layer + 1, start + args.window_size)
        layers = tuple(range(start, stop))
        if len(layers) < 2:
            raise RuntimeError(
                "final window has fewer than two layers; choose a window size/end-layer that exercises two slots"
            )

        window_started = time.monotonic()
        disk_before_download = shutil.disk_usage(root).free
        shards = needed_shards(weight_map, layers)
        for shard in shards:
            hf_hub_download(
                MODEL_ID,
                filename=shard,
                revision=PINNED_REVISION,
                local_dir=str(model_dir),
            )
        disk_after_download = shutil.disk_usage(root).free

        out_bin = trunk_dir / f"layers{start}-{stop - 1}.trunk.bin"
        out_idx = trunk_dir / f"layers{start}-{stop - 1}.trunk.json"
        manifest = pack_layers(
            model_dir,
            weight_map,
            layers,
            out_bin,
            out_idx,
            model_id=MODEL_ID,
            revision=PINNED_REVISION,
        )
        meta_by_layer = {int(item["layer"]): item for item in manifest["layers"]}
        max_layer_bytes = max(int(item["read_bytes"]) for item in manifest["layers"])
        expected_window_bytes = sum(int(meta_by_layer[layer]["read_bytes"]) for layer in layers)
        disk_after_pack = shutil.disk_usage(root).free

        reference_layers: list[dict] = []
        for layer in layers:
            reference = advance_reference_layer(
                reference,
                cfg,
                model_dir,
                weight_map,
                layer,
                position_embeddings,
                position_ids,
                full_mask,
            )
            reference_layers.append({
                "layer": layer,
                "type": text_config["layer_types"][layer],
                "rss_gib": max_rss_gib(),
            })

        deleted_source_bytes = 0
        deleted_source_files: list[str] = []
        for shard in shards:
            path = model_dir / shard
            removed = remove_file(path)
            if removed:
                deleted_source_bytes += removed
                deleted_source_files.append(shard)
        total_deleted_source_bytes += deleted_source_bytes
        gc.collect()
        disk_after_source_cleanup = shutil.disk_usage(root).free

        prefetch_events: list[dict] = []
        budget = 2 * max_layer_bytes
        with K3Trunk(
            out_bin,
            out_idx,
            budget_bytes=budget,
            want_ring=2,
            max_pinned=0,
            prefer_direct_io=True,
        ) as trunk:
            if trunk.plan.ring_slots != 2 or not trunk.report()["async_prefetch_enabled"]:
                raise RuntimeError("window requires two safe ring slots with async prefetch enabled")

            layer_view = trunk.bind(layers[0])
            for pos, layer in enumerate(layers):
                next_layer = layers[pos + 1] if pos + 1 < len(layers) else None
                accepted = trunk.prefetch(next_layer) if next_layer is not None else False
                if next_layer is not None:
                    total_prefetch_expected += 1
                    if not accepted:
                        raise RuntimeError(
                            f"prefetch of layer {next_layer} was not accepted while layer {layer} was active"
                        )
                    total_prefetch_accepted += 1

                state, keepalive = streamed_state(
                    layer_view,
                    meta_by_layer[layer],
                    weight_map,
                    layer,
                )
                layer_type = text_config["layer_types"][layer]
                with torch.inference_mode():
                    if layer_type == "linear_attention":
                        candidate = candidate_linear_layer(candidate, state, cfg)
                    elif layer_type == "full_attention":
                        candidate = candidate_full_attention_layer(
                            candidate,
                            state,
                            cfg,
                            position_embeddings,
                            full_mask,
                        )
                    else:
                        raise RuntimeError(f"unsupported layer type {layer_type!r}")

                prefetch_events.append({
                    "layer": layer,
                    "next_layer": next_layer,
                    "prefetch_accepted": bool(accepted),
                    "rss_gib": max_rss_gib(),
                })
                state.clear()
                keepalive.clear()
                del layer_view
                if next_layer is not None:
                    layer_view = trunk.bind(next_layer)
            runtime_report = trunk.report()

        if runtime_report["bytes_read"] != expected_window_bytes:
            raise RuntimeError(
                f"window {start}-{stop - 1} read {runtime_report['bytes_read']} bytes, expected {expected_window_bytes}"
            )
        total_stream_bytes += runtime_report["bytes_read"]

        window_metrics = metrics(reference, candidate)
        all_exact = all_exact and window_metrics["exact_equal"]

        deleted_trunk_bytes = remove_file(out_bin) + remove_file(out_idx)
        total_deleted_trunk_bytes += deleted_trunk_bytes
        disk_after_window_cleanup = shutil.disk_usage(root).free

        windows.append({
            "start_layer": start,
            "end_layer": stop - 1,
            "layers": list(layers),
            "source_shards": shards,
            "packed_file_bytes": manifest["packed_file_bytes"],
            "expected_stream_bytes": expected_window_bytes,
            "runtime": runtime_report,
            "prefetch_events": prefetch_events,
            "reference_layers": reference_layers,
            "output_metrics": window_metrics,
            "source_cleanup": {
                "deleted_files": deleted_source_files,
                "deleted_bytes": deleted_source_bytes,
            },
            "trunk_cleanup_bytes": deleted_trunk_bytes,
            "disk_free": {
                "before_download": disk_before_download,
                "after_download": disk_after_download,
                "after_pack": disk_after_pack,
                "after_source_cleanup": disk_after_source_cleanup,
                "after_window_cleanup": disk_after_window_cleanup,
            },
            "seconds": time.monotonic() - window_started,
        })

        if not window_metrics["exact_equal"]:
            break

    final_metrics = metrics(reference, candidate)
    passed = (
        all_exact
        and final_metrics["exact_equal"]
        and total_prefetch_accepted == total_prefetch_expected
        and len(windows) == ((args.end_layer + 1 + args.window_size - 1) // args.window_size)
    )

    result = {
        "schema": "qwen38-k3-windowed-chain-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "end_layer": args.end_layer,
        "window_size": args.window_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "layers": list(layers_all),
        "layer_types": layer_types,
        "windows": windows,
        "final_output_metrics": final_metrics,
        "total_stream_bytes": total_stream_bytes,
        "total_prefetch_expected": total_prefetch_expected,
        "total_prefetch_accepted": total_prefetch_accepted,
        "total_deleted_source_bytes": total_deleted_source_bytes,
        "total_deleted_trunk_bytes": total_deleted_trunk_bytes,
        "max_rss_gib": max_rss_gib(),
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "total_seconds": time.monotonic() - started,
        "reference": f"official Transformers decoder layers 0-{args.end_layer}, sequential per window",
        "candidate": (
            f"two-slot K3Trunk layers 0-{args.end_layer} in temporary {args.window_size}-layer CI windows"
        ),
        "window_boundary_prefetch": False,
        "generation_attempted": False,
        "logits_claimed": False,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
