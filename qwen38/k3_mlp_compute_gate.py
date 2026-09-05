#!/usr/bin/env python3
"""Real-weight compute equivalence gate for the Qwen3.8 K3-style trunk.

This intentionally tests only the dense per-layer RMSNorm + SwiGLU MLP path.
It does not generate text and does not claim full decoder-layer equivalence.
The candidate path consumes BF16 tensors directly from a K3Trunk ring buffer;
the reference path reads the same official tensors from SafeTensors.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from core import HIDDEN, MODEL_ID, PINNED_REVISION, validate_official_metadata
from k3_stream import K3Trunk, needed_shards, pack_layers

PREFIX = "model.language_model.layers"
EPS = 1e-6


def tensor_name(layer: int, suffix: str) -> str:
    return f"{PREFIX}.{layer}.{suffix}"


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def normalized_hidden(samples: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn((samples, HIDDEN), generator=gen, dtype=torch.float32)
    x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    return x.to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + EPS)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)


def swiglu_mlp(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    denom = torch.linalg.vector_norm(ref).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "exact_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)),
    }


def metadata(root: Path) -> tuple[dict, dict, dict]:
    config_path = Path(hf_hub_download(MODEL_ID, filename="config.json", revision=PINNED_REVISION, local_dir=str(root)))
    index_path = Path(hf_hub_download(
        MODEL_ID, filename="model.safetensors.index.json", revision=PINNED_REVISION, local_dir=str(root)
    ))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return config, index, validate_official_metadata(config, index)


def source_tensor(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    return tensor


def streamed_tensor(layer_view: memoryview, layer_meta: dict, name: str) -> tuple[torch.Tensor, memoryview]:
    tensor_meta = next((item for item in layer_meta["tensors"] if item["name"] == name), None)
    if tensor_meta is None:
        raise KeyError(name)
    if tensor_meta["dtype"] != "BF16":
        raise ValueError(f"{name}: streamed dtype {tensor_meta['dtype']} is not BF16")
    start = int(tensor_meta["offset"])
    nbytes = int(tensor_meta["nbytes"])
    shape = tuple(int(v) for v in tensor_meta["shape"])
    view = layer_view[start : start + nbytes]
    tensor = torch.frombuffer(view, dtype=torch.bfloat16, count=math.prod(shape)).reshape(shape)
    return tensor, view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer < 64:
        raise SystemExit("--layer must be in [0, 63]")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    _, index, validated = metadata(model_dir)
    weight_map = index["weight_map"]
    shards = needed_shards(weight_map, [args.layer])
    for shard in shards:
        hf_hub_download(MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(model_dir))

    out_bin = trunk_dir / "layer.trunk.bin"
    out_idx = trunk_dir / "layer.trunk.json"
    manifest = pack_layers(
        model_dir, weight_map, [args.layer], out_bin, out_idx,
        model_id=MODEL_ID, revision=PINNED_REVISION,
    )
    layer_meta = manifest["layers"][0]
    layer_bytes = int(layer_meta["read_bytes"])

    names = {
        "input_norm": tensor_name(args.layer, "input_layernorm.weight"),
        "post_norm": tensor_name(args.layer, "post_attention_layernorm.weight"),
        "gate": tensor_name(args.layer, "mlp.gate_proj.weight"),
        "up": tensor_name(args.layer, "mlp.up_proj.weight"),
        "down": tensor_name(args.layer, "mlp.down_proj.weight"),
    }
    for name in names.values():
        if name not in weight_map:
            raise KeyError(name)

    x = normalized_hidden(args.samples, args.seed)

    ref_input_norm = source_tensor(model_dir, weight_map, names["input_norm"])
    ref_post_norm = source_tensor(model_dir, weight_map, names["post_norm"])
    ref_gate = source_tensor(model_dir, weight_map, names["gate"])
    ref_up = source_tensor(model_dir, weight_map, names["up"])
    ref_down = source_tensor(model_dir, weight_map, names["down"])
    ref_norm_out = rms_norm(x, ref_input_norm)
    ref_mlp_in = rms_norm(x, ref_post_norm)
    reference = swiglu_mlp(ref_mlp_in, ref_gate, ref_up, ref_down)
    rss_after_reference = max_rss_gib()

    with K3Trunk(
        out_bin, out_idx,
        budget_bytes=layer_bytes,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        if trunk.plan.ring_slots != 1 or trunk.report()["async_prefetch_enabled"]:
            raise RuntimeError("one-slot compute gate must not enable async prefetch")
        layer_view = trunk.bind(args.layer)
        keepalive: list[memoryview] = []
        streamed: dict[str, torch.Tensor] = {}
        for key, name in names.items():
            tensor, view = streamed_tensor(layer_view, layer_meta, name)
            streamed[key] = tensor
            keepalive.append(view)
        del tensor, view

        stream_norm_out = rms_norm(x, streamed["input_norm"])
        stream_mlp_in = rms_norm(x, streamed["post_norm"])
        candidate = swiglu_mlp(stream_mlp_in, streamed["gate"], streamed["up"], streamed["down"])
        norm_metrics = metrics(ref_norm_out, stream_norm_out)
        mlp_metrics = metrics(reference, candidate)
        runtime_report = trunk.report()
        del stream_norm_out, stream_mlp_in, candidate
        streamed.clear()
        keepalive.clear()
        del layer_view

    result = {
        "schema": "qwen38-k3-mlp-compute-gate-v1",
        "status": "PASS" if norm_metrics["exact_equal"] and mlp_metrics["exact_equal"] else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "layer": args.layer,
        "samples": args.samples,
        "seed": args.seed,
        "source_shards": shards,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "layer_read_bytes": layer_bytes,
        "input_rmsnorm": norm_metrics,
        "post_norm_swiglu_mlp": mlp_metrics,
        "runtime": runtime_report,
        "rss_after_reference_gib": rss_after_reference,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "generation_attempted": False,
        "full_decoder_layer_claimed": False,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
