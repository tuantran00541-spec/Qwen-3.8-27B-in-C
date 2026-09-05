#!/usr/bin/env python3
"""Exact real-weight decoder-layer gate over the K3-style Qwen trunk.

Scope is intentionally one official *linear_attention* decoder layer and one
short synthetic hidden-state sequence. The reference is Transformers'
Qwen3_5DecoderLayer loaded with official BF16 weights. The candidate performs
the same decoder wiring using zero-copy tensor views over a K3Trunk ring slot.
No tokenizer, LM head, vision encoder, or prompt generation is involved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import time
from pathlib import Path

# Force the pure PyTorch CPU fallbacks before importing Transformers modeling.
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    causal_conv1d_fn,
    torch_chunk_gated_delta_rule,
)

from core import HIDDEN, MODEL_ID, PINNED_REVISION, PREFIX, validate_official_metadata
from k3_stream import K3Trunk, needed_shards, pack_layers


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def deterministic_hidden(seq_len: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn((1, seq_len, HIDDEN), generator=gen, dtype=torch.float32)
    x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True))
    return x.to(torch.bfloat16)


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


def download_metadata(root: Path) -> tuple[dict, dict, dict]:
    config_path = Path(hf_hub_download(
        MODEL_ID, filename="config.json", revision=PINNED_REVISION, local_dir=str(root)
    ))
    index_path = Path(hf_hub_download(
        MODEL_ID, filename="model.safetensors.index.json", revision=PINNED_REVISION, local_dir=str(root)
    ))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return config, index, validate_official_metadata(config, index)


def source_tensor(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(str(root / weight_map[name]), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    return tensor


def streamed_tensor(layer_view: memoryview, layer_meta: dict, name: str) -> tuple[torch.Tensor, memoryview]:
    meta = next((item for item in layer_meta["tensors"] if item["name"] == name), None)
    if meta is None:
        raise KeyError(name)
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name}: expected streamed BF16, got {meta['dtype']}")
    start = int(meta["offset"])
    nbytes = int(meta["nbytes"])
    shape = tuple(int(v) for v in meta["shape"])
    view = layer_view[start : start + nbytes]
    tensor = torch.frombuffer(view, dtype=torch.bfloat16, count=math.prod(shape)).reshape(shape)
    return tensor, view


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)


def gated_rms_norm(hidden: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden.dtype
    y = hidden.float()
    variance = y.pow(2).mean(-1, keepdim=True)
    y = y * torch.rsqrt(variance + eps)
    y = weight * y.to(input_dtype)
    y = y * F.silu(gate.float())
    return y.to(input_dtype)


def candidate_linear_layer(x: torch.Tensor, w: dict[str, torch.Tensor], cfg: Qwen3_5TextConfig) -> torch.Tensor:
    residual = x
    hidden = rms_norm(x, w["input_layernorm.weight"], cfg.rms_norm_eps)

    num_k_heads = int(cfg.linear_num_key_heads)
    num_v_heads = int(cfg.linear_num_value_heads)
    head_k_dim = int(cfg.linear_key_head_dim)
    head_v_dim = int(cfg.linear_value_head_dim)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed_qkv = F.linear(hidden, w["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv,
        w["linear_attn.conv1d.weight"].squeeze(1),
        None,
        activation=cfg.hidden_act,
    ).transpose(1, 2)

    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    batch, seq_len, _ = query.shape
    query = query.reshape(batch, seq_len, num_k_heads, head_k_dim)
    key = key.reshape(batch, seq_len, num_k_heads, head_k_dim)
    value = value.reshape(batch, seq_len, num_v_heads, head_v_dim)

    z = F.linear(hidden, w["linear_attn.in_proj_z.weight"]).reshape(batch, seq_len, num_v_heads, head_v_dim)
    b = F.linear(hidden, w["linear_attn.in_proj_b.weight"])
    a = F.linear(hidden, w["linear_attn.in_proj_a.weight"])
    beta = b.sigmoid()
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"])

    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core, _ = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    core = gated_rms_norm(
        core.reshape(-1, head_v_dim),
        z.reshape(-1, head_v_dim),
        w["linear_attn.norm.weight"],
        cfg.rms_norm_eps,
    ).reshape(batch, seq_len, value_dim)
    hidden = residual + F.linear(core, w["linear_attn.out_proj.weight"])

    residual = hidden
    hidden = rms_norm(hidden, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    mlp = F.linear(
        F.silu(F.linear(hidden, w["mlp.gate_proj.weight"])) * F.linear(hidden, w["mlp.up_proj.weight"]),
        w["mlp.down_proj.weight"],
    )
    return residual + mlp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer < 64:
        raise SystemExit("--layer must be in [0,63]")
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be positive")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_config_dict = config["text_config"]
    if text_config_dict["layer_types"][args.layer] != "linear_attention":
        raise ValueError(f"layer {args.layer} is not linear_attention")
    cfg = Qwen3_5TextConfig(**text_config_dict)
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
    prefix = f"{PREFIX}.{args.layer}."
    layer_names = sorted(name for name in weight_map if name.startswith(prefix))

    # Reference: official Transformers decoder layer + official BF16 state.
    ref_layer = Qwen3_5DecoderLayer(cfg, args.layer).to(dtype=torch.bfloat16).eval()
    expected_keys = set(ref_layer.state_dict().keys())
    source_state = {name[len(prefix):]: source_tensor(model_dir, weight_map, name) for name in layer_names}
    if set(source_state) != expected_keys:
        missing = sorted(expected_keys - set(source_state))
        unexpected = sorted(set(source_state) - expected_keys)
        raise RuntimeError(f"reference state mismatch missing={missing} unexpected={unexpected}")
    ref_layer.load_state_dict(source_state, strict=True)
    del source_state

    x = deterministic_hidden(args.seq_len, args.seed)
    with torch.inference_mode():
        dummy = torch.empty(0, dtype=torch.float32)
        reference = ref_layer(
            x.clone(),
            position_embeddings=(dummy, dummy),
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
        )
    rss_after_reference = max_rss_gib()

    # Candidate: zero-copy BF16 views over one O_DIRECT-backed ring load.
    with K3Trunk(
        out_bin, out_idx,
        budget_bytes=layer_bytes,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        layer_view = trunk.bind(args.layer)
        keepalive: list[memoryview] = []
        streamed: dict[str, torch.Tensor] = {}
        for full_name in layer_names:
            local_name = full_name[len(prefix):]
            tensor, view = streamed_tensor(layer_view, layer_meta, full_name)
            streamed[local_name] = tensor
            keepalive.append(view)
        del tensor, view
        if set(streamed) != expected_keys:
            raise RuntimeError("streamed state keys do not match decoder state keys")
        with torch.inference_mode():
            candidate = candidate_linear_layer(x.clone(), streamed, cfg)
        result_metrics = metrics(reference, candidate)
        runtime_report = trunk.report()
        del candidate
        streamed.clear()
        keepalive.clear()
        del layer_view

    passed = result_metrics["exact_equal"]
    result = {
        "schema": "qwen38-k3-linear-layer-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "layer": args.layer,
        "layer_type": text_config_dict["layer_types"][args.layer],
        "seq_len": args.seq_len,
        "seed": args.seed,
        "tensor_count": len(layer_names),
        "source_shards": shards,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "layer_read_bytes": layer_bytes,
        "output_metrics": result_metrics,
        "runtime": runtime_report,
        "rss_after_reference_gib": rss_after_reference,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "reference": "Transformers Qwen3_5DecoderLayer CPU PyTorch fallback",
        "candidate": "manual decoder wiring over zero-copy K3Trunk BF16 views",
        "generation_attempted": False,
        "logits_claimed": False,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
