#!/usr/bin/env python3
"""Exact real-weight full-attention decoder-layer gate over the K3-style Qwen trunk.

This gate validates an official Qwen3.8 full_attention layer with a non-trivial
multi-token causal sequence. The reference is Transformers' Qwen3_5DecoderLayer
loaded with official BF16 weights. The candidate manually wires the same block
using zero-copy BF16 tensor views over one K3Trunk ring slot.

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

os.environ.setdefault("USE_HUB_KERNELS", "NO")

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)
from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb

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


def decoder_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)


def attention_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Qwen3Next q_norm/k_norm are zero-centered RMSNorm too: (1 + weight).
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def optional_bias(w: dict[str, torch.Tensor], stem: str) -> torch.Tensor | None:
    return w.get(f"{stem}.bias")


def candidate_full_attention_layer(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    cfg: Qwen3_5TextConfig,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    residual = x
    hidden = decoder_rms_norm(x, w["input_layernorm.weight"], cfg.rms_norm_eps)

    input_shape = hidden.shape[:-1]
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    num_kv_groups = num_heads // num_kv_heads
    scaling = head_dim ** -0.5

    q_projected = F.linear(
        hidden,
        w["self_attn.q_proj.weight"],
        optional_bias(w, "self_attn.q_proj"),
    ).view(*input_shape, -1, head_dim * 2)
    query, gate = torch.chunk(q_projected, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query = attention_rms_norm(query, w["self_attn.q_norm.weight"], cfg.rms_norm_eps).transpose(1, 2)
    key = attention_rms_norm(
        F.linear(hidden, w["self_attn.k_proj.weight"], optional_bias(w, "self_attn.k_proj")).view(
            *input_shape, -1, head_dim
        ),
        w["self_attn.k_norm.weight"],
        cfg.rms_norm_eps,
    ).transpose(1, 2)
    value = F.linear(
        hidden,
        w["self_attn.v_proj.weight"],
        optional_bias(w, "self_attn.v_proj"),
    ).view(*input_shape, -1, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    key_states = repeat_kv(key, num_kv_groups)
    value_states = repeat_kv(value, num_kv_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = F.linear(
        attn_output,
        w["self_attn.o_proj.weight"],
        optional_bias(w, "self_attn.o_proj"),
    )
    hidden = residual + attn_output

    residual = hidden
    hidden = decoder_rms_norm(hidden, w["post_attention_layernorm.weight"], cfg.rms_norm_eps)
    mlp = F.linear(
        F.silu(F.linear(hidden, w["mlp.gate_proj.weight"])) * F.linear(hidden, w["mlp.up_proj.weight"]),
        w["mlp.down_proj.weight"],
    )
    return residual + mlp


def causal_mask(seq_len: int) -> torch.Tensor:
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float32)
    blocked = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
    return mask.masked_fill(blocked, float("-inf"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.layer < 64:
        raise SystemExit("--layer must be in [0,63]")
    if args.seq_len < 2:
        raise SystemExit("--seq-len must be >= 2 for a meaningful full-attention gate")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_config_dict = config["text_config"]
    if text_config_dict["layer_types"][args.layer] != "full_attention":
        raise ValueError(f"layer {args.layer} is not full_attention")
    cfg = Qwen3_5TextConfig(**text_config_dict)
    cfg._attn_implementation = "eager"
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
    position_ids = torch.arange(args.seq_len, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.inference_mode():
        position_embeddings = rotary(x, position_ids)
    mask = causal_mask(args.seq_len)

    with torch.inference_mode():
        reference = ref_layer(
            x.clone(),
            position_embeddings=position_embeddings,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=None,
        )
    rss_after_reference = max_rss_gib()

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
            candidate = candidate_full_attention_layer(
                x.clone(), streamed, cfg, position_embeddings, mask
            )
        result_metrics = metrics(reference, candidate)
        runtime_report = trunk.report()
        del candidate
        streamed.clear()
        keepalive.clear()
        del layer_view

    passed = result_metrics["exact_equal"]
    result = {
        "schema": "qwen38-k3-full-attention-layer-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "layer": args.layer,
        "layer_type": text_config_dict["layer_types"][args.layer],
        "seq_len": args.seq_len,
        "seed": args.seed,
        "attention_impl": "eager",
        "causal_mask": True,
        "rope": True,
        "tensor_count": len(layer_names),
        "source_shards": shards,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "layer_read_bytes": layer_bytes,
        "output_metrics": result_metrics,
        "runtime": runtime_report,
        "rss_after_reference_gib": rss_after_reference,
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "reference": "Transformers Qwen3_5DecoderLayer eager CPU BF16",
        "candidate": "manual full-attention decoder wiring over zero-copy K3Trunk BF16 views",
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
