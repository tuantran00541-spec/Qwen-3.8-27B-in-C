#!/usr/bin/env python3
"""Real-weight prefill -> cached one-token decode equivalence for Qwen3.8.

This is the bridge between exact stateless logits and autoregressive generation.
It validates both cache families used by Qwen3.8 on official BF16 weights:

* layer 0 linear_attention: convolution state + FP32 recurrent DeltaNet state
* layer 3 full_attention: rotated key/value cache

The reference uses Transformers Qwen3_5DecoderLayer + DynamicCache.  The
candidate manually executes the same equations from zero-copy K3Trunk BF16
weight views and owns its cache tensors explicitly.  It compares prefill output,
cache contents, one-token decode output, and updated cache contents exactly.
No tokenizer, LM head, or prompt generation is involved.
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import DynamicCache, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb

from core import MODEL_ID, PINNED_REVISION, PREFIX
from k3_full_attention_gate import (
    attention_rms_norm,
    causal_mask,
    decoder_rms_norm,
    deterministic_hidden,
    optional_bias,
    repeat_kv,
)
from k3_linear_layer_gate import (
    download_metadata,
    gated_rms_norm,
    metrics,
    rms_norm,
    source_tensor,
    streamed_tensor,
)
from k3_stream import K3Trunk, needed_shards, pack_layers

LINEAR_LAYER = 0
FULL_LAYER = 3


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def tensor_bytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())


def state_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    out = metrics(reference, candidate)
    out.update({
        "shape": list(reference.shape),
        "dtype": str(reference.dtype).removeprefix("torch."),
        "bytes": tensor_bytes(reference),
    })
    return out


def load_reference_layer(
    cfg: Qwen3_5TextConfig,
    layer: int,
    model_dir: Path,
    weight_map: dict[str, str],
) -> Qwen3_5DecoderLayer:
    prefix = f"{PREFIX}.{layer}."
    names = sorted(name for name in weight_map if name.startswith(prefix))
    module = Qwen3_5DecoderLayer(cfg, layer).to(dtype=torch.bfloat16).eval()
    state = {name[len(prefix):]: source_tensor(model_dir, weight_map, name) for name in names}
    expected = set(module.state_dict())
    if set(state) != expected:
        raise RuntimeError(
            f"layer {layer} reference state mismatch: missing={sorted(expected-set(state))} "
            f"unexpected={sorted(set(state)-expected)}"
        )
    module.load_state_dict(state, strict=True)
    del state
    gc.collect()
    return module


def streamed_state(
    layer_view: memoryview,
    layer_meta: dict,
    layer: int,
    weight_map: dict[str, str],
) -> tuple[dict[str, torch.Tensor], list[memoryview]]:
    prefix = f"{PREFIX}.{layer}."
    state: dict[str, torch.Tensor] = {}
    keepalive: list[memoryview] = []
    for full_name in sorted(name for name in weight_map if name.startswith(prefix)):
        tensor, view = streamed_tensor(layer_view, layer_meta, full_name)
        state[full_name[len(prefix):]] = tensor
        keepalive.append(view)
    del tensor, view
    return state, keepalive


def candidate_linear_cached(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    cfg: Qwen3_5TextConfig,
    conv_state: torch.Tensor | None,
    recurrent_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual linear-attention decoder layer with explicit cache ownership."""
    residual = x
    hidden = rms_norm(x, w["input_layernorm.weight"], cfg.rms_norm_eps)

    num_k_heads = int(cfg.linear_num_key_heads)
    num_v_heads = int(cfg.linear_num_value_heads)
    head_k_dim = int(cfg.linear_key_head_dim)
    head_v_dim = int(cfg.linear_value_head_dim)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    kernel = int(w["linear_attn.conv1d.weight"].shape[-1])

    raw_qkv = F.linear(hidden, w["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)
    z = F.linear(hidden, w["linear_attn.in_proj_z.weight"]).reshape(
        x.shape[0], x.shape[1], num_v_heads, head_v_dim
    )
    b = F.linear(hidden, w["linear_attn.in_proj_b.weight"])
    a = F.linear(hidden, w["linear_attn.in_proj_a.weight"])

    if conv_state is None:
        # DynamicCache prefill: left-pad only when shorter than the kernel, cache
        # exactly the last kernel raw projected states, then causal-convolve the
        # full prefill and retain only the current sequence outputs.
        full_qkv = raw_qkv
        if full_qkv.shape[-1] < kernel:
            full_qkv = F.pad(full_qkv, (kernel - full_qkv.shape[-1], 0), value=0)
        new_conv_state = full_qkv[..., -kernel:].clone()
        mixed_qkv = causal_conv1d_fn(
            full_qkv,
            w["linear_attn.conv1d.weight"].squeeze(1),
            None,
            activation=cfg.hidden_act,
        )[..., -x.shape[1] :]
        recurrent_fn = torch_chunk_gated_delta_rule
        initial_recurrent = None
    else:
        if x.shape[1] != 1:
            raise ValueError("cached linear decode gate expects exactly one token")
        new_conv_state = conv_state.clone()
        mixed_qkv = causal_conv1d_update(
            raw_qkv,
            new_conv_state,
            w["linear_attn.conv1d.weight"].squeeze(1),
            None,
            cfg.hidden_act,
        )
        recurrent_fn = torch_recurrent_gated_delta_rule
        initial_recurrent = recurrent_state

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    batch, seq_len, _ = query.shape
    query = query.reshape(batch, seq_len, num_k_heads, head_k_dim)
    key = key.reshape(batch, seq_len, num_k_heads, head_k_dim)
    value = value.reshape(batch, seq_len, num_v_heads, head_v_dim)

    beta = b.sigmoid()
    g = -w["linear_attn.A_log"].float().exp() * F.softplus(a.float() + w["linear_attn.dt_bias"])
    repeat = num_v_heads // num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core, new_recurrent_state = recurrent_fn(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_recurrent,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    if new_recurrent_state is None:
        raise RuntimeError("linear cache candidate did not produce recurrent state")

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
    return residual + mlp, new_conv_state, new_recurrent_state


def candidate_full_cached(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    cfg: Qwen3_5TextConfig,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key: torch.Tensor | None,
    past_value: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Manual full-attention decoder layer with explicit rotated KV cache."""
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
    if past_key is not None:
        if past_value is None:
            raise ValueError("past key/value must be provided together")
        key = torch.cat([past_key, key], dim=-2)
        value = torch.cat([past_value, value], dim=-2)
    new_key = key.clone()
    new_value = value.clone()

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
    return residual + mlp, new_key, new_value


def reference_linear(
    cfg: Qwen3_5TextConfig,
    module: Qwen3_5DecoderLayer,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
) -> dict:
    cache = DynamicCache(config=cfg)
    dummy = torch.empty(0, dtype=torch.float32)
    with torch.inference_mode():
        prefill = module(
            x_prefill.clone(),
            position_embeddings=(dummy, dummy),
            attention_mask=None,
            position_ids=None,
            past_key_values=cache,
        )
    layer_cache = cache.layers[LINEAR_LAYER]
    conv_prefill = layer_cache.conv_states[0].clone()
    recurrent_prefill = layer_cache.recurrent_states[0].clone()
    if not cache.has_previous_state(LINEAR_LAYER, state_idx=0):
        raise RuntimeError("reference linear cache did not mark previous state")
    with torch.inference_mode():
        decode = module(
            x_decode.clone(),
            position_embeddings=(dummy, dummy),
            attention_mask=None,
            position_ids=None,
            past_key_values=cache,
        )
    return {
        "prefill": prefill,
        "decode": decode,
        "conv_prefill": conv_prefill,
        "recurrent_prefill": recurrent_prefill,
        "conv_decode": layer_cache.conv_states[0].clone(),
        "recurrent_decode": layer_cache.recurrent_states[0].clone(),
    }


def reference_full(
    cfg: Qwen3_5TextConfig,
    module: Qwen3_5DecoderLayer,
    rotary: Qwen3_5TextRotaryEmbedding,
    x_prefill: torch.Tensor,
    x_decode: torch.Tensor,
) -> dict:
    cache = DynamicCache(config=cfg)
    prefill_ids = torch.arange(x_prefill.shape[1], dtype=torch.long).unsqueeze(0)
    decode_ids = torch.tensor([[x_prefill.shape[1]]], dtype=torch.long)
    with torch.inference_mode():
        prefill_pos = rotary(x_prefill, prefill_ids)
        prefill = module(
            x_prefill.clone(),
            position_embeddings=prefill_pos,
            attention_mask=causal_mask(x_prefill.shape[1]),
            position_ids=prefill_ids,
            past_key_values=cache,
        )
    layer_cache = cache.layers[FULL_LAYER]
    key_prefill = layer_cache.keys.clone()
    value_prefill = layer_cache.values.clone()
    with torch.inference_mode():
        decode_pos = rotary(x_decode, decode_ids)
        decode = module(
            x_decode.clone(),
            position_embeddings=decode_pos,
            attention_mask=torch.zeros((1, 1, 1, x_prefill.shape[1] + 1), dtype=torch.float32),
            position_ids=decode_ids,
            past_key_values=cache,
        )
    return {
        "prefill": prefill,
        "decode": decode,
        "key_prefill": key_prefill,
        "value_prefill": value_prefill,
        "key_decode": layer_cache.keys.clone(),
        "value_decode": layer_cache.values.clone(),
        "prefill_pos": prefill_pos,
        "decode_pos": decode_pos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--prefill-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prefill_len < 2:
        raise SystemExit("--prefill-len must be >=2")

    root = args.work_dir.resolve()
    model_dir = root / "model"
    trunk_dir = root / "trunk"
    model_dir.mkdir(parents=True, exist_ok=True)
    trunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    config, index, validated = download_metadata(model_dir)
    text_dict = config["text_config"]
    if text_dict["layer_types"][LINEAR_LAYER] != "linear_attention":
        raise RuntimeError("expected layer 0 to be linear_attention")
    if text_dict["layer_types"][FULL_LAYER] != "full_attention":
        raise RuntimeError("expected layer 3 to be full_attention")
    cfg = Qwen3_5TextConfig(**text_dict)
    cfg._attn_implementation = "eager"
    weight_map = index["weight_map"]

    layers = [LINEAR_LAYER, FULL_LAYER]
    shards = needed_shards(weight_map, layers)
    for shard in shards:
        hf_hub_download(
            MODEL_ID,
            filename=shard,
            revision=PINNED_REVISION,
            local_dir=str(model_dir),
        )

    out_bin = trunk_dir / "cache-gate.trunk.bin"
    out_idx = trunk_dir / "cache-gate.trunk.json"
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

    x_prefill_linear = deterministic_hidden(args.prefill_len, args.seed)
    x_decode_linear = deterministic_hidden(1, args.seed + 1)
    x_prefill_full = deterministic_hidden(args.prefill_len, args.seed + 2)
    x_decode_full = deterministic_hidden(1, args.seed + 3)

    # Official references first, then release module weights before K3 candidate.
    ref_linear_module = load_reference_layer(cfg, LINEAR_LAYER, model_dir, weight_map)
    linear_ref = reference_linear(cfg, ref_linear_module, x_prefill_linear, x_decode_linear)
    del ref_linear_module
    gc.collect()

    ref_full_module = load_reference_layer(cfg, FULL_LAYER, model_dir, weight_map)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    full_ref = reference_full(cfg, ref_full_module, rotary, x_prefill_full, x_decode_full)
    del ref_full_module
    gc.collect()

    with K3Trunk(
        out_bin,
        out_idx,
        budget_bytes=max_layer_bytes,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as trunk:
        # Linear candidate prefill and decode, reusing explicit states while the
        # streamed BF16 weights remain bound in the single safe slot.
        linear_view = trunk.bind(LINEAR_LAYER)
        linear_w, linear_keepalive = streamed_state(
            linear_view, meta_by_layer[LINEAR_LAYER], LINEAR_LAYER, weight_map
        )
        with torch.inference_mode():
            linear_prefill, conv_prefill, recurrent_prefill = candidate_linear_cached(
                x_prefill_linear.clone(), linear_w, cfg, None, None
            )
            linear_decode, conv_decode, recurrent_decode = candidate_linear_cached(
                x_decode_linear.clone(), linear_w, cfg, conv_prefill, recurrent_prefill
            )
        linear_metrics = {
            "prefill_output": metrics(linear_ref["prefill"], linear_prefill),
            "decode_output": metrics(linear_ref["decode"], linear_decode),
            "conv_prefill": state_metrics(linear_ref["conv_prefill"], conv_prefill),
            "recurrent_prefill": state_metrics(linear_ref["recurrent_prefill"], recurrent_prefill),
            "conv_decode": state_metrics(linear_ref["conv_decode"], conv_decode),
            "recurrent_decode": state_metrics(linear_ref["recurrent_decode"], recurrent_decode),
        }
        linear_w.clear()
        linear_keepalive.clear()
        del linear_view

        # One slot means bind is synchronous and cannot overwrite an active
        # layer asynchronously. Load the full-attention layer only after all
        # layer-0 views have been released.
        full_view = trunk.bind(FULL_LAYER)
        full_w, full_keepalive = streamed_state(full_view, meta_by_layer[FULL_LAYER], FULL_LAYER, weight_map)
        with torch.inference_mode():
            full_prefill, key_prefill, value_prefill = candidate_full_cached(
                x_prefill_full.clone(),
                full_w,
                cfg,
                full_ref["prefill_pos"],
                causal_mask(args.prefill_len),
                None,
                None,
            )
            full_decode, key_decode, value_decode = candidate_full_cached(
                x_decode_full.clone(),
                full_w,
                cfg,
                full_ref["decode_pos"],
                torch.zeros((1, 1, 1, args.prefill_len + 1), dtype=torch.float32),
                key_prefill,
                value_prefill,
            )
        full_metrics = {
            "prefill_output": metrics(full_ref["prefill"], full_prefill),
            "decode_output": metrics(full_ref["decode"], full_decode),
            "key_prefill": state_metrics(full_ref["key_prefill"], key_prefill),
            "value_prefill": state_metrics(full_ref["value_prefill"], value_prefill),
            "key_decode": state_metrics(full_ref["key_decode"], key_decode),
            "value_decode": state_metrics(full_ref["value_decode"], value_decode),
        }
        full_w.clear()
        full_keepalive.clear()
        del full_view
        runtime = trunk.report()

    if runtime["async_prefetch_enabled"]:
        raise RuntimeError("one-slot cache gate must never enable async prefetch")
    expected_read = int(meta_by_layer[LINEAR_LAYER]["read_bytes"]) + int(meta_by_layer[FULL_LAYER]["read_bytes"])
    if int(runtime["bytes_read"]) != expected_read:
        raise RuntimeError(f"read {runtime['bytes_read']} bytes, expected {expected_read}")

    exact_flags = [
        item["exact_equal"]
        for group in (linear_metrics, full_metrics)
        for item in group.values()
    ]
    passed = all(exact_flags)
    result = {
        "schema": "qwen38-k3-cache-gate-v1",
        "status": "PASS" if passed else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "official_metadata": validated,
        "prefill_len": args.prefill_len,
        "decode_len": 1,
        "linear_layer": LINEAR_LAYER,
        "linear_metrics": linear_metrics,
        "full_attention_layer": FULL_LAYER,
        "full_attention_metrics": full_metrics,
        "runtime": runtime,
        "source_shards": shards,
        "packed_file_bytes": manifest["packed_file_bytes"],
        "expected_stream_bytes": expected_read,
        "cache_bytes_after_prefill": {
            "linear_conv": tensor_bytes(conv_prefill),
            "linear_recurrent": tensor_bytes(recurrent_prefill),
            "full_key": tensor_bytes(key_prefill),
            "full_value": tensor_bytes(value_prefill),
        },
        "cache_bytes_after_decode": {
            "linear_conv": tensor_bytes(conv_decode),
            "linear_recurrent": tensor_bytes(recurrent_decode),
            "full_key": tensor_bytes(key_decode),
            "full_value": tensor_bytes(value_decode),
        },
        "max_rss_gib": max_rss_gib(),
        "total_seconds": time.monotonic() - started,
        "disk_free_bytes_after": shutil.disk_usage(root).free,
        "reference": "Transformers Qwen3_5DecoderLayer + DynamicCache",
        "candidate": "manual decoder equations over one-slot O_DIRECT K3Trunk + explicit cache tensors",
        "cache_claimed": bool(passed),
        "generation_attempted": False,
        "logits_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
