#!/usr/bin/env python3
"""Low-cost official-weight Q5/Q4 distortion probe for Qwen3.8-27B MLP matrices.

Downloads only the shards needed for one decoder layer and reads a bounded row
slice from gate/up/down projections. This is a measurement gate, not a model
quality claim and not full-model quantization.
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from core import MODEL_ID, PINNED_REVISION

PREFIX = "model.language_model.layers"


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    ref = ref.float(); cand = cand.float(); diff = cand - ref
    denom = torch.linalg.vector_norm(ref).clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_abs": float(diff.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff))),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denom),
        "cosine": float(F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0)),
    }


def quant_dequant_rowwise(weight: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, int, int]:
    if bits not in (4, 5):
        raise ValueError("bits must be 4 or 5")
    w = weight.float().contiguous()
    rows, cols = w.shape
    groups = math.ceil(cols / group_size)
    padded_cols = groups * group_size
    if padded_cols != cols:
        wpad = F.pad(w, (0, padded_cols - cols))
    else:
        wpad = w
    grouped = wpad.reshape(rows, groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    scales = grouped.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny) / qmax
    q = torch.round(grouped / scales[..., None]).clamp(-qmax, qmax)
    restored = (q * scales[..., None]).reshape(rows, padded_cols)[:, :cols]
    # Production target: tightly packed codes + FP16/BF16 scale per row-group.
    code_bytes = math.ceil(rows * groups * group_size * bits / 8)
    scale_bytes = rows * groups * 2
    return restored, code_bytes, scale_bytes


def tensor_slice(path: Path, name: str, rows: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        sl = handle.get_slice(name)
        shape = tuple(int(v) for v in sl.get_shape())
        if len(shape) != 2:
            raise ValueError(f"{name}: expected matrix, got {shape}")
        take = min(rows, shape[0])
        out = sl[:take, :]
    if out.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16, got {out.dtype}")
    return out, shape


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260831)
    a = p.parse_args()
    if not 0 <= a.layer < 64 or a.rows <= 0 or a.group_size <= 0 or a.samples <= 0:
        raise SystemExit("invalid arguments")

    root = a.work_dir.resolve(); root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    index_path = Path(hf_hub_download(MODEL_ID, filename="model.safetensors.index.json", revision=PINNED_REVISION, local_dir=str(root)))
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    names = {
        "gate": f"{PREFIX}.{a.layer}.mlp.gate_proj.weight",
        "up": f"{PREFIX}.{a.layer}.mlp.up_proj.weight",
        "down": f"{PREFIX}.{a.layer}.mlp.down_proj.weight",
    }
    missing = [name for name in names.values() if name not in weight_map]
    if missing:
        raise KeyError(f"missing official tensors: {missing}")

    shards = sorted({weight_map[name] for name in names.values()})
    local = {shard: Path(hf_hub_download(MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(root))) for shard in shards}
    gen = torch.Generator(device="cpu"); gen.manual_seed(a.seed)
    projections = {}
    for key, name in names.items():
        weight, full_shape = tensor_slice(local[weight_map[name]], name, a.rows)
        x = torch.randn((a.samples, full_shape[1]), generator=gen, dtype=torch.float32)
        ref_y = F.linear(x, weight.float())
        cases = []
        for bits in (5, 4):
            restored, slice_code_bytes, slice_scale_bytes = quant_dequant_rowwise(weight, bits, a.group_size)
            cand_y = F.linear(x, restored)
            full_rows, full_cols = full_shape
            full_groups = math.ceil(full_cols / a.group_size)
            full_code_bytes = math.ceil(full_rows * full_groups * a.group_size * bits / 8)
            full_scale_bytes = full_rows * full_groups * 2
            bf16_bytes = full_rows * full_cols * 2
            cases.append({
                "bits": bits,
                "group_size": a.group_size,
                "weight_error": metrics(weight, restored),
                "linear_error": metrics(ref_y, cand_y),
                "sampled_quant_bytes": slice_code_bytes + slice_scale_bytes,
                "projected_full_quant_bytes": full_code_bytes + full_scale_bytes,
                "projected_full_bf16_bytes": bf16_bytes,
                "projected_compression_vs_bf16": bf16_bytes / (full_code_bytes + full_scale_bytes),
            })
        projections[key] = {
            "name": name,
            "full_shape": list(full_shape),
            "sampled_rows": int(weight.shape[0]),
            "source_shard": weight_map[name],
            "cases": cases,
        }
        del weight, x, ref_y

    q5_better = all(pj["cases"][0]["linear_error"]["relative_l2"] <= pj["cases"][1]["linear_error"]["relative_l2"] for pj in projections.values())
    result = {
        "schema": "qwen38-k3-real-mlp-quant-probe-v1",
        "status": "PASS" if q5_better else "FAIL",
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "layer": a.layer,
        "rows": a.rows,
        "group_size": a.group_size,
        "samples": a.samples,
        "source_shards": shards,
        "projections": projections,
        "q5_lower_or_equal_linear_error_than_q4": q5_better,
        "real_weights_attempted": True,
        "full_model_quantized": False,
        "generation_attempted": False,
        "quality_claimed": False,
        "max_rss_gib": max_rss_gib(),
        "seconds": time.monotonic() - started,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
