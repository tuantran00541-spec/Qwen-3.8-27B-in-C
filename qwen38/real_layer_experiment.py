#!/usr/bin/env python3
"""Pinned real-BF16 Qwen3.8-27B MLP pruning/quantization sensitivity experiment.

This is deliberately layer-local. Calibration and evaluation inputs are
synthetic RMS-normalized hidden states with disjoint seeds. Real model weights
are measured, but this is not a semantic/full-model/generation quality test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from core import (
    HIDDEN,
    INTERMEDIATE,
    MODEL_ID,
    PINNED_REVISION,
    activation_energy_scores_from_hidden,
    activation_guided_reconstruction_refine,
    aligned_prune_count,
    candidate_output_from_pruned,
    keep_to_pruned,
    linear_down,
    mlp_forward,
    norm_channel_scores,
    output_metrics,
    parse_csv_floats,
    parse_csv_ints,
    projected_qbytes,
    select_keep_indices,
    swiglu_hidden,
    tensor_name,
    validate_official_metadata,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_metadata(root: Path) -> tuple[dict, dict, dict]:
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename in ("config.json", "model.safetensors.index.json"):
        paths[filename] = Path(hf_hub_download(MODEL_ID, filename=filename, revision=PINNED_REVISION, local_dir=str(root)))
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    index = json.loads(paths["model.safetensors.index.json"].read_text(encoding="utf-8"))
    validated = validate_official_metadata(config, index)
    validated["metadata_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    return config, index, validated


def download_layer_shards(root: Path, index: dict, layer: int) -> dict[str, tuple[str, str]]:
    weight_map = index["weight_map"]
    parts: dict[str, tuple[str, str]] = {}
    for part in ("gate_proj", "up_proj", "down_proj"):
        name = tensor_name(layer, part)
        shard = weight_map.get(name)
        if not shard:
            raise KeyError(name)
        parts[part] = (name, shard)
    for shard in sorted({shard for _, shard in parts.values()}):
        hf_hub_download(MODEL_ID, filename=shard, revision=PINNED_REVISION, local_dir=str(root))
    return parts


def load_bf16_tensor(root: Path, name: str, shard: str) -> torch.Tensor:
    with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected BF16, got {tensor.dtype}")
    return tensor.contiguous()


def normalized_random(samples: int, hidden: int, seed: int) -> torch.Tensor:
    if samples <= 0:
        raise ValueError("samples must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((samples, hidden), generator=generator, dtype=torch.float32)
    return x / torch.sqrt(torch.mean(x * x, dim=1, keepdim=True))


def pruned_set(keep: torch.Tensor, intermediate: int) -> set[int]:
    kept = set(int(v) for v in keep.tolist())
    return set(range(intermediate)) - kept


def overlap_metrics(a: set[int], b: set[int]) -> dict:
    intersection = len(a & b)
    union = len(a | b)
    return {
        "intersection": intersection,
        "union": union,
        "jaccard": intersection / union if union else 1.0,
        "same_fraction_of_pruned": intersection / len(a) if a else 1.0,
    }


def log_score_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("score vectors must be matching rank-1 tensors")
    eps = torch.finfo(torch.float64).tiny
    x = torch.log(a.double().clamp_min(eps)); y = torch.log(b.double().clamp_min(eps))
    x = x - x.mean(); y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float(torch.dot(x, y) / denom) if float(denom) else 1.0


def calibration_metric(reference: torch.Tensor, hidden: torch.Tensor, down: torch.Tensor, keep: torch.Tensor) -> dict:
    pruned = keep_to_pruned(keep, hidden.shape[1])
    candidate = candidate_output_from_pruned(reference, hidden, down, pruned)
    return output_metrics(reference, candidate)


def run_layer(
    root: Path,
    index: dict,
    layer: int,
    prune_fractions: list[float],
    bits_list: list[int],
    group_size: int,
    calibration_samples: int,
    evaluation_samples: int,
    seed: int,
) -> dict:
    parts = download_layer_shards(root, index, layer)
    source_shards = sorted({shard for _, shard in parts.values()})
    gate = load_bf16_tensor(root, *parts["gate_proj"])
    up = load_bf16_tensor(root, *parts["up_proj"])
    down = load_bf16_tensor(root, *parts["down_proj"])
    expected_shapes = ((INTERMEDIATE, HIDDEN), (INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE))
    actual_shapes = (tuple(gate.shape), tuple(up.shape), tuple(down.shape))
    if actual_shapes != expected_shapes:
        raise ValueError(f"layer {layer}: unexpected MLP shapes {actual_shapes}")

    calibration_x = normalized_random(calibration_samples, HIDDEN, seed)
    evaluation_x = normalized_random(evaluation_samples, HIDDEN, seed + 1_000_003)

    score_started = time.monotonic()
    norm_scores = norm_channel_scores(gate, up, down)
    calibration_hidden = swiglu_hidden(calibration_x, gate, up)
    activation_scores, activation_diagnostics = activation_energy_scores_from_hidden(calibration_hidden, down)
    half = calibration_samples // 2
    if half < 2:
        raise ValueError("calibration_samples must be at least 4 for split-half diagnostics")
    first_scores, _ = activation_energy_scores_from_hidden(calibration_hidden[:half], down)
    second_scores, _ = activation_energy_scores_from_hidden(calibration_hidden[half:], down)
    calibration_reference = linear_down(calibration_hidden, down)
    score_seconds = time.monotonic() - score_started

    eval_started = time.monotonic()
    reference = mlp_forward(evaluation_x, gate, up, down)
    qonly: dict[str, dict] = {}
    for bits in bits_list:
        qonly[f"q{bits}_only"] = output_metrics(reference, mlp_forward(evaluation_x, gate, up, down, bits=bits, group_size=group_size))

    pruning_results: list[dict] = []
    full_params = 3 * HIDDEN * INTERMEDIATE
    full_bf16_bytes = full_params * 2

    for requested_fraction in prune_fractions:
        prune_channels = aligned_prune_count(INTERMEDIATE, requested_fraction, group_size)
        keep_n = INTERMEDIATE - prune_channels
        actual_fraction = prune_channels / INTERMEDIATE
        norm_keep = select_keep_indices(norm_scores, prune_channels, alignment=group_size)
        activation_keep = select_keep_indices(activation_scores, prune_channels, alignment=group_size)
        refined_keep, refinement = activation_guided_reconstruction_refine(
            calibration_reference, calibration_hidden, down, norm_keep, activation_scores
        )
        first_keep = select_keep_indices(first_scores, prune_channels, alignment=group_size)
        second_keep = select_keep_indices(second_scores, prune_channels, alignment=group_size)

        norm_pruned = pruned_set(norm_keep, INTERMEDIATE)
        activation_pruned = pruned_set(activation_keep, INTERMEDIATE)
        refined_pruned = pruned_set(refined_keep, INTERMEDIATE)
        first_pruned = pruned_set(first_keep, INTERMEDIATE)
        second_pruned = pruned_set(second_keep, INTERMEDIATE)

        scorers: dict[str, dict] = {}
        for scorer_name, keep in (
            ("weight_norm", norm_keep),
            ("activation_energy", activation_keep),
            ("reconstruction_refined", refined_keep),
        ):
            entry = {
                "calibration_prune_only_bf16": calibration_metric(calibration_reference, calibration_hidden, down, keep),
                "prune_only_bf16": output_metrics(reference, mlp_forward(evaluation_x, gate, up, down, keep=keep)),
                "quantized": {},
            }
            for bits in bits_list:
                combo = mlp_forward(evaluation_x, gate, up, down, keep=keep, bits=bits, group_size=group_size)
                entry["quantized"][f"prune_plus_q{bits}"] = output_metrics(reference, combo)
            scorers[scorer_name] = entry

        storage = {}
        for bits in bits_list:
            projected = projected_qbytes(HIDDEN, keep_n, bits, group_size)
            storage[f"q{bits}"] = {
                "projection_only": True,
                "projected_bytes": projected["projected_total_bytes"],
                "reduction_vs_full_mlp_bf16": 1.0 - projected["projected_total_bytes"] / full_bf16_bytes,
            }

        pruning_results.append({
            "requested_prune_fraction": requested_fraction,
            "pruned_channels": prune_channels,
            "actual_prune_fraction": actual_fraction,
            "intermediate_after": keep_n,
            "removed_mlp_params": 3 * HIDDEN * prune_channels,
            "pruned_overlap": {
                "norm_vs_activation": overlap_metrics(norm_pruned, activation_pruned),
                "norm_vs_refined": overlap_metrics(norm_pruned, refined_pruned),
                "activation_split_half": overlap_metrics(first_pruned, second_pruned),
            },
            "reconstruction_refinement": refinement,
            "scorers": scorers,
            "storage_projection": storage,
        })

    return {
        "layer": layer,
        "source_shards": source_shards,
        "weight_dtype": "bfloat16",
        "calibration": {
            "kind": "synthetic_rms_normalized_hidden_states",
            "samples": calibration_samples,
            "seed": seed,
            "used_for_ranking_and_reconstruction_selection": True,
        },
        "evaluation": {
            "kind": "synthetic_rms_normalized_hidden_states",
            "samples": evaluation_samples,
            "seed": seed + 1_000_003,
            "disjoint_from_calibration": True,
        },
        "activation_scorer": {
            "formula": "E[(SiLU(gate_j(x))*up_j(x))^2] * ||down[:,j]||_2^2",
            "diagnostics": activation_diagnostics,
            "split_half_log_score_pearson": log_score_pearson(first_scores, second_scores),
        },
        "refinement": {
            "kind": "activation-guided channel swaps selected by exact calibration MLP reconstruction",
            "fallback": "weight_norm baseline is always a candidate",
        },
        "score_seconds": score_seconds,
        "qonly_metrics": qonly,
        "pruning_results": pruning_results,
        "evaluation_seconds": time.monotonic() - eval_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--prune-fractions", default="0.05,0.10,0.15")
    parser.add_argument("--bits", default="6,5,4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--evaluation-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prune_fractions = parse_csv_floats(args.prune_fractions)
    bits_list = parse_csv_ints(args.bits)
    if any(not 0 < fraction < 0.5 for fraction in prune_fractions):
        raise SystemExit("prune fractions must be in (0, 0.5)")
    if any(not 2 <= bits <= 8 for bits in bits_list):
        raise SystemExit("bits must be in [2,8]")
    if args.group_size != 128:
        raise SystemExit("this pilot pins group/alignment size to 128")
    if args.calibration_samples < 4:
        raise SystemExit("calibration-samples must be >= 4")

    started = time.monotonic()
    _, index, metadata = download_metadata(args.work_dir)
    result = {
        "schema": "qwen38-sparse-aware-knife-v2",
        "scope": "single real BF16 MLP layer; synthetic RMS-normalized calibration/evaluation; no autoregressive generation; no semantic or full-model quality claim",
        "model": metadata,
        "group_size": args.group_size,
        "bits": bits_list,
        "layer_result": run_layer(
            args.work_dir, index, args.layer, prune_fractions, bits_list, args.group_size,
            args.calibration_samples, args.evaluation_samples, args.seed,
        ),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    layer = result["layer_result"]
    print(f"QWEN38_REAL_LAYER_DONE layer={layer['layer']} revision={PINNED_REVISION} activation_split_pearson={layer['activation_scorer']['split_half_log_score_pearson']:.6g}")
    for name, metric in layer["qonly_metrics"].items():
        print(f"QWEN38_QONLY variant={name} rel_l2={metric['relative_l2']:.9g} cosine={metric['cosine']:.9g} rmse={metric['rmse']:.9g} max_abs={metric['max_abs']:.9g}")
    for pruning in layer["pruning_results"]:
        refinement = pruning["reconstruction_refinement"]
        print(f"QWEN38_REFINE actual={pruning['actual_prune_fraction']:.6f} swap={refinement['chosen_swap_channels']} baseline_cal_rel_l2={refinement['baseline_calibration_metrics']['relative_l2']:.9g} chosen_cal_rel_l2={refinement['chosen_calibration_metrics']['relative_l2']:.9g}")
        for scorer_name, scorer in pruning["scorers"].items():
            metric = scorer["prune_only_bf16"]
            cal = scorer["calibration_prune_only_bf16"]
            print(f"QWEN38_PRUNE scorer={scorer_name} requested={pruning['requested_prune_fraction']:.4f} actual={pruning['actual_prune_fraction']:.6f} channels={pruning['pruned_channels']} cal_rel_l2={cal['relative_l2']:.9g} rel_l2={metric['relative_l2']:.9g} cosine={metric['cosine']:.9g} rmse={metric['rmse']:.9g} max_abs={metric['max_abs']:.9g}")
            for variant, qmetric in scorer["quantized"].items():
                print(f"QWEN38_COMBO scorer={scorer_name} variant={variant} actual={pruning['actual_prune_fraction']:.6f} rel_l2={qmetric['relative_l2']:.9g} cosine={qmetric['cosine']:.9g} rmse={qmetric['rmse']:.9g} max_abs={qmetric['max_abs']:.9g}")


if __name__ == "__main__":
    main()
