from __future__ import annotations

import math

import torch

MODEL_ID = "Qwen/Qwen3.8-27B"
PINNED_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PREFIX = "model.language_model.layers"
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
MLP_PARTS = ("gate_proj", "up_proj", "down_proj")


def validate_official_metadata(config: dict, index: dict) -> dict:
    errors: list[str] = []
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        errors.append(f"architectures={config.get('architectures')!r}")
    if config.get("model_type") != "qwen3_5":
        errors.append(f"model_type={config.get('model_type')!r}")
    tc = config.get("text_config") or {}
    expected = {
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "full_attention_interval": 4,
    }
    for key, wanted in expected.items():
        if tc.get(key) != wanted:
            errors.append(f"text_config.{key}={tc.get(key)!r} expected={wanted!r}")
    layer_types = tc.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != LAYERS:
        errors.append("text_config.layer_types must contain 64 entries")
    else:
        for layer, kind in enumerate(layer_types):
            wanted = "full_attention" if layer % 4 == 3 else "linear_attention"
            if kind != wanted:
                errors.append(f"layer_types[{layer}]={kind!r} expected={wanted!r}")
                break

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("model.safetensors.index.json has no weight_map")
    else:
        for layer in range(LAYERS):
            for part in MLP_PARTS:
                name = f"{PREFIX}.{layer}.mlp.{part}.weight"
                if name not in weight_map:
                    errors.append(f"missing tensor {name}")
                    break
            if errors:
                break

    if errors:
        raise ValueError("official Qwen3.8-27B metadata validation failed: " + "; ".join(errors))
    return {
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "weight_map_entries": len(weight_map),
        "source_shards": len(set(weight_map.values())),
    }


def tensor_name(layer: int, part: str) -> str:
    if layer < 0 or layer >= LAYERS:
        raise ValueError(f"layer must be in [0,{LAYERS - 1}]")
    if part not in MLP_PARTS:
        raise ValueError(f"unsupported MLP part: {part}")
    return f"{PREFIX}.{layer}.mlp.{part}.weight"


def qdq_rows(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Symmetric RTN fake-quantization, grouping along each row's input dimension."""
    if w.ndim != 2:
        raise ValueError("w must be rank-2")
    if not 2 <= bits <= 8:
        raise ValueError("bits must be in [2,8]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, cols = w.shape
    pad = (-cols) % group_size
    w2 = torch.nn.functional.pad(w, (0, pad)) if pad else w
    blocks = w2.reshape(rows, -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    amax = blocks.abs().amax(dim=2, keepdim=True)
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.round(blocks / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(rows, -1)[:, :cols]


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    a = reference.reshape(-1).double()
    b = candidate.reshape(-1).double()
    err = b - a
    anorm = torch.linalg.vector_norm(a)
    enorm = torch.linalg.vector_norm(err)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom):
        cosine = float(torch.dot(a, b) / denom)
    else:
        cosine = 1.0 if torch.equal(a, b) else 0.0
    mse = torch.mean(err * err)
    return {
        "mse": float(mse),
        "rmse": float(torch.sqrt(mse)),
        "max_abs": float(err.abs().max()),
        "relative_l2": float(enorm / anorm) if float(anorm) else float(enorm),
        "cosine": cosine,
    }


def norm_channel_scores(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Legacy baseline: combined L2 weight energy for each SwiGLU intermediate channel."""
    _validate_mlp_shapes(gate, up, down)
    score = torch.zeros(gate.shape[0], dtype=torch.float64)
    for w in (gate, up):
        for start in range(0, w.shape[0], chunk_size):
            c = w[start:start + chunk_size].float().double()
            score[start:start + c.shape[0]] += (c * c).sum(dim=1)
    for start in range(0, down.shape[0], chunk_size):
        c = down[start:start + chunk_size].float().double()
        score += (c * c).sum(dim=0)
    return score


def swiglu_hidden(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    if gate.ndim != 2 or up.ndim != 2 or gate.shape != up.shape:
        raise ValueError("gate/up weights must be rank-2 with matching shapes")
    if x.ndim != 2 or x.shape[1] != gate.shape[1]:
        raise ValueError("x shape must be [samples, hidden]")
    g = linear_rows(x, gate, chunk_size=chunk_size)
    u = linear_rows(x, up, chunk_size=chunk_size)
    return torch.nn.functional.silu(g) * u


def activation_energy_scores_from_hidden(
    hidden: torch.Tensor,
    down: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, dict]:
    if hidden.ndim != 2 or down.ndim != 2 or hidden.shape[1] != down.shape[1]:
        raise ValueError("hidden/down shapes are incompatible")
    if hidden.shape[0] == 0:
        raise ValueError("hidden must contain at least one sample")
    inter = hidden.shape[1]
    h64 = hidden.double()
    activation_energy = (h64 * h64).mean(dim=0)
    rms = torch.sqrt(activation_energy)
    mean_abs = h64.abs().mean(dim=0)
    down_energy = torch.zeros(inter, dtype=torch.float64)
    for start in range(0, down.shape[0], chunk_size):
        c = down[start:start + chunk_size].float().double()
        down_energy += (c * c).sum(dim=0)
    score = activation_energy * down_energy
    return score, {
        "post_gate_rms_mean": float(rms.mean()),
        "post_gate_rms_median": float(rms.median()),
        "post_gate_mean_abs_mean": float(mean_abs.mean()),
        "score_min": float(score.min()),
        "score_median": float(score.median()),
        "score_max": float(score.max()),
    }


def activation_energy_channel_scores(
    calibration_x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, dict]:
    """Estimate expected squared output contribution per intermediate channel.

    score_j = E[(SiLU(gate_j(x)) * up_j(x))^2] * ||down[:,j]||_2^2
    Cross-channel cancellation is ignored by this ranking signal.
    """
    _validate_mlp_shapes(gate, up, down)
    hidden = swiglu_hidden(calibration_x, gate, up, chunk_size=chunk_size)
    return activation_energy_scores_from_hidden(hidden, down, chunk_size=chunk_size)


def keep_to_pruned(keep: torch.Tensor, intermediate: int) -> torch.Tensor:
    if keep.ndim != 1:
        raise ValueError("keep must be rank-1")
    mask = torch.ones(intermediate, dtype=torch.bool)
    mask[keep.long()] = False
    return mask.nonzero(as_tuple=False).flatten()


def pruned_to_keep(pruned: torch.Tensor, intermediate: int) -> torch.Tensor:
    if pruned.ndim != 1:
        raise ValueError("pruned must be rank-1")
    mask = torch.ones(intermediate, dtype=torch.bool)
    mask[pruned.long()] = False
    return mask.nonzero(as_tuple=False).flatten()


def candidate_output_from_pruned(
    full_output: torch.Tensor,
    hidden: torch.Tensor,
    down: torch.Tensor,
    pruned: torch.Tensor,
) -> torch.Tensor:
    if hidden.ndim != 2 or down.ndim != 2 or hidden.shape[1] != down.shape[1]:
        raise ValueError("hidden/down shapes are incompatible")
    if pruned.numel() == 0:
        return full_output.clone()
    hp = hidden.index_select(1, pruned.long()).float()
    dp = down.index_select(1, pruned.long()).float()
    removed = hp @ dp.T
    return full_output.float() - removed


def activation_guided_reconstruction_refine(
    full_output: torch.Tensor,
    hidden: torch.Tensor,
    down: torch.Tensor,
    baseline_keep: torch.Tensor,
    activation_scores: torch.Tensor,
    swap_sizes: tuple[int, ...] = (32, 64, 128, 256, 512),
) -> tuple[torch.Tensor, dict]:
    """Refine norm pruning with activation-guided swaps chosen by calibration reconstruction.

    The baseline set is always a candidate, so the selected set cannot be worse
    than baseline on the calibration reconstruction metric used for selection.
    Held-out evaluation is still required to detect overfitting.
    """
    inter = activation_scores.numel()
    if hidden.shape[1] != inter or down.shape[1] != inter:
        raise ValueError("activation score width mismatch")
    baseline_pruned = keep_to_pruned(baseline_keep, inter)
    if baseline_pruned.numel() == 0:
        raise ValueError("baseline must prune at least one channel")

    pruned_scores = activation_scores.index_select(0, baseline_pruned)
    rescue_order = baseline_pruned.index_select(0, torch.argsort(pruned_scores, descending=True))
    kept_scores = activation_scores.index_select(0, baseline_keep)
    replacement_order = baseline_keep.index_select(0, torch.argsort(kept_scores, descending=False))

    candidates: list[tuple[float, int, torch.Tensor, dict]] = []

    def add_candidate(swap: int, pruned: torch.Tensor) -> None:
        pruned = pruned.sort().values
        candidate = candidate_output_from_pruned(full_output, hidden, down, pruned)
        metric = output_metrics(full_output, candidate)
        candidates.append((metric["relative_l2"], swap, pruned, metric))

    add_candidate(0, baseline_pruned)
    for swap in swap_sizes:
        if swap <= 0 or swap > baseline_pruned.numel() or swap > baseline_keep.numel():
            continue
        proposed = torch.cat((rescue_order[swap:], replacement_order[:swap]))
        add_candidate(int(swap), proposed)

    best = min(candidates, key=lambda item: (item[0], item[1]))
    best_keep = pruned_to_keep(best[2], inter)
    diagnostics = {
        "chosen_swap_channels": best[1],
        "baseline_calibration_metrics": candidates[0][3],
        "chosen_calibration_metrics": best[3],
        "candidates": [
            {"swap_channels": swap, "calibration_metrics": metric}
            for _, swap, _, metric in candidates
        ],
    }
    return best_keep, diagnostics


def select_keep_indices(scores: torch.Tensor, prune_channels: int, alignment: int = 128) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    inter = scores.numel()
    if prune_channels <= 0 or prune_channels >= inter:
        raise ValueError("prune_channels must be in (0, intermediate)")
    keep_n = inter - prune_channels
    if alignment > 1 and keep_n % alignment:
        raise ValueError(f"kept intermediate {keep_n} is not aligned to {alignment}")
    keep = torch.topk(scores, k=keep_n, largest=True, sorted=False).indices
    return keep.sort().values


def linear_rows(
    x: torch.Tensor,
    w: torch.Tensor,
    row_idx: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
    chunk_size: int = 256,
) -> torch.Tensor:
    nout = w.shape[0] if row_idx is None else row_idx.numel()
    out = torch.empty((x.shape[0], nout), dtype=torch.float32)
    for start in range(0, nout, chunk_size):
        if row_idx is None:
            wc = w[start:start + chunk_size].float()
        else:
            idx = row_idx[start:start + chunk_size]
            wc = w.index_select(0, idx).float()
        if bits is not None:
            wc = qdq_rows(wc, bits, group_size)
        out[:, start:start + wc.shape[0]] = x.float() @ wc.T
    return out


def linear_down(
    x: torch.Tensor,
    down: torch.Tensor,
    keep: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
    chunk_size: int = 256,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], down.shape[0]), dtype=torch.float32)
    for start in range(0, down.shape[0], chunk_size):
        wc = down[start:start + chunk_size].float()
        if keep is not None:
            wc = wc.index_select(1, keep)
        if bits is not None:
            wc = qdq_rows(wc, bits, group_size)
        out[:, start:start + wc.shape[0]] = x.float() @ wc.T
    return out


def mlp_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    keep: torch.Tensor | None = None,
    bits: int | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    g = linear_rows(x, gate, keep, bits, group_size)
    u = linear_rows(x, up, keep, bits, group_size)
    h = torch.nn.functional.silu(g) * u
    return linear_down(h, down, keep, bits, group_size)


def projected_qbytes(hidden: int, intermediate: int, bits: int, group_size: int) -> dict:
    if hidden <= 0 or intermediate <= 0:
        raise ValueError("dimensions must be positive")
    params = 3 * hidden * intermediate
    scales = 2 * intermediate * math.ceil(hidden / group_size) + hidden * math.ceil(intermediate / group_size)
    payload_bytes = (params * bits + 7) // 8
    fp16_scale_bytes = scales * 2
    return {
        "params": params,
        "payload_bytes": payload_bytes,
        "fp16_scale_bytes": fp16_scale_bytes,
        "projected_total_bytes": payload_bytes + fp16_scale_bytes,
    }


def aligned_prune_count(intermediate: int, fraction: float, alignment: int = 128) -> int:
    """Return a prune count that leaves the kept width aligned, nearest to target fraction."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0,1)")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    target_keep = intermediate * (1.0 - fraction)
    aligned_keep = int(round(target_keep / alignment) * alignment)
    aligned_keep = min(intermediate - alignment, max(alignment, aligned_keep))
    return intermediate - aligned_keep


def parse_csv_ints(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("expected at least one float")
    return values


def _validate_mlp_shapes(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
    if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("MLP weights must be rank-2")
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} vs {tuple(up.shape)}")
    if down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError(f"down shape {tuple(down.shape)} incompatible with gate {tuple(gate.shape)}")
