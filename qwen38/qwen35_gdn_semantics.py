#!/usr/bin/env python3
"""Qwen3.5 Gated DeltaNet head-order/reference helpers.

This is deliberately zero-model code.  It captures the semantic transform used
by llama.cpp's Qwen3.5 GGUF converter when linear-attention V heads outnumber K
heads, plus a tiny one-token recurrence oracle.  Real-weight gates can import
these helpers without re-deriving the grouped-vs-tiled head mapping.
"""
from __future__ import annotations

import json
import math
import random
from typing import Sequence

LLAMA_CPP_REFERENCE_REVISION = "557614e0296ff4a5b6f649737a65ae2076eea2fd"
QWEN38_KEY_HEADS = 16
QWEN38_VALUE_HEADS = 48
QWEN38_HEAD_DIM = 128
QWEN38_CONV_KERNEL = 4


def head_permutation(num_k_heads: int, num_v_heads: int) -> list[int]:
    """New tiled V-head order expressed as indices into HF grouped order."""
    if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be a positive multiple of num_k_heads")
    per_k = num_v_heads // num_k_heads
    return [k * per_k + r for r in range(per_k) for k in range(num_k_heads)]


def expanded_head_permutation(num_k_heads: int, num_v_heads: int, head_dim: int) -> list[int]:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    return [h * head_dim + d for h in head_permutation(num_k_heads, num_v_heads) for d in range(head_dim)]


def reorder_vector_heads(values: Sequence[float], num_k_heads: int, num_v_heads: int, head_dim: int) -> list[float]:
    perm = expanded_head_permutation(num_k_heads, num_v_heads, head_dim)
    if len(values) != len(perm):
        raise ValueError("vector width does not match V-head layout")
    return [float(values[i]) for i in perm]


def reorder_matrix_rows(matrix: Sequence[Sequence[float]], num_k_heads: int, num_v_heads: int, head_dim: int) -> list[list[float]]:
    perm = expanded_head_permutation(num_k_heads, num_v_heads, head_dim)
    if len(matrix) != len(perm):
        raise ValueError("matrix row count does not match V-head layout")
    return [[float(v) for v in matrix[i]] for i in perm]


def reorder_matrix_columns(matrix: Sequence[Sequence[float]], num_k_heads: int, num_v_heads: int, head_dim: int) -> list[list[float]]:
    perm = expanded_head_permutation(num_k_heads, num_v_heads, head_dim)
    out: list[list[float]] = []
    for row in matrix:
        if len(row) != len(perm):
            raise ValueError("matrix column count does not match V-head layout")
        out.append([float(row[i]) for i in perm])
    return out


def repeat_k_heads_grouped(heads: Sequence[Sequence[float]], repeats: int) -> list[list[float]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return [[float(v) for v in head] for head in heads for _ in range(repeats)]


def repeat_k_heads_tiled(heads: Sequence[Sequence[float]], repeats: int) -> list[list[float]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return [[float(v) for v in head] for _ in range(repeats) for head in heads]


def split_heads(values: Sequence[float], heads: int, head_dim: int) -> list[list[float]]:
    if len(values) != heads * head_dim:
        raise ValueError("head reshape mismatch")
    return [[float(v) for v in values[h * head_dim:(h + 1) * head_dim]] for h in range(heads)]


def flatten(heads: Sequence[Sequence[float]]) -> list[float]:
    return [float(v) for head in heads for v in head]


def linear(weight: Sequence[Sequence[float]], x: Sequence[float]) -> list[float]:
    return [math.fsum(float(a) * float(b) for a, b in zip(row, x)) for row in weight]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def silu(x: float) -> float:
    return x * sigmoid(x)


def l2norm(x: Sequence[float], eps: float = 1e-6) -> list[float]:
    denom = math.sqrt(math.fsum(float(v) * float(v) for v in x) + eps)
    return [float(v) / denom for v in x]


def depthwise_conv_step(
    current: Sequence[float],
    history: Sequence[Sequence[float]],
    kernels: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    """One causal depthwise-conv step; history is oldest -> newest."""
    channels = len(current)
    if len(kernels) != channels or not kernels:
        raise ValueError("kernel channel mismatch")
    kernel = len(kernels[0])
    if kernel <= 0 or any(len(row) != kernel for row in kernels):
        raise ValueError("invalid depthwise kernels")
    if len(history) != kernel - 1 or any(len(row) != channels for row in history):
        raise ValueError("history shape mismatch")
    window = [list(map(float, row)) for row in history] + [list(map(float, current))]
    out = [
        silu(math.fsum(float(kernels[c][t]) * window[t][c] for t in range(kernel)))
        for c in range(channels)
    ]
    return out, window[1:]


def recurrent_step(
    q_heads: Sequence[Sequence[float]],
    k_heads: Sequence[Sequence[float]],
    v_heads: Sequence[Sequence[float]],
    log_decay: Sequence[float],
    beta: Sequence[float],
    state: Sequence[Sequence[Sequence[float]]],
    *,
    eps: float = 1e-6,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """Single-token GDN recurrence matching HF/llama.cpp autoregressive math."""
    heads = len(v_heads)
    if not (len(q_heads) == len(k_heads) == len(log_decay) == len(beta) == len(state) == heads):
        raise ValueError("recurrent head count mismatch")
    if heads == 0:
        raise ValueError("at least one head is required")
    dim = len(v_heads[0])
    if dim <= 0:
        raise ValueError("head_dim must be positive")
    scale = 1.0 / math.sqrt(dim)
    outputs: list[list[float]] = []
    next_state: list[list[list[float]]] = []
    for h in range(heads):
        q = [v * scale for v in l2norm(q_heads[h], eps)]
        k = l2norm(k_heads[h], eps)
        v = [float(x) for x in v_heads[h]]
        if len(q) != dim or len(k) != dim or len(state[h]) != dim or any(len(row) != dim for row in state[h]):
            raise ValueError("recurrent state/head shape mismatch")
        decay = math.exp(float(log_decay[h]))
        s = [[float(state[h][i][j]) * decay for j in range(dim)] for i in range(dim)]
        predicted = [math.fsum(s[i][j] * k[i] for i in range(dim)) for j in range(dim)]
        delta = [(v[j] - predicted[j]) * float(beta[h]) for j in range(dim)]
        for i in range(dim):
            for j in range(dim):
                s[i][j] += k[i] * delta[j]
        out = [math.fsum(s[i][j] * q[i] for i in range(dim)) for j in range(dim)]
        outputs.append(out)
        next_state.append(s)
    return outputs, next_state


def gated_rms_heads(core: Sequence[Sequence[float]], z: Sequence[Sequence[float]], weight: Sequence[float], eps: float = 1e-6) -> list[list[float]]:
    if len(core) != len(z) or not core:
        raise ValueError("gated RMS head mismatch")
    dim = len(weight)
    out: list[list[float]] = []
    for c, gate in zip(core, z):
        if len(c) != dim or len(gate) != dim:
            raise ValueError("gated RMS width mismatch")
        rms = math.sqrt(math.fsum(v * v for v in c) / dim + eps)
        out.append([c[i] / rms * float(weight[i]) * silu(float(gate[i])) for i in range(dim)])
    return out


def _matrix(rows: int, cols: int, rng: random.Random, scale: float = 0.2) -> list[list[float]]:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _vector(n: int, rng: random.Random, scale: float = 0.2) -> list[float]:
    return [rng.uniform(-scale, scale) for _ in range(n)]


def _permute_head_state(state, perm):
    return [[[float(v) for v in row] for row in state[h]] for h in perm]


def _max_diff(a, b) -> float:
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((_max_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(float(a) - float(b))


def semantic_sanity(seed: int = 20260901) -> dict:
    """Prove grouped-HF and tiled-GGUF one-token GDN are permutation-equivalent."""
    rng = random.Random(seed)
    hidden = 7
    num_k = 2
    num_v = 6
    dim = 4
    repeats = num_v // num_k
    key_dim = num_k * dim
    value_dim = num_v * dim
    conv_dim = 2 * key_dim + value_dim
    kernel = 4
    perm_heads = head_permutation(num_k, num_v)
    perm_values = expanded_head_permutation(num_k, num_v, dim)

    x = _vector(hidden, rng)
    w_qkv = _matrix(2 * key_dim + value_dim, hidden, rng)
    w_z = _matrix(value_dim, hidden, rng)
    w_b = _matrix(num_v, hidden, rng)
    w_a = _matrix(num_v, hidden, rng)
    w_out = _matrix(hidden, value_dim, rng)
    conv_w = _matrix(conv_dim, kernel, rng)
    dt_bias = _vector(num_v, rng)
    a_log = _vector(num_v, rng, scale=0.4)
    norm_w = [1.0 + v for v in _vector(dim, rng, scale=0.1)]
    history = [_vector(conv_dim, rng) for _ in range(kernel - 1)]
    state = [[_vector(dim, rng, scale=0.05) for _ in range(dim)] for _ in range(num_v)]

    # HF/grouped path.
    qkv_hf = linear(w_qkv, x)
    z_hf = linear(w_z, x)
    b_hf = linear(w_b, x)
    a_hf = linear(w_a, x)
    conv_hf, next_hist_hf = depthwise_conv_step(qkv_hf, history, conv_w)
    q_hf = split_heads(conv_hf[:key_dim], num_k, dim)
    k_hf = split_heads(conv_hf[key_dim:2 * key_dim], num_k, dim)
    v_hf = split_heads(conv_hf[2 * key_dim:], num_v, dim)
    q_hf = repeat_k_heads_grouped(q_hf, repeats)
    k_hf = repeat_k_heads_grouped(k_hf, repeats)
    beta_hf = [sigmoid(v) for v in b_hf]
    decay_hf = [-math.exp(a_log[h]) * softplus(a_hf[h] + dt_bias[h]) for h in range(num_v)]
    core_hf, state_hf = recurrent_step(q_hf, k_hf, v_hf, decay_hf, beta_hf, state)
    gated_hf = gated_rms_heads(core_hf, split_heads(z_hf, num_v, dim), norm_w)
    out_hf = linear(w_out, flatten(gated_hf))

    # Converter-equivalent tiled weights/parameters.
    q_rows = w_qkv[:key_dim]
    k_rows = w_qkv[key_dim:2 * key_dim]
    v_rows = reorder_matrix_rows(w_qkv[2 * key_dim:], num_k, num_v, dim)
    w_qkv_t = q_rows + k_rows + v_rows
    w_z_t = reorder_matrix_rows(w_z, num_k, num_v, dim)
    w_b_t = reorder_matrix_rows(w_b, num_k, num_v, 1)
    w_a_t = reorder_matrix_rows(w_a, num_k, num_v, 1)
    w_out_t = reorder_matrix_columns(w_out, num_k, num_v, dim)
    dt_t = reorder_vector_heads(dt_bias, num_k, num_v, 1)
    alog_t = reorder_vector_heads(a_log, num_k, num_v, 1)
    conv_w_t = conv_w[:2 * key_dim] + reorder_matrix_rows(conv_w[2 * key_dim:], num_k, num_v, dim)
    history_t = [row[:2 * key_dim] + [row[2 * key_dim + i] for i in perm_values] for row in history]
    state_t = _permute_head_state(state, perm_heads)

    qkv_t = linear(w_qkv_t, x)
    z_t = linear(w_z_t, x)
    b_t = linear(w_b_t, x)
    a_t = linear(w_a_t, x)
    conv_t, next_hist_t = depthwise_conv_step(qkv_t, history_t, conv_w_t)
    q_t = repeat_k_heads_tiled(split_heads(conv_t[:key_dim], num_k, dim), repeats)
    k_t = repeat_k_heads_tiled(split_heads(conv_t[key_dim:2 * key_dim], num_k, dim), repeats)
    v_t = split_heads(conv_t[2 * key_dim:], num_v, dim)
    beta_t = [sigmoid(v) for v in b_t]
    decay_t = [-math.exp(alog_t[h]) * softplus(a_t[h] + dt_t[h]) for h in range(num_v)]
    core_t, state_next_t = recurrent_step(q_t, k_t, v_t, decay_t, beta_t, state_t)
    gated_t = gated_rms_heads(core_t, split_heads(z_t, num_v, dim), norm_w)
    out_t = linear(w_out_t, flatten(gated_t))

    expected_qkv_t = qkv_hf[:2 * key_dim] + [qkv_hf[2 * key_dim + i] for i in perm_values]
    expected_conv_t = conv_hf[:2 * key_dim] + [conv_hf[2 * key_dim + i] for i in perm_values]
    expected_hist_t = [row[:2 * key_dim] + [row[2 * key_dim + i] for i in perm_values] for row in next_hist_hf]
    expected_state_t = _permute_head_state(state_hf, perm_heads)
    expected_core_t = [core_hf[h] for h in perm_heads]

    diffs = {
        "qkv_projection": _max_diff(qkv_t, expected_qkv_t),
        "z_projection": _max_diff(z_t, reorder_vector_heads(z_hf, num_k, num_v, dim)),
        "beta_projection": _max_diff(b_t, reorder_vector_heads(b_hf, num_k, num_v, 1)),
        "alpha_projection": _max_diff(a_t, reorder_vector_heads(a_hf, num_k, num_v, 1)),
        "conv_output": _max_diff(conv_t, expected_conv_t),
        "conv_history": _max_diff(next_hist_t, expected_hist_t),
        "recurrent_core": _max_diff(core_t, expected_core_t),
        "recurrent_state": _max_diff(state_next_t, expected_state_t),
        "final_output": _max_diff(out_t, out_hf),
    }
    limit = 2e-12
    status = "PASS" if all(v <= limit for v in diffs.values()) else "FAIL"
    return {
        "schema": "qwen35-gdn-head-order-sanity-v1",
        "status": status,
        "llama_cpp_reference_revision": LLAMA_CPP_REFERENCE_REVISION,
        "model_weights_downloaded": False,
        "fixture": {
            "hidden": hidden,
            "num_k_heads": num_k,
            "num_v_heads": num_v,
            "value_heads_per_key": repeats,
            "head_dim": dim,
            "conv_kernel": kernel,
            "head_permutation": perm_heads,
        },
        "qwen38_layout": {
            "num_k_heads": QWEN38_KEY_HEADS,
            "num_v_heads": QWEN38_VALUE_HEADS,
            "value_heads_per_key": QWEN38_VALUE_HEADS // QWEN38_KEY_HEADS,
            "head_dim": QWEN38_HEAD_DIM,
            "conv_kernel": QWEN38_CONV_KERNEL,
            "state_f32_bytes_per_recurrent_layer": QWEN38_VALUE_HEADS * QWEN38_HEAD_DIM * QWEN38_HEAD_DIM * 4,
        },
        "max_abs_diffs": diffs,
        "error_limit": limit,
        "converter_transforms_covered": [
            "in_proj_qkv_v_rows",
            "in_proj_z_rows",
            "in_proj_b_rows",
            "in_proj_a_rows",
            "A_log_heads",
            "dt_bias_heads",
            "conv1d_v_channels",
            "out_proj_columns",
            "recurrent_state_heads",
        ],
    }


def main() -> None:
    result = semantic_sanity()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
