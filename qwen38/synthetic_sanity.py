#!/usr/bin/env python3
from __future__ import annotations

import torch

from core import (
    activation_energy_channel_scores,
    activation_energy_scores_from_hidden,
    activation_guided_reconstruction_refine,
    aligned_prune_count,
    mlp_forward,
    norm_channel_scores,
    output_metrics,
    qdq_rows,
    select_keep_indices,
    swiglu_hidden,
    tensor_name,
    validate_official_metadata,
)


def test_activation_score_beats_norm_on_functionally_dead_heavy_channel() -> None:
    # Hidden=4, intermediate=4. Channel 0 has huge weights but its gate is
    # effectively shut for this positive calibration/evaluation distribution.
    gate = torch.tensor([
        [-20.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
    ])
    up = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    down = torch.zeros((4, 4))
    down[0] = torch.tensor([10.0, 1.0, 1.0, 1.0])

    calibration = torch.tensor([
        [0.8, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0, 0.0],
    ])
    evaluation = torch.tensor([
        [0.9, 0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0, 0.0],
        [1.3, 0.0, 0.0, 0.0],
    ])

    old = norm_channel_scores(gate, up, down, chunk_size=2)
    new, _ = activation_energy_channel_scores(calibration, gate, up, down, chunk_size=2)
    old_keep = select_keep_indices(old, prune_channels=1, alignment=1)
    new_keep = select_keep_indices(new, prune_channels=1, alignment=1)

    assert 0 in old_keep.tolist(), old_keep.tolist()
    assert 0 not in new_keep.tolist(), new_keep.tolist()

    ref = mlp_forward(evaluation, gate, up, down)
    old_out = mlp_forward(evaluation, gate, up, down, keep=old_keep)
    new_out = mlp_forward(evaluation, gate, up, down, keep=new_keep)
    old_err = output_metrics(ref, old_out)["relative_l2"]
    new_err = output_metrics(ref, new_out)["relative_l2"]
    assert new_err < old_err * 0.05, (old_err, new_err)

    hidden = swiglu_hidden(calibration, gate, up, chunk_size=2)
    activation_from_hidden, _ = activation_energy_scores_from_hidden(hidden, down, chunk_size=2)
    assert torch.allclose(new, activation_from_hidden)
    calibration_ref = hidden @ down.float().T
    refined_keep, diagnostics = activation_guided_reconstruction_refine(
        calibration_ref,
        hidden,
        down,
        old_keep,
        activation_from_hidden,
        swap_sizes=(1,),
    )
    assert 0 not in refined_keep.tolist(), refined_keep.tolist()
    assert diagnostics["chosen_swap_channels"] == 1
    refined_err = output_metrics(
        ref, mlp_forward(evaluation, gate, up, down, keep=refined_keep)
    )["relative_l2"]
    assert refined_err < old_err * 0.05, (old_err, refined_err)


def test_qdq_and_alignment_helpers() -> None:
    w = torch.tensor([[0.0, -1.0, 0.5, 1.0], [0.2, 0.4, -0.3, -0.1]])
    q8 = qdq_rows(w, bits=8, group_size=2)
    q4 = qdq_rows(w, bits=4, group_size=2)
    assert q8.shape == w.shape == q4.shape
    assert torch.mean((q8 - w) ** 2) <= torch.mean((q4 - w) ** 2) + 1e-12
    assert aligned_prune_count(17408, 0.05, 128) == 896
    assert (17408 - aligned_prune_count(17408, 0.10, 128)) % 128 == 0
    assert (17408 - aligned_prune_count(17408, 0.15, 128)) % 128 == 0


def test_pinned_metadata_and_tensor_names() -> None:
    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "layer_types": layer_types,
        },
    }
    weight_map = {}
    for layer in range(64):
        for part in ("gate_proj", "up_proj", "down_proj"):
            weight_map[tensor_name(layer, part)] = f"model-{layer // 4:05d}.safetensors"
    result = validate_official_metadata(config, {"weight_map": weight_map})
    assert result["hidden_size"] == 5120
    assert result["intermediate_size"] == 17408
    assert tensor_name(63, "down_proj").endswith("layers.63.mlp.down_proj.weight")


def main() -> None:
    test_activation_score_beats_norm_on_functionally_dead_heavy_channel()
    test_qdq_and_alignment_helpers()
    test_pinned_metadata_and_tensor_names()
    print("QWEN38_SYNTHETIC_SANITY_PASS")


if __name__ == "__main__":
    main()
