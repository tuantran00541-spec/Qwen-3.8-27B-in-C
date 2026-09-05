#!/usr/bin/env python3
"""Measure K3 sub-layer readiness frontiers without changing runtime semantics.

The current GGUF K3 packer writes tensors in lexicographic tensor-name order.
This probe reconstructs that exact in-layer layout from the pinned GGUF and
compares it with a hypothetical execution-order layout.  No tensor bytes are
rewritten and no inference arithmetic is touched.

The useful quantity is the page-aligned prefix that must have completed before
a compute stage can safely consume all tensors it needs.  Smaller frontiers are
potential I/O/compute overlap headroom; they are not claimed speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from gguf_k3_layout import partition_tensors
from gguf_stream import parse_gguf
from k3_stream import ALIGN, TENSOR_ALIGN, align_up

MODEL_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
N_LAYER = 64
SCHEMA = "qwen38-k3-sub-layer-frontier-v1"

GDN_ORDER = (
    "attn_norm.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ssm_dt.bias",
    "ssm_a",
    "ssm_conv1d.weight",
    "ssm_norm.weight",
    "ssm_out.weight",
    "post_attention_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)
GDN_STAGES = (
    ("attn_norm", ("attn_norm.weight",)),
    ("qkv", ("attn_qkv.weight",)),
    ("input_projections", ("attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight")),
    ("state_inputs", ("ssm_dt.bias", "ssm_a", "ssm_conv1d.weight")),
    ("state_output_norm", ("ssm_norm.weight",)),
    ("ssm_output", ("ssm_out.weight",)),
    ("post_attention_norm", ("post_attention_norm.weight",)),
    ("ffn_gate_up", ("ffn_gate.weight", "ffn_up.weight")),
    ("ffn_down", ("ffn_down.weight",)),
)

ATTN_ORDER = (
    "attn_norm.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)
ATTN_STAGES = (
    ("attn_norm", ("attn_norm.weight",)),
    ("q_projection", ("attn_q.weight",)),
    ("q_norm", ("attn_q_norm.weight",)),
    ("kv_projections", ("attn_k.weight", "attn_v.weight")),
    ("k_norm", ("attn_k_norm.weight",)),
    ("attn_output", ("attn_output.weight",)),
    ("post_attention_norm", ("post_attention_norm.weight",)),
    ("ffn_gate_up", ("ffn_gate.weight", "ffn_up.weight")),
    ("ffn_down", ("ffn_down.weight",)),
)


def _suffix(layer: int, name: str) -> str:
    prefix = f"blk.{layer}."
    if not name.startswith(prefix):
        raise ValueError(f"layer {layer}: unexpected tensor name {name}")
    return name[len(prefix):]


def _layout(sizes: dict[str, int], order: Iterable[str]) -> tuple[dict[str, dict[str, int]], int]:
    pos = 0
    out: dict[str, dict[str, int]] = {}
    for suffix in order:
        if suffix not in sizes:
            raise KeyError(suffix)
        start = align_up(pos, TENSOR_ALIGN)
        end = start + int(sizes[suffix])
        out[suffix] = {"offset": start, "nbytes": int(sizes[suffix]), "end": end}
        pos = end
    return out, align_up(pos, ALIGN)


def _frontiers(layout: dict[str, dict[str, int]], stages, read_bytes: int) -> list[dict[str, Any]]:
    frontier = 0
    consumed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for stage_name, needed in stages:
        for suffix in needed:
            consumed.add(suffix)
            frontier = max(frontier, int(layout[suffix]["end"]))
        ready = min(int(read_bytes), align_up(frontier, ALIGN))
        rows.append({
            "stage": stage_name,
            "new_tensors": list(needed),
            "cumulative_tensors": len(consumed),
            "ready_prefix_bytes": ready,
            "ready_fraction": ready / float(read_bytes),
            "bytes_remaining_after_ready": int(read_bytes) - ready,
        })
    return rows


def _layer_result(layer: int, tensors) -> dict[str, Any]:
    kind = "attention" if layer % 4 == 3 else "gdn"
    execution_order = ATTN_ORDER if kind == "attention" else GDN_ORDER
    stages = ATTN_STAGES if kind == "attention" else GDN_STAGES
    sizes = {_suffix(layer, t.name): int(t.nbytes) for t in tensors}
    expected = set(execution_order)
    actual = set(sizes)
    if actual != expected:
        raise RuntimeError(
            f"layer {layer} ({kind}) tensor contract mismatch: "
            f"missing={sorted(expected-actual)} unexpected={sorted(actual-expected)}"
        )

    current_order = tuple(sorted(sizes))
    current, current_read = _layout(sizes, current_order)
    proposed, proposed_read = _layout(sizes, execution_order)
    if sum(sizes.values()) != sum(x["nbytes"] for x in current.values()):
        raise AssertionError("tensor byte accounting mismatch")

    current_frontiers = _frontiers(current, stages, current_read)
    proposed_frontiers = _frontiers(proposed, stages, proposed_read)
    stage_rows = []
    for cur, prop in zip(current_frontiers, proposed_frontiers):
        if cur["stage"] != prop["stage"]:
            raise AssertionError("stage alignment mismatch")
        stage_rows.append({
            "stage": cur["stage"],
            "current_ready_prefix_bytes": cur["ready_prefix_bytes"],
            "proposed_ready_prefix_bytes": prop["ready_prefix_bytes"],
            "current_ready_fraction": cur["ready_fraction"],
            "proposed_ready_fraction": prop["ready_fraction"],
            "frontier_bytes_saved": cur["ready_prefix_bytes"] - prop["ready_prefix_bytes"],
            "current_bytes_remaining_after_ready": cur["bytes_remaining_after_ready"],
            "proposed_bytes_remaining_after_ready": prop["bytes_remaining_after_ready"],
        })

    first_heavy_stage = "q_projection" if kind == "attention" else "qkv"
    first = next(x for x in stage_rows if x["stage"] == first_heavy_stage)
    ffn = next(x for x in stage_rows if x["stage"] == "ffn_gate_up")
    return {
        "layer": layer,
        "kind": kind,
        "tensor_count": len(sizes),
        "tensor_bytes": sum(sizes.values()),
        "current_read_bytes": current_read,
        "proposed_read_bytes": proposed_read,
        "read_bytes_delta": proposed_read - current_read,
        "current_order": list(current_order),
        "execution_order": list(execution_order),
        "first_heavy_stage": first_heavy_stage,
        "first_heavy_current_prefix_bytes": first["current_ready_prefix_bytes"],
        "first_heavy_proposed_prefix_bytes": first["proposed_ready_prefix_bytes"],
        "first_heavy_frontier_bytes_saved": first["frontier_bytes_saved"],
        "ffn_gate_up_current_prefix_bytes": ffn["current_ready_prefix_bytes"],
        "ffn_gate_up_proposed_prefix_bytes": ffn["proposed_ready_prefix_bytes"],
        "ffn_gate_up_frontier_bytes_saved": ffn["frontier_bytes_saved"],
        "stages": stage_rows,
    }


def _aggregate(layers: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {
        "gdn": [x for x in layers if x["kind"] == "gdn"],
        "attention": [x for x in layers if x["kind"] == "attention"],
    }
    out: dict[str, Any] = {}
    for kind, rows in by_kind.items():
        total_read = sum(int(x["current_read_bytes"]) for x in rows)
        first_cur = sum(int(x["first_heavy_current_prefix_bytes"]) for x in rows)
        first_new = sum(int(x["first_heavy_proposed_prefix_bytes"]) for x in rows)
        ffn_cur = sum(int(x["ffn_gate_up_current_prefix_bytes"]) for x in rows)
        ffn_new = sum(int(x["ffn_gate_up_proposed_prefix_bytes"]) for x in rows)
        out[kind] = {
            "layers": len(rows),
            "current_read_bytes": total_read,
            "first_heavy_current_prefix_bytes": first_cur,
            "first_heavy_proposed_prefix_bytes": first_new,
            "first_heavy_frontier_bytes_saved": first_cur - first_new,
            "first_heavy_current_fraction": first_cur / float(total_read),
            "first_heavy_proposed_fraction": first_new / float(total_read),
            "ffn_gate_up_current_prefix_bytes": ffn_cur,
            "ffn_gate_up_proposed_prefix_bytes": ffn_new,
            "ffn_gate_up_frontier_bytes_saved": ffn_cur - ffn_new,
            "ffn_gate_up_current_fraction": ffn_cur / float(total_read),
            "ffn_gate_up_proposed_fraction": ffn_new / float(total_read),
            "read_bytes_delta": sum(int(x["read_bytes_delta"]) for x in rows),
        }
    return out


def sanity() -> None:
    sizes = {name: (i + 1) * 4096 for i, name in enumerate(GDN_ORDER)}
    cur, cur_read = _layout(sizes, sorted(sizes))
    prop, prop_read = _layout(sizes, GDN_ORDER)
    cf = _frontiers(cur, GDN_STAGES, cur_read)
    pf = _frontiers(prop, GDN_STAGES, prop_read)
    if len(cf) != len(GDN_STAGES) or len(pf) != len(GDN_STAGES):
        raise AssertionError("frontier stage count mismatch")
    if cf[-1]["ready_prefix_bytes"] != cur_read or pf[-1]["ready_prefix_bytes"] != prop_read:
        raise AssertionError("final frontier must cover full layer")
    print("QWEN38_K3_SUB_LAYER_FRONTIER_SANITY PASS")


def real(model: Path, output: Path) -> dict[str, Any]:
    directory = parse_gguf(model)
    grouped, auxiliary, _globals = partition_tensors(directory, expected_layers=N_LAYER)
    if 64 not in auxiliary:
        raise RuntimeError("expected blk.64 auxiliary MTP block is missing")
    layers = [_layer_result(il, grouped[il]) for il in range(N_LAYER)]
    total_current = sum(int(x["current_read_bytes"]) for x in layers)
    total_proposed = sum(int(x["proposed_read_bytes"]) for x in layers)
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "model_sha256_expected": MODEL_SHA256,
        "decoder_layers": N_LAYER,
        "gdn_layers": sum(x["kind"] == "gdn" for x in layers),
        "attention_layers": sum(x["kind"] == "attention" for x in layers),
        "current_total_read_bytes": total_current,
        "proposed_total_read_bytes": total_proposed,
        "total_read_bytes_delta": total_proposed - total_current,
        "alignment": ALIGN,
        "tensor_alignment": TENSOR_ALIGN,
        "aggregate": _aggregate(layers),
        "layers": layers,
        "interpretation": {
            "frontier_bytes_are_measured_layout_bytes_not_time": True,
            "speedup_claimed": False,
            "runtime_modified": False,
            "ring_residency_modified": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "current_total_read_bytes": total_current,
        "proposed_total_read_bytes": total_proposed,
        "total_read_bytes_delta": result["total_read_bytes_delta"],
        "aggregate": result["aggregate"],
    }, indent=2, sort_keys=True))
    print("QWEN38_K3_SUB_LAYER_FRONTIER_REAL_PASS")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    rp = sub.add_parser("real")
    rp.add_argument("--model", type=Path, required=True)
    rp.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        real(args.model, args.output)


if __name__ == "__main__":
    main()
