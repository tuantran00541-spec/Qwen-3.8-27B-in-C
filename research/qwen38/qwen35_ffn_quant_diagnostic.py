#!/usr/bin/env python3
"""Focused Qwen3.8 layer-0 FFN diagnostic against pinned llama.cpp checkpoints.

This intentionally starts from llama.cpp's exact ``attn_post_norm-0`` vector so
any discrepancy is isolated from the already-proven Gated DeltaNet path.  It
splits the dense FFN into four comparisons:

1. Q6_K x Q8_K up projection,
2. Q6_K x Q8_K gate projection,
3. SwiGLU itself, including a function-only comparison fed with oracle inputs,
4. Q6_K x Q8_K down projection, including a kernel-only comparison fed with
   llama.cpp's exact ``ffn_swiglu-0`` vector.

No full matrix is dequantized and the existing one-slot K3 reader/native quant
bridge are reused unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time

from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk
from qwen35_gdn_quant_layer_gate import (
    DECODER_LAYERS,
    HIDDEN,
    INTERMEDIATE,
    MODEL_ID,
    REVISION,
    SHA256,
    QuantRuntime,
    _layer_meta,
    _load_native,
    atomic_json,
    metrics,
    silu,
)

ORACLE_SCHEMA = "qwen38-llama-layer0-oracle-v3"
SCHEMA = "qwen38-k3-layer0-ffn-diagnostic-v1"


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _require_vector(checkpoints: dict, name: str, size: int) -> list[float]:
    values = checkpoints.get(name)
    if not isinstance(values, list) or len(values) != size:
        got = None if not isinstance(values, list) else len(values)
        raise RuntimeError(f"oracle {name} has {got} elements, expected {size}")
    return [float(v) for v in values]


def diagnose(model: Path, native_lib: Path, inventory_json: Path, oracle_json: Path,
             work_dir: Path, output: Path) -> dict:
    started = time.monotonic()
    inventory = json.loads(inventory_json.read_text(encoding="utf-8"))
    if inventory.get("status") != "PASS" or inventory.get("sha256") != SHA256:
        raise RuntimeError("inventory contract is not a PASS for the pinned GGUF")

    oracle = json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema") != ORACLE_SCHEMA or not oracle.get("captured_complete_layer"):
        raise RuntimeError(f"expected complete {ORACLE_SCHEMA} oracle")
    ref = oracle.get("checkpoints", {})
    post_norm = _require_vector(ref, "attn_post_norm-0", HIDDEN)
    ref_up = _require_vector(ref, "ffn_up-0", INTERMEDIATE)
    ref_gate = _require_vector(ref, "ffn_gate-0", INTERMEDIATE)
    ref_swiglu = _require_vector(ref, "ffn_swiglu-0", INTERMEDIATE)
    ref_out = _require_vector(ref, "ffn_out-0", HIDDEN)

    directory = parse_gguf(model)
    work_dir.mkdir(parents=True, exist_ok=True)
    trunk = work_dir / "ffn-layer0.k3.bin"
    manifest_path = work_dir / "ffn-layer0.k3.json"
    manifest = pack_gguf_layers(
        directory,
        trunk,
        manifest_path,
        layers=[0],
        model_id=MODEL_ID,
        revision=REVISION,
        source_sha256=SHA256,
        expected_layers=DECODER_LAYERS,
    )
    layer_entry = manifest["layers"][0]
    budget = int(layer_entry["read_bytes"])
    runtime = QuantRuntime(_load_native(native_lib))

    with K3Trunk(
        trunk,
        manifest_path,
        budget_bytes=budget,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as reader:
        layer = reader.bind(0)
        metas = _layer_meta(manifest, 0)

        def view(name: str) -> memoryview:
            return reader.tensor_view(layer, name)

        # Gate and up share exactly one Q8_K activation, matching the production
        # candidate and llama.cpp's common input semantic boundary.
        prepared = runtime.quantize(post_norm, "Q6_K")
        cand_gate = runtime.matvec(
            view("blk.0.ffn_gate.weight"),
            metas["blk.0.ffn_gate.weight"],
            post_norm,
            prepared,
        )
        cand_up = runtime.matvec(
            view("blk.0.ffn_up.weight"),
            metas["blk.0.ffn_up.weight"],
            post_norm,
            prepared,
        )

        cand_swiglu = [silu(cand_gate[i]) * cand_up[i] for i in range(INTERMEDIATE)]
        # This removes both Q6 projections from the equation. Any remaining
        # mismatch here is specifically the scalar SiLU/SwiGLU implementation.
        oracle_input_swiglu = [silu(ref_gate[i]) * ref_up[i] for i in range(INTERMEDIATE)]

        # Full local FFN from exact oracle post-norm input.
        cand_out = runtime.matvec(
            view("blk.0.ffn_down.weight"),
            metas["blk.0.ffn_down.weight"],
            cand_swiglu,
        )
        # Down-kernel-only probe: feed llama.cpp's exact SwiGLU output through
        # the custom Q6_K x Q8_K down projection.
        down_from_oracle_swiglu = runtime.matvec(
            view("blk.0.ffn_down.weight"),
            metas["blk.0.ffn_down.weight"],
            ref_swiglu,
        )

        reader_report = reader.report()
        layer.release()

    comparisons = {
        "gate_q6_projection": metrics(ref_gate, cand_gate),
        "up_q6_projection": metrics(ref_up, cand_up),
        "swiglu_total_from_custom_projections": metrics(ref_swiglu, cand_swiglu),
        "swiglu_function_only_from_oracle_inputs": metrics(ref_swiglu, oracle_input_swiglu),
        "down_q6_kernel_only_from_oracle_swiglu": metrics(ref_out, down_from_oracle_swiglu),
        "full_ffn_from_oracle_post_norm": metrics(ref_out, cand_out),
    }
    ranking = sorted(
        (
            {"name": name, "relative_l2": float(value["relative_l2"]), "max_abs": float(value["max_abs"])}
            for name, value in comparisons.items()
        ),
        key=lambda item: item["relative_l2"],
        reverse=True,
    )

    result = {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETE",
        "model_sha256": SHA256,
        "oracle_schema": ORACLE_SCHEMA,
        "input_source": "llama.cpp attn_post_norm-0",
        "full_matrix_dequantized": False,
        "comparisons": comparisons,
        "largest_relative_l2_first": ranking,
        "candidate": {
            "activation_quantizations": runtime.activation_quantizations,
            "matvec_rows": runtime.matvec_rows,
            "reader_report": reader_report,
        },
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    }
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def sanity() -> None:
    gate = [-2.5, -0.5, 0.0, 0.5, 2.5]
    up = [0.25, -0.5, 1.0, 2.0, -3.0]
    out = [silu(g) * u for g, u in zip(gate, up)]
    if len(out) != len(gate) or not all(math.isfinite(v) for v in out):
        raise SystemExit("FFN diagnostic SwiGLU sanity failed")
    exact = metrics(out, list(out))
    if exact["max_abs"] != 0.0 or exact["relative_l2"] != 0.0:
        raise SystemExit("FFN diagnostic metrics sanity failed")
    print(json.dumps({
        "schema": "qwen38-k3-layer0-ffn-diagnostic-sanity-v1",
        "status": "PASS",
        "oracle_schema": ORACLE_SCHEMA,
        "diagnostic_splits": 6,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--oracle", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        diagnose(args.model, args.native_lib, args.inventory, args.oracle, args.work_dir, args.output)


if __name__ == "__main__":
    main()
