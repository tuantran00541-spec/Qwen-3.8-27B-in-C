#!/usr/bin/env python3
"""Real Qwen3.8 layer-0 multi-vector quantized matvec gate.

The purpose of this probe is deliberately narrow: validate the already-proven
synthetic ``matvec_many`` kernels on real Q6_K/Q8_0 tensors and real prompt
activations before changing the layer scheduler.

We use the 11-token essay prompt at layer 0.  Embeddings are independent before
layer 0, so both the Q8 ``attn_qkv`` projection and Q6 ``attn_gate`` projection
can be evaluated across all tokens without touching recurrent state.  The gate
pre-quantizes activations once, then A/Bs the current one-vector exact AVX2 ABI
against one weight traversal serving all 11 vectors.  Every output float must
match bit-for-bit.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
from pathlib import Path
import resource
import struct
import time
from typing import Any

import qwen35_gdn_quant_layer_gate as gdn
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

PROMPT_IDS = [7734, 264, 220, 22, 15, 15, 36093, 8627, 383, 38896, 13]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def load_many(path: Path):
    lib = ctypes.CDLL(str(path))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    fp = ctypes.POINTER(ctypes.c_float)
    args = [
        u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        u8p, ctypes.c_size_t, ctypes.c_size_t, fp,
    ]
    for name in (
        "qwen_matvec_many_q8_0_q8_0_bridge",
        "qwen_matvec_many_q6_k_q8_k_bridge",
    ):
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = ctypes.c_int
    return lib


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack("<%df" % len(values), *values)


def run_one(
    runtime: gdn.QuantRuntime,
    many_lib,
    weights: memoryview,
    meta: dict[str, Any],
    xs: list[list[float]],
    kind: str,
) -> dict[str, Any]:
    prepared = [runtime.quantize(x, kind) for x in xs]
    act_bytes = int(prepared[0][1])
    if any(int(n) != act_bytes for _, n in prepared):
        raise RuntimeError("activation byte sizes differ")
    blob = b"".join(bytes(buf) for buf, _ in prepared)
    acts = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
    ne0, rows = map(int, meta["shape"])
    nv = len(xs)
    w_arr = (ctypes.c_uint8 * len(weights)).from_buffer(weights)
    many_out = (ctypes.c_float * (nv * rows))()
    many_fn = (
        many_lib.qwen_matvec_many_q6_k_q8_k_bridge
        if kind == "Q6_K"
        else many_lib.qwen_matvec_many_q8_0_q8_0_bridge
    )

    def sequential_once() -> tuple[float, bytes]:
        t0 = time.monotonic()
        rows_out = [
            runtime.matvec(weights, meta, x, prepared=p)
            for x, p in zip(xs, prepared)
        ]
        elapsed = time.monotonic() - t0
        flat = [v for row in rows_out for v in row]
        return elapsed, f32_bytes(flat)

    def many_once() -> tuple[float, bytes]:
        t0 = time.monotonic()
        rc = many_fn(
            w_arr, len(weights), rows, ne0,
            acts, act_bytes, nv, many_out,
        )
        elapsed = time.monotonic() - t0
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: many-vector rc={rc}")
        raw = ctypes.string_at(ctypes.addressof(many_out), ctypes.sizeof(many_out))
        return elapsed, raw

    # ABBA reduces sensitivity to cache/runner drift while keeping the gate
    # cheap enough for a single real layer.  Exactness is checked on every B.
    seq1_s, ref1 = sequential_once()
    many1_s, cand1 = many_once()
    many2_s, cand2 = many_once()
    seq2_s, ref2 = sequential_once()
    if ref1 != cand1 or ref1 != cand2 or ref1 != ref2:
        raise RuntimeError(f"{meta['name']}: many-vector output is not bitwise exact")

    seq_med = statistics.median([seq1_s, seq2_s])
    many_med = statistics.median([many1_s, many2_s])
    return {
        "tensor": meta["name"],
        "kind": kind,
        "shape": [ne0, rows],
        "n_vec": nv,
        "activation_bytes_each": act_bytes,
        "weight_bytes": len(weights),
        "sequential_seconds_ab": [seq1_s, seq2_s],
        "many_seconds_bb": [many1_s, many2_s],
        "sequential_median_seconds": seq_med,
        "many_median_seconds": many_med,
        "speedup": seq_med / many_med,
        "output_bytes_compared": len(ref1),
        "bitwise_exact": True,
    }


def run(args) -> dict[str, Any]:
    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inv.get("status") != "PASS" or inv.get("sha256") != gdn.SHA256:
        raise RuntimeError("inventory is not a PASS for pinned GGUF")

    directory = parse_gguf(args.model)
    trunk = args.work_dir / "layer0.k3.bin"
    manifest_path = args.work_dir / "layer0.k3.json"
    manifest = pack_gguf_layers(
        directory, trunk, manifest_path, layers=[0],
        model_id=gdn.MODEL_ID, revision=gdn.REVISION,
        source_sha256=gdn.SHA256, expected_layers=gdn.DECODER_LAYERS,
    )
    budget = int(manifest["layers"][0]["read_bytes"])
    base_lib = gdn._load_native(args.many_lib)
    runtime = gdn.QuantRuntime(base_lib)
    many_lib = load_many(args.many_lib)

    embeddings = [gdn._embedding_row(args.model, directory, tid) for tid in PROMPT_IDS]
    started = time.monotonic()
    with K3Trunk(
        trunk, manifest_path, budget_bytes=budget, want_ring=1,
        max_pinned=0, prefer_direct_io=True,
    ) as reader:
        layer = reader.bind(0)
        try:
            metas = gdn._layer_meta(manifest, 0)

            def view(name: str) -> memoryview:
                return reader.tensor_view(layer, name)

            norm_w = gdn.f32_vector(view("blk.0.attn_norm.weight"))
            xs = [gdn.rms_norm(x, norm_w) for x in embeddings]

            q8 = run_one(
                runtime, many_lib,
                view("blk.0.attn_qkv.weight"),
                metas["blk.0.attn_qkv.weight"],
                xs, "Q8_0",
            )
            q6 = run_one(
                runtime, many_lib,
                view("blk.0.attn_gate.weight"),
                metas["blk.0.attn_gate.weight"],
                xs, "Q6_K",
            )
            reader_report = reader.report()
        finally:
            layer.release()

    if not bool(reader_report.get("direct_io")):
        raise RuntimeError("real matvec-many gate requires direct I/O")
    payload = {
        "schema": "qwen38-matvec-many-real-layer0-v1",
        "status": "PASS",
        "model_sha256": gdn.SHA256,
        "prompt_token_ids": PROMPT_IDS,
        "prompt_token_count": len(PROMPT_IDS),
        "q8_attn_qkv": q8,
        "q6_attn_gate": q6,
        "activation_quantizations": runtime.activation_quantizations,
        "reader": reader_report,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "q8_speedup": q8["speedup"],
        "q6_speedup": q6["speedup"],
        "q8_seq_s": q8["sequential_median_seconds"],
        "q8_many_s": q8["many_median_seconds"],
        "q6_seq_s": q6["sequential_median_seconds"],
        "q6_many_s": q6["many_median_seconds"],
        "max_rss_gib": payload["max_rss_gib"],
    }, indent=2))
    print("QWEN38_MATVEC_MANY_REAL_LAYER0_BITWISE_PASS")
    return payload


def sanity() -> None:
    assert len(PROMPT_IDS) == 11
    assert gdn.HIDDEN == 5120
    assert gdn.CONV_DIM == 10240
    assert gdn.VALUE_DIM == 6144
    print(json.dumps({
        "schema": "qwen38-matvec-many-real-layer0-sanity-v1",
        "status": "PASS",
        "scope": ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"],
        "n_vec": len(PROMPT_IDS),
        "recurrence": "not entered",
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--many-lib", type=Path, required=True)
    r.add_argument("--inventory", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        run(args)


if __name__ == "__main__":
    main()
