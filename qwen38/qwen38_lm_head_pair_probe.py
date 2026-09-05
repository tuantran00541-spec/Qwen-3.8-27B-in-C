#!/usr/bin/env python3
"""Exact two-vector global LM-head stream-reuse probe for Qwen3.8-27B.

The target decoder can now evaluate two consecutive positions with one K3
weight stream.  On an accepted speculative pair, however, the target still
needs logits for both hidden A (verify B) and hidden B (choose C).  Calling the
existing LM-head helper twice reads the 1.35 GB Q8_0 output matrix twice.

This isolated probe keeps each vector's arithmetic unchanged while reading each
LM-head chunk only once and applying it to A then B.  It requires bitwise-equal
full logits and exact 2:1 physical pread bytes versus two sequential streams.
The default generator remains untouched.
"""
from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_full64_one_token as base
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as reuse
from gguf_quant_ref import row_nbytes


PREFIX_TOKENS = [12675, 11]
PAIR_TOKENS = [353, 2688]
EXPECTED_TOP1 = [2688, 264]


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _f32_bytes(values: Sequence[float]) -> bytes:
    return array("f", (float(x) for x in values)).tobytes()


def _geometry(tensor) -> tuple[int, int, int]:
    if tensor.type_name != "Q8_0":
        raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
    ne0, rows = map(int, tensor.shape)
    if ne0 != gdn.HIDDEN or rows != base.VOCAB:
        raise ValueError(f"unexpected LM-head shape {list(tensor.shape)}")
    stride = row_nbytes("Q8_0", ne0)
    if stride * rows != int(tensor.nbytes):
        raise ValueError("LM-head Q8_0 byte geometry mismatch")
    return ne0, rows, stride


def _stream_one_counted(
    model: Path,
    tensor,
    runtime,
    hidden: Sequence[float],
    output_norm_w: Sequence[float],
    chunk_rows: int = base.LM_HEAD_CHUNK_ROWS,
) -> tuple[list[float], int]:
    ne0, rows, stride = _geometry(tensor)
    norm = gdn.rms_norm(hidden, output_norm_w)
    prepared = runtime.quantize(norm, "Q8_0")
    logits: list[float] = []
    bytes_read = 0
    fd = os.open(model, os.O_RDONLY)
    try:
        for row0 in range(0, rows, chunk_rows):
            nrows = min(chunk_rows, rows - row0)
            nbytes = nrows * stride
            raw = bytearray(os.pread(fd, nbytes, int(tensor.data_offset) + row0 * stride))
            if len(raw) != nbytes:
                raise EOFError(f"short LM-head read at row {row0}")
            bytes_read += len(raw)
            view = memoryview(raw)
            meta = {"name": tensor.name, "type_name": "Q8_0", "shape": [ne0, nrows]}
            logits.extend(runtime.matvec(view, meta, norm, prepared))
            view.release()
    finally:
        os.close(fd)
    return logits, bytes_read


def _stream_pair_counted(
    model: Path,
    tensor,
    runtime,
    hidden_a: Sequence[float],
    hidden_b: Sequence[float],
    output_norm_w: Sequence[float],
    chunk_rows: int = base.LM_HEAD_CHUNK_ROWS,
) -> tuple[list[float], list[float], int]:
    """Read every output-weight chunk once, preserving per-vector math order."""
    ne0, rows, stride = _geometry(tensor)
    norm_a = gdn.rms_norm(hidden_a, output_norm_w)
    norm_b = gdn.rms_norm(hidden_b, output_norm_w)
    prep_a = runtime.quantize(norm_a, "Q8_0")
    prep_b = runtime.quantize(norm_b, "Q8_0")
    logits_a: list[float] = []
    logits_b: list[float] = []
    bytes_read = 0
    fd = os.open(model, os.O_RDONLY)
    try:
        for row0 in range(0, rows, chunk_rows):
            nrows = min(chunk_rows, rows - row0)
            nbytes = nrows * stride
            raw = bytearray(os.pread(fd, nbytes, int(tensor.data_offset) + row0 * stride))
            if len(raw) != nbytes:
                raise EOFError(f"short LM-head pair read at row {row0}")
            bytes_read += len(raw)
            view = memoryview(raw)
            meta = {"name": tensor.name, "type_name": "Q8_0", "shape": [ne0, nrows]}
            logits_a.extend(runtime.matvec(view, meta, norm_a, prep_a))
            logits_b.extend(runtime.matvec(view, meta, norm_b, prep_b))
            view.release()
    finally:
        os.close(fd)
    return logits_a, logits_b, bytes_read


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.native_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    try:
        for token in PREFIX_TOKENS:
            engine.step(token)
        hidden_a, hidden_b = reuse.step_pair(engine, *PAIR_TOKENS)
        tensor = engine.tensors["output.weight"]
        lm_bytes = int(tensor.nbytes)
        if lm_bytes != 1_350_860_800:
            raise RuntimeError(f"unexpected global LM-head bytes: {lm_bytes}")

        t0 = time.monotonic()
        seq_a, seq_a_bytes = _stream_one_counted(
            args.model, tensor, engine.runtime, hidden_a, engine.output_norm_w)
        seq_b, seq_b_bytes = _stream_one_counted(
            args.model, tensor, engine.runtime, hidden_b, engine.output_norm_w)
        sequential_seconds = time.monotonic() - t0
        sequential_bytes = seq_a_bytes + seq_b_bytes

        t0 = time.monotonic()
        pair_a, pair_b, pair_bytes = _stream_pair_counted(
            args.model, tensor, engine.runtime, hidden_a, hidden_b, engine.output_norm_w)
        pair_seconds = time.monotonic() - t0

        a_exact = _f32_bytes(seq_a) == _f32_bytes(pair_a)
        b_exact = _f32_bytes(seq_b) == _f32_bytes(pair_b)
        seq_top5_a = base._topk(seq_a, 5)
        seq_top5_b = base._topk(seq_b, 5)
        pair_top5_a = base._topk(pair_a, 5)
        pair_top5_b = base._topk(pair_b, 5)
        top1 = [int(pair_top5_a[0]["token"]), int(pair_top5_b[0]["token"])]

        if not a_exact or not b_exact:
            raise RuntimeError("pair LM-head logits are not bitwise exact")
        if seq_top5_a != pair_top5_a or seq_top5_b != pair_top5_b:
            raise RuntimeError("pair LM-head top5 differs despite full-logit equality check")
        if top1 != EXPECTED_TOP1:
            raise RuntimeError(f"accepted-context target top1 anchor changed: {top1}")
        if seq_a_bytes != lm_bytes or seq_b_bytes != lm_bytes:
            raise RuntimeError("sequential LM-head read count does not equal tensor bytes")
        if pair_bytes != lm_bytes or sequential_bytes != 2 * pair_bytes:
            raise RuntimeError(
                f"pair LM-head did not halve bytes: seq={sequential_bytes} pair={pair_bytes}")

        payload = {
            "schema": "qwen38-lm-head-pair-probe-v1",
            "status": "PASS",
            "model_sha256": gdn.SHA256,
            "prefix_token_ids": PREFIX_TOKENS,
            "pair_token_ids": PAIR_TOKENS,
            "expected_top1": EXPECTED_TOP1,
            "top5_a": pair_top5_a,
            "top5_b": pair_top5_b,
            "logits_a_bitwise_exact": a_exact,
            "logits_b_bitwise_exact": b_exact,
            "lm_head_tensor_bytes": lm_bytes,
            "sequential_lm_head_bytes": sequential_bytes,
            "pair_lm_head_bytes": pair_bytes,
            "lm_head_bytes_saved": sequential_bytes - pair_bytes,
            "exact_two_to_one_lm_head_bytes": sequential_bytes == 2 * pair_bytes,
            "sequential_seconds": sequential_seconds,
            "pair_seconds": pair_seconds,
            "pair_speedup_vs_two_sequential_heads": sequential_seconds / pair_seconds,
            "target_k3_reader": engine.reader.report(),
            "elapsed_seconds": time.monotonic() - started,
            "max_rss_gib": rss_gib(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": "PASS",
            "top1": top1,
            "logits_a_bitwise_exact": a_exact,
            "logits_b_bitwise_exact": b_exact,
            "sequential_lm_head_bytes": sequential_bytes,
            "pair_lm_head_bytes": pair_bytes,
            "lm_head_bytes_saved": sequential_bytes - pair_bytes,
            "sequential_seconds": sequential_seconds,
            "pair_seconds": pair_seconds,
            "speedup": sequential_seconds / pair_seconds,
            "max_rss_gib": payload["max_rss_gib"],
        }, indent=2))
        print("QWEN38_LM_HEAD_PAIR_EXACT_PASS")
        return payload
    finally:
        engine.close()


def sanity() -> None:
    assert PREFIX_TOKENS == [12675, 11]
    assert PAIR_TOKENS == [353, 2688]
    assert EXPECTED_TOP1 == [2688, 264]
    assert base.VOCAB == 248320
    assert gdn.HIDDEN == 5120
    print(json.dumps({
        "schema": "qwen38-lm-head-pair-sanity-v1",
        "status": "PASS",
        "schedule": "one Q8_0 chunk -> A matvec -> B matvec",
        "vocab": base.VOCAB,
        "hidden": gdn.HIDDEN,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--native-lib", type=Path, required=True)
    r.add_argument("--state-lib", type=Path, required=True)
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
