#!/usr/bin/env python3
"""Exact structural contract for the Qwen3.8-27B MTP/NextN block.

The main decoder contract intentionally treats blk.64 as auxiliary. This file
spells out the MTP block that llama.cpp executes for LLM_GRAPH_TYPE_DECODER_MTP
and reports the byte footprint relevant to an SSD-resident draft path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from gguf_stream import parse_gguf
from qwen35_gguf_decoder_contract import (
    FILE,
    HIDDEN,
    INTERMEDIATE,
    SHA256,
    VALUE_DIM,
    VOCAB,
)

MTP_LAYER = 64

# llama.cpp qwen35.cpp load_block_mtp() + graph_mtp contract.
# The bartowski Q6_K_L file stores the large MTP projections as Q4_0 while
# keeping normalization weights in F32. Optional private embedding/head tensors
# are absent in the pinned GGUF and graph_mtp falls back to the global weights.
MTP_REQUIRED = {
    "attn_k.weight": ({"Q4_0"}, [HIDDEN, 1024]),
    "attn_k_norm.weight": ({"F32"}, [256]),
    "attn_norm.weight": ({"F32"}, [HIDDEN]),
    "attn_output.weight": ({"Q4_0"}, [VALUE_DIM, HIDDEN]),
    "attn_q.weight": ({"Q4_0"}, [HIDDEN, 12288]),
    "attn_q_norm.weight": ({"F32"}, [256]),
    "attn_v.weight": ({"Q4_0"}, [HIDDEN, 1024]),
    "ffn_down.weight": ({"Q4_0"}, [INTERMEDIATE, HIDDEN]),
    "ffn_gate.weight": ({"Q4_0"}, [HIDDEN, INTERMEDIATE]),
    "ffn_up.weight": ({"Q4_0"}, [HIDDEN, INTERMEDIATE]),
    "post_attention_norm.weight": ({"F32"}, [HIDDEN]),
    "nextn.eh_proj.weight": ({"Q4_0"}, [2 * HIDDEN, HIDDEN]),
    "nextn.enorm.weight": ({"F32"}, [HIDDEN]),
    "nextn.hnorm.weight": ({"F32"}, [HIDDEN]),
}
MTP_OPTIONAL = {
    "nextn.embed_tokens.weight": [HIDDEN, VOCAB],
    "nextn.shared_head_head.weight": [HIDDEN, VOCAB],
    "nextn.shared_head_norm.weight": [HIDDEN],
}
GLOBAL_REQUIRED = {
    "token_embd.weight": [HIDDEN, VOCAB],
    "output_norm.weight": [HIDDEN],
    "output.weight": [HIDDEN, VOCAB],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_tensor(t: Any, allowed: Iterable[str], shape: Sequence[int], errors: list[str]) -> None:
    if t.type_name not in set(allowed):
        errors.append(f"{t.name}: type={t.type_name} allowed={sorted(set(allowed))}")
    if list(t.shape) != list(shape):
        errors.append(f"{t.name}: shape={list(t.shape)} expected={list(shape)}")


def tensor_json(t: Any) -> dict[str, Any]:
    return {
        "name": t.name,
        "type": t.type_name,
        "shape": list(t.shape),
        "nbytes": int(t.nbytes),
        "source_offset": int(t.data_offset),
    }


def build(model: Path, output: Path) -> dict[str, Any]:
    errors: list[str] = []
    digest = sha256_file(model)
    if digest != SHA256:
        errors.append(f"sha256={digest} expected={SHA256}")

    directory = parse_gguf(model)
    if directory.metadata.get("general.architecture") != "qwen35":
        errors.append(f"architecture={directory.metadata.get('general.architecture')!r} expected='qwen35'")

    prefix = f"blk.{MTP_LAYER}."
    mtp_tensors = [t for t in directory.tensors if t.name.startswith(prefix)]
    actual = {t.name.removeprefix(prefix): t for t in mtp_tensors}

    required_names = set(MTP_REQUIRED)
    optional_names = set(MTP_OPTIONAL)
    missing = sorted(required_names - set(actual))
    extra = sorted(set(actual) - required_names - optional_names)
    if missing:
        errors.append(f"mtp missing={missing}")
    if extra:
        errors.append(f"mtp unexpected={extra}")

    for suffix, (allowed, shape) in MTP_REQUIRED.items():
        t = actual.get(suffix)
        if t is not None:
            validate_tensor(t, allowed, shape, errors)
    for suffix, shape in MTP_OPTIONAL.items():
        t = actual.get(suffix)
        if t is not None and list(t.shape) != list(shape):
            errors.append(f"{t.name}: shape={list(t.shape)} expected={list(shape)}")

    globals_map = {t.name: t for t in directory.tensors if t.name in GLOBAL_REQUIRED}
    for name, shape in GLOBAL_REQUIRED.items():
        t = globals_map.get(name)
        if t is None:
            errors.append(f"missing global fallback tensor {name}")
        elif list(t.shape) != list(shape):
            errors.append(f"{name}: shape={list(t.shape)} expected={shape}")

    has_private_embed = "nextn.embed_tokens.weight" in actual
    has_private_head = "nextn.shared_head_head.weight" in actual
    has_private_head_norm = "nextn.shared_head_norm.weight" in actual

    mtp_bytes = sum(int(t.nbytes) for t in mtp_tensors)
    q4_bytes = sum(int(t.nbytes) for t in mtp_tensors if t.type_name == "Q4_0")
    f32_bytes = sum(int(t.nbytes) for t in mtp_tensors if t.type_name == "F32")
    global_embed_bytes = int(globals_map["token_embd.weight"].nbytes) if "token_embd.weight" in globals_map else None
    global_head_bytes = int(globals_map["output.weight"].nbytes) if "output.weight" in globals_map else None
    global_head_norm_bytes = int(globals_map["output_norm.weight"].nbytes) if "output_norm.weight" in globals_map else None

    metadata_subset = {
        str(k): v for k, v in directory.metadata.items()
        if any(s in str(k).lower() for s in ("nextn", "mtp", "block_count", "recurrent", "full_attention"))
    }

    result: dict[str, Any] = {
        "schema": "qwen38-mtp-contract-v1",
        "status": "PASS" if not errors else "FAIL",
        "model_file": FILE,
        "sha256": digest,
        "sha256_match": digest == SHA256,
        "architecture": directory.metadata.get("general.architecture"),
        "mtp_layer": MTP_LAYER,
        "mtp_tensor_count": len(mtp_tensors),
        "mtp_type_counts": dict(sorted(Counter(t.type_name for t in mtp_tensors).items())),
        "mtp_bytes": mtp_bytes,
        "mtp_q4_bytes": q4_bytes,
        "mtp_f32_bytes": f32_bytes,
        "mtp_tensors": [tensor_json(t) for t in mtp_tensors],
        "private_embedding_present": has_private_embed,
        "private_lm_head_present": has_private_head,
        "private_lm_head_norm_present": has_private_head_norm,
        "uses_global_embedding_fallback": not has_private_embed,
        "uses_global_lm_head_fallback": not has_private_head,
        "uses_global_lm_head_norm_fallback": not has_private_head_norm,
        "global_embedding_bytes": global_embed_bytes,
        "global_lm_head_bytes": global_head_bytes,
        "global_lm_head_norm_bytes": global_head_norm_bytes,
        "metadata_subset": metadata_subset,
        "errors": errors,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "mtp_tensor_count": result["mtp_tensor_count"],
        "mtp_type_counts": result["mtp_type_counts"],
        "mtp_bytes": result["mtp_bytes"],
        "global_lm_head_bytes": result["global_lm_head_bytes"],
        "uses_global_embedding_fallback": result["uses_global_embedding_fallback"],
        "uses_global_lm_head_fallback": result["uses_global_lm_head_fallback"],
        "uses_global_lm_head_norm_fallback": result["uses_global_lm_head_norm_fallback"],
        "errors": errors,
    }, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    return result


def sanity() -> None:
    assert len(MTP_REQUIRED) == 14
    assert {name for name, (types, _) in MTP_REQUIRED.items() if types == {"Q4_0"}} == {
        "attn_k.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_down.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "nextn.eh_proj.weight",
    }
    assert MTP_REQUIRED["nextn.eh_proj.weight"][1] == [10240, 5120]
    print(json.dumps({
        "schema": "qwen38-mtp-contract-sanity-v1",
        "status": "PASS",
        "mtp_layer": MTP_LAYER,
        "required_tensor_count": len(MTP_REQUIRED),
        "optional_private_tensor_count": len(MTP_OPTIONAL),
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    real = sub.add_parser("real")
    real.add_argument("--model", type=Path, required=True)
    real.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
    else:
        build(args.model, args.output)


if __name__ == "__main__":
    main()
