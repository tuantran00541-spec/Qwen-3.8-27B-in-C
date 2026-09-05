#!/usr/bin/env python3
"""Machine-readable Qwen3.8-27B GGUF decoder contract.

This validates *semantic roles and shapes* while preserving the real per-layer
mixed quantization chosen by the Q6_K_L file. In particular, later layers may
promote/demote selected projections between Q8_0 and Q6_K; layer 0 or layer 3
must not be treated as a universal quant-type template for all 64 blocks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:
    from qwen38_win32_bootstrap import install_resource_compat

    resource = install_resource_compat()

from gguf_k3_layout import partition_tensors
from gguf_stream import parse_gguf

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
DECODER_LAYERS = 64
HIDDEN = 5120
INTERMEDIATE = 17408
K_HEADS = 16
V_HEADS = 48
HEAD_DIM = 128
KEY_DIM = K_HEADS * HEAD_DIM
VALUE_DIM = V_HEADS * HEAD_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
CONV_KERNEL = 4
VOCAB = 248320

# (allowed ggml types, ggml shape). The mixed-type allowances below are not
# guesses: the first real contract run exposed Q6_K ssm_out in selected GDN
# layers and Q6_K q/v projections in selected full-attention layers.
RECURRENT = {
    "attn_gate.weight": ({"Q6_K"}, [HIDDEN, VALUE_DIM]),
    "attn_norm.weight": ({"F32"}, [HIDDEN]),
    "attn_qkv.weight": ({"Q8_0"}, [HIDDEN, CONV_DIM]),
    "ffn_down.weight": ({"Q6_K"}, [INTERMEDIATE, HIDDEN]),
    "ffn_gate.weight": ({"Q6_K"}, [HIDDEN, INTERMEDIATE]),
    "ffn_up.weight": ({"Q6_K"}, [HIDDEN, INTERMEDIATE]),
    "post_attention_norm.weight": ({"F32"}, [HIDDEN]),
    "ssm_a": ({"F32"}, [V_HEADS]),
    "ssm_alpha.weight": ({"F32"}, [HIDDEN, V_HEADS]),
    "ssm_beta.weight": ({"F32"}, [HIDDEN, V_HEADS]),
    "ssm_conv1d.weight": ({"F32"}, [CONV_KERNEL, CONV_DIM]),
    "ssm_dt.bias": ({"F32"}, [V_HEADS]),
    "ssm_norm.weight": ({"F32"}, [HEAD_DIM]),
    "ssm_out.weight": ({"Q6_K", "Q8_0"}, [VALUE_DIM, HIDDEN]),
}
FULL_ATTENTION = {
    "attn_k.weight": ({"Q8_0"}, [HIDDEN, 1024]),
    "attn_k_norm.weight": ({"F32"}, [256]),
    "attn_norm.weight": ({"F32"}, [HIDDEN]),
    "attn_output.weight": ({"Q8_0"}, [VALUE_DIM, HIDDEN]),
    "attn_q.weight": ({"Q6_K", "Q8_0"}, [HIDDEN, 12288]),
    "attn_q_norm.weight": ({"F32"}, [256]),
    "attn_v.weight": ({"Q6_K", "Q8_0"}, [HIDDEN, 1024]),
    "ffn_down.weight": ({"Q6_K"}, [INTERMEDIATE, HIDDEN]),
    "ffn_gate.weight": ({"Q6_K"}, [HIDDEN, INTERMEDIATE]),
    "ffn_up.weight": ({"Q6_K"}, [HIDDEN, INTERMEDIATE]),
    "post_attention_norm.weight": ({"F32"}, [HIDDEN]),
}
GLOBALS = {
    "token_embd.weight": ({"Q8_0"}, [HIDDEN, VOCAB]),
    "output_norm.weight": ({"F32"}, [HIDDEN]),
    "output.weight": ({"Q8_0"}, [HIDDEN, VOCAB]),
}
EXPECTED_Q4_MTP = {
    "blk.64.attn_k.weight",
    "blk.64.attn_output.weight",
    "blk.64.attn_q.weight",
    "blk.64.attn_v.weight",
    "blk.64.ffn_down.weight",
    "blk.64.ffn_gate.weight",
    "blk.64.ffn_up.weight",
    "blk.64.nextn.eh_proj.weight",
}
DECODER_ALLOWED_TYPES = {"F32", "Q6_K", "Q8_0"}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_json(t) -> dict[str, Any]:
    return {
        "name": t.name,
        "type": t.type_name,
        "ggml_type": t.ggml_type,
        "shape": list(t.shape),
        "nbytes": t.nbytes,
        "source_offset": t.data_offset,
    }


def validate_tensor(t, allowed: Iterable[str], shape: Sequence[int], errors: list[str]) -> None:
    allowed_set = set(allowed)
    if t.type_name not in allowed_set:
        errors.append(f"{t.name}: type={t.type_name} allowed={sorted(allowed_set)}")
    if list(t.shape) != list(shape):
        errors.append(f"{t.name}: shape={list(t.shape)} expected={list(shape)}")


def build_contract(model: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    state: dict[str, Any] = {
        "schema": "qwen38-gguf-decoder-contract-v2",
        "status": "INCOMPLETE",
        "repo": REPO,
        "file": FILE,
        "expected_sha256": SHA256,
        "file_bytes": model.stat().st_size,
    }
    atomic_json(output, state)

    digest = sha256_file(model)
    state.update(sha256=digest, sha256_match=(digest == SHA256))
    atomic_json(output, state)
    if digest != SHA256:
        state.update(status="FAIL", failure_class="parser/layout", errors=["SHA256 mismatch"])
        atomic_json(output, state)
        raise RuntimeError(f"GGUF SHA256 mismatch: {digest}")

    directory = parse_gguf(model)
    grouped, auxiliary, globals_ = partition_tensors(directory, expected_layers=DECODER_LAYERS)
    errors: list[str] = []
    if directory.metadata.get("general.architecture") != "qwen35":
        errors.append(f"architecture={directory.metadata.get('general.architecture')!r} expected='qwen35'")
    if sorted(grouped) != list(range(DECODER_LAYERS)):
        errors.append(f"decoder blocks={sorted(grouped)} expected=0..63")
    if sorted(auxiliary) != [64]:
        errors.append(f"auxiliary blocks={sorted(auxiliary)} expected=[64]")

    decoder_type_counts: Counter[str] = Counter()
    layers: list[dict[str, Any]] = []
    mixed_quant_schedule: list[dict[str, Any]] = []
    for layer in range(DECODER_LAYERS):
        kind = "full_attention" if layer % 4 == 3 else "gated_deltanet"
        spec = FULL_ATTENTION if kind == "full_attention" else RECURRENT
        tensors = grouped.get(layer, [])
        actual = {t.name.removeprefix(f"blk.{layer}."): t for t in tensors}
        missing = sorted(set(spec) - set(actual))
        extra = sorted(set(actual) - set(spec))
        if missing or extra:
            errors.append(f"blk.{layer}: missing={missing} extra={extra}")
        for suffix, (allowed, shape) in spec.items():
            tensor = actual.get(suffix)
            if tensor is not None:
                validate_tensor(tensor, allowed, shape, errors)
                if len(allowed) > 1:
                    mixed_quant_schedule.append({
                        "layer": layer,
                        "tensor": tensor.name,
                        "type": tensor.type_name,
                        "allowed_types": sorted(allowed),
                    })
        decoder_type_counts.update(t.type_name for t in tensors)
        layers.append({
            "layer": layer,
            "kind": kind,
            "tensor_count": len(tensors),
            "type_counts": dict(sorted(Counter(t.type_name for t in tensors).items())),
            "tensors": [tensor_json(t) for t in tensors],
        })

    global_map = {t.name: t for t in globals_}
    if set(global_map) != set(GLOBALS):
        errors.append(f"global tensors={sorted(global_map)} expected={sorted(GLOBALS)}")
    for name, (allowed, shape) in GLOBALS.items():
        if name in global_map:
            validate_tensor(global_map[name], allowed, shape, errors)

    q4_names = {t.name for t in directory.tensors if t.type_name == "Q4_0"}
    if q4_names != EXPECTED_Q4_MTP:
        errors.append(f"Q4_0 names={sorted(q4_names)} expected={sorted(EXPECTED_Q4_MTP)}")
    unsupported = set(decoder_type_counts) - DECODER_ALLOWED_TYPES
    if unsupported:
        errors.append(f"unsupported decoder types={sorted(unsupported)}")
    if decoder_type_counts.get("Q4_0", 0):
        errors.append("decoder main path contains Q4_0")

    state.update({
        "status": "PASS" if not errors else "FAIL",
        "failure_class": None if not errors else "parser/layout",
        "architecture": directory.metadata.get("general.architecture"),
        "tensor_count": directory.tensor_count,
        "tensor_type_counts": dict(sorted(Counter(t.type_name for t in directory.tensors).items())),
        "decoder_layer_count": len(grouped),
        "decoder_type_counts": dict(sorted(decoder_type_counts.items())),
        "decoder_allowed_types": sorted(DECODER_ALLOWED_TYPES),
        "auxiliary_block_ids": sorted(auxiliary),
        "q4_0_names": sorted(q4_names),
        "globals": [tensor_json(t) for t in globals_],
        "layers": layers,
        "mixed_quant_schedule": mixed_quant_schedule,
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_gib": rss_gib(),
    })
    atomic_json(output, state)
    print(json.dumps({
        "schema": state["schema"],
        "status": state["status"],
        "sha256_match": state["sha256_match"],
        "decoder_layer_count": state["decoder_layer_count"],
        "decoder_type_counts": state["decoder_type_counts"],
        "q4_0_names": state["q4_0_names"],
        "mixed_quant_entries": len(mixed_quant_schedule),
        "errors": errors,
    }, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)
    return state


def sanity() -> None:
    assert RECURRENT["ssm_out.weight"][0] == {"Q6_K", "Q8_0"}
    assert FULL_ATTENTION["attn_q.weight"][0] == {"Q6_K", "Q8_0"}
    assert FULL_ATTENTION["attn_v.weight"][0] == {"Q6_K", "Q8_0"}
    assert EXPECTED_Q4_MTP == {
        "blk.64.attn_k.weight", "blk.64.attn_output.weight", "blk.64.attn_q.weight",
        "blk.64.attn_v.weight", "blk.64.ffn_down.weight", "blk.64.ffn_gate.weight",
        "blk.64.ffn_up.weight", "blk.64.nextn.eh_proj.weight",
    }
    print(json.dumps({
        "schema": "qwen38-gguf-decoder-contract-sanity-v1",
        "status": "PASS",
        "decoder_allowed_types": sorted(DECODER_ALLOWED_TYPES),
        "mixed_quant_roles": ["gdn.ssm_out", "full_attention.q", "full_attention.v"],
        "q4_decoder_required": False,
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
        build_contract(args.model, args.output)


if __name__ == "__main__":
    main()
