#!/usr/bin/env python3
"""Lossless chat-history state capsules for the Bonsai 2 Qwen3.8 runtime.

A capsule is a resumable runtime snapshot, not a replacement for the durable
text transcript.  It stores the fixed-size GDN recurrent state, causal-conv
history, a bounded full-attention KV tail, and the absolute token position.
"""
from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import bonsai2_prompt_spike as prompt

SCHEMA = "qwen38-bonsai2-chat-capsule-v1"


def _ctypes_bytes(obj) -> bytes:
    c = prompt.t2.ctypes
    return c.string_at(c.addressof(obj), c.sizeof(obj))


def _f32_bytes(values) -> bytes:
    buf = values if isinstance(values, array) else array("f", map(float, values))
    if buf.typecode != "f":
        raise ValueError("expected F32 array")
    return buf.tobytes()


def _f16_row_bytes(values) -> bytes:
    vals = list(map(float, values))
    return struct.pack(f"<{len(vals)}e", *vals)


def _np():
    try:
        import numpy
    except Exception as exc:
        raise RuntimeError(
            "numpy is required for FP16/BF16 capsule state storage"
        ) from exc
    return numpy


def _encode_state_bytes(raw_f32: bytes, state_dtype: str) -> bytes:
    state_dtype = str(state_dtype).lower()
    if state_dtype == "f32":
        return raw_f32
    np = _np()
    src = np.frombuffer(raw_f32, dtype="<f4")
    if state_dtype == "f16":
        return src.astype("<f2").tobytes()
    if state_dtype == "bf16":
        bits = src.view("<u4")
        # Round-to-nearest-even before dropping the low 16 mantissa bits.
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        return (rounded >> 16).astype("<u2").tobytes()
    raise ValueError(f"unsupported state_dtype={state_dtype!r}")


def _decode_state_bytes(payload: bytes, state_dtype: str, expected_f32_bytes: int) -> bytes:
    state_dtype = str(state_dtype).lower()
    if state_dtype == "f32":
        raw = payload
    else:
        np = _np()
        if state_dtype == "f16":
            raw = np.frombuffer(payload, dtype="<f2").astype("<f4").tobytes()
        elif state_dtype == "bf16":
            words = np.frombuffer(payload, dtype="<u2").astype("<u4")
            raw = (words << 16).view("<f4").tobytes()
        else:
            raise ValueError(f"unsupported state_dtype={state_dtype!r}")
    if len(raw) != expected_f32_bytes:
        raise RuntimeError(
            f"decoded recurrent-state bytes={len(raw)} expected={expected_f32_bytes}"
        )
    return raw


def _engine_fingerprint(engine: prompt.StatefulBonsai2Generator) -> str:
    h = hashlib.sha256()
    h.update(struct.pack("<Q", int(engine.position)))

    for layer in sorted(engine.states):
        h.update(struct.pack("<I", int(layer)))
        h.update(_ctypes_bytes(engine.states[layer]))

    for layer in sorted(engine.conv_history):
        rows = engine.conv_history[layer]
        h.update(struct.pack("<II", int(layer), len(rows)))
        for row in rows:
            h.update(struct.pack("<I", len(row)))
            h.update(_f32_bytes(row))

    for layer in sorted(engine.caches):
        cache = engine.caches[layer]
        if len(cache["k"]) != len(cache["v"]):
            raise RuntimeError("attention K/V cache length mismatch")
        h.update(struct.pack("<II", int(layer), len(cache["k"])))
        for key in ("k", "v"):
            for row in cache[key]:
                h.update(struct.pack("<I", len(row)))
                h.update(_f16_row_bytes(row))
    return h.hexdigest()


def save_chat_capsule(
    engine: prompt.StatefulBonsai2Generator,
    path: Path,
    *,
    attention_tail_tokens: int = 31,
    state_dtype: str = "f32",
) -> dict:
    """Save a resumable state capsule and bound future attention to N tokens.

    engine.step() appends the new token before attention runs, therefore a
    checkpoint keeps at most N-1 prior K/V rows per full-attention layer.
    """
    attention_tail_tokens = int(attention_tail_tokens)
    state_dtype = str(state_dtype).lower()
    if attention_tail_tokens < 1:
        raise ValueError("attention_tail_tokens must be at least 1")
    if state_dtype not in {"f32", "f16", "bf16"}:
        raise ValueError("state_dtype must be one of: f32, f16, bf16")

    keep_prior_rows = max(0, attention_tail_tokens - 1)
    dropped_rows = engine.trim_attention_kv(keep_prior_rows)

    state_layers = sorted(engine.states)
    conv_layers = sorted(engine.conv_history)
    attention_layers = sorted(engine.caches)

    states_blob = bytearray()
    state_f32_bytes_per_layer = None
    state_storage_bytes_per_layer = None
    for layer in state_layers:
        raw_f32 = _ctypes_bytes(engine.states[layer])
        if state_f32_bytes_per_layer is None:
            state_f32_bytes_per_layer = len(raw_f32)
        elif len(raw_f32) != state_f32_bytes_per_layer:
            raise RuntimeError("non-uniform recurrent state size")
        stored = _encode_state_bytes(raw_f32, state_dtype)
        if state_storage_bytes_per_layer is None:
            state_storage_bytes_per_layer = len(stored)
        elif len(stored) != state_storage_bytes_per_layer:
            raise RuntimeError("non-uniform stored recurrent state size")
        states_blob.extend(stored)

    conv_blob = bytearray()
    conv_rows = {}
    conv_widths = {}
    for layer in conv_layers:
        rows = engine.conv_history[layer]
        conv_rows[str(layer)] = len(rows)
        widths = []
        for row in rows:
            widths.append(len(row))
            conv_blob.extend(_f32_bytes(row))
        conv_widths[str(layer)] = widths

    kv_blob = bytearray()
    kv_rows = {}
    kv_widths = {}
    for layer in attention_layers:
        cache = engine.caches[layer]
        if len(cache["k"]) != len(cache["v"]):
            raise RuntimeError("attention K/V cache length mismatch")
        kv_rows[str(layer)] = len(cache["k"])
        widths = {"k": [], "v": []}
        for key in ("k", "v"):
            for row in cache[key]:
                widths[key].append(len(row))
                kv_blob.extend(_f16_row_bytes(row))
        kv_widths[str(layer)] = widths

    fingerprint = _engine_fingerprint(engine)
    metadata = {
        "schema": SCHEMA,
        "model_sha256": prompt.base.MODEL_SHA256,
        "position": int(engine.position),
        "attention_tail_tokens": attention_tail_tokens,
        "kept_prior_attention_rows": keep_prior_rows,
        "dropped_attention_rows": dropped_rows,
        "state_layers": state_layers,
        "state_dtype": state_dtype,
        "state_f32_bytes_per_layer": int(state_f32_bytes_per_layer or 0),
        "state_storage_bytes_per_layer": int(state_storage_bytes_per_layer or 0),
        "conv_layers": conv_layers,
        "conv_rows": conv_rows,
        "conv_widths": conv_widths,
        "attention_layers": attention_layers,
        "kv_rows": kv_rows,
        "kv_widths": kv_widths,
        "fingerprint": fingerprint,
        "payload_bytes": {
            f"states_{state_dtype}": len(states_blob),
            "conv_f32": len(conv_blob),
            "kv_f16": len(kv_blob),
        },
        "payload_sha256": {
            "states": hashlib.sha256(states_blob).hexdigest(),
            "conv": hashlib.sha256(conv_blob).hexdigest(),
            "kv": hashlib.sha256(kv_blob).hexdigest(),
        },
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        zf.writestr(f"states.{state_dtype}", states_blob)
        zf.writestr("conv.f32", conv_blob)
        zf.writestr("kv.f16", kv_blob)

    metadata["capsule_file_bytes"] = path.stat().st_size
    return metadata


def load_chat_capsule(
    engine: prompt.StatefulBonsai2Generator,
    path: Path,
) -> dict:
    """Restore a capsule exactly into a freshly constructed runtime engine."""
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        metadata = json.loads(zf.read("metadata.json"))
        if metadata.get("schema") != SCHEMA:
            raise ValueError(f"unsupported capsule schema: {metadata.get('schema')!r}")
        if metadata.get("model_sha256") != prompt.base.MODEL_SHA256:
            raise ValueError("capsule model SHA does not match runtime model")
        state_dtype = str(metadata.get("state_dtype", "f32")).lower()
        states_blob = zf.read(f"states.{state_dtype}")
        conv_blob = zf.read("conv.f32")
        kv_blob = zf.read("kv.f16")

    payload_sha = metadata.get("payload_sha256")
    if payload_sha:
        observed = {
            "states": hashlib.sha256(states_blob).hexdigest(),
            "conv": hashlib.sha256(conv_blob).hexdigest(),
            "kv": hashlib.sha256(kv_blob).hexdigest(),
        }
        if observed != payload_sha:
            raise RuntimeError("capsule payload SHA256 mismatch")

    c = prompt.t2.ctypes
    state_f32_size = int(
        metadata.get("state_f32_bytes_per_layer", metadata.get("state_bytes_per_layer", 0))
    )
    state_storage_size = int(
        metadata.get("state_storage_bytes_per_layer", state_f32_size)
    )
    off = 0
    for layer in metadata["state_layers"]:
        layer = int(layer)
        state = engine.states[layer]
        if c.sizeof(state) != state_f32_size:
            raise RuntimeError("capsule recurrent-state size mismatch")
        chunk = states_blob[off : off + state_storage_size]
        if len(chunk) != state_storage_size:
            raise RuntimeError("truncated recurrent-state payload")
        raw_f32 = _decode_state_bytes(chunk, state_dtype, state_f32_size)
        c.memmove(c.addressof(state), raw_f32, state_f32_size)
        off += state_storage_size
    if off != len(states_blob):
        raise RuntimeError("unexpected recurrent-state payload tail")

    off = 0
    for layer in metadata["conv_layers"]:
        layer = int(layer)
        rows = []
        for width in metadata["conv_widths"][str(layer)]:
            width = int(width)
            nbytes = width * 4
            chunk = conv_blob[off : off + nbytes]
            if len(chunk) != nbytes:
                raise RuntimeError("truncated conv-history payload")
            row = array("f")
            row.frombytes(chunk)
            if len(row) != width:
                raise RuntimeError("conv-history width mismatch")
            rows.append(row)
            off += nbytes
        engine.conv_history[layer] = rows
    if off != len(conv_blob):
        raise RuntimeError("unexpected conv-history payload tail")

    off = 0
    for layer in metadata["attention_layers"]:
        layer = int(layer)
        cache = {"k": [], "v": []}
        widths = metadata["kv_widths"][str(layer)]
        for key in ("k", "v"):
            for width in widths[key]:
                width = int(width)
                nbytes = width * 2
                chunk = kv_blob[off : off + nbytes]
                if len(chunk) != nbytes:
                    raise RuntimeError("truncated attention-KV payload")
                row = list(struct.unpack(f"<{width}e", chunk))
                cache[key].append(row)
                off += nbytes
        engine.caches[layer] = cache
    if off != len(kv_blob):
        raise RuntimeError("unexpected attention-KV payload tail")

    engine.position = int(metadata["position"])
    engine.runtime._attention_cache.clear()

    actual = _engine_fingerprint(engine)
    expected = str(metadata["fingerprint"])
    if state_dtype == "f32" and actual != expected:
        raise RuntimeError(
            f"capsule fingerprint mismatch expected={expected} actual={actual}"
        )

    metadata["capsule_file_bytes"] = path.stat().st_size
    metadata["restored_fingerprint"] = actual
    metadata["fingerprint_exact"] = actual == expected
    return metadata


def engine_fingerprint(engine: prompt.StatefulBonsai2Generator) -> str:
    return _engine_fingerprint(engine)
