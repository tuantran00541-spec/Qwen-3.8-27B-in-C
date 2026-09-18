#!/usr/bin/env python3
"""Real-weight Bonsai 2 -> K3 -> native C streaming gate.

The probe downloads nothing itself. It receives a verified Bonsai 2 GGUF,
packs only decoder layer 0 into the existing K3 layout, reads that layer through
K3Trunk's bounded ring, applies the Prism Hadamard activation transform, and
runs one real low-bit FFN matvec through the native C AVX2 kernel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_quant_runtime import Bonsai2NativeRuntime  # noqa: E402
from gguf_k3_layout import pack_gguf_layers, partition_tensors  # noqa: E402
from gguf_stream import TensorSpan, parse_gguf  # noqa: E402
from k3_stream import K3Trunk  # noqa: E402


MODEL_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "6ed5e12bf84b7a63069882c91dd9e9218647d17b"


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f16(raw: bytes) -> float:
    return float(struct.unpack("<e", raw)[0])


def signed_i8(raw: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(raw)}b", raw))


def decode_ptq1(block: bytes) -> list[int]:
    if len(block) != 28:
        raise ValueError("PTQ1_0 block must be 28 bytes")
    qs = block[:24]
    qh = block[24:26]
    pow3 = (1, 3, 9, 27, 81)
    out: list[int] = []

    for n in range(5):
        for m in range(16):
            v = (qs[m] * pow3[n]) & 0xFF
            out.append(((v * 3) >> 8) - 1)
    for n in range(5):
        for m in range(16, 24):
            v = (qs[m] * pow3[n]) & 0xFF
            out.append(((v * 3) >> 8) - 1)
    for n in range(4):
        for h in range(2):
            v = (qh[h] * pow3[n]) & 0xFF
            out.append(((v * 3) >> 8) - 1)

    if len(out) != 128:
        raise AssertionError(len(out))
    return out


def decode_pq2(block: bytes) -> list[int]:
    if len(block) != 34:
        raise ValueError("PQ2_0 block must be 34 bytes")
    out: list[int] = []
    for byte in block[2:34]:
        out.extend((
            ((byte >> 0) & 3) - 1,
            ((byte >> 2) & 3) - 1,
            ((byte >> 4) & 3) - 1,
            ((byte >> 6) & 3) - 1,
        ))
    return out


def reference_first_row(kind: str, weight_row: bytes, activation: bytes, n: int) -> float:
    if n % 128:
        raise ValueError("Bonsai 2 reference width must be divisible by 128")
    wblock = 28 if kind == "PTQ1_0" else 34
    expected_wbytes = (n // 128) * wblock
    expected_abytes = (n // 32) * 34
    if len(weight_row) != expected_wbytes or len(activation) != expected_abytes:
        raise ValueError("reference row byte length mismatch")

    total = 0.0
    for ib in range(n // 128):
        wb = weight_row[ib*wblock : (ib+1)*wblock]
        q = decode_ptq1(wb) if kind == "PTQ1_0" else decode_pq2(wb)
        d0 = f16(wb[26:28] if kind == "PTQ1_0" else wb[0:2])
        sumi = 0.0
        for k in range(4):
            ab = activation[(ib*4+k)*34 : (ib*4+k+1)*34]
            d1 = f16(ab[0:2])
            aq = signed_i8(ab[2:34])
            dot = sum(q[k*32+j] * aq[j] for j in range(32))
            sumi = f32(sumi + f32(d1 * dot))
        total = f32(total + f32(d0 * sumi))
    return total


def pick_tensor(layer: list[TensorSpan], folded: set[str]) -> TensorSpan:
    candidates = [
        t for t in layer
        if t.type_name in {"PTQ1_0", "PQ2_0"}
        and t.name in folded
        and ".ssm_out." not in t.name
        and len(t.shape) == 2
    ]
    if not candidates:
        raise RuntimeError("layer 0 has no non-GDN folded PTQ1/PQ2 matrix")
    preferred = [t for t in candidates if t.name.endswith(".ffn_gate.weight")]
    return preferred[0] if preferred else candidates[0]


def deterministic_activation(n: int) -> list[float]:
    return [((i * 131 + 17) % 1021 - 510) / 511.0 for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--native-lib", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    directory = parse_gguf(args.model)
    grouped, _, _ = partition_tensors(directory, expected_layers=64)

    runtime = Bonsai2NativeRuntime(
        args.native_lib, directory.metadata, threads=args.threads
    )
    tensor = pick_tensor(grouped[0], set(runtime.folded_weights))
    ne0, rows = map(int, tensor.shape)

    trunk = args.work_dir / "layer0.k3.bin"
    index = args.work_dir / "layer0.k3.json"
    pack_started = time.perf_counter()
    manifest = pack_gguf_layers(
        directory,
        trunk,
        index,
        layers=[0],
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        source_sha256=args.source_sha256,
    )
    pack_seconds = time.perf_counter() - pack_started

    layer0 = manifest["layers"][0]
    budget = int(layer0["read_bytes"]) + 4096
    meta = {
        "name": tensor.name,
        "type_name": tensor.type_name,
        "ggml_type": tensor.ggml_type,
        "shape": list(tensor.shape),
        "nbytes": tensor.nbytes,
    }

    x = deterministic_activation(ne0)
    with K3Trunk(
        trunk,
        index,
        budget_bytes=budget,
        want_ring=1,
        max_pinned=0,
        prefer_direct_io=True,
    ) as reader:
        t0 = time.perf_counter()
        bound = reader.bind(0)
        bind_seconds = time.perf_counter() - t0
        weights = reader.tensor_view(bound, tensor.name)

        t0 = time.perf_counter()
        prepared = runtime.prepare_activation(tensor.name, x)
        prepare_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        out = runtime.matvec_prepared(weights, meta, prepared)
        matvec_seconds = time.perf_counter() - t0

        weight_row_bytes = tensor.nbytes // rows
        first_row = bytes(weights[:weight_row_bytes])
        activation_bytes = bytes(prepared[0])
        ref0 = reference_first_row(tensor.type_name, first_row, activation_bytes, ne0)
        got0 = out[0]
        abs_error = abs(got0 - ref0)
        tolerance = 3e-5 * max(1.0, abs(ref0))
        if not math.isfinite(got0) or abs_error > tolerance:
            raise SystemExit(
                f"native first-row parity failed got={got0:.9g} ref={ref0:.9g} "
                f"abs_error={abs_error:.9g} tolerance={tolerance:.9g}"
            )

        reader_report = reader.report()
        del weights
        del bound

    max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    state = {
        "schema": "qwen38-bonsai2-native-k3-stream-v1",
        "status": "PASS",
        "model": args.model.name,
        "architecture": directory.metadata.get("general.architecture"),
        "file_type": directory.metadata.get("general.file_type"),
        "hadamard": {
            "version": directory.metadata.get("prism.hadamard.version"),
            "block_size": runtime.block_size,
            "transform": directory.metadata.get("prism.hadamard.transform"),
            "axis": directory.metadata.get("prism.hadamard.axis"),
            "sign_mode": runtime.sign_mode,
            "gdn_v_grouped": runtime.gdn_v_grouped,
        },
        "tensor": meta,
        "layer0": {
            "read_bytes": int(layer0["read_bytes"]),
            "data_bytes": int(layer0["data_bytes"]),
            "tensor_count": int(layer0["tensor_count"]),
        },
        "timing": {
            "pack_seconds": pack_seconds,
            "bind_seconds": bind_seconds,
            "activation_prepare_seconds": prepare_seconds,
            "matvec_seconds": matvec_seconds,
            "matvec_rows_per_second": rows / matvec_seconds,
            "encoded_weight_gib_per_second": tensor.nbytes / matvec_seconds / (1024**3),
        },
        "parity": {
            "first_row_native": got0,
            "first_row_reference": ref0,
            "absolute_error": abs_error,
            "tolerance": tolerance,
        },
        "output_probe": {
            "first8": out[:8],
            "sum_first64": f32(sum(out[:64])),
        },
        "k3": reader_report,
        "native": runtime.report(),
        "max_rss_gib": max_rss_kib / (1024**2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime.close()
    print(json.dumps(state, indent=2, sort_keys=True))
    print("QWEN38_BONSAI2_NATIVE_K3_STREAM_PASS")


if __name__ == "__main__":
    main()
