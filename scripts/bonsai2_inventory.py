#!/usr/bin/env python3
"""Fast GGUF directory inventory for Bonsai 2 without reading tensor payloads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from gguf_stream import parse_gguf  # noqa: E402


def build_inventory(model: Path) -> dict[str, object]:
    directory = parse_gguf(model)
    type_counts = Counter(t.type_name for t in directory.tensors)
    layer_types: dict[int, Counter[str]] = defaultdict(Counter)

    for tensor in directory.tensors:
        if tensor.name.startswith("blk."):
            parts = tensor.name.split(".", 2)
            if len(parts) >= 3 and parts[1].isdigit():
                layer_types[int(parts[1])][tensor.type_name] += 1

    global_names = {"token_embd.weight", "output.weight"}
    globals_found = {
        t.name: {
            "type": t.type_name,
            "ggml_type": t.ggml_type,
            "shape": list(t.shape),
            "nbytes": t.nbytes,
        }
        for t in directory.tensors
        if t.name in global_names
    }

    lowbit = [t for t in directory.tensors if t.type_name in {"PQ2_0", "PTQ1_0"}]
    return {
        "schema": "qwen38-bonsai2-gguf-inventory-v1",
        "status": "PASS",
        "model": str(model),
        "file_bytes": directory.file_bytes,
        "architecture": directory.metadata.get("general.architecture"),
        "file_type": directory.metadata.get("general.file_type"),
        "tensor_count": directory.tensor_count,
        "alignment": directory.alignment,
        "type_counts": dict(sorted(type_counts.items())),
        "layer_count": len(layer_types),
        "layer_type_counts": {
            str(layer): dict(sorted(counts.items()))
            for layer, counts in sorted(layer_types.items())
        },
        "globals": globals_found,
        "lowbit_tensor_count": len(lowbit),
        "lowbit_types": sorted({t.type_name for t in lowbit}),
        "lowbit_bytes": sum(t.nbytes for t in lowbit),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    state = build_inventory(args.model)
    text = json.dumps(state, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
