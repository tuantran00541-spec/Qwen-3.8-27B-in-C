#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

import bonsai2_two_token as t2
import qwen35_gdn_quant_layer_gate as gdn


class ProbeRuntime:
    def __init__(self) -> None:
        self.ffn_calls = 0

    def ffn(
        self,
        x,
        gate_weights,
        gate_meta,
        up_weights,
        up_meta,
        down_weights,
        down_meta,
    ):
        self.ffn_calls += 1
        assert len(x) == gdn.HIDDEN
        assert gate_weights == "ffn_gate.weight"
        assert up_weights == "ffn_up.weight"
        assert down_weights == "ffn_down.weight"
        assert gate_meta["name"].endswith("ffn_gate.weight")
        assert up_meta["name"].endswith("ffn_up.weight")
        assert down_meta["name"].endswith("ffn_down.weight")
        return [0.0] * gdn.HIDDEN

def main() -> None:
    runtime = ProbeRuntime()
    prefix = "blk.0"
    metas = {
        f"{prefix}.ffn_gate.weight": {"name": f"{prefix}.ffn_gate.weight"},
        f"{prefix}.ffn_up.weight": {"name": f"{prefix}.ffn_up.weight"},
        f"{prefix}.ffn_down.weight": {"name": f"{prefix}.ffn_down.weight"},
    }

    def view(suffix: str):
        return suffix

    out = t2.ffn(runtime, view, metas, prefix, [0.0] * gdn.HIDDEN)
    assert len(out) == gdn.HIDDEN
    assert runtime.ffn_calls == 1, (
        f"FFN helper must delegate the native superkernel once, calls={runtime.ffn_calls}"
    )
    print("QWEN38_BONSAI2_FFN_NATIVE_SWIGLU_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
