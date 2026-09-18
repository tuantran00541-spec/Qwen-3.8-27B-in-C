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
        self.swiglu_calls = 0
        self.down_input = None

    def prepare_activation(self, weight_name, x):
        return ("prepared", weight_name, len(x))

    def matvec_prepared(self, weights, meta, prepared):
        name = str(meta["name"])
        if name.endswith("ffn_gate.weight"):
            return [0.25] * gdn.INTERMEDIATE
        if name.endswith("ffn_up.weight"):
            return [-0.5] * gdn.INTERMEDIATE
        raise AssertionError(name)

    def swiglu(self, gate, up):
        self.swiglu_calls += 1
        assert len(gate) == gdn.INTERMEDIATE
        assert len(up) == gdn.INTERMEDIATE
        return [0.125] * gdn.INTERMEDIATE

    def matvec(self, weights, meta, x):
        name = str(meta["name"])
        assert name.endswith("ffn_down.weight"), name
        self.down_input = list(x)
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
    assert runtime.swiglu_calls == 1, (
        f"FFN must delegate pointwise SwiGLU once, calls={runtime.swiglu_calls}"
    )
    assert runtime.down_input == [0.125] * gdn.INTERMEDIATE
    print("QWEN38_BONSAI2_FFN_NATIVE_SWIGLU_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
