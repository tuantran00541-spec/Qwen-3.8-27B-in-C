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
        self.conv_calls = 0
        self.norm_gate_calls = 0
        self.ssm_out_input = None
        self.rms_norm_calls = 0
        self.residual_add_calls = 0
        self.gdn_repeat_scale_calls = 0
        self.ffn_calls = 0

    def rms_norm(self, values, weight, *, rows=1, eps=1e-6):
        self.rms_norm_calls += 1
        width = len(weight)
        assert rows >= 1
        assert len(values) == rows * width
        out = []
        for row in range(rows):
            start = row * width
            out.extend(gdn.rms_norm(values[start:start + width], weight, eps))
        return out

    def gdn_repeat_scale(self, q, k, *, repeats, scale):
        self.gdn_repeat_scale_calls += 1
        assert len(q) == gdn.KEY_DIM
        assert len(k) == gdn.KEY_DIM
        assert repeats == gdn.V_HEADS // gdn.K_HEADS
        q_out = []
        k_out = []
        for _ in range(repeats):
            q_out.extend(t2.mulf(v, scale) for v in q)
            k_out.extend(float(v) for v in k)
        return q_out, k_out

    def residual_add(self, a, b):
        self.residual_add_calls += 1
        assert len(a) == len(b)
        return [float(a[i]) + float(b[i]) for i in range(len(a))]

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

    def prepare_activation(self, weight_name, x):
        return ("prepared", weight_name, len(x))

    def matvec_prepared(self, weights, meta, prepared):
        name = str(meta["name"])
        if name.endswith("attn_qkv.weight"):
            return [0.0] * gdn.CONV_DIM
        if name.endswith("attn_gate.weight"):
            return [0.0] * gdn.VALUE_DIM
        if name.endswith("ffn_gate.weight"):
            return [0.0] * gdn.INTERMEDIATE
        if name.endswith("ffn_up.weight"):
            return [0.0] * gdn.INTERMEDIATE
        raise AssertionError(name)

    def matvec(self, weights, meta, x):
        name = str(meta["name"])
        if name.endswith("ssm_beta.weight") or name.endswith("ssm_alpha.weight"):
            return [0.0] * gdn.V_HEADS
        if name.endswith("ssm_out.weight"):
            self.ssm_out_input = list(x)
            return [0.0] * gdn.HIDDEN
        if name.endswith("ffn_down.weight"):
            return [0.0] * gdn.HIDDEN
        raise AssertionError(name)

    def swiglu(self, gate, up):
        return [0.0] * gdn.INTERMEDIATE

    def gdn_conv_silu(self, qkv, history, kernels):
        self.conv_calls += 1
        assert len(qkv) == gdn.CONV_DIM
        assert len(history) == 1
        assert len(history[0]) == gdn.CONV_DIM
        assert len(kernels) == gdn.CONV_DIM * gdn.CONV_KERNEL
        return [0.0] * gdn.CONV_DIM

    def gdn_norm_gate(self, core, norm_weight, z, *, eps=1e-6):
        self.norm_gate_calls += 1
        assert len(core) == gdn.VALUE_DIM
        assert len(norm_weight) == gdn.HEAD_DIM
        assert len(z) == gdn.VALUE_DIM
        return [0.125] * gdn.VALUE_DIM


class ProbeState:
    def step(self, state, q, k, v, gate, beta, out):
        for i in range(gdn.VALUE_DIM):
            out[i] = 0.0
        return 0


def main() -> None:
    runtime = ProbeRuntime()
    state_lib = ProbeState()
    prefix = "blk.0"

    suffixes = (
        "attn_qkv.weight",
        "attn_gate.weight",
        "ssm_beta.weight",
        "ssm_alpha.weight",
        "ssm_out.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    )
    metas = {
        f"{prefix}.{suffix}": {"name": f"{prefix}.{suffix}"}
        for suffix in suffixes
    }

    vectors = {
        "attn_norm.weight": [1.0] * gdn.HIDDEN,
        "ssm_dt.bias": [0.0] * gdn.V_HEADS,
        "ssm_a": [0.0] * gdn.V_HEADS,
        "ssm_conv1d.weight": [0.0] * (gdn.CONV_DIM * gdn.CONV_KERNEL),
        "ssm_norm.weight": [1.0] * gdn.HEAD_DIM,
        "post_attention_norm.weight": [1.0] * gdn.HIDDEN,
    }

    def view(suffix: str):
        return suffix

    def vec(suffix: str):
        return vectors[suffix]

    out, qkv = t2.recurrent_step(
        runtime,
        state_lib,
        object(),
        [0.0] * gdn.CONV_DIM,
        view,
        metas,
        vec,
        [0.0] * gdn.HIDDEN,
        0,
        1,
    )

    assert len(out) == gdn.HIDDEN
    assert len(qkv) == gdn.CONV_DIM
    assert runtime.conv_calls == 1, (
        f"recurrent_step must delegate conv+SiLU exactly once, calls={runtime.conv_calls}"
    )
    assert runtime.norm_gate_calls == 1, (
        "recurrent_step must delegate state norm+gate exactly once, "
        f"calls={runtime.norm_gate_calls}"
    )
    assert runtime.rms_norm_calls == 2, (
        "recurrent_step must delegate both layer RMSNorms to native runtime, "
        f"calls={runtime.rms_norm_calls}"
    )
    assert runtime.residual_add_calls == 2, (
        "recurrent_step must delegate both residual adds to native runtime, "
        f"calls={runtime.residual_add_calls}"
    )
    assert runtime.gdn_repeat_scale_calls == 1, (
        "recurrent_step must delegate GDN q/k repeat-scale exactly once, "
        f"calls={runtime.gdn_repeat_scale_calls}"
    )
    assert runtime.ssm_out_input == [0.125] * gdn.VALUE_DIM, (
        "ssm_out must consume native norm+gate output"
    )
    assert runtime.ffn_calls == 1, (
        f"recurrent path must delegate FFN exactly once, calls={runtime.ffn_calls}"
    )

    print("QWEN38_BONSAI2_GDN_VECTOR_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
