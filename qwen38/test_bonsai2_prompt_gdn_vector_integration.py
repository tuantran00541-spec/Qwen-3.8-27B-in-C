#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

import bonsai2_prompt_spike as spike
import qwen35_gdn_quant_layer_gate as gdn


class ProbeRuntime:
    def __init__(self) -> None:
        self.conv_calls = 0
        self.norm_gate_calls = 0
        self.history_lengths: list[int] = []
        self.ssm_out_input = None
        self.rms_norm_calls = 0
        self.residual_add_calls = 0
        self.gdn_repeat_scale_calls = 0
        self.recurrent_mid_calls = 0

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
            q_out.extend(spike.mulf(v, scale) for v in q)
            k_out.extend(float(v) for v in k)
        return q_out, k_out

    def residual_add(self, a, b):
        self.residual_add_calls += 1
        assert len(a) == len(b)
        return [float(a[i]) + float(b[i]) for i in range(len(a))]

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

    def recurrent_mid(
        self,
        state_lib,
        state,
        qkv,
        history,
        kernels,
        alpha,
        beta_raw,
        dt,
        a,
        z,
        norm_weight,
        *,
        eps,
        scale,
    ):
        self.recurrent_mid_calls += 1
        assert len(qkv) == gdn.CONV_DIM
        assert len(history) == 3
        assert kernels == "ssm_conv1d.weight"
        assert len(alpha) == gdn.V_HEADS
        assert len(beta_raw) == gdn.V_HEADS
        assert dt == "ssm_dt.bias"
        assert a == "ssm_a"
        assert len(z) == gdn.VALUE_DIM
        assert norm_weight == "ssm_norm.weight"
        assert eps == gdn.RMS_EPS
        assert scale == spike.t2.SCALE_GDN
        return [0.25] * gdn.VALUE_DIM

    def gdn_conv_silu(self, qkv, history, kernels):
        self.conv_calls += 1
        self.history_lengths.append(len(history))
        assert len(qkv) == gdn.CONV_DIM
        assert len(history) == 3
        assert all(len(row) == gdn.CONV_DIM for row in history)
        assert len(kernels) == gdn.CONV_DIM * gdn.CONV_KERNEL
        return [0.0] * gdn.CONV_DIM

    def gdn_norm_gate(self, core, norm_weight, z, *, eps=1e-6):
        self.norm_gate_calls += 1
        assert len(core) == gdn.VALUE_DIM
        assert len(norm_weight) == gdn.HEAD_DIM
        assert len(z) == gdn.VALUE_DIM
        return [0.25] * gdn.VALUE_DIM


class ProbeState:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, state, q, k, v, gate, beta, out):
        self.calls += 1
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

    history = [
        [0.0] * gdn.CONV_DIM,
        [0.0] * gdn.CONV_DIM,
        [0.0] * gdn.CONV_DIM,
    ]

    out, qkv = spike.recurrent_step(
        runtime,
        state_lib,
        object(),
        history,
        view,
        metas,
        vec,
        [0.0] * gdn.HIDDEN,
        0,
    )

    assert len(out) == gdn.HIDDEN
    assert len(qkv) == gdn.CONV_DIM
    assert runtime.recurrent_mid_calls == 1, (
        "prompt recurrent_step must delegate the recurrent middle path "
        f"exactly once, calls={runtime.recurrent_mid_calls}"
    )
    assert runtime.conv_calls == 0, (
        "prompt recurrent_step must not call standalone conv after mid fusion, "
        f"calls={runtime.conv_calls}"
    )
    assert runtime.history_lengths == [], runtime.history_lengths
    assert runtime.norm_gate_calls == 0, (
        "prompt recurrent_step must not call standalone norm+gate after mid fusion, "
        f"calls={runtime.norm_gate_calls}"
    )
    assert state_lib.calls == 0, (
        "prompt recurrent_step must not re-enter Python state step after mid fusion"
    )
    assert runtime.rms_norm_calls == 2, (
        "prompt recurrent_step must delegate both layer RMSNorms to native runtime, "
        f"calls={runtime.rms_norm_calls}"
    )
    assert runtime.residual_add_calls == 2, (
        "prompt recurrent_step must delegate both residual adds to native runtime, "
        f"calls={runtime.residual_add_calls}"
    )
    assert runtime.gdn_repeat_scale_calls == 0, (
        "prompt recurrent_step must not call standalone repeat-scale after mid fusion, "
        f"calls={runtime.gdn_repeat_scale_calls}"
    )
    assert runtime.ssm_out_input == [0.25] * gdn.VALUE_DIM

    print("QWEN38_BONSAI2_PROMPT_GDN_VECTOR_INTEGRATION_PASS")


if __name__ == "__main__":
    main()
