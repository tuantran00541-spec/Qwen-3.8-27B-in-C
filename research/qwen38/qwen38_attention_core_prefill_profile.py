#!/usr/bin/env python3
"""44-token staged-prefill profile with the exact native attention core enabled.

This is a performance profile, not a new semantic baseline.  The native core has
already passed the real 11-token hidden/state bitwise A/B gate.  Here we keep the
same staged matvec-many schedule and profiler, replace only the full-attention
causal score/softmax/V accumulation, and measure whether the O(n^2) Python work
shrinks at a longer prompt.

The ExactAttentionCore wrapper still mirrors canonical F16-rounded K/V into an
auxiliary contiguous F32 probe cache.  That duplicate cache is evidence-only and
must not be promoted as the production long-context layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import qwen38_attention_core_exact_probe as attn_probe
import qwen38_k3_prompt_block_many_probe as prefill_many
import qwen38_prefill_layer_profile as profile
from attention_core_runtime import ExactAttentionCore


def run(args):
    core = ExactAttentionCore(args.attn_lib)
    stats = {"native_attention_core_seconds": 0.0}
    original_attention = prefill_many._full_attention_layer_many

    def native_attention(engine, hidden, il, pos0, view, metas, vec):
        return attn_probe._full_attention_layer_native(
            engine, hidden, il, pos0, view, metas, vec, core, stats)

    # profile.run installs its own timing wrapper around whatever attention
    # implementation is present at this point, so patch before calling it.
    prefill_many._full_attention_layer_many = native_attention
    try:
        payload = profile.run(args)
    finally:
        prefill_many._full_attention_layer_many = original_attention

    payload["schema"] = "qwen38-attention-core-prefill-profile-v1"
    payload["status"] = "COMPLETE"
    payload["candidate"] = "exact-native-attention-core"
    payload["native_attention_core_seconds"] = stats["native_attention_core_seconds"]
    payload["native_attention_core"] = core.report()
    payload["comparison_note"] = (
        "Candidate-only 44-token hosted profile. Compare with the earlier 44-token "
        "reference only directionally because runner hardware may differ; this is "
        "not a same-run speedup claim."
    )
    payload["probe_cache_note"] = (
        "Auxiliary contiguous F32 K/V mirror is evidence-only; production must "
        "consume canonical packed F16 cache directly to preserve low-RAM behavior."
    )
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    breakdown = payload["breakdown"]
    print(json.dumps({
        "status": payload["status"],
        "profile_token_count": payload["profile_token_count"],
        "prefill_seconds": payload["prefill_seconds"],
        "gdn_seconds": breakdown["gdn"]["seconds"],
        "gdn_non_matvec_seconds": breakdown["gdn"]["non_matvec_seconds"],
        "attention_seconds": breakdown["attention"]["seconds"],
        "attention_non_matvec_seconds": breakdown["attention"]["non_matvec_seconds"],
        "native_attention_core_seconds": payload["native_attention_core_seconds"],
        "native_attention_core": payload["native_attention_core"],
        "k3_bytes": payload["k3_bytes"],
        "max_rss_gib": payload["max_rss_gib"],
    }, indent=2))
    print("QWEN38_ATTENTION_CORE_PREFILL_PROFILE_COMPLETE")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--quant-lib", type=Path, required=True)
    ap.add_argument("--many-lib", type=Path, required=True)
    ap.add_argument("--state-lib", type=Path, required=True)
    ap.add_argument("--f32-lib", type=Path, required=True)
    ap.add_argument("--attn-lib", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--profile-tokens", type=int, default=44)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
