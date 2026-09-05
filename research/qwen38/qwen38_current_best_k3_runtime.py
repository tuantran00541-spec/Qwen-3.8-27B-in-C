#!/usr/bin/env python3
"""Promoted Linux experimental current-best K3 policy for Qwen3.8.

The policy is the exact progressive execution-ordered two-slot reader proven by
same-run warm ABBA.  It changes only K3 packing/read readiness: model tensor
bytes, decoder arithmetic, ring residency, and storage I/O concurrency remain
unchanged.
"""
from __future__ import annotations

import sys

import qwen35_k3_generate as gen
from qwen38_k3_progressive import (
    K3_STREAM_BYTES,
    LAYOUT_POLICY,
    ProgressiveK3Trunk,
    pack_gguf_layers_progressive,
    synthetic_sanity,
)

CURRENT_BEST_K3_LAYOUT_POLICY = LAYOUT_POLICY
CURRENT_BEST_K3_RING_SLOTS = 2
CURRENT_BEST_K3_PLANNED_BYTES = 672_899_072
CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY = 1
CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS = 1

PROOF_PROGRESSIVE_ABBA_RUN = 33867794166
PROOF_PROGRESSIVE_ABBA_COMMIT = "08053d5bab238c21804b64b47b509f00d2c09f0a"
PROOF_PROGRESSIVE_ABBA_ARTIFACT_ID = 9935226379
PROOF_PROGRESSIVE_ABBA_ARTIFACT_SHA256 = (
    "97e4c5b49c77ca1dcd8659fcd703af43201f242e9946d6c19b16f9fafe1fc8d7"
)
PROOF_PROGRESSIVE_ABBA_SPEEDUP = 1.0117934584383415
PROOF_PROGRESSIVE_ABBA_SECONDS_SAVED = 0.6040127894999046


class Qwen38CurrentBestK3Overlay:
    """Temporarily select the promoted progressive K3 packer/reader."""

    def __init__(self) -> None:
        self._active = False
        self._old_pack = None
        self._old_reader = None

    def __enter__(self):
        if not sys.platform.startswith("linux"):
            raise RuntimeError(
                "promoted progressive K3 current-best is Linux experimental; "
                "Windows production reader is not promoted yet")
        if self._active:
            raise RuntimeError("current-best K3 overlay is already active")
        self._old_pack = gen.pack_gguf_layers
        self._old_reader = gen.K3Trunk
        gen.pack_gguf_layers = pack_gguf_layers_progressive
        gen.K3Trunk = ProgressiveK3Trunk
        self._active = True
        return self

    def close(self) -> None:
        if not self._active:
            return
        assert self._old_pack is not None
        assert self._old_reader is not None
        gen.pack_gguf_layers = self._old_pack
        gen.K3Trunk = self._old_reader
        self._active = False

    def report(self) -> dict:
        return {
            "platform": sys.platform,
            "layout_policy": CURRENT_BEST_K3_LAYOUT_POLICY,
            "progressive_readiness": True,
            "ring_slots": CURRENT_BEST_K3_RING_SLOTS,
            "planned_bytes": CURRENT_BEST_K3_PLANNED_BYTES,
            "storage_io_concurrency": CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY,
            "max_deferred_layer_requests": CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS,
            "k3_stream_bytes": K3_STREAM_BYTES,
            "tensor_byte_change": False,
            "arithmetic_change": False,
            "proof_progressive_abba_run": PROOF_PROGRESSIVE_ABBA_RUN,
            "proof_progressive_abba_commit": PROOF_PROGRESSIVE_ABBA_COMMIT,
            "proof_progressive_abba_artifact_id": PROOF_PROGRESSIVE_ABBA_ARTIFACT_ID,
            "proof_progressive_abba_artifact_sha256": PROOF_PROGRESSIVE_ABBA_ARTIFACT_SHA256,
            "proof_progressive_abba_speedup": PROOF_PROGRESSIVE_ABBA_SPEEDUP,
            "proof_progressive_abba_seconds_saved": PROOF_PROGRESSIVE_ABBA_SECONDS_SAVED,
        }

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def sanity() -> None:
    if CURRENT_BEST_K3_RING_SLOTS != 2:
        raise SystemExit("current-best K3 ring-slot contract changed")
    if CURRENT_BEST_K3_PLANNED_BYTES != 672_899_072:
        raise SystemExit("current-best K3 residency contract changed")
    if CURRENT_BEST_K3_STORAGE_IO_CONCURRENCY != 1:
        raise SystemExit("current-best K3 storage I/O concurrency changed")
    if CURRENT_BEST_K3_MAX_DEFERRED_LAYER_REQUESTS != 1:
        raise SystemExit("current-best K3 deferred-request contract changed")
    if PROOF_PROGRESSIVE_ABBA_RUN != 33867794166:
        raise SystemExit("current-best K3 promotion proof changed")
    synthetic_sanity()
    print("QWEN38_CURRENT_BEST_K3_RUNTIME_SANITY PASS")


if __name__ == "__main__":
    sanity()
