#!/usr/bin/env python3
"""Measurement-only K3 direct-I/O service profile on promoted Qwen3.8 current-best.

The decoder/runtime is unchanged.  During one exact 11-token layer-major pass
this probe temporarily instruments:
  * K3Trunk._load_layer() -- total service wall time per layer;
  * os.preadv()/os.pread() -- low-level read call count/bytes/wall time;
  * K3Trunk.bind() -- main-thread visible wait per requested layer.

This separates storage service time from the portion already hidden behind
compute.  Ring slots, prefetch depth, direct-I/O policy, model bytes, arithmetic,
and worker counts are not changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import threading
import time
from typing import Any

import k3_stream
import qwen35_gdn_quant_layer_gate as gdn
import qwen35_k3_generate as gen
import qwen38_k3_pair_reuse_probe as pair
import qwen38_k3_prompt_block_prefill_probe as block
import qwen38_q6_persistent_pool_exact_probe as q6
from native_f32_runtime import enable_native_f32
from qwen38_current_best_runtime import (
    CURRENT_BEST_Q6_WORKERS,
    Qwen38CurrentBestQuantStack,
)

PROMPT_IDS = list(q6.PROMPT_IDS)
K3_STREAM_BYTES = q6.K3_STREAM_BYTES
KNOWN_HIDDEN_SHA256 = q6.KNOWN_HIDDEN_SHA256
KNOWN_STATE_SHA256 = q6.KNOWN_STATE_SHA256
EXPECTED_READER = dict(q6.EXPECTED_READER)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def sanity() -> None:
    from qwen38_current_best_runtime import sanity as runtime_sanity

    runtime_sanity()
    q6.sanity()
    if not hasattr(k3_stream.os, "preadv"):
        raise SystemExit("K3 I/O service profile requires os.preadv on Linux")
    if CURRENT_BEST_Q6_WORKERS != 2:
        raise SystemExit("K3 I/O profile requires promoted Q6 pool2 current-best")
    print("QWEN38_K3_IO_SERVICE_PROFILE_SANITY PASS")


class IOInstrumentation:
    def __init__(self, reader) -> None:
        self.reader = reader
        self.lock = threading.Lock()
        self.local = threading.local()
        self.layers: dict[int, dict[str, Any]] = defaultdict(lambda: {
            "read_bytes": 0,
            "load_calls": 0,
            "load_seconds": 0.0,
            "preadv_calls": 0,
            "preadv_requested_bytes": 0,
            "preadv_bytes": 0,
            "preadv_seconds": 0.0,
            "pread_calls": 0,
            "pread_requested_bytes": 0,
            "pread_bytes": 0,
            "pread_seconds": 0.0,
            "bind_calls": 0,
            "bind_seconds": 0.0,
            "bind_pending_calls": 0,
            "bind_pending_seconds": 0.0,
        })
        self.original_load = reader._load_layer
        self.original_bind = reader.bind
        self.original_preadv = getattr(k3_stream.os, "preadv", None)
        self.original_pread = k3_stream.os.pread
        self.installed = False

    def _layer(self) -> int:
        value = getattr(self.local, "layer", None)
        return -1 if value is None else int(value)

    def install(self) -> None:
        if self.installed:
            raise RuntimeError("I/O instrumentation already installed")
        self.installed = True

        def timed_load(layer: int, target):
            il = int(layer)
            previous = getattr(self.local, "layer", None)
            self.local.layer = il
            t0 = time.monotonic()
            try:
                got = self.original_load(il, target)
            finally:
                dt = time.monotonic() - t0
                with self.lock:
                    rec = self.layers[il]
                    rec["load_calls"] += 1
                    rec["load_seconds"] += dt
                self.local.layer = previous
            with self.lock:
                self.layers[il]["read_bytes"] += int(got)
            return got

        def timed_bind(layer: int):
            il = int(layer)
            pending = getattr(self.reader, "_pending", None)
            pending_requested = bool(pending is not None and int(pending[0]) == il)
            t0 = time.monotonic()
            out = self.original_bind(il)
            dt = time.monotonic() - t0
            with self.lock:
                rec = self.layers[il]
                rec["bind_calls"] += 1
                rec["bind_seconds"] += dt
                if pending_requested:
                    rec["bind_pending_calls"] += 1
                    rec["bind_pending_seconds"] += dt
            return out

        def timed_preadv(fd, buffers, offset, *args, **kwargs):
            il = self._layer()
            requested = sum(len(buf) for buf in buffers)
            t0 = time.monotonic()
            got = self.original_preadv(fd, buffers, offset, *args, **kwargs)
            dt = time.monotonic() - t0
            with self.lock:
                rec = self.layers[il]
                rec["preadv_calls"] += 1
                rec["preadv_requested_bytes"] += int(requested)
                rec["preadv_bytes"] += int(got)
                rec["preadv_seconds"] += dt
            return got

        def timed_pread(fd, n, offset):
            il = self._layer()
            t0 = time.monotonic()
            data = self.original_pread(fd, n, offset)
            dt = time.monotonic() - t0
            with self.lock:
                rec = self.layers[il]
                rec["pread_calls"] += 1
                rec["pread_requested_bytes"] += int(n)
                rec["pread_bytes"] += len(data)
                rec["pread_seconds"] += dt
            return data

        self.reader._load_layer = timed_load
        self.reader.bind = timed_bind
        k3_stream.os.preadv = timed_preadv
        k3_stream.os.pread = timed_pread

    def restore(self) -> None:
        if not self.installed:
            return
        self.reader._load_layer = self.original_load
        self.reader.bind = self.original_bind
        if self.original_preadv is not None:
            k3_stream.os.preadv = self.original_preadv
        k3_stream.os.pread = self.original_pread
        self.installed = False

    def report(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for layer in sorted(k for k in self.layers if k >= 0):
            raw = dict(self.layers[layer])
            service = float(raw["load_seconds"])
            bind = float(raw["bind_seconds"])
            read_bytes = int(raw["read_bytes"])
            preadv_seconds = float(raw["preadv_seconds"])
            throughput_gib_s = (
                (read_bytes / (1024.0 ** 3)) / service if service > 0 else None
            )
            preadv_gib_s = (
                (int(raw["preadv_bytes"]) / (1024.0 ** 3)) / preadv_seconds
                if preadv_seconds > 0 else None
            )
            rows.append({
                "layer": layer,
                **raw,
                "service_gib_s": throughput_gib_s,
                "preadv_gib_s": preadv_gib_s,
                "estimated_io_hidden_by_compute_seconds": max(0.0, service - bind),
                "visible_bind_fraction_of_service": (
                    bind / service if service > 0 else None
                ),
            })

        total_service = sum(float(r["load_seconds"]) for r in rows)
        total_bind = sum(float(r["bind_seconds"]) for r in rows)
        total_read = sum(int(r["read_bytes"]) for r in rows)
        total_preadv_time = sum(float(r["preadv_seconds"]) for r in rows)
        total_preadv_bytes = sum(int(r["preadv_bytes"]) for r in rows)
        total_calls = sum(int(r["preadv_calls"]) for r in rows)
        total_pread_calls = sum(int(r["pread_calls"]) for r in rows)
        hidden = sum(float(r["estimated_io_hidden_by_compute_seconds"]) for r in rows)
        ranked_service = sorted(rows, key=lambda r: float(r["load_seconds"]), reverse=True)
        ranked_bind = sorted(rows, key=lambda r: float(r["bind_seconds"]), reverse=True)
        return {
            "layers": rows,
            "layer_count": len(rows),
            "read_bytes": total_read,
            "load_service_seconds_total": total_service,
            "visible_bind_seconds_total": total_bind,
            "estimated_io_hidden_by_compute_seconds_total": hidden,
            "preadv_calls_total": total_calls,
            "preadv_bytes_total": total_preadv_bytes,
            "preadv_seconds_total": total_preadv_time,
            "pread_calls_total": total_pread_calls,
            "aggregate_service_gib_s": (
                (total_read / (1024.0 ** 3)) / total_service if total_service > 0 else None
            ),
            "aggregate_preadv_gib_s": (
                (total_preadv_bytes / (1024.0 ** 3)) / total_preadv_time
                if total_preadv_time > 0 else None
            ),
            "visible_bind_fraction_of_service": (
                total_bind / total_service if total_service > 0 else None
            ),
            "top_service_layers": ranked_service[:12],
            "top_visible_bind_layers": ranked_bind[:12],
        }


def run(args) -> dict[str, Any]:
    engine = gen.StatefulK3Generator(
        args.model, args.quant_lib, args.state_lib, args.inventory, args.work_dir)
    started = time.monotonic()
    stack: Qwen38CurrentBestQuantStack | None = None
    instrumentation: IOInstrumentation | None = None
    try:
        native_f32 = enable_native_f32(engine, args.f32_lib)
        stack = Qwen38CurrentBestQuantStack(
            engine,
            args.pool_lib,
            q6_workers=CURRENT_BEST_Q6_WORKERS,
            max_vec=len(PROMPT_IDS),
        )
        runtime = stack.runtime
        if runtime is None:
            raise RuntimeError("current-best quant runtime did not initialize")

        if not bool(engine.reader.direct_io):
            raise RuntimeError("K3 I/O service profile requires direct I/O before run")
        instrumentation = IOInstrumentation(engine.reader)
        instrumentation.install()
        try:
            rec = q6._run_stack(engine, args, runtime)
        finally:
            instrumentation.restore()

        hidden_sha = block._digest_hidden_rows(rec["hidden"])
        state_sha = pair.snapshot_digest(pair.capture_state(engine))
        fast_report = stack.fast_quant.report()
        q6._check_common("k3-io-service-profile", rec, fast_report)
        if hidden_sha != KNOWN_HIDDEN_SHA256:
            raise RuntimeError(f"hidden anchor changed: {hidden_sha}")
        if state_sha != KNOWN_STATE_SHA256:
            raise RuntimeError(f"state anchor changed: {state_sha}")

        io_report = instrumentation.report()
        if int(io_report["layer_count"]) != 64:
            raise RuntimeError(f"expected 64 measured layers, got {io_report['layer_count']}")
        if int(io_report["read_bytes"]) != K3_STREAM_BYTES:
            raise RuntimeError(
                f"instrumented read bytes={io_report['read_bytes']} expected={K3_STREAM_BYTES}")
        if int(io_report["preadv_bytes_total"]) != K3_STREAM_BYTES:
            raise RuntimeError(
                f"preadv bytes={io_report['preadv_bytes_total']} expected={K3_STREAM_BYTES}")
        if int(io_report["pread_calls_total"]) != 0:
            raise RuntimeError(
                f"direct-I/O profile unexpectedly used os.pread {io_report['pread_calls_total']} times")

        reader = engine.reader.report()
        if not bool(reader.get("direct_io")):
            raise RuntimeError("reader fell back from direct I/O during profile")
        for key, expected in EXPECTED_READER.items():
            if int(reader.get(key, -1)) != int(expected):
                raise RuntimeError(f"reader {key}={reader.get(key)} expected={expected}")
        if int(reader.get("bytes_read", -1)) != K3_STREAM_BYTES:
            raise RuntimeError(
                f"reader bytes_read={reader.get('bytes_read')} expected={K3_STREAM_BYTES}")

        pool_report = runtime.pool_report()
        if int(pool_report["threads"]) != CURRENT_BEST_Q6_WORKERS:
            raise RuntimeError(f"unexpected Q6 pool report {pool_report}")

        service = float(io_report["load_service_seconds_total"])
        bind = float(io_report["visible_bind_seconds_total"])
        if bind > service + 0.25:
            raise RuntimeError(
                f"visible bind time {bind} unexpectedly exceeds service time {service}")

        payload = {
            "schema": "qwen38-k3-io-service-profile-v1",
            "status": "PASS",
            "claim": "measurement-only direct-I/O syscall/service/bind profile on promoted exact Q6-pool2 current-best path",
            "model_sha256": gdn.SHA256,
            "prompt_token_count": len(PROMPT_IDS),
            "hidden_sha256": hidden_sha,
            "state_sha256": state_sha,
            "prefill_seconds": rec["seconds"],
            "k3_bytes": rec["k3_bytes"],
            "q6_boundary_seconds": rec["q6_timing"]["seconds"],
            "io": io_report,
            "q6_pool": pool_report,
            "fast_quantize": fast_report,
            "native_f32": native_f32.report(),
            "reader": reader,
            "max_rss_gib": rss_gib(),
            "elapsed_seconds": time.monotonic() - started,
            "optimization": {
                "measurement_only": True,
                "reader_code_change": False,
                "reader_policy_change": False,
                "ring_change": False,
                "prefetch_depth_change": False,
                "q6_worker_change": False,
                "arithmetic_change": False,
            },
            "baseline_current_best_q6_profile_run": 33855832226,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("QWEN38_K3_IO_SERVICE_PROFILE_REAL_BITWISE_PASS")
        return payload
    finally:
        if instrumentation is not None:
            instrumentation.restore()
        if stack is not None:
            stack.close()
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("sanity")
    r = sub.add_parser("run")
    for name in (
        "model", "quant-lib", "pool-lib", "state-lib", "batch-state-lib",
        "f32-lib", "attn-lib", "conv-lib", "gate-lib", "swiglu-lib",
        "rmsnorm-lib", "rmsnorm-many-lib", "head-rmsnorm-many-lib",
        "residual-lib", "attention-gate-lib", "repeat-lib", "inventory",
        "work-dir", "output",
    ):
        r.add_argument(f"--{name}", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "sanity":
        sanity()
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
