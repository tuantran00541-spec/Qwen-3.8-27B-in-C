#!/usr/bin/env python3
"""Opt-in aggregate profiler for the cached Qwen3.8 K3 generator.

This file deliberately does not replace the proven generator.  Without
``--profile`` it delegates to the existing cached-trunk path.  With profiling
enabled it wraps the same semantic functions and native ABI while collecting
fixed-size aggregate counters.  Timing calls never alter tensor math, model
state, cache precision, quantization or token selection.

Main-thread wall categories are additive at the top level:
  engine_init + decoder_steps + logits + topk + frontend/other ~= total.
Nested layer/kernel/storage counters are reported separately and MUST NOT be
summed with the top-level categories.  In particular async K3 read service can
overlap layer compute and is explicitly labelled non-additive.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Sequence

import qwen35_k3_generate as gen
import qwen35_k3_generate_cached as cached
from gguf_quant_ref import row_nbytes


BASE_QUANT_RUNTIME = gen.gdn.QuantRuntime
BASE_K3_TRUNK = gen.K3Trunk
BASE_EMBEDDING_ROW = gen.gdn._embedding_row
BASE_F32_VECTOR = gen.gdn.f32_vector
BASE_F32_MATVEC = gen.gdn.f32_matvec
BASE_CARR = gen.t2.carr
BASE_READ_F32_TENSOR = gen.base._read_f32_tensor
BASE_STREAM_Q8_LOGITS = gen.base._stream_q8_logits
BASE_TOPK = gen.base._topk
BASE_STATEFUL_GENERATOR = gen.StatefulK3Generator

_ACTIVE_PROFILE: "AggregateProfile | None" = None


def _profile() -> "AggregateProfile":
    if _ACTIVE_PROFILE is None:
        raise RuntimeError("profiling wrapper used without an active profile")
    return _ACTIVE_PROFILE


def _role(meta: dict[str, Any]) -> str:
    name = str(meta.get("name", ""))
    if name == "output.weight":
        return "lm_head"
    if ".ffn_" in name:
        return "ffn"
    if any(tag in name for tag in (".attn_qkv.", ".attn_gate.", ".ssm_beta.", ".ssm_alpha.", ".ssm_out.")):
        return "gdn_projection"
    if any(tag in name for tag in (".attn_q.", ".attn_k.", ".attn_v.", ".attn_output.")):
        return "full_attention_projection"
    return "other"


class AggregateProfile:
    def __init__(self) -> None:
        self.main_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        self.seconds: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.bytes: defaultdict[str, int] = defaultdict(int)
        self.elements: defaultdict[str, int] = defaultdict(int)
        self.role_stats: dict[str, dict[str, float | int]] = {}
        self.layer_wall_seconds = [0.0] * gen.N_LAYER
        self.layer_calls = [0] * gen.N_LAYER

    def add(self, key: str, *, seconds: float = 0.0, count: int = 0,
            nbytes: int = 0, elements: int = 0) -> None:
        with self._lock:
            if seconds:
                self.seconds[key] += float(seconds)
            if count:
                self.counts[key] += int(count)
            if nbytes:
                self.bytes[key] += int(nbytes)
            if elements:
                self.elements[key] += int(elements)

    def add_role(self, role: str, kind: str, *, native_seconds: float,
                 marshal_seconds: float, rows: int, weight_bytes: int) -> None:
        key = f"{role}:{kind}"
        with self._lock:
            stats = self.role_stats.setdefault(key, {
                "calls": 0,
                "rows": 0,
                "weight_bytes": 0,
                "native_seconds": 0.0,
                "marshal_seconds": 0.0,
            })
            stats["calls"] = int(stats["calls"]) + 1
            stats["rows"] = int(stats["rows"]) + int(rows)
            stats["weight_bytes"] = int(stats["weight_bytes"]) + int(weight_bytes)
            stats["native_seconds"] = float(stats["native_seconds"]) + float(native_seconds)
            stats["marshal_seconds"] = float(stats["marshal_seconds"]) + float(marshal_seconds)

    def add_layer(self, layer: int, seconds: float) -> None:
        with self._lock:
            self.layer_wall_seconds[int(layer)] += float(seconds)
            self.layer_calls[int(layer)] += 1

    def snapshot(self, *, total_wall_seconds: float, total_cpu_seconds: float) -> dict[str, Any]:
        with self._lock:
            sec = dict(self.seconds)
            counts = dict(self.counts)
            bts = dict(self.bytes)
            elems = dict(self.elements)
            roles = {k: dict(v) for k, v in self.role_stats.items()}
            layers = list(self.layer_wall_seconds)
            layer_calls = list(self.layer_calls)

        engine_init = sec.get("engine_init_wall", 0.0)
        decoder = sec.get("decoder_step_wall", 0.0)
        logits = sec.get("logits_wall", 0.0)
        topk = sec.get("topk_wall", 0.0)
        accounted = engine_init + decoder + logits + topk
        other = max(0.0, float(total_wall_seconds) - accounted)

        engine_init_cpu = sec.get("engine_init_cpu", 0.0)
        decoder_cpu = sec.get("decoder_step_cpu", 0.0)
        logits_cpu = sec.get("logits_cpu", 0.0)
        topk_cpu = sec.get("topk_cpu", 0.0)
        accounted_cpu = engine_init_cpu + decoder_cpu + logits_cpu + topk_cpu
        other_cpu = max(0.0, float(total_cpu_seconds) - accounted_cpu)

        return {
            "schema": "qwen38-runtime-aggregate-profile-v1",
            "timing_semantics": {
                "main_thread_categories_additive": True,
                "nested_counters_additive_to_main": False,
                "io_service_seconds_nonadditive": True,
                "clock": "time.perf_counter",
                "cpu_clock": "time.process_time",
            },
            "total": {
                "wall_seconds": float(total_wall_seconds),
                "process_cpu_seconds": float(total_cpu_seconds),
            },
            "main_thread_additive": {
                "engine_init_wall_seconds": engine_init,
                "decoder_steps_wall_seconds": decoder,
                "logits_wall_seconds": logits,
                "topk_wall_seconds": topk,
                "frontend_and_other_wall_seconds": other,
                "accounted_wall_seconds": accounted,
                "engine_init_cpu_seconds": engine_init_cpu,
                "decoder_steps_cpu_seconds": decoder_cpu,
                "logits_cpu_seconds": logits_cpu,
                "topk_cpu_seconds": topk_cpu,
                "frontend_and_other_cpu_seconds": other_cpu,
                "accounted_cpu_seconds": accounted_cpu,
            },
            "storage": {
                "k3_logical_bytes": bts.get("k3_read", 0),
                "k3_io_service_seconds_nonadditive": sec.get("k3_io_service", 0.0),
                "k3_worker_io_service_seconds_nonadditive": sec.get("k3_worker_io_service", 0.0),
                "k3_sync_io_service_seconds": sec.get("k3_sync_io_service", 0.0),
                "bind_wall_seconds": sec.get("bind", 0.0),
                "bind_wait_for_prefetch_seconds": sec.get("bind_wait", 0.0),
                "bind_calls": counts.get("bind", 0),
                "bind_wait_for_prefetch_calls": counts.get("bind_wait", 0),
                "prefetch_issued": counts.get("prefetch_issued", 0),
                "prefetch_rejected_or_unneeded": counts.get("prefetch_rejected", 0),
                "prefetch_ready_before_bind": counts.get("prefetch_ready_before_bind", 0),
                "compute_with_prefetch_active_seconds": sec.get("compute_with_prefetch_active", 0.0),
                "lm_head_logical_bytes": bts.get("lm_head_read", 0),
                "lm_head_read_wall_seconds": sec.get("lm_head_read", 0.0),
                "lm_head_read_calls": counts.get("lm_head_read", 0),
                "embedding_logical_bytes": bts.get("embedding_read", 0),
                "embedding_wall_seconds": sec.get("embedding_read", 0.0),
                "global_f32_logical_bytes": bts.get("global_f32_read", 0),
                "global_f32_read_wall_seconds": sec.get("global_f32_read", 0.0),
                "total_profiled_logical_model_bytes": (
                    bts.get("k3_read", 0)
                    + bts.get("lm_head_read", 0)
                    + bts.get("embedding_read", 0)
                    + bts.get("global_f32_read", 0)
                ),
            },
            "nested_compute": {
                "layer_envelope_wall_seconds": sec.get("layer_envelope", 0.0),
                "recurrent_layer_semantic_wall_seconds": sec.get("recurrent_semantic", 0.0),
                "full_attention_semantic_wall_seconds": sec.get("full_attention_semantic", 0.0),
                "gdn_native_state_wall_seconds": sec.get("gdn_native_state", 0.0),
                "gdn_ctypes_input_marshal_wall_seconds": sec.get("gdn_carr", 0.0),
                "history_update_wall_seconds": sec.get("history_update", 0.0),
                "result_norm_wall_seconds": sec.get("result_norm", 0.0),
                "f32_tensor_decode_wall_seconds": sec.get("f32_vector", 0.0),
                "f32_tensor_decode_bytes": bts.get("f32_vector", 0),
                "lm_head_assemble_wall_seconds": sec.get("lm_head_assemble", 0.0),
            },
            "quant_and_matvec": {
                "activation_quantize_native_wall_seconds": sec.get("quant_native", 0.0),
                "activation_input_marshal_wall_seconds": sec.get("quant_input_marshal", 0.0),
                "activation_buffer_alloc_wall_seconds": sec.get("quant_output_alloc", 0.0),
                "activation_quantize_calls": counts.get("quant_native", 0),
                "matvec_native_wall_seconds": sec.get("matvec_native", 0.0),
                "matvec_weight_view_wall_seconds": sec.get("matvec_weight_view", 0.0),
                "matvec_output_alloc_wall_seconds": sec.get("matvec_output_alloc", 0.0),
                "matvec_output_to_list_wall_seconds": sec.get("matvec_output_list", 0.0),
                "matvec_calls": counts.get("matvec_native", 0),
                "matvec_rows": counts.get("matvec_rows", 0),
                "matvec_weight_bytes": bts.get("matvec_weight", 0),
                "f32_matvec_python_wall_seconds": sec.get("f32_matvec", 0.0),
                "by_role_and_weight_kind": roles,
            },
            "call_counts": counts,
            "element_counts": elems,
            "layer_wall_seconds": layers,
            "layer_calls": layer_calls,
        }


class ProfiledQuantRuntime(BASE_QUANT_RUNTIME):
    def quantize(self, x: Sequence[float], kind: str):
        p = _profile()
        n = len(x)
        t = time.perf_counter()
        x_arr = (gen.gdn.ctypes.c_float * n)(*map(float, x))
        p.add("quant_input_marshal", seconds=time.perf_counter() - t, count=1, elements=n)

        if kind == "Q6_K":
            if n % 256:
                raise ValueError("Q6_K activation width must be divisible by 256")
            nbytes = (n // 256) * 292
            fn = self.lib.qwen_quantize_q8_k_scalar
        elif kind == "Q8_0":
            if n % 32:
                raise ValueError("Q8_0 activation width must be divisible by 32")
            nbytes = (n // 32) * 34
            fn = self.lib.qwen_quantize_q8_0_scalar
        else:
            raise ValueError(kind)

        t = time.perf_counter()
        buf = (gen.gdn.ctypes.c_uint8 * nbytes)()
        p.add("quant_output_alloc", seconds=time.perf_counter() - t, count=1, nbytes=nbytes)
        t = time.perf_counter()
        rc = fn(x_arr, n, buf, nbytes)
        native = time.perf_counter() - t
        p.add("quant_native", seconds=native, count=1, nbytes=nbytes, elements=n)
        p.add(f"quant_native_{kind}", seconds=native, count=1, nbytes=nbytes, elements=n)
        if rc != 0:
            raise RuntimeError(f"activation quantization {kind} failed rc={rc}")
        self.activation_quantizations += 1
        return buf, nbytes

    def matvec(self, weights: memoryview, meta: dict[str, Any], x: Sequence[float], prepared=None) -> list[float]:
        p = _profile()
        kind = meta["type_name"]
        ne0, rows = map(int, meta["shape"])
        if len(x) != ne0:
            raise ValueError(f"{meta['name']}: x={len(x)} ne0={ne0}")
        if kind == "F32":
            t = time.perf_counter()
            out = BASE_F32_MATVEC(weights, ne0, rows, x)
            p.add("f32_matvec", seconds=time.perf_counter() - t, count=1,
                  nbytes=len(weights), elements=ne0 * rows)
            return out
        if kind not in {"Q6_K", "Q8_0"}:
            raise ValueError(f"unsupported matvec type {kind}")
        if prepared is None:
            prepared = self.quantize(x, kind)
        activation, activation_bytes = prepared

        t = time.perf_counter()
        w_arr = (gen.gdn.ctypes.c_uint8 * len(weights)).from_buffer(weights)
        weight_view_seconds = time.perf_counter() - t
        p.add("matvec_weight_view", seconds=weight_view_seconds, count=1)

        t = time.perf_counter()
        out = (gen.gdn.ctypes.c_float * rows)()
        output_alloc_seconds = time.perf_counter() - t
        p.add("matvec_output_alloc", seconds=output_alloc_seconds, count=1)

        fn = (self.lib.qwen_matvec_q6_k_q8_k_scalar if kind == "Q6_K"
              else self.lib.qwen_matvec_q8_0_q8_0_scalar)
        t = time.perf_counter()
        rc = fn(w_arr, len(weights), rows, ne0, activation, activation_bytes, out)
        native_seconds = time.perf_counter() - t
        if rc != 0:
            raise RuntimeError(f"{meta['name']}: native matvec failed rc={rc}")

        t = time.perf_counter()
        result = [float(out[i]) for i in range(rows)]
        output_list_seconds = time.perf_counter() - t
        p.add("matvec_output_list", seconds=output_list_seconds, count=1, elements=rows)
        p.add("matvec_native", seconds=native_seconds, count=1,
              nbytes=len(weights), elements=ne0 * rows)
        p.add("matvec_rows", count=rows)
        p.add("matvec_weight", nbytes=len(weights))
        p.add(f"matvec_native_{kind}", seconds=native_seconds, count=1,
              nbytes=len(weights), elements=ne0 * rows)
        p.add_role(
            _role(meta), kind,
            native_seconds=native_seconds,
            marshal_seconds=weight_view_seconds + output_alloc_seconds + output_list_seconds,
            rows=rows,
            weight_bytes=len(weights),
        )
        self.matvec_rows += rows
        return result


class ProfiledK3Trunk(BASE_K3_TRUNK):
    def _load_layer(self, layer: int, target) -> int:
        p = _profile()
        t = time.perf_counter()
        got = super()._load_layer(layer, target)
        elapsed = time.perf_counter() - t
        p.add("k3_io_service", seconds=elapsed, count=1, nbytes=got)
        if threading.get_ident() == p.main_thread_id:
            p.add("k3_sync_io_service", seconds=elapsed, count=1, nbytes=got)
        else:
            p.add("k3_worker_io_service", seconds=elapsed, count=1, nbytes=got)
        p.add("k3_read", nbytes=got)
        return got

    def bind(self, layer: int) -> memoryview:
        p = _profile()
        pending = self._pending
        waits_for_prefetch = False
        if pending is not None and int(pending[0]) == int(layer):
            if pending[2].done():
                p.add("prefetch_ready_before_bind", count=1)
            else:
                waits_for_prefetch = True
        t = time.perf_counter()
        view = super().bind(layer)
        elapsed = time.perf_counter() - t
        p.add("bind", seconds=elapsed, count=1)
        if waits_for_prefetch:
            p.add("bind_wait", seconds=elapsed, count=1)
        return view

    def prefetch(self, layer: int) -> bool:
        issued = super().prefetch(layer)
        _profile().add("prefetch_issued" if issued else "prefetch_rejected", count=1)
        return issued


class ProfiledStateLib:
    def __init__(self, lib) -> None:
        self._lib = lib

    def qwen_gdn_ar_step_f32(self, *args):
        t = time.perf_counter()
        rc = self._lib.qwen_gdn_ar_step_f32(*args)
        _profile().add("gdn_native_state", seconds=time.perf_counter() - t, count=1)
        return rc


def profiled_embedding_row(model: Path, directory, token_id: int) -> list[float]:
    t = time.perf_counter()
    result = BASE_EMBEDDING_ROW(model, directory, token_id)
    elapsed = time.perf_counter() - t
    tensor = directory.by_name()["token_embd.weight"]
    stride = row_nbytes("Q8_0", int(tensor.shape[0]))
    _profile().add("embedding_read", seconds=elapsed, count=1, nbytes=stride)
    return result


def profiled_f32_vector(view: memoryview) -> list[float]:
    t = time.perf_counter()
    out = BASE_F32_VECTOR(view)
    _profile().add("f32_vector", seconds=time.perf_counter() - t, count=1,
                   nbytes=len(view), elements=len(out))
    return out


def profiled_carr(values: Sequence[float]):
    t = time.perf_counter()
    out = BASE_CARR(values)
    _profile().add("gdn_carr", seconds=time.perf_counter() - t, count=1, elements=len(values))
    return out


def profiled_read_f32_tensor(model: Path, tensor) -> list[float]:
    t = time.perf_counter()
    out = BASE_READ_F32_TENSOR(model, tensor)
    _profile().add("global_f32_read", seconds=time.perf_counter() - t, count=1,
                   nbytes=int(tensor.nbytes), elements=len(out))
    return out


def profiled_stream_q8_logits(model: Path, tensor, runtime: BASE_QUANT_RUNTIME,
                              hidden: Sequence[float], chunk_rows: int = gen.base.LM_HEAD_CHUNK_ROWS) -> list[float]:
    if tensor.type_name != "Q8_0":
        raise ValueError(f"{tensor.name}: expected Q8_0, got {tensor.type_name}")
    ne0, rows = map(int, tensor.shape)
    if ne0 != gen.base.HIDDEN or rows != gen.base.VOCAB:
        raise ValueError(f"unexpected LM head shape {list(tensor.shape)}")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    stride = row_nbytes("Q8_0", ne0)
    if stride * rows != tensor.nbytes:
        raise ValueError("LM head Q8_0 byte geometry mismatch")

    prepared = runtime.quantize(hidden, "Q8_0")
    logits: list[float] = []
    fd = os.open(model, os.O_RDONLY)
    try:
        for row0 in range(0, rows, chunk_rows):
            nrows = min(chunk_rows, rows - row0)
            nbytes = nrows * stride
            t = time.perf_counter()
            raw = bytearray(os.pread(fd, nbytes, tensor.data_offset + row0 * stride))
            read_seconds = time.perf_counter() - t
            _profile().add("lm_head_read", seconds=read_seconds, count=1, nbytes=nbytes)
            if len(raw) != nbytes:
                raise EOFError(f"short LM-head read at row {row0}")
            weight_view = memoryview(raw)
            meta = {"name": tensor.name, "type_name": "Q8_0", "shape": [ne0, nrows]}
            chunk = runtime.matvec(weight_view, meta, hidden, prepared)
            t = time.perf_counter()
            logits.extend(chunk)
            _profile().add("lm_head_assemble", seconds=time.perf_counter() - t,
                           count=1, elements=len(chunk))
            weight_view.release()
    finally:
        os.close(fd)
    return logits


def profiled_topk(values: Sequence[float], k: int = 10):
    t_wall = time.perf_counter()
    t_cpu = time.process_time()
    out = BASE_TOPK(values, k)
    _profile().add("topk_wall", seconds=time.perf_counter() - t_wall, count=1)
    _profile().add("topk_cpu", seconds=time.process_time() - t_cpu, count=1)
    return out


class ProfiledStatefulK3Generator(cached.CachedStatefulK3Generator):
    def __init__(self, *args, **kwargs) -> None:
        t_wall = time.perf_counter()
        t_cpu = time.process_time()
        super().__init__(*args, **kwargs)
        self.state_lib = ProfiledStateLib(self.state_lib)
        _profile().add("engine_init_wall", seconds=time.perf_counter() - t_wall, count=1)
        _profile().add("engine_init_cpu", seconds=time.process_time() - t_cpu, count=1)

    def step(self, token_id: int) -> list[float]:
        p = _profile()
        t_step_wall = time.perf_counter()
        t_step_cpu = time.process_time()
        hidden = gen.gdn._embedding_row(self.model, self.directory, int(token_id))
        pos = self.position
        for il in range(gen.N_LAYER):
            layer_start = time.perf_counter()
            bound = self.reader.bind(il)
            issued = False
            if il + 1 < gen.N_LAYER:
                issued = self.reader.prefetch(il + 1)
            metas = gen.base._layer_meta(self.manifest, il)
            prefix = f"blk.{il}"

            def view(suffix: str):
                return self.reader.tensor_view(bound, f"{prefix}.{suffix}")

            def vec(suffix: str):
                return gen.gdn.f32_vector(view(suffix))

            compute_start = time.perf_counter()
            if il % 4 == 3:
                t = time.perf_counter()
                hidden = gen.full_attn_step(
                    self.runtime, self.caches[il], view, metas, vec, hidden, il, pos
                )
                p.add("full_attention_semantic", seconds=time.perf_counter() - t, count=1)
            else:
                t = time.perf_counter()
                hidden, qkv = gen.recurrent_step(
                    self.runtime,
                    self.state_lib,
                    self.states[il],
                    self.conv_history[il],
                    view,
                    metas,
                    vec,
                    hidden,
                    il,
                )
                p.add("recurrent_semantic", seconds=time.perf_counter() - t, count=1)
                t = time.perf_counter()
                hist = self.conv_history[il]
                hist.append(array("f", qkv))
                if len(hist) > 3:
                    del hist[0]
                p.add("history_update", seconds=time.perf_counter() - t, count=1)
            compute_seconds = time.perf_counter() - compute_start
            if issued or getattr(self.reader, "_pending", None) is not None:
                p.add("compute_with_prefetch_active", seconds=compute_seconds, count=1)
            bound.release()
            layer_elapsed = time.perf_counter() - layer_start
            p.add("layer_envelope", seconds=layer_elapsed, count=1)
            p.add_layer(il, layer_elapsed)
        self.position += 1
        p.add("decoder_step_wall", seconds=time.perf_counter() - t_step_wall, count=1)
        p.add("decoder_step_cpu", seconds=time.process_time() - t_step_cpu, count=1)
        return hidden

    def logits(self, hidden: Sequence[float]) -> list[float]:
        p = _profile()
        t_wall = time.perf_counter()
        t_cpu = time.process_time()
        t = time.perf_counter()
        result_norm = gen.gdn.rms_norm(hidden, self.output_norm_w)
        p.add("result_norm", seconds=time.perf_counter() - t, count=1)
        out = gen.base._stream_q8_logits(
            self.model, self.tensors["output.weight"], self.runtime, result_norm
        )
        p.add("logits_wall", seconds=time.perf_counter() - t_wall, count=1)
        p.add("logits_cpu", seconds=time.process_time() - t_cpu, count=1)
        return out


def _install_profile_wrappers() -> None:
    gen.gdn.QuantRuntime = ProfiledQuantRuntime
    gen.K3Trunk = ProfiledK3Trunk
    gen.gdn._embedding_row = profiled_embedding_row
    gen.gdn.f32_vector = profiled_f32_vector
    gen.t2.carr = profiled_carr
    gen.base._read_f32_tensor = profiled_read_f32_tensor
    gen.base._stream_q8_logits = profiled_stream_q8_logits
    gen.base._topk = profiled_topk
    gen.StatefulK3Generator = ProfiledStatefulK3Generator


def _restore_wrappers() -> None:
    gen.gdn.QuantRuntime = BASE_QUANT_RUNTIME
    gen.K3Trunk = BASE_K3_TRUNK
    gen.gdn._embedding_row = BASE_EMBEDDING_ROW
    gen.gdn.f32_vector = BASE_F32_VECTOR
    gen.t2.carr = BASE_CARR
    gen.base._read_f32_tensor = BASE_READ_F32_TENSOR
    gen.base._stream_q8_logits = BASE_STREAM_Q8_LOGITS
    gen.base._topk = BASE_TOPK
    gen.StatefulK3Generator = BASE_STATEFUL_GENERATOR


def run_generator(*, model: Path, native_lib: Path, state_lib: Path, inventory: Path,
                  tokenizer_json: Path, prompt: str, raw_prompt: bool,
                  max_new_tokens: int, max_prompt_tokens: int, work_dir: Path,
                  output: Path, enable_profile: bool, profile_output: Path | None) -> dict[str, Any]:
    global _ACTIVE_PROFILE
    if not enable_profile:
        previous = gen.StatefulK3Generator
        gen.StatefulK3Generator = cached.CachedStatefulK3Generator
        try:
            return gen.generate(
                model, native_lib, state_lib, inventory, tokenizer_json, prompt,
                raw_prompt, max_new_tokens, max_prompt_tokens, work_dir, output,
            )
        finally:
            gen.StatefulK3Generator = previous

    profile = AggregateProfile()
    _ACTIVE_PROFILE = profile
    total_cpu_start = time.process_time()
    _install_profile_wrappers()
    try:
        result = gen.generate(
            model, native_lib, state_lib, inventory, tokenizer_json, prompt,
            raw_prompt, max_new_tokens, max_prompt_tokens, work_dir, output,
        )
        total_cpu = time.process_time() - total_cpu_start
        snapshot = profile.snapshot(
            total_wall_seconds=float(result["elapsed_seconds"]),
            total_cpu_seconds=total_cpu,
        )
        result["profile"] = snapshot
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if profile_output is not None:
            profile_output.parent.mkdir(parents=True, exist_ok=True)
            profile_output.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "profile_schema": snapshot["schema"],
            "total_wall_seconds": snapshot["total"]["wall_seconds"],
            "decoder_wall_seconds": snapshot["main_thread_additive"]["decoder_steps_wall_seconds"],
            "logits_wall_seconds": snapshot["main_thread_additive"]["logits_wall_seconds"],
            "k3_bytes": snapshot["storage"]["k3_logical_bytes"],
            "lm_head_bytes": snapshot["storage"]["lm_head_logical_bytes"],
            "bind_wait_seconds": snapshot["storage"]["bind_wait_for_prefetch_seconds"],
            "matvec_native_seconds": snapshot["quant_and_matvec"]["matvec_native_wall_seconds"],
            "gdn_native_seconds": snapshot["nested_compute"]["gdn_native_state_wall_seconds"],
        }, indent=2, sort_keys=True))
        return result
    finally:
        _restore_wrappers()
        _ACTIVE_PROFILE = None


def sanity() -> None:
    assert _role({"name": "blk.0.ffn_down.weight"}) == "ffn"
    assert _role({"name": "blk.0.attn_qkv.weight"}) == "gdn_projection"
    assert _role({"name": "blk.3.attn_q.weight"}) == "full_attention_projection"
    assert _role({"name": "output.weight"}) == "lm_head"
    p = AggregateProfile()
    p.add("engine_init_wall", seconds=1.0)
    p.add("decoder_step_wall", seconds=2.0)
    p.add("logits_wall", seconds=3.0)
    p.add("topk_wall", seconds=0.5)
    p.add("k3_read", nbytes=123)
    snap = p.snapshot(total_wall_seconds=7.0, total_cpu_seconds=5.0)
    assert snap["main_thread_additive"]["frontend_and_other_wall_seconds"] == 0.5
    assert snap["storage"]["k3_logical_bytes"] == 123
    assert snap["timing_semantics"]["io_service_seconds_nonadditive"] is True
    print(json.dumps({
        "schema": "qwen38-runtime-profile-sanity-v1",
        "status": "PASS",
        "profile_schema": snap["schema"],
        "top_level_additive": True,
        "io_service_nonadditive": True,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sanity")
    run = sub.add_parser("run")
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--native-lib", type=Path, required=True)
    run.add_argument("--state-lib", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--tokenizer-json", type=Path, required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--raw-prompt", action="store_true")
    run.add_argument("--max-new-tokens", type=int, default=4)
    run.add_argument("--max-prompt-tokens", type=int, default=gen.DEFAULT_MAX_PROMPT_TOKENS)
    run.add_argument("--work-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--profile", action="store_true")
    run.add_argument("--profile-output", type=Path)
    args = ap.parse_args()
    if args.cmd == "sanity":
        sanity()
        return
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_generator(
        model=args.model,
        native_lib=args.native_lib,
        state_lib=args.state_lib,
        inventory=args.inventory,
        tokenizer_json=args.tokenizer_json,
        prompt=args.prompt,
        raw_prompt=args.raw_prompt,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        work_dir=args.work_dir,
        output=args.output,
        enable_profile=args.profile,
        profile_output=args.profile_output,
    )


if __name__ == "__main__":
    main()
