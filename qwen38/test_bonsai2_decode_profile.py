#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qwen38"))

from bonsai2_prompt_spike import profile_delta


def close(a: float, b: float, tol: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{a} != {b}")


class FakeReporter:
    def __init__(self, payload):
        self.payload = payload

    def report(self):
        return self.payload


def verify_profile_snapshot_contract() -> None:
    engine = object.__new__(__import__("bonsai2_prompt_spike").StatefulBonsai2Generator)
    engine.runtime = FakeReporter({
        "timing_seconds": {"ptq1_matvec": 2.5, "output_copy": 0.2},
        "timing_calls": {"ptq1_matvec": 4, "output_copy": 3},
    })
    engine.state_lib = FakeReporter({
        "timing_seconds": {"step": 0.7},
        "timing_calls": {"step": 2},
    })
    engine.reader = FakeReporter({
        "bytes_read": 8192,
        "hits": 6,
        "misses": 1,
    })
    engine.section_timing_seconds = {
        "recurrent_layer": 1.25,
        "attention_layer": 0.5,
        "logits": 0.1,
    }
    engine.section_timing_calls = {
        "recurrent_layer": 48,
        "attention_layer": 16,
        "logits": 1,
    }

    snap = engine.profile_snapshot()
    if snap["lowbit_runtime_timing_seconds"]["ptq1_matvec"] != 2.5:
        raise AssertionError(snap)
    if snap["lowbit_runtime_timing_calls"]["ptq1_matvec"] != 4:
        raise AssertionError(snap)
    if snap["gdn_state_timing_seconds"]["step"] != 0.7:
        raise AssertionError(snap)
    if snap["gdn_state_timing_calls"]["step"] != 2:
        raise AssertionError(snap)
    if snap["section_timing_seconds"]["recurrent_layer"] != 1.25:
        raise AssertionError(snap)
    if snap["section_timing_calls"]["attention_layer"] != 16:
        raise AssertionError(snap)
    if snap["reader"] != {"bytes_read": 8192, "hits": 6, "misses": 1}:
        raise AssertionError(snap)


def verify_decode_loop_wiring() -> None:
    source = (ROOT / "qwen38" / "bonsai2_prompt_spike.py").read_text(encoding="utf-8")
    required = (
        'profile_before = engine.profile_snapshot()',
        'report["critical_path"] = profile_delta(',
        '"decode_critical_path"',
        'self._record_section("recurrent_layer"',
        'self._record_section("attention_layer"',
        'self._record_section("logits"',
    )
    for needle in required:
        if needle not in source:
            raise AssertionError(f"decode profile wiring missing: {needle}")


def main() -> None:
    before = {
        "lowbit_runtime_timing_seconds": {
            "ptq1_matvec": 1.0,
            "hadamard": 0.2,
            "output_copy": 0.3,
        },
        "lowbit_runtime_timing_calls": {
            "ptq1_matvec": 10,
            "hadamard": 5,
            "output_copy": 7,
        },
        "gdn_state_timing_seconds": {"step": 0.4},
        "gdn_state_timing_calls": {"step": 3},
        "section_timing_seconds": {
            "recurrent_layer": 4.0,
            "attention_layer": 1.0,
            "logits": 0.2,
        },
        "section_timing_calls": {
            "recurrent_layer": 48,
            "attention_layer": 16,
            "logits": 1,
        },
        "reader": {"bytes_read": 1000, "hits": 2, "misses": 1},
    }
    after = {
        "lowbit_runtime_timing_seconds": {
            "ptq1_matvec": 3.0,
            "hadamard": 0.6,
            "output_copy": 0.4,
        },
        "lowbit_runtime_timing_calls": {
            "ptq1_matvec": 14,
            "hadamard": 7,
            "output_copy": 9,
        },
        "gdn_state_timing_seconds": {"step": 0.9},
        "gdn_state_timing_calls": {"step": 5},
        "section_timing_seconds": {
            "recurrent_layer": 7.4,
            "attention_layer": 1.8,
            "logits": 0.35,
        },
        "section_timing_calls": {
            "recurrent_layer": 96,
            "attention_layer": 32,
            "logits": 2,
        },
        "reader": {"bytes_read": 5096, "hits": 5, "misses": 2},
    }

    out = profile_delta(before, after, wall_seconds=4.0)

    if out["reader"] != {
        "bytes_read": 4096,
        "hits": 3,
        "misses": 1,
    }:
        raise AssertionError(out["reader"])

    components = out["components_seconds"]
    close(components["ptq1_matvec"], 2.0)
    close(components["hadamard"], 0.4)
    close(components["output_copy"], 0.1)
    close(components["gdn_state_step"], 0.5)

    calls = out["component_calls"]
    if calls["ptq1_matvec"] != 4:
        raise AssertionError(calls)
    if calls["gdn_state_step"] != 2:
        raise AssertionError(calls)

    sections = out["sections_seconds"]
    close(sections["recurrent_layer"], 3.4)
    close(sections["attention_layer"], 0.8)
    close(sections["logits"], 0.15)
    section_calls = out["section_calls"]
    if section_calls["recurrent_layer"] != 48:
        raise AssertionError(section_calls)
    if section_calls["attention_layer"] != 16:
        raise AssertionError(section_calls)
    if section_calls["logits"] != 1:
        raise AssertionError(section_calls)

    close(out["wall_seconds"], 4.0)
    close(out["tracked_seconds"], 3.0)
    close(out["untracked_seconds"], 1.0)
    close(out["tracked_fraction"], 0.75)

    verify_profile_snapshot_contract()
    verify_decode_loop_wiring()
    print("QWEN38_BONSAI2_DECODE_PROFILE_DELTA_PASS")


if __name__ == "__main__":
    main()
