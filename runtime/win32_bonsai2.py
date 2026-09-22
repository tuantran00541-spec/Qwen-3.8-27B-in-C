#!/usr/bin/env python3
"""Native-Windows launcher for the Bonsai 2 PTQ1 Qwen3.8 runtime."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
QWEN38 = ROOT / "qwen38"
if str(QWEN38) not in sys.path:
    sys.path.insert(0, str(QWEN38))

MODEL_NAME = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MODEL_SHA256 = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
N_LAYER = 64
_PREAD_LOCK = threading.Lock()


def _require_windows() -> None:
    if sys.platform != "win32":
        raise SystemExit("This launcher requires native Windows Python.")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _install_pread_compat() -> None:
    if hasattr(os, "pread"):
        return
    _require_windows()
    import msvcrt

    def pread(fd: int, nbytes: int, offset: int) -> bytes:
        if nbytes < 0 or offset < 0:
            raise ValueError("pread nbytes/offset must be non-negative")
        with _PREAD_LOCK:
            msvcrt.setmode(fd, os.O_BINARY)
            restore = os.lseek(fd, 0, os.SEEK_CUR)
            try:
                os.lseek(fd, int(offset), os.SEEK_SET)
                parts: list[bytes] = []
                done = 0
                while done < int(nbytes):
                    chunk = os.read(fd, int(nbytes) - done)
                    if not chunk:
                        break
                    parts.append(chunk)
                    done += len(chunk)
                return b"".join(parts)
            finally:
                os.lseek(fd, restore, os.SEEK_SET)

    setattr(os, "pread", pread)


def _load_runtime(build_dir: Path):
    _require_windows()
    build_dir = _resolve(build_dir)

    compat = build_dir / "qwen_glibc_expf_compat.dll"
    quant = build_dir / "qwen_bonsai2_quant.dll"
    state = build_dir / "qwen_bonsai2_gdn_state.dll"
    for path in (compat, quant, state):
        if not path.is_file():
            raise RuntimeError(f"missing Bonsai runtime DLL: {path}")

    os.environ["QWEN38_EXPF_COMPAT_LIB"] = str(compat)

    from qwen38_win32_bootstrap import (
        _bind_win32_expf,
        install_resource_compat,
    )

    install_resource_compat()
    _install_pread_compat()

    import bonsai2_prompt_spike as prompt
    import qwen35_k3_generate as textgen

    _bind_win32_expf(prompt.exact)

    original_pack = prompt.pack_gguf_layers

    def cached_pack(directory, out_bin: Path, out_index: Path, **kwargs):
        out_bin = Path(out_bin)
        out_index = Path(out_index)
        if out_bin.is_file() and out_index.is_file():
            try:
                manifest = json.loads(out_index.read_text(encoding="utf-8"))
                reusable = (
                    manifest.get("source", {}).get("sha256") == MODEL_SHA256
                    and len(manifest.get("layers", [])) == N_LAYER
                    and int(manifest.get("packed_file_bytes", -1)) == out_bin.stat().st_size
                )
            except Exception:
                reusable = False
            if reusable:
                print(f"Reusing Bonsai K3 trunk: {out_bin}", file=sys.stderr)
                return manifest
        print(f"Packing Bonsai K3 trunk once: {out_bin}", file=sys.stderr)
        return original_pack(directory, out_bin, out_index, **kwargs)

    prompt.pack_gguf_layers = cached_pack
    return prompt, textgen, quant, state


def sanity(args) -> None:
    prompt, _textgen, quant, state = _load_runtime(args.build_dir)

    # Validate the full Python <-> native ABI before a user downloads
    # gigabytes of model weights. The earlier partial check missed
    # qwen_bonsai2_permute_gdn_ssm_out_f32 and both row-dequantizers.
    import re

    qlib = ctypes.CDLL(str(quant))
    adapter_source = (QWEN38 / "bonsai2_quant_runtime.py").read_text(
        encoding="utf-8"
    )
    required_quant = set(
        re.findall(r"self\.lib\.(qwen_[A-Za-z0-9_]+)", adapter_source)
    )
    required_quant.update(
        re.findall(r'"(qwen_bonsai2_[A-Za-z0-9_]+)"', adapter_source)
    )
    required_quant.update({
        "qwen_bonsai2_permute_gdn_ssm_out_f32",
        "qwen_bonsai2_dequantize_ptq1_0_row",
        "qwen_bonsai2_dequantize_pq2_0_row",
    })
    for name in sorted(required_quant):
        getattr(qlib, name)
    print(
        f"QWEN38_BONSAI2_WINDOWS_ABI_EXPORTS_PASS "
        f"symbols={len(required_quant)}"
    )

    # Construct the same ctypes adapter that the real first token uses,
    # but with a tiny synthetic config; no GGUF download is needed.
    adapter = prompt.Bonsai2NativeRuntime(
        quant,
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 128,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "identity",
        },
        threads=2,
        max_rows=1024,
    )
    try:
        if adapter.pool is None:
            raise RuntimeError("Bonsai native quant pool was not created")
    finally:
        adapter.close()
    print("QWEN38_BONSAI2_WINDOWS_ADAPTER_INIT_PASS")

    # Prove that the previously missing permute function can actually run.
    src = (ctypes.c_float * 12)(*range(12))
    dst = (ctypes.c_float * 12)()
    permute = qlib.qwen_bonsai2_permute_gdn_ssm_out_f32
    permute.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    permute.restype = ctypes.c_int
    rc = permute(src, dst, 12, 2, 2, 3)
    expected = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    if rc != 0 or list(dst) != expected:
        raise RuntimeError(
            f"Bonsai native GDN permutation mismatch rc={rc} "
            f"actual={list(dst)} expected={expected}"
        )
    print("QWEN38_BONSAI2_WINDOWS_GDN_PERMUTE_PASS")

    # Synthetic 64-layer memory policy check; no GGUF/model download needed.
    from k3_stream import plan_memory

    fake_layer_mib = 84
    fake_layer_size = fake_layer_mib * 1024**2
    fake_layers = [
        {
            "layer": i,
            "read_bytes": fake_layer_size,
            "data_bytes": fake_layer_size,
            "file_offset": i * fake_layer_size,
        }
        for i in range(64)
    ]
    fake_manifest = {
        "schema": "qwen38-k3-trunk-v1",
        "alignment": 4096,
        "layers": fake_layers,
    }
    modes = {}
    for mode in ("low", "medium", "full"):
        budget, max_pinned = prompt.select_decoder_memory_plan(fake_layers, mode)
        modes[mode] = plan_memory(
            fake_manifest,
            budget,
            want_ring=2,
            max_pinned=max_pinned,
        )
    if len(modes["low"].pinned_layers) != 0 or modes["low"].ring_slots != 2:
        raise RuntimeError("low memory mode must use 2-slot streaming only")
    medium_pins = len(modes["medium"].pinned_layers)
    if not 0 < medium_pins < 64 or modes["medium"].ring_slots != 2:
        raise RuntimeError("medium memory mode must pin a partial prefix and preserve 2 rings")
    if modes["medium"].planned_bytes > prompt.MEDIUM_K3_BUDGET_BYTES:
        raise RuntimeError("medium memory mode exceeded 2.5 GiB K3 budget")
    if len(modes["full"].pinned_layers) != 64 or modes["full"].ring_slots != 0:
        raise RuntimeError("full memory mode must pin all decoder layers")
    print(
        f"QWEN38_BONSAI2_WINDOWS_MEMORY_MODES_PASS "
        f"low_pins=0 medium_pins={medium_pins} full_pins=64"
    )

    state_runtime = prompt.t2.load_state_lib(state, 2)
    try:
        if int(state_runtime.report().get("threads", 0)) != 2:
            raise RuntimeError("Bonsai GDN state pool did not start with 2 threads")
    finally:
        state_runtime.close()

    if not hasattr(os, "pread"):
        raise RuntimeError("Win32 positional-read compatibility was not installed")
    print("QWEN38_BONSAI2_NATIVE_WINDOWS_SANITY_PASS")


def run_once(args) -> None:
    prompt, _textgen, quant, state = _load_runtime(args.build_dir)
    model = _resolve(args.model)
    tokenizer = _resolve(args.tokenizer_json)
    work_dir = _resolve(args.work_dir)
    output = _resolve(args.output)
    for label, path in (("model", model), ("tokenizer", tokenizer)):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")

    mode = "full" if args.resident_decoder else args.memory_mode
    result = prompt.run(
        model,
        quant,
        state,
        tokenizer,
        args.prompt,
        int(args.max_new_tokens),
        work_dir,
        output,
        int(args.threads),
        bool(args.resident_decoder),
        None,
        True,
        False,
        False,
        None,
        mode,
    )
    print(json.dumps({
        "status": result["status"],
        "memory_mode": result["memory_mode"],
        "reader_planned_gib": round(result["state"]["reader"]["planned_bytes"] / 1024**3, 3),
        "reader_pinned_layers": len(result["state"]["reader"]["pinned_layers"]),
        "generated_text": result["generated_text"],
        "tokens_per_second": result["timing"]["tokens_per_second"],
        "max_rss_gib": result["max_rss_gib"],
    }, indent=2, ensure_ascii=False))
    print("QWEN38_BONSAI2_NATIVE_WINDOWS_GENERATION_PASS")


def _reset_engine(engine) -> None:
    for state in engine.states.values():
        ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))
    for hist in engine.conv_history.values():
        hist.clear()
    for cache in engine.caches.values():
        cache["k"].clear()
        cache["v"].clear()
    engine.runtime._attention_cache.clear()
    engine.position = 0


def _generate(engine, prompt, tokenizer, text: str, max_new_tokens: int) -> str:
    _rendered, prompt_ids = prompt.textgen.encode_prompt(tokenizer, text, raw=False)
    hidden = None
    for token_id in prompt_ids:
        hidden = engine.step(int(token_id))
    if hidden is None:
        return ""

    generated: list[int] = []
    for index in range(max_new_tokens):
        logits = engine.logits(hidden)
        token_id = int(prompt.base.topk(logits, 1)[0]["token"])
        generated.append(token_id)
        if token_id in prompt.EOS_IDS or index + 1 >= max_new_tokens:
            break
        hidden = engine.step(token_id)
    return tokenizer.decode(generated, skip_special_tokens=False)


def chat(args) -> None:
    prompt, textgen, quant, state = _load_runtime(args.build_dir)
    model = _resolve(args.model)
    tokenizer_json = _resolve(args.tokenizer_json)
    work_dir = _resolve(args.work_dir)
    for label, path in (("model", model), ("tokenizer", tokenizer_json)):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")

    mode = "full" if args.resident_decoder else args.memory_mode
    engine = prompt.StatefulBonsai2Generator(
        model,
        quant,
        state,
        work_dir,
        int(args.threads),
        resident_decoder=bool(args.resident_decoder),
        memory_mode=mode,
    )
    tokenizer = textgen.load_tokenizer(tokenizer_json)
    print(
        f"Bonsai 2 PTQ1 native Windows ready. "
        f"Memory mode={engine.reader.memory_mode}, "
        f"pinned layers={len(engine.reader.plan.pinned_layers)}, "
        f"K3 budget={engine.reader.plan.budget_bytes / 1024**3:.2f} GiB. "
        f"Type /exit to quit."
    )
    print("This shell resets model state between prompts; chat-history capsules remain experimental.")
    try:
        while True:
            try:
                user = input("You > ").strip()
            except EOFError:
                break
            if not user:
                continue
            if user.lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            _reset_engine(engine)
            try:
                answer = _generate(engine, prompt, tokenizer, user, int(args.max_new_tokens))
                print(f"Qwen > {answer}")
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                _reset_engine(engine)
    finally:
        engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Native Windows Bonsai 2 PTQ1 launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sanity")
    s.add_argument("--build-dir", type=Path, default=Path("build/win32"))

    def common(p):
        p.add_argument("--build-dir", type=Path, default=Path("build/win32"))
        p.add_argument(
            "--model",
            type=Path,
            default=Path("models/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
        )
        p.add_argument(
            "--tokenizer-json",
            type=Path,
            default=Path("models/qwen-official/tokenizer.json"),
        )
        p.add_argument("--work-dir", type=Path, default=Path("work/bonsai2-k3"))
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--max-new-tokens", type=int, default=32)
        p.add_argument(
            "--memory-mode",
            choices=("low", "medium", "full"),
            default="medium",
            help="low=2 SSD ring slots, medium=2.5 GiB pinned+ring, full=all decoder",
        )
        p.add_argument(
            "--resident-decoder",
            action="store_true",
            help="legacy alias for --memory-mode full",
        )

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--output", type=Path, default=Path("work/bonsai2-generation.json"))

    c = sub.add_parser("chat")
    common(c)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.cmd == "sanity":
        sanity(args)
    elif args.cmd == "run":
        run_once(args)
    else:
        chat(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
