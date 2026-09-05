#!/usr/bin/env python3
"""User-facing native-Windows launcher for the migrated Qwen3.8 runtime.

The proven qwen38 subtree is intentionally left byte-for-byte untouched.  This
wrapper installs the existing Win32 compatibility plumbing, reuses the proven
execution-ordered K3 packer and NO_BUFFERING|OVERLAPPED reader, and exposes the
stateful text generator through a small Windows-facing CLI.

This first packaged generator keeps the generator's proven single-vector quant
runtime.  The separately proven two-worker current-best prefill/profile stack is
not silently substituted here; that promotion needs its own end-to-end text
regression gate.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QWEN38 = ROOT / "qwen38"
if str(QWEN38) not in sys.path:
    sys.path.insert(0, str(QWEN38))

MODEL_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
K3_STREAM_BYTES = 21_127_430_144
N_LAYER = 64


def _require_windows() -> None:
    if sys.platform != "win32":
        raise SystemExit("This launcher requires native Windows Python (not WSL).")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_modules(build_dir: Path):
    _require_windows()
    build_dir = _resolve(build_dir)
    compat = build_dir / "qwen_glibc_expf_compat.dll"
    if not compat.is_file():
        raise RuntimeError(f"missing exact expf compatibility DLL: {compat}")
    os.environ["QWEN38_EXPF_COMPAT_LIB"] = str(compat)

    from qwen38_win32_bootstrap import install_resource_compat, import_generator

    install_resource_compat()
    gen = import_generator()
    import qwen38_k3_progressive as progressive
    import qwen38_progressive_current_best_profile_win32 as win_profile
    import qwen38_k3_progressive_win32 as win_k3
    return gen, progressive, win_profile, win_k3


def _manifest_is_reusable(manifest: dict[str, Any], trunk: Path, directory, progressive) -> bool:
    try:
        return (
            manifest.get("source", {}).get("sha256") == MODEL_SHA256
            and int(manifest.get("source", {}).get("file_bytes", -1)) == int(Path(directory.path).stat().st_size)
            and manifest.get("layout_policy") == progressive.LAYOUT_POLICY
            and bool(manifest.get("progressive_readiness"))
            and len(manifest.get("layers", [])) == N_LAYER
            and int(manifest.get("total_read_bytes", -1)) == K3_STREAM_BYTES
            and int(manifest.get("packed_file_bytes", -1)) == int(trunk.stat().st_size)
        )
    except (OSError, TypeError, ValueError):
        return False


def _cached_packer(win_profile, progressive):
    def pack(directory, out_bin: Path, out_index: Path, **kwargs):
        out_bin = Path(out_bin)
        out_index = Path(out_index)
        if out_bin.is_file() and out_index.is_file():
            try:
                manifest = json.loads(out_index.read_text(encoding="utf-8"))
            except Exception:
                manifest = None
            if isinstance(manifest, dict) and _manifest_is_reusable(manifest, out_bin, directory, progressive):
                print(f"Reusing packed K3 trunk: {out_bin}", file=sys.stderr)
                return manifest
        print(f"Packing K3 trunk once: {out_bin}", file=sys.stderr)
        return win_profile.pack_gguf_layers_progressive_win32(
            directory, out_bin, out_index, **kwargs)
    return pack


def _paths(args):
    build = _resolve(args.build_dir)
    return {
        "build": build,
        "quant": build / "qwen_quant_base.dll",
        "state": build / "qwen_gdn_state.dll",
        "direct": build / "qwen_win32_direct_io.dll",
    }


def _check_runtime_files(paths: dict[str, Path]) -> None:
    for key in ("quant", "state", "direct"):
        if not paths[key].is_file():
            raise RuntimeError(f"missing {key} DLL: {paths[key]}")


def sanity(args) -> None:
    gen, _progressive, _win_profile, win_k3 = _load_modules(args.build_dir)
    paths = _paths(args)
    _check_runtime_files(paths)

    from qwen38_win32_bootstrap import sanity as bootstrap_sanity

    bootstrap_sanity()
    gen.gdn._load_native(paths["quant"])
    gen.t2.load_state_lib(paths["state"])
    gen.sanity()
    result = win_k3.synthetic_lifecycle(paths["direct"])
    reader = result["reader"]
    if not reader.get("direct_io") or not reader.get("win32_no_buffering") or not reader.get("win32_overlapped"):
        raise RuntimeError(f"native Win32 direct-I/O sanity failed: {reader}")
    print("QWEN38_NATIVE_WINDOWS_GENERATOR_SANITY PASS")


def prepare(args) -> None:
    gen, progressive, win_profile, _win_k3 = _load_modules(args.build_dir)
    model = _resolve(args.model)
    work_dir = _resolve(args.work_dir)
    if not model.is_file():
        raise RuntimeError(f"model not found: {model}")
    work_dir.mkdir(parents=True, exist_ok=True)
    directory = gen.parse_gguf(model)
    trunk = work_dir / "decoder64.k3.bin"
    manifest_path = work_dir / "decoder64.k3.json"
    manifest = _cached_packer(win_profile, progressive)(
        directory,
        trunk,
        manifest_path,
        layers=range(N_LAYER),
        model_id=gen.gdn.MODEL_ID,
        revision=gen.gdn.REVISION,
        source_sha256=gen.gdn.SHA256,
        expected_layers=N_LAYER,
    )
    print(json.dumps({
        "status": "PASS",
        "packed_file": str(trunk),
        "packed_file_bytes": int(manifest["packed_file_bytes"]),
        "k3_read_bytes": int(manifest["total_read_bytes"]),
        "layout_policy": manifest.get("layout_policy"),
    }, indent=2))
    print("QWEN38_NATIVE_WINDOWS_PREPARE_PASS")


def _install_win32_generator_hooks(gen, progressive, win_profile, direct_lib: Path):
    old_pack = gen.pack_gguf_layers
    old_reader = gen.K3Trunk
    win_profile._install_pread_compat()
    gen.pack_gguf_layers = _cached_packer(win_profile, progressive)
    gen.K3Trunk = win_profile._reader_factory(direct_lib)
    return old_pack, old_reader


def _restore_hooks(gen, old_pack, old_reader) -> None:
    gen.pack_gguf_layers = old_pack
    gen.K3Trunk = old_reader


def run_once(args) -> None:
    gen, progressive, win_profile, _win_k3 = _load_modules(args.build_dir)
    paths = _paths(args)
    _check_runtime_files(paths)
    model = _resolve(args.model)
    inventory = _resolve(args.inventory)
    tokenizer = _resolve(args.tokenizer_json)
    work_dir = _resolve(args.work_dir)
    output = _resolve(args.output)
    for label, path in (("model", model), ("inventory", inventory), ("tokenizer", tokenizer)):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")
    work_dir.mkdir(parents=True, exist_ok=True)

    old_pack, old_reader = _install_win32_generator_hooks(gen, progressive, win_profile, paths["direct"])
    try:
        result = gen.generate(
            model,
            paths["quant"],
            paths["state"],
            inventory,
            tokenizer,
            args.prompt,
            bool(args.raw_prompt),
            int(args.max_new_tokens),
            int(args.max_prompt_tokens),
            work_dir,
            output,
        )
    finally:
        _restore_hooks(gen, old_pack, old_reader)

    reader = result.get("state", {}).get("reader", {})
    if reader.get("platform") != "win32" or not reader.get("direct_io"):
        raise RuntimeError(f"generation did not use native Win32 direct I/O: {reader}")
    print("QWEN38_NATIVE_WINDOWS_GENERATION_PASS")


def _reset_engine(engine) -> None:
    for state in engine.states.values():
        ctypes.memset(ctypes.addressof(state), 0, ctypes.sizeof(state))
    for hist in engine.conv_history.values():
        hist.clear()
    for cache in engine.caches.values():
        cache["k"].clear()
        cache["v"].clear()
    engine.position = 0


def _generate_with_engine(gen, engine, tokenizer, prompt: str, max_new_tokens: int, max_prompt_tokens: int) -> str:
    _rendered, prompt_ids = gen.encode_prompt(tokenizer, prompt, raw=False)
    if len(prompt_ids) > max_prompt_tokens:
        raise RuntimeError(f"prompt has {len(prompt_ids)} tokens; limit is {max_prompt_tokens}")
    hidden = None
    for token_id in prompt_ids:
        hidden = engine.step(token_id)
    if hidden is None:
        return ""

    generated: list[int] = []
    for new_index in range(max_new_tokens):
        logits = engine.logits(hidden)
        token_id = int(gen.base._topk(logits, 1)[0]["token"])
        generated.append(token_id)
        if token_id in gen.EOS_IDS:
            break
        if new_index + 1 < max_new_tokens:
            hidden = engine.step(token_id)
    return tokenizer.decode(generated, skip_special_tokens=False)


def chat(args) -> None:
    gen, progressive, win_profile, _win_k3 = _load_modules(args.build_dir)
    paths = _paths(args)
    _check_runtime_files(paths)
    model = _resolve(args.model)
    inventory = _resolve(args.inventory)
    tokenizer_json = _resolve(args.tokenizer_json)
    work_dir = _resolve(args.work_dir)
    for label, path in (("model", model), ("inventory", inventory), ("tokenizer", tokenizer_json)):
        if not path.is_file():
            raise RuntimeError(f"{label} not found: {path}")

    old_pack, old_reader = _install_win32_generator_hooks(gen, progressive, win_profile, paths["direct"])
    engine = None
    try:
        engine = gen.StatefulK3Generator(model, paths["quant"], paths["state"], inventory, work_dir)
        tokenizer = gen.load_tokenizer(tokenizer_json)
    finally:
        _restore_hooks(gen, old_pack, old_reader)

    print("Qwen3.8-27B native Windows ready. Type /exit to quit.")
    print("Current chat shell is stateless between prompts and uses greedy decoding.")
    try:
        while True:
            try:
                prompt = input("You > ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            _reset_engine(engine)
            try:
                text = _generate_with_engine(
                    gen, engine, tokenizer, prompt,
                    int(args.max_new_tokens), int(args.max_prompt_tokens))
                print(f"Qwen > {text}")
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                _reset_engine(engine)
    finally:
        if engine is not None:
            engine.close()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Native Windows Qwen3.8 launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sanity")
    s.add_argument("--build-dir", type=Path, default=Path("build/win32"))

    p = sub.add_parser("prepare")
    p.add_argument("--build-dir", type=Path, default=Path("build/win32"))
    p.add_argument("--model", type=Path, default=Path("models/Qwen3.8-27B-Q6_K_L.gguf"))
    p.add_argument("--work-dir", type=Path, default=Path("work/k3"))

    def add_runtime_args(r):
        r.add_argument("--build-dir", type=Path, default=Path("build/win32"))
        r.add_argument("--model", type=Path, default=Path("models/Qwen3.8-27B-Q6_K_L.gguf"))
        r.add_argument("--inventory", type=Path, default=Path("work/inventory.json"))
        r.add_argument("--tokenizer-json", type=Path, default=Path("models/qwen-official/tokenizer.json"))
        r.add_argument("--work-dir", type=Path, default=Path("work/k3"))
        r.add_argument("--max-new-tokens", type=int, default=4)
        r.add_argument("--max-prompt-tokens", type=int, default=64)

    r = sub.add_parser("run")
    add_runtime_args(r)
    r.add_argument("--prompt", required=True)
    r.add_argument("--raw-prompt", action="store_true")
    r.add_argument("--output", type=Path, default=Path("work/generation.json"))

    c = sub.add_parser("chat")
    add_runtime_args(c)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.cmd == "sanity":
        sanity(args)
    elif args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        run_once(args)
    else:
        chat(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
