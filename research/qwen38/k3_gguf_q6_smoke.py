#!/usr/bin/env python3
"""Qwen3.8-27B Q6_K_L GGUF CPU first-token viability gate.

This gate downloads only the pinned community Q6_K_L GGUF, verifies SHA256,
parses the GGUF directory without mmap'ing tensor data, then asks pinned
llama.cpp for exactly one deterministic token. The CLI is forced into a single
non-interactive turn and its default warmup/repack passes are disabled so the
smoke measures the requested first-token path rather than accidental extra
whole-model work.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import resource
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from gguf_stream import parse_gguf

REPO = "bartowski/Qwen3.8-27B-GGUF"
FILE = "Qwen3.8-27B-Q6_K_L.gguf"
EXPECTED_SHA256 = "a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a"
PROMPT = "2+2="


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def emit(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    print("QWEN38_EVIDENCE " + json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--llama-cli", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state: dict[str, Any] = {
        "schema": "qwen38-k3-gguf-q6-first-token-v4",
        "status": "INCOMPLETE",
        "phase": "start",
        "repo": REPO,
        "file": FILE,
        "expected_sha256": EXPECTED_SHA256,
        "prompt": PROMPT,
        "requested_tokens": 1,
        "context": 32,
        "temperature": 0,
        "load_mode": "mmap",
        "single_turn": True,
        "warmup": False,
        "repack": False,
        "bf16_downloaded": False,
    }

    def checkpoint(phase: str, **updates: Any) -> None:
        state.update(updates)
        state["phase"] = phase
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output, state)
        emit(phase, **updates)

    child: subprocess.Popen[str] | None = None

    def on_signal(signum: int, _frame: Any) -> None:
        checkpoint(
            "terminated_by_signal",
            status="INFRA_INTERRUPTED",
            signal=signum,
            max_child_rss_gib=rss_gib(),
        )
        if child is not None and child.poll() is None:
            child.terminate()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        checkpoint("llama_version_started")
        version_proc = subprocess.run(
            [str(args.llama_cli), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        version_text = version_proc.stdout.strip()
        checkpoint(
            "llama_version_completed",
            llama_version_returncode=version_proc.returncode,
            llama_version=version_text[-1000:],
        )
        if version_proc.returncode != 0:
            raise RuntimeError(f"llama-cli --version failed: {version_proc.returncode}")

        checkpoint("download_started")
        download_started = time.monotonic()
        model = Path(hf_hub_download(REPO, filename=FILE, local_dir=str(args.work_dir)))
        checkpoint(
            "download_completed",
            model_path=str(model),
            file_bytes=model.stat().st_size,
            download_seconds=time.monotonic() - download_started,
        )

        checkpoint("sha256_started")
        sha_started = time.monotonic()
        digest = sha256(model)
        sha_seconds = time.monotonic() - sha_started
        if digest != EXPECTED_SHA256:
            checkpoint(
                "sha256_mismatch",
                status="FAIL",
                sha256=digest,
                sha256_seconds=sha_seconds,
            )
            raise RuntimeError(f"GGUF sha256 mismatch: {digest}")
        checkpoint("sha256_verified", sha256=digest, sha256_seconds=sha_seconds)

        checkpoint("gguf_directory_started")
        directory = parse_gguf(model)
        type_counts = Counter(tensor.type_name for tensor in directory.tensors)
        type_id_counts = Counter(tensor.ggml_type for tensor in directory.tensors)
        by_name = directory.by_name()
        token_embd = by_name.get("token_embd.weight")
        output = by_name.get("output.weight")
        architecture = directory.metadata.get("general.architecture")
        if architecture != "qwen35":
            raise RuntimeError(f"unexpected GGUF architecture: {architecture!r}")
        checkpoint(
            "gguf_directory_verified",
            gguf_version=directory.version,
            architecture=architecture,
            tensor_count=directory.tensor_count,
            kv_count=directory.kv_count,
            alignment=directory.alignment,
            data_offset=directory.data_offset,
            tensor_type_counts=dict(sorted(type_counts.items())),
            tensor_type_id_counts={str(k): v for k, v in sorted(type_id_counts.items())},
            token_embd_type=token_embd.type_name if token_embd else None,
            output_type=output.type_name if output else None,
        )

        cmd = [
            str(args.llama_cli),
            "-m", str(model),
            "-ngl", "0",
            "-c", "32",
            "-b", "8",
            "-ub", "8",
            "-n", "1",
            "--temp", "0",
            "--load-mode", "mmap",
            "--no-repack",
            "--no-warmup",
            "--single-turn",
            "--no-display-prompt",
            "--special",
            "--no-mmproj",
            "--log-disable",
            "-p", PROMPT,
        ]
        checkpoint("inference_started", command=cmd)
        infer_started = time.monotonic()
        child = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if child.stdout is None:
            raise RuntimeError("llama-cli stdout pipe unavailable")

        tail: deque[str] = deque(maxlen=64)
        first_output_seen = False
        for line in child.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            if not first_output_seen and line:
                first_output_seen = True
                checkpoint("first_token_output_seen", first_output_line=line[:1000])

        returncode = child.wait()
        infer_seconds = time.monotonic() - infer_started
        text = "".join(tail)
        output_nonempty = bool(text)
        semantic_four = "4" in text
        status = "PASS" if returncode == 0 and output_nonempty else "FAIL"
        checkpoint(
            "inference_completed",
            status=status,
            returncode=returncode,
            output_nonempty=output_nonempty,
            semantic_four=semantic_four,
            generated_output=text[-2000:],
            inference_seconds=infer_seconds,
            max_child_rss_gib=rss_gib(),
            total_seconds=time.monotonic() - started,
        )
        if status != "PASS":
            raise SystemExit(1)
    except BaseException as exc:
        if state.get("status") == "INCOMPLETE":
            checkpoint(
                "exception",
                status="FAIL",
                error_type=type(exc).__name__,
                error=str(exc),
                max_child_rss_gib=rss_gib(),
            )
        raise


if __name__ == "__main__":
    main()
