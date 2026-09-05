from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

from core import MODEL_ID, PINNED_REVISION


PROMPTS = {
    "vi": "Hãy trả lời đúng một câu ngắn bằng tiếng Việt: Việt Nam có thủ đô là thành phố nào?",
    "en": "Answer in exactly one short English sentence: What is the capital of France?",
}


def max_rss_gib() -> float:
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def write_progress(output: Path, result: dict[str, object]) -> None:
    """Persist evidence after every expensive phase so runner timeout still leaves metrics."""
    result["timing"]["total_seconds"] = time.perf_counter() - result["_started"]
    serializable = {key: value for key, value in result.items() if not key.startswith("_")}
    output.write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.8-27B official BF16 low-RAM text baseline smoke")
    parser.add_argument("--offload-dir", required=True)
    parser.add_argument("--cpu-memory", default="9GiB")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    offload_dir = Path(args.offload_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    proc = psutil.Process()
    started = time.perf_counter()
    print(f"QWEN38_BASELINE_START model={MODEL_ID} revision={PINNED_REVISION}", flush=True)
    print(
        f"QWEN38_HOST python={platform.python_version()} torch={torch.__version__} "
        f"cpu_count={os.cpu_count()} ram_gib={psutil.virtual_memory().total / 2**30:.3f}",
        flush=True,
    )

    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=PINNED_REVISION, use_fast=True)
    tokenizer_seconds = time.perf_counter() - tokenizer_started

    load_started = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        revision=PINNED_REVISION,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={"cpu": args.cpu_memory},
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    device_map = {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    placements: dict[str, int] = {}
    for placement in device_map.values():
        placements[placement] = placements.get(placement, 0) + 1
    print(
        "QWEN38_LOADED "
        f"load_seconds={load_seconds:.3f} rss_gib={proc.memory_info().rss / 2**30:.3f} "
        f"max_rss_gib={max_rss_gib():.3f} placements={json.dumps(placements, sort_keys=True)}",
        flush=True,
    )

    result: dict[str, object] = {
        "schema": "qwen38-official-baseline-smoke-v2",
        "status": "loaded",
        "model": {
            "id": MODEL_ID,
            "revision": PINNED_REVISION,
            "dtype": "bfloat16",
            "cpu_weight_budget": args.cpu_memory,
            "thinking": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "ram_gib": psutil.virtual_memory().total / 2**30,
        },
        "timing": {
            "tokenizer_seconds": tokenizer_seconds,
            "model_load_seconds": load_seconds,
            "total_seconds": 0.0,
        },
        "memory": {
            "rss_gib_end": proc.memory_info().rss / 2**30,
            "max_rss_gib": max_rss_gib(),
        },
        "device_map": device_map,
        "placement_counts": placements,
        "replies": {},
        "_started": started,
    }
    write_progress(output, result)

    replies = result["replies"]
    assert isinstance(replies, dict)
    for lang, prompt in PROMPTS.items():
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        encoded = tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to("cpu") if torch.is_tensor(value) else value for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        gen_started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        gen_seconds = time.perf_counter() - gen_started
        continuation = generated[0][prompt_tokens:]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        output_tokens = int(continuation.numel())
        tok_per_s = output_tokens / gen_seconds if gen_seconds > 0 else 0.0
        replies[lang] = {
            "prompt": prompt,
            "reply": text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "generation_seconds": gen_seconds,
            "tokens_per_second": tok_per_s,
        }
        result["status"] = f"generated_{lang}"
        result["memory"] = {
            "rss_gib_end": proc.memory_info().rss / 2**30,
            "max_rss_gib": max_rss_gib(),
        }
        write_progress(output, result)
        print(
            f"QWEN38_REPLY lang={lang} prompt_tokens={prompt_tokens} output_tokens={output_tokens} "
            f"seconds={gen_seconds:.3f} tok_s={tok_per_s:.6f} text={json.dumps(text, ensure_ascii=False)}",
            flush=True,
        )

    result["status"] = "pass"
    result["memory"] = {
        "rss_gib_end": proc.memory_info().rss / 2**30,
        "max_rss_gib": max_rss_gib(),
    }
    write_progress(output, result)
    print(f"QWEN38_BASELINE_PASS output={output}", flush=True)


if __name__ == "__main__":
    main()
