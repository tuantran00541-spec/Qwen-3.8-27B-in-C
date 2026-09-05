# Qwen3.8-27B low-RAM CPU runtime

Experimental CPU-only runtime for **Qwen/Qwen3.8-27B** that keeps the model primarily on SSD/NVMe instead of loading the full GGUF into RAM.

The repository contains native C kernels plus Python orchestration used to develop and verify the runtime. It is **not yet a polished llama.cpp/Ollama-style application**: the current text-generation entry point is developer-facing and greedy-only, while the fastest native-Windows path is currently exposed as an exact validation/profile runner.

## Read this first if you are on Windows

The current easiest text-generation path is **Ubuntu inside WSL2**.

Do **not** paste commands such as `sudo apt update` into ordinary Windows PowerShell. `sudo` and `apt` are Linux commands.

From PowerShell, first inspect the installed WSL distributions:

```powershell
wsl -l -v
```

You want an Ubuntu distribution running as **WSL version 2**, for example:

```text
  NAME              STATE           VERSION
* Ubuntu            Running         2
  docker-desktop    Stopped         2
```

If Ubuntu is already installed, enter it explicitly:

```powershell
wsl -d Ubuntu
```

If its name is different, such as `Ubuntu-24.04`, use that exact name:

```powershell
wsl -d Ubuntu-24.04
```

If Ubuntu is not listed, open PowerShell as Administrator and install it:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested, open Ubuntu once to finish the first-time username/password setup, then optionally make it the default WSL distribution:

```powershell
wsl --set-default Ubuntu
```

After entering WSL, verify that you are actually inside Ubuntu:

```bash
cat /etc/os-release
```

It should identify itself as Ubuntu.

If plain `wsl` drops you into another backend or utility distribution instead, exit it:

```bash
exit
```

Then use `wsl -l -v` in PowerShell and enter Ubuntu explicitly with `wsl -d <Ubuntu-name>`.

A path such as `/mnt/host/c/...` is **not** where this README recommends running the model. Likewise, although Ubuntu can access your Windows files under `/mnt/c/...`, the current K3 runtime is I/O-sensitive.

For the quick start below, clone and run the repository **inside the Ubuntu filesystem**, under your Linux home directory, for example:

```text
/home/your-user/Qwen-3.8-27B-in-C
```

rather than:

```text
/mnt/c/Users/your-user/Qwen-3.8-27B-in-C
/mnt/host/c/Users/your-user/Qwen-3.8-27B-in-C
```

Keeping a separate Windows clone is fine; just do not use that mounted copy for the current Linux/WSL generation path unless you deliberately want to test mounted-filesystem I/O behavior.

## What works today

- Qwen3.8-27B Q6_K_L GGUF parsing and contract validation.
- 64-layer decoder execution: 48 Gated-DeltaNet layers + 16 full-attention layers.
- Stateful multi-token text generation with GDN recurrent state, convolution history and F16 attention KV cache.
- SSD-resident K3 layer streaming with a small ring buffer.
- Native C quantized matvec and recurrent-state kernels.
- Native Windows `NO_BUFFERING | OVERLAPPED` direct-I/O backend.
- Exact two-worker Q6 current-best path and Q8 no-allocation path.
- A full real 24 GB Windows gate has matched the Linux hidden/state anchors bit-for-bit.

Current limitations:

- Text generation is greedy-only; sampling is not wired into the public entry point yet.
- The basic text generator is currently easiest to run on **Linux or WSL2**.
- Native Windows current-best is verified, but still packaged as a developer profile/validation stack rather than a one-command chat executable.
- The research entry point currently rebuilds its packed K3 decoder trunk when starting a run, so keep the work directory on a fast SSD and leave enough free disk space.
- Default prompt limit in the basic generator is 64 tokens.

## Hardware and software

Recommended for the current runtime:

- 64-bit x86 CPU. AVX2 is used by the optimized/current-best kernels; the minimal quick-start build below uses the portable scalar quant bridge.
- SSD/NVMe storage.
- At least **~50 GB free disk space** for the ~24 GB GGUF plus the packed decoder trunk and temporary files.
- Python **3.10+**.
- Clang/LLVM or another C11 compiler.

The project was designed around a 16 GB RAM consumer laptop. The proven hosted current-best Windows profile stayed around ~1 GiB process RSS, but your OS cache, filesystem and other applications still consume memory, so treat that number as a runtime measurement rather than a total-system RAM guarantee.

## Quick start: generate text on Ubuntu / WSL2

All commands in this section are Linux commands and must be run **inside Ubuntu**, not in Windows PowerShell.

### 1. Install the required packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv clang git
```

### 2. Clone the repository inside the Ubuntu filesystem

Start from your Linux home directory:

```bash
cd ~
git clone https://github.com/tuantran00541-spec/Qwen-3.8-27B-in-C.git
cd Qwen-3.8-27B-in-C
```

Check where you are:

```bash
pwd
```

A good result looks similar to:

```text
/home/your-user/Qwen-3.8-27B-in-C
```

Avoid running the model from `/mnt/c/...` or `/mnt/host/c/...` for this quick start.

### 3. Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tokenizers huggingface_hub hf_xet
```

When you open a new Ubuntu terminal later, reactivate the environment with:

```bash
cd ~/Qwen-3.8-27B-in-C
source .venv/bin/activate
```

### 4. Download the pinned GGUF and official tokenizer

```bash
mkdir -p models

python - <<'PY'
from huggingface_hub import hf_hub_download

model = hf_hub_download(
    "bartowski/Qwen3.8-27B-GGUF",
    filename="Qwen3.8-27B-Q6_K_L.gguf",
    local_dir="models",
)

tokenizer = hf_hub_download(
    "Qwen/Qwen3.8-27B",
    filename="tokenizer.json",
    revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    local_dir="models/qwen-official",
)

print("GGUF:", model)
print("Tokenizer:", tokenizer)
PY
```

The runtime is pinned to this GGUF SHA256:

```text
a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a
```

Verify it before running:

```bash
sha256sum models/Qwen3.8-27B-Q6_K_L.gguf
```

Do not continue if the hash differs.

### 5. Build the minimal native libraries

The basic generator needs the scalar Q6_K/Q8_0 quantized matvec bridge and the recurrent GDN state kernel.

```bash
mkdir -p build work/k3

clang -O3 -std=c11 -shared -fPIC \
  -fno-fast-math -fno-associative-math -ffp-contract=off \
  qwen38/gguf_quant_dot.c -lm \
  -o build/libqwen_quant.so

clang -O3 -std=c11 -shared -fPIC \
  -fno-fast-math -fno-associative-math -ffp-contract=off \
  qwen38/gdn_state_ar.c -lm \
  -o build/libqwen_gdn_state.so
```

Do **not** add `-ffast-math`: exact arithmetic behavior is part of the validated runtime contract.

### 6. Run code sanity checks

```bash
export PYTHONPATH="$PWD/qwen38"

python -u qwen38/qwen35_gguf_decoder_contract.py sanity
python -u qwen38/qwen35_k3_generate.py sanity
```

### 7. Validate the real GGUF and create its inventory

This reads and hashes the real model, then checks all 64 decoder layers, tensor roles, shapes and mixed quantization types.

```bash
python -u qwen38/qwen35_gguf_decoder_contract.py real \
  --model models/Qwen3.8-27B-Q6_K_L.gguf \
  --output work/inventory.json
```

`work/inventory.json` must report `"status": "PASS"` and the pinned SHA256 before generation.

### 8. Generate text

```bash
python -u qwen38/qwen35_k3_generate.py run \
  --model models/Qwen3.8-27B-Q6_K_L.gguf \
  --native-lib build/libqwen_quant.so \
  --state-lib build/libqwen_gdn_state.so \
  --inventory work/inventory.json \
  --tokenizer-json models/qwen-official/tokenizer.json \
  --prompt "Explain why the sky is blue in one short paragraph." \
  --max-new-tokens 4 \
  --work-dir work/k3 \
  --output work/generation.json
```

The command prints a compact JSON result containing `generated_text`, token IDs, elapsed time and RSS. The complete record is written to `work/generation.json`.

Increase `--max-new-tokens` to generate more text. The current generator uses greedy decoding, so there is no temperature/top-p flag yet.

If you already constructed the exact chat-template text yourself, add `--raw-prompt`; otherwise the runtime wraps `--prompt` in its plain-user Qwen chat envelope automatically.

### WSL storage note

The K3 reader prefers direct I/O when the filesystem supports it and falls back to buffered reads when it does not. For better I/O behavior, keep the repository, model and `work/` directory inside the Ubuntu filesystem on the WSL virtual disk when following this quick start.

Windows-mounted paths such as `/mnt/c/...` can have different filesystem and I/O behavior. `/mnt/host/c/...` usually indicates that you are not following the intended Ubuntu quick-start environment at all; verify your active WSL distribution before continuing.

## Native Windows status

Native Windows is **not hypothetical**: the current-best stack has completed a full real 24 GB exact run using:

- Win32 `CreateFileW` with `NO_BUFFERING | OVERLAPPED`;
- two K3 ring slots;
- one storage-I/O request at a time;
- portable exact Q6 worker pool with two workers;
- Q8 no-allocation path;
- glibc-compatible `expf` where required for Linux/Windows bitwise identity.

The Windows entry point is currently:

```text
qwen38/qwen38_progressive_current_best_profile_win32.py
```

Its sanity mode is:

```powershell
$env:PYTHONPATH = "$PWD\qwen38"
python -u qwen38\qwen35_gguf_decoder_contract.py sanity
python -u qwen38\qwen38_win32_bootstrap.py
python -u qwen38\qwen38_progressive_current_best_profile_win32.py sanity
```

However, that path expects the complete native DLL bundle (`qwen_quant_base.dll`, portable Q6 pool, GDN/attention/RMSNorm helpers, Win32 direct-I/O DLL, expf compatibility DLL, etc.). A standalone build-and-run PowerShell wrapper has not been packaged yet, so **for text generation today, use the Ubuntu/WSL2 quick start above**. The next packaging milestone is a native-Windows launcher that builds/locates those DLLs and exposes the current-best stack as a normal chat/generation command.

Do not run the raw Linux generator directly from ordinary Windows Python and assume it is equivalent to the verified Win32 path: the Windows adapter provides platform-specific positional reads, direct I/O and exact `expf` plumbing.

## Repository layout

```text
.
├── qwen38/                 # runtime, native C kernels, probes and exact gates
├── .github/workflows/      # migration/provenance tooling
└── README.md
```

The `qwen38/` directory intentionally still contains the historical probes that established correctness and performance. They are kept because several proven runtime modules are shared with those gates. A smaller user-facing layout will be extracted only after it can be re-gated without changing arithmetic or storage behavior.

## Pinned model and verified baseline

Model:

- `Qwen/Qwen3.8-27B`
- official revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- GGUF: `bartowski/Qwen3.8-27B-GGUF / Qwen3.8-27B-Q6_K_L.gguf`
- GGUF SHA256: `a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a`

Exact source-baseline anchors:

- hidden SHA256: `e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04`
- persistent-state SHA256: `41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06`
- exact 11-token K3 bytes: `21,127,430,144`
- K3 ring slots: `2`
- slot bytes: `336,449,536`
- planned ring bytes: `672,899,072`
- storage I/O concurrency: `1`
- Q6 workers: `2`

The standalone code was migrated byte-for-byte from:

- source repo: `tuantran00541-spec/manga-translator`
- source branch: `research/qwen3.8-27b-runtime-finish`
- source commit: `8f378cf13786e5e62b64ba540e3a187e351b2160`
- proven Windows full-GGUF run: `33954444876`
- evidence artifact: `9966107719`
- evidence digest: `sha256:f810dab4d61d429383cdc9bb01835a6acc49523ac175183f8769e32faebdb8d2`

The original migrated `qwen38/` subtree SHA is:

```text
a0e03e8bffc282e30aa3b0664b039f99c6bcead9
```

Large model weights, packed K3 trunks and generated native binaries are intentionally not committed to Git.
