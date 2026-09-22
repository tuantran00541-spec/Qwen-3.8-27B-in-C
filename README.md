# Qwen3.8-27B low-RAM CPU runtime

## Recommended Windows path: Ternary Bonsai 2 PTQ1

The current recommended local model is now **Ternary Bonsai 2 27B PTQ1_0**, not the older ~24 GB Q6_K_L package.

Pinned model:

- repository: `prism-ml/Ternary-Bonsai-2-27B-gguf`
- revision: `6ed5e12bf84b7a63069882c91dd9e9218647d17b`
- file: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
- size: `5,946,648,928` bytes (~5.95 GB)
- SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`
- tokenizer: pinned `Qwen/Qwen3.8-27B` tokenizer at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

On Windows PowerShell:

```powershell
git switch main
git pull --ff-only origin main
Set-ExecutionPolicy -Scope Process Bypass
.\setup-bonsai2.ps1
.\run-bonsai2.ps1 "Hello" -MaxNewTokens 8
```

**Default memory mode: Medium** (balanced for a 16-GiB Windows laptop). It allocates a
**2.5-GiB K3 layer-cache budget** that pins an early decoder-layer prefix in RAM
while streaming the remaining layers through two SSD ring slots. In addition to
that budget, allow room for recurrent state, Python, Windows and filesystem cache.

```powershell
# Balanced default, equivalent to -MemoryMode Medium:
.\run-bonsai2.ps1 "Hello" -MaxNewTokens 8 -MemoryMode Medium

# Minimal RAM (~170 MiB streaming rings plus state/runtime overhead):
.\run-bonsai2.ps1 "Hello" -MaxNewTokens 8 -MemoryMode Low

# Backward-compatible alias for -MemoryMode Low:
.\run-bonsai2.ps1 "Hello" -MaxNewTokens 8 -LowRam

# Pin the complete ~5.38-GB decoder:
.\run-bonsai2.ps1 "Hello" -MaxNewTokens 8 -MemoryMode Full
```

Medium is a RAM/I/O trade-off, not a guaranteed speedup; compare the reported
`tokens_per_second`, `reader.pinned_layers`, `reader.bytes_read` and
`max_rss_gib` on the same prompt. Memory modes do not change quantization,
model weights or attention-history behavior.

The first Bonsai run builds an execution-ordered K3 trunk once under
`work\bonsai2-k3`; later runs reuse it. Changing memory modes **does not**
repack K3 or redownload the GGUF.

An interactive shell is also available:

```powershell
.\chat-bonsai2.ps1 -MaxNewTokens 32 -MemoryMode Medium
```

The chat-history capsule work is now in-tree. The validated low-precision capsule stores GDN recurrent state as **BF16**, keeps convolution history in F32 and the bounded attention tail in F16. The BF16 capsule measured about **83.4 MB** in the release probe and resumed token-for-token identically to the F32 control for that probe. It remains an experimental history acceleration layer; the durable text transcript should still be kept separately.

### Legacy Q6 package

The older `Qwen3.8-27B-Q6_K_L.gguf` Windows package and `setup.ps1/run.ps1/chat.ps1` are retained for regression/reference use. They are no longer the recommended first download.


Experimental CPU-only runtime for **Qwen/Qwen3.8-27B** that keeps the model primarily on SSD/NVMe instead of loading the full GGUF into RAM.

## Native Windows is now the default user path

You do **not** need WSL for the normal setup anymore. The repository now has a Windows packaging layer around the preserved `qwen38/` research runtime:

- PowerShell setup and launchers;
- exact native DLL build;
- Win32 `NO_BUFFERING | OVERLAPPED` direct I/O;
- glibc-compatible `expf` compatibility where the proven exact path requires it;
- low-RAM, resumable model download through `curl.exe` instead of `hf_xet`;
- one-time GGUF validation and K3 trunk preparation;
- a simple prompt launcher and a simple interactive chat shell.

The original migrated `qwen38/` subtree is intentionally kept untouched as the exact research/reference baseline. The user-facing Windows layer lives outside it.

## Quick start on Windows

Open **PowerShell** in the repository directory.

If you have not cloned the repository yet:

```powershell
git clone https://github.com/tuantran00541-spec/Qwen-3.8-27B-in-C.git
cd Qwen-3.8-27B-in-C
```

Then run the one-time setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` will:

1. find Python 3 and LLVM/Clang;
2. install Python 3.12 and/or LLVM through `winget` when they are missing;
3. create `.venv` and install the `tokenizers` Python package;
4. compile the complete exact Windows DLL bundle;
5. run a native Windows DLL/direct-I/O sanity gate;
6. download `Qwen3.8-27B-Q6_K_L.gguf` with resumable `curl.exe`;
7. verify the pinned GGUF SHA256;
8. download the pinned official tokenizer;
9. validate the real 64-layer GGUF contract;
10. create the execution-ordered K3 trunk once and reuse it on later launches.

The model is about 24 GB and the packed K3 trunk is another large SSD file, so the first setup is intentionally much slower than later starts. Keep at least roughly **55–60 GB free**; more headroom is recommended.

### Chat shell

After setup:

```powershell
.\chat.ps1
```

Example:

```text
Qwen3.8-27B native Windows ready. Type /exit to quit.
You > Explain why the sky is blue.
Qwen > ...
```

The first packaged chat shell is **stateless between prompts** and uses greedy decoding. Exit with `/exit` or `/quit`.

Increase the generated-token limit if desired:

```powershell
.\chat.ps1 -MaxNewTokens 8
```

### One prompt

```powershell
.\run.ps1 "Explain why the sky is blue in one short paragraph."
```

or:

```powershell
.\run.ps1 "Hello" -MaxNewTokens 8
```

The detailed generation record is written to:

```text
work\generation.json
```

## Download behavior

The Windows installer deliberately does **not** use the default `hf_xet` model downloader. On a 16 GB machine, the adaptive Xet download path can consume much more RAM than is desirable for a simple model installation.

Instead, the installer downloads the GGUF to:

```text
models\Qwen3.8-27B-Q6_K_L.gguf.part
```

using Windows `curl.exe` with HTTP resume enabled. If the connection is interrupted, rerun:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

and the `.part` file is kept for resume. After the download completes, setup verifies the exact pinned SHA256 before renaming it to the final GGUF path.

Pinned GGUF SHA256:

```text
a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a
```

## What works today

- Qwen3.8-27B Q6_K_L GGUF parsing and contract validation.
- 64-layer decoder execution: 48 Gated-DeltaNet layers + 16 full-attention layers.
- Stateful multi-token text generation with persistent GDN state, convolution history and F16 attention KV cache.
- SSD-resident K3 layer streaming with a two-slot ring.
- Native C quantized matvec and recurrent-state kernels.
- Native Windows `CreateFileW` direct I/O with `NO_BUFFERING | OVERLAPPED`.
- Exact Windows glibc-compatible `expf` compatibility path.
- Exact two-worker Q6 current-best profile path and Q8 no-allocation path.
- Native Windows installer, resumable downloader, DLL builder, prompt launcher and chat shell.

## Important validation status

There are two different levels of evidence and they should not be confused.

### Proven source baseline

The original source branch completed a real full ~24 GB Windows current-best exact gate and matched the Linux hidden/state anchors bit-for-bit.

Proven source run:

```text
33954444876
```

Marker:

```text
QWEN38_WIN32_FULL_GGUF_CURRENT_BEST_REAL_BITWISE_PASS
```

### Standalone Windows package

The new standalone Windows packaging layer has passed its lightweight hosted Windows gate. That gate verifies:

- every DLL in the packaged Windows bundle compiles;
- exact `expf` compatibility binds successfully;
- the stateful generator sanity contract passes;
- the synthetic Win32 K3 lifecycle passes;
- `NO_BUFFERING` is active;
- `OVERLAPPED` is active;
- direct I/O uses two native ring slots;
- storage I/O concurrency remains one;
- the synthetic tensor SHA checks pass.

Standalone package sanity run:

```text
33962254079
```

Markers include:

```text
QWEN38_NATIVE_WINDOWS_DLL_BUILD_PASS
QWEN38_WIN32_GLIBC_EXPF_COMPAT_BIND_PASS
QWEN38_WIN32_PROGRESSIVE_K3_LIFECYCLE_PASS
QWEN38_NATIVE_WINDOWS_GENERATOR_SANITY PASS
```

The new standalone wrapper has **not yet been re-gated with the real 24 GB GGUF end-to-end after packaging**. The source runtime is proven; the standalone migration is exact; the Windows packaging sanity is green; the next real milestone is running `setup.ps1` and text generation on the target Windows laptop and then adding a standalone full-GGUF text gate.

## Current performance caveat

The first user-facing Windows text launcher intentionally prioritizes getting a reliable native Windows path working without changing the proven `qwen38/` reference tree.

For text generation it currently uses the generator's proven **single-vector quant runtime** plus the proven Win32 progressive direct-I/O reader. The full two-worker current-best prefill/profile stack is compiled into the DLL bundle, but it is **not yet silently promoted into stateful text generation**.

That means:

- native Windows setup and text generation are now packaged;
- storage uses the intended Win32 direct-I/O path;
- arithmetic compatibility is preserved;
- but the launcher is not yet claiming the maximum performance measured by the current-best profile gate.

Promoting the two-worker/current-best helpers into real multi-token generation requires an explicit output/exactness A/B gate first.

## Hardware and software

Recommended:

- Windows 10/11 x64;
- x86-64 CPU with AVX2 for the optimized bundle;
- SSD/NVMe;
- roughly 55–60 GB or more free disk space;
- 16 GB RAM is the target class for this project.

`setup.ps1` can install these through `winget` when missing:

```text
Python.Python.3.12
LLVM.LLVM
```

If you already have compatible Python and Clang, setup reuses them.

## Useful Windows commands

Rebuild only the DLLs:

```powershell
.\scripts\build_win32.ps1
```

Download/resume only the model and tokenizer:

```powershell
.\scripts\download_model.ps1
```

Run the lightweight native direct-I/O sanity manually:

```powershell
$env:PYTHONPATH = "$PWD\qwen38"
$env:QWEN38_EXPF_COMPAT_LIB = "$PWD\build\win32\qwen_glibc_expf_compat.dll"
.\.venv\Scripts\python.exe -u runtime\win32_generate.py sanity --build-dir build\win32
```

Rebuild the K3 trunk only:

```powershell
.\.venv\Scripts\python.exe -u runtime\win32_generate.py prepare `
  --model models\Qwen3.8-27B-Q6_K_L.gguf `
  --work-dir work\k3 `
  --build-dir build\win32
```

If you deliberately want setup to skip the K3 preparation step:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -SkipPrepare
```

The first actual run will then prepare it automatically.

## Repository layout

```text
.
├── qwen38/                  # exact migrated research/runtime baseline
├── runtime/
│   └── win32_generate.py    # Windows-facing generator adapter
├── scripts/
│   ├── build_win32.ps1      # exact DLL bundle builder
│   └── download_model.ps1   # low-RAM resumable downloader
├── setup.ps1                # one-time Windows setup
├── run.ps1                  # one-prompt launcher
├── chat.ps1                 # simple interactive shell
├── .github/workflows/
└── README.md
```

Large GGUF files, K3 trunks, generated DLLs, virtual environments and runtime work files are intentionally excluded from Git.

## WSL / Linux

The historical Linux/WSL developer path still exists in `qwen38/`, but it is no longer the recommended user installation path for this repository. The project now targets native Windows packaging first.

Do not interpret this as removing Linux evidence: Linux remains the arithmetic reference used by the cross-platform exactness work.

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
- current-best Q6 workers: `2`

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
