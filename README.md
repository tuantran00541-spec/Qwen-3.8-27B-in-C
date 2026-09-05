# Qwen3.8-27B low-RAM CPU runtime

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
