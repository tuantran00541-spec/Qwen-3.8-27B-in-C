# Qwen3.8-27B low-RAM CPU runtime

Standalone home for the Qwen3.8-27B CPU-only, SSD-resident runtime originally developed inside `tuantran00541-spec/manga-translator`.

## Proven migration baseline

This repository is initially migrated from:

- source repository: `tuantran00541-spec/manga-translator`
- source branch: `research/qwen3.8-27b-runtime-finish`
- source commit: `8f378cf13786e5e62b64ba540e3a187e351b2160`
- source commit message: `[qwen-win32] bind exact expf in recurrent GDN state`
- proven Windows full-GGUF Actions run: `33954444876`
- evidence artifact: `9966107719`
- evidence artifact SHA256: `f810dab4d61d429383cdc9bb01835a6acc49523ac175183f8769e32faebdb8d2`

The source baseline passed the full real Windows current-best exact gate with the pinned 24 GB GGUF and matched the Linux arithmetic anchors bit-for-bit.

## Pinned model and exact anchors

- model: `Qwen/Qwen3.8-27B`
- official model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- GGUF: `bartowski/Qwen3.8-27B-GGUF / Qwen3.8-27B-Q6_K_L.gguf`
- GGUF SHA256: `a487690b9f17de581857c4ae484dab50800335bb9eb978a4fb02c0465629dc0a`
- hidden SHA256: `e40dfb2d14456006608b095dd0c6bd018cdeed4214fdc573c8e352fb463f2e04`
- persistent-state SHA256: `41f6fcd8f9947833956aaad0175da197456a3e678e0e31b40c5d7a08560fda06`
- exact 11-token K3 bytes: `21127430144`

## Proven runtime contracts

The migration baseline preserves these constraints:

- CPU-only runtime
- low-RAM / SSD-resident weights
- Windows native direct I/O with no-buffering + overlapped I/O
- ring slots: 2
- slot bytes: 336449536
- planned ring bytes: 672899072
- storage I/O concurrency: 1
- max deferred layer requests: 1
- Q6 workers: 2
- Q8 no-allocation path enabled
- no fast-math / reassociation / FMA contraction changes to the exact arithmetic path

## Repository state

The first migration intentionally keeps the proven source layout under `research/qwen38/`. This avoids changing imports and runtime paths while separating the project from the manga application. Cleanup into a user-facing `src/`/`runtime/` layout should happen only after the migrated baseline is re-gated in this repository.

Large model weights are not committed to Git.
