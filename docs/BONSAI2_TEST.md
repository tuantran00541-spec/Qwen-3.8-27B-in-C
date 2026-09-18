# Bonsai 2 27B experiment

This branch tests PrismML's Ternary Bonsai 2 27B as a possible replacement for
the SSD-streamed Qwen3.8-27B Q6_K_L baseline.

The first phase is intentionally reference-first. It does **not** replace the
proven Q6 runtime. We benchmark PrismML's own llama.cpp fork on the same Windows
machine, then use the existing Qwen3.8 GGUF tooling to inventory the new model.
Only after the reference numbers are known should the project port the ternary
kernels and Hadamard activation path into the native C runtime.

## Pinned external inputs

Bonsai 2 model repository:

- `prism-ml/Ternary-Bonsai-2-27B-gguf`
- revision `6ed5e12bf84b7a63069882c91dd9e9218647d17b`

Reference runtime:

- `PrismML-Eng/llama.cpp`
- release `prism-b10683-d8f26ee`

Model packs under test:

| pack | bytes | SHA256 |
|---|---:|---|
| PTQ1_0 | 5,946,648,928 | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| PQ2_0 | 7,206,168,928 | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

The vision projector is deliberately excluded from this phase. This is a
text-runtime and CPU throughput experiment.

## Why the old Q6 runtime cannot simply load this model

Bonsai 2 keeps the Qwen3.8 architecture but changes the low-bit representation
and runtime math.

The Prism fork defines:

- `GGML_TYPE_PQ2_0 = 142`, group 128, 34 encoded bytes;
- `GGML_TYPE_PTQ1_0 = 143`, group 128, 28 encoded bytes.

The weights are stored in a rotated basis. The matching blockwise Hadamard
transform must be applied to activations at runtime. Adding only the two GGUF
type ids would make the file parseable but would not make inference correct.

This branch therefore adds directory parsing first, while leaving the proven Q6
matvec/generation path untouched.

## Windows reference setup

Download the pinned Prism CPU runtime and the smaller PTQ1_0 model:

```powershell
.\scripts\setup_bonsai2.ps1 -Quant PTQ1_0
```

Download both formats for an A/B comparison:

```powershell
.\scripts\setup_bonsai2.ps1 -Quant Both
```

Downloads are resumable and the GGUF files are checked against their exact
published SHA256 and byte length.

## Fast GGUF inventory

This scans only GGUF metadata/directory data and does not read all model weights:

```powershell
.\.venv\Scripts\python.exe .\scripts\bonsai2_inventory.py \
  --model .\models\bonsai2\Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --output .\work\bonsai2\ptq1-inventory.json
```

It reports model architecture, tensor type counts, per-layer type counts,
embedding/output types, and the low-bit payload size.

## CPU benchmark

Full comparison on the target laptop:

```powershell
.\scripts\bench_bonsai2.ps1 -Quant Both
```

The default benchmark is the standard llama.cpp pair:

- PP512: prompt processing over 512 tokens;
- TG128: token generation over 128 tokens.

The script automatically sweeps useful thread counts up to the machine's logical
CPU count and writes raw output plus peak process working set to
`work\bonsai2\`.

For a quick smoke test:

```powershell
.\scripts\bench_bonsai2.ps1 -Quant PTQ1_0 -Threads 4 -PromptTokens 32 -GenTokens 8 -Repetitions 1
```

## Decision gate before porting kernels

Port Bonsai 2 into the native C runtime only if the reference run is compelling
on the target i7-1355U/16 GB Windows laptop.

Record at minimum:

1. PTQ1_0 vs PQ2_0 PP512 and TG128.
2. Peak working set.
3. Best thread count.
4. Cold-load behavior.
5. Output sanity on a small fixed prompt set.
6. Comparison with the existing Q6_K_L runtime on the same prompts/hardware.

If Bonsai 2 wins, the next implementation phase is:

1. preserve these new GGUF parser types;
2. add exact scalar reference decoders for PQ2_0/PTQ1_0;
3. reproduce Prism's activation Hadamard transform exactly;
4. add x86 CPU matvec kernels behind a separate runtime path;
5. A/B hidden-state/logit outputs against the pinned Prism reference;
6. only then consider replacing K3 SSD streaming for the resident ternary model.

Do not route Bonsai 2 through the existing Q6 kernels or treat successful GGUF
parsing as proof of correct inference.
