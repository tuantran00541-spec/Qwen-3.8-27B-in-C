# Real-weight trigger record

Current trigger scope: Qwen sparse-aware knife V2 layer-0 validation.

- scope: layer 0 MLP only
- model: Qwen/Qwen3.8-27B
- revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
- pruning: aligned ~5%, ~10%, ~15%
- scorers: legacy weight norm, raw activation energy, activation-guided reconstruction refinement
- refinement: norm baseline is always a fallback candidate
- quantization: Q6/Q5/Q4, group size 128, symmetric RTN simulation
- calibration: 64 deterministic synthetic RMS-normalized hidden states
- held-out evaluation: 16 disjoint deterministic synthetic RMS-normalized hidden states
- evidence: JSON metrics artifact only; no model weights

A workflow success only means the experiment completed. It is not a full-model or semantic quality PASS.
