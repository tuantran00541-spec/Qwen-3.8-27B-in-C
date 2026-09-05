# Qwen3.8-27B pruning/compression research lane

This directory is isolated research code for `test/qwen3.8-27b`. It is not part of Manga Translator production inference paths.

## Provenance and pin

The lane continues the earlier `tuantran00541-spec/In-c-test` experiment at commit `a33cff1b35c419dcc97ec52e874c9b56fa3c7eb9`; it does not restart the investigation or copy unrelated Kimi runtime code.

Pinned model source:

- repository: `Qwen/Qwen3.8-27B`
- revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- language hidden size: 5120
- SwiGLU intermediate size: 17408
- language layers: 64
- MLP tensor prefix: `model.language_model.layers.{layer}.mlp`

The runner validates config and all MLP tensor names before loading real weights. Tokenizer/config/weights must not be mixed across revisions.

## Historical baseline from In-c-test

The earlier real-weight layer-0 MLP run measured deterministic synthetic RMS-normalized activations:

| Variant | Relative L2 | Output cosine |
| --- | ---: | ---: |
| Q6 only | ~0.01145 | ~0.999936 |
| 5.15% norm prune only | ~0.03937 | ~0.999265 |
| 5.15% norm prune + Q6 | ~0.04051 | ~0.999202 |

These are one-layer measurements, not a full-model quality PASS. Storage estimates are projections, not physically packed model sizes.

## Manga-side v1 real result: raw activation score failed

Run `33314915626`, job `99266497668`, artifact `qwen38-sparse-aware-evidence-33314915626` (artifact ID `9733151720`, digest `sha256:85c3e44414cbb7887932723e9be3d3786b0fb35bc6a93661c7c800b83a0f01fa`) completed successfully on real BF16 layer-0 weights at the pinned revision.

Measured Q-only relative L2:

- Q6: 0.0110832
- Q5: 0.0234378
- Q4: 0.0492856

Measured prune-only relative L2:

| Actual prune | Weight norm | Raw activation energy |
| ---: | ---: | ---: |
| 5.147% | 0.0378318 | 0.0581582 |
| 10.294% | 0.0590623 | 0.0982869 |
| 14.706% | 0.0781544 | 0.1323941 |

So the first activation-only ranking is worse than the old norm baseline at every tested ratio. At 5.147% pruning, only 94/896 pruned channels overlap between the two rankings (~10.5% of the pruned set). This is evidence against blindly replacing norm ranking with the raw 8-sample activation score.

## V2 knife: activation-guided reconstruction refinement

Raw activation energy remains a diagnostic/proposal signal:

`E[(SiLU(gate_j(x)) * up_j(x))^2] * ||down[:,j]||_2^2`

V2 does not let that signal replace the baseline directly. It:

1. starts from the weight-norm prune set;
2. uses activation energy to identify baseline-pruned channels worth rescuing and baseline-kept channels worth replacing;
3. evaluates fixed-size channel swaps with exact calibration MLP reconstruction;
4. selects the lowest reconstruction-error candidate;
5. always includes the original norm set as fallback, so calibration reconstruction cannot regress by construction;
6. evaluates the selected set on disjoint held-out synthetic hidden states to detect overfitting.

Calibration is increased to 64 samples and evaluation to 16 samples. The runner also records split-half activation-score correlation and split-half pruned-set overlap.

Current comparisons remain:

- pruning around 5%, 10%, 15%, aligned to 128 channels;
- old weight norm vs raw activation energy vs reconstruction-refined pruning;
- BF16 prune-only;
- Q6/Q5/Q4-only;
- prune + Q6/Q5/Q4;
- relative L2, cosine, RMSE, max absolute error;
- measured reconstruction diagnostics and overlap;
- storage projections explicitly marked projection-only.

The calibration/evaluation inputs are still synthetic hidden states. Real prompt/image activations, hidden-state/logit divergence, deterministic generation and vision semantics remain future gates.

## CI discipline

Normal pushes run only synthetic sanity. Heavy real-weight work is opt-in:

- commit marker `[qwen-real]`: run layer 0 only;
- commit marker `[qwen-reps]`: run representative layers `0,3,16,31,47,63` sequentially in one job;
- manual dispatch: choose a comma-separated layer list.

Each real layer uses an isolated temporary model directory which is deleted after the layer finishes. Artifacts contain JSON metrics only; model shards/weights are never uploaded or committed.
