# Evaluation protocol v1

## Scope

Original COCO POPE plus zero/single-target coordinate grounding. For each grounding sample, optionally ask a separate existence question about its complete expression. Requests are independent: no previous answer, label, or ground-truth box is supplied to the backend.

## Data

- Preserve original POPE random/popular/adversarial sets; no benchmark resampling.
- gRefCOCO: exclude multi-target, crowd, invalid-box, empty-query samples. Deduplicate image/normalized-query pairs. At most one query per image per presence condition.
- Default debug source is official train, formal source is official testA. The audited official revision has zero single-target positives in val. Exclude debug image IDs from formal even when source-split settings change. Verify exact image-byte duplicates across splits. Near-duplicate detection beyond byte identity is a remaining limitation.
- Write eligible query and unique-image counts before sampling; insufficient pools fail, without replacement.
- If debug images overlap standard POPE, combined validation fails. Change the debug selection before prompt tuning; do not drop official POPE questions silently.
- All variants stay with the original image's split. Only transform explicitly reviewed positive IDs; verify that the full expression stays valid and uniquely identifies the same target.
- No-target means no match to the expression. It does not establish category absence. Imported negatives remain `unspecified_no_match` until separately annotated.
- Human review of source label correctness/ambiguity and transform eligibility remains necessary.

## Grounding contract

Canonical ground truth is normalized xyxy. No EXIF rotation: annotations use the stored pixel grid. Transformed boxes use the new canvas dimensions. Predictions declare their coordinate units per config; units are not guessed. Nonfinite, degenerate, out-of-range, malformed or contradictory status/box output is invalid, never repaired into a success.

Output JSON has `status` (`found`, `absent`, `uncertain`) and `boxes`. Empty/uncertain answers are explicitly allowed. Main runs use unconstrained generation with an output instruction. Multiple predicted boxes do not count as successful single-target localization. Preserve raw answers and truncation flags.

## Position intervention

v1 `pad_shift` preserves all source pixels in a larger neutral canvas with asymmetric margins. Update GT analytically. Include original and transformed views, with the same expression and original-image ID. Padding still changes image context. No arbitrary geometric-invariance claim; no conclusion that a failure proves the model ignored vision. Cropping/flipping require separate validity checks and a new protocol version. Normalized coordinates remain unchanged under pure resize.

## Outcomes and denominators

- Positive samples: correct box at IoU >= 0.5, wrong box, empty/absent, uncertain, invalid, inference error. Each rate uses all positives; outcomes sum to one. Empty is a task failure, distinct from a wrong box, and is not automatically calibrated honesty.
- Negative samples: valid nonempty false box, empty/absent, uncertain, invalid, inference error. Rates use all negatives. Invalid coordinates are separately visible, not successful abstention.
- Transform metric: both original and transformed views must localize correctly. Stable wrong outputs receive no credit. This joint rate reflects capability plus robustness; inspect the separate original/transform groups too.
- Existence/grounding disagreements: `No + valid nonempty box`, `Yes + empty/absent`. Agreement is not correctness; retain invalid/uncertain counts.
- Strict POPE: leading yes/no parsing. Invalid/error outputs fail accuracy and positive recall denominators. Precision is conditional on valid positive predictions and must accompany response coverage. F1 includes uncompleted positive tasks as missed positives.
- Original POPE compatibility: independently reproduce its first-sentence token heuristic, including default Yes for unrecognized generated responses. Do not coerce inference failures to a label.
- Zero denominators are null. Missing planned requests count as errors. No aggregate hallucination score across tasks.
- Confidence intervals: percentile bootstrap over original image IDs; related questions/variants are not independent samples. Intervals do not quantify GPU decoding nondeterminism.

## Freeze list and comparison

Freeze data/sample IDs, image hashes, transform parameters and reviewed-ID hash, model/tokenizer/processor revision, template text/hash, complete prompt, coordinate units, processor settings, dtype, attention implementation, decoding/token budget, seed, package versions, source-code hashes, and scoring. Save resolved model SHA and generation defaults. Greedy generation and a fixed seed do not guarantee bitwise reproducibility across hardware/libraries.

Candidates: Qwen3-VL 4B/8B Instruct for scale; InternVL3.5 8B official HF format for cross-family comparison. These are full-system comparisons, not isolated encoder/training effects. Native visual-token budgets differ; processor settings must be audited before any equal-compute claim. v1: no thinking prompt, greedy, batch size 1, BF16 where supported.

Formal CLI evaluation requires a pinned revision. Resume rejects changed fingerprints or resolved backend environment. Comparison requires identical sample/protocol fingerprints; inspect legitimately model-specific processor/template metadata separately. Sequential request latency includes input processing and transfer as well as generation; it is not a serving-throughput benchmark.

## Failure taxonomy

Existence false positive/negative; no-target false box; wrong box; empty; uncertain; invalid; inference error; transform joint failure; existence/grounding disagreement. Prior override and perception failure are possible interpretations for reviewed cases, not automatically assigned causal labels.

## Release gates

1. Core scoring/data/resume tests pass.
2. Replay fixture exercises success and failure cases, labelled synthetic.
3. Each real checkpoint passes processor/generation/debug checks; verify coordinate interpretation and prompts.
4. Real pool audit and human review complete.
5. Formal data, model revisions, prompts, preprocessing and scoring frozen.
6. Actual model runs complete; publish tables, error counts, per-sample evidence, intervals, limitations.

Implementation completion is not benchmark-study completion. Passing gates 1–2 does not imply gates 3–6 passed.
