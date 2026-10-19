# Project scope and roadmap

## Purpose

Evaluate the reliability of VLM object-existence judgments and coordinate grounding under a reproducible protocol. The software harness is the implementation; the comparative study is the intended outcome.

## Fixed scope for the first study

| Part | Definition | Output |
|---|---|---|
| Standard baseline | Original COCO POPE random/popular/adversarial, 3,000 questions each | Separate strict and official-compatible metrics |
| Grounding probe | gRefCOCO zero/single-target expressions; 500 positive + 500 no-target formal samples | Correct box, wrong box, empty, uncertain, invalid, inference-error rates |
| Paired existence | Independent existence question on each grounding sample | Existence/coordinate agreement and disagreement cases |
| Position intervention | Explicitly reviewed positive queries; whole-scene padded translation | Original/transformed localization and joint correctness |
| Model comparison | Qwen3-VL 4B/8B Instruct and InternVL3.5 8B official HF format | Matched-protocol tables, image-cluster intervals, raw cases |

Debug and formal data are separated by source image, including all variants. Formal gRefCOCO defaults to testA because the audited official revision has no single-target positives in val. No-target means no match to the complete expression, not necessarily absence of the category.

## Current status

- [x] Data import, source hashes, unique-image pool audit, disjoint sampling.
- [x] Explicit coordinate units and zero/single-target scoring.
- [x] Separate empty, uncertain, wrong-box, invalid and inference-error outcomes.
- [x] Config-driven Transformers inference, checkpoint/template snapshots, resumable JSONL outputs.
- [x] Reports, matched-protocol comparison, image-cluster bootstrap intervals.
- [x] 38 tests and a synthetic replay fixture.
- [x] Qwen3-VL-4B real debug inference, including a reviewed position pair.
- [x] Full source-data preflight: 11,000 planned requests per model before formal transforms.
- [ ] Review formal source-label ambiguity and approve eligible transform queries.
- [ ] Verify weight loading, generation, memory fit, and coordinate contract on both 8B candidates.
- [ ] Freeze the final manifests and model-specific image budgets.
- [ ] Execute all real formal model runs.
- [ ] Publish comparison tables, uncertainty, failure cases and limitations.

## What counts as a result

A completed request is not necessarily a correct answer. A replay fixture is not a VLM measurement. Debug cases are not an unbiased benchmark estimate. Model configurations are not completed evaluations. The first complete study requires real runs and reviewable per-sample evidence for every reported model.

## Outside this release

Model training/fine-tuning; CHAIR; a full MMVP suite; production serving; a new object detector; and causal claims that a particular failure proves image neglect or language-prior override. New research novelty is not claimed for no-target grounding itself.
