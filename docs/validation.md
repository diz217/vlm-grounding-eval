# Implementation validation — 2026-10-18

This is a development validation record, not a completed multi-model benchmark.

## Host

Windows; NVIDIA RTX 3090 Ti (24 GB); Python 3.11; PyTorch 2.5.1+cu121; Transformers 5.2.0. Actual single-GPU inference works without vLLM or Accelerate.

## Software checks

- Core tests cover coordinate-unit conversion, IoU, strict and original-compatible POPE parsing, empty/uncertain/wrong/invalid outcomes, malformed model responses, split/image-byte leakage, sampling without replacement, reviewed transform geometry, incomplete run denominators, crash recovery, resume invariants, and incompatible-run comparisons.
- Replay fixture covers all three POPE variant paths, paired existence questions, false boxes, incorrect localization, uncertainty, invalid output, and a position-transform pair. Reports explicitly say synthetic, not model measurements.
- Formal preflight succeeds: 9,000 POPE requests + 1,000 grounding requests + 1,000 independent existence requests = 11,000 planned requests per model, before any reviewed formal transformations are added.

## Actual checkpoint verification

Qwen/Qwen3-VL-4B-Instruct, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`:

- Weights explicitly downloaded into the workspace cache.
- Native processor/chat rendering verified offline.
- Initial 4-image debug smoke: 8/8 requests completed.
- Reviewed debug integration: 12/12 requests completed, covering genuine image grounding/existence, two custom yes/no contract checks, explicit no-target examples, and one visually reviewed padded-position pair. These custom yes/no examples are **not the official POPE benchmark**.
- Completion means generation succeeded, not that every answer was correct. The transformed grounding answer contained an extra nesting level in its box JSON and was recorded as invalid, not silently repaired or presented as a geometric failure.
- Raw outputs, full prompts, normalized scores, resolved processor/template, checkpoint revision, timings, and metadata are under `artifacts/eval/qwen3vl-4b-reviewed-debug/` on the development host.
- The initial smoke contained an ambiguous no-target expression (`first guy with red shorts on top`): a person visibly wears red shorts. It is flagged in `artifacts/eval/debug-label-review.json` and excluded from the separately reviewed debug check. This is debug annotation review, not a post-hoc formal benchmark exclusion. Do not use that example as proof of hallucination.

OpenGVLab/InternVL3_5-8B-HF, revision `741a7d03020411e666c6109218ab71e08151ef86`:

- Native Transformers config/processor metadata downloaded and loaded.
- Image + rendered-chat encoding succeeds using `InternVLProcessor`.
- Weight loading and generation are **not yet verified**. Its default tiling differs substantially from Qwen; audit the memory/token budget before a formal comparison.

Qwen/Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`: revision resolved; weight loading and generation **not yet verified**.

## Actual dataset audit

Annotations fetched from official sources and hashed in `data/downloads/sources.json`:

- gRefCOCO revision `81eede59b3ac070049f597d023c0ff08d1fb80e9`.
- POPE revision `08d957b917e5a378a2f99d35b6293c536a66298b`.

After the importer's zero/single-target eligibility filtering and image/query deduplication:

| Official split | Positive query pairs | Positive unique images | No-target query pairs | No-target unique images |
|---|---:|---:|---:|---:|
| train | 113,311 | 16,994 | 19,118 | 11,466 |
| val | 0 | 0 | 8,866 | 1,499 |
| testA | 5,647 | 750 | 4,315 | 749 |
| testB | 5,303 | 749 | 4,340 | 749 |

The downloaded revision has no single-target positives in val. The implementation therefore defaults formal evaluation to **testA**, not val, and retains train for debug. It selects 50 positive + 50 negative debug samples and 500 positive + 500 negative formal samples without repeating an image within a condition. Positive/negative conditions may share an image; confidence intervals cluster by original image.

Original COCO POPE: 3,000 questions per variant, 9,000 total. All selected POPE/gRefCOCO images have been downloaded: **1,253 distinct source files**. Full combined image verification and debug/formal split checks pass. The pool audit is at `data/eval/gref/pool_audit.json`.

## Remaining study work

- Source-label and formal transform-eligibility review. No formal transform IDs have been auto-approved.
- Actual 8B checkpoint generation/memory checks.
- Freeze reviewed formal manifests and complete real runs for all candidate models.
- Publish a real cross-model table with intervals, error counts, case review, and model-specific preprocessing limitations.

The three-model comparative study remains in progress; this validation record establishes the current implementation and debug evidence only.
