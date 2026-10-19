# VLM Grounding Eval

A config-driven evaluation harness for **POPE object-existence questions + a coordinate grounding probe**, using Hugging Face Transformers.

The central question is: **when a VLM says an object exists, can it locate that object—and when no object matches, does it still invent coordinates?** The project compares object-existence answers, spatial predictions, and responses to reviewed image-position changes across open-weight VLMs.

**Status: implementation + real-model debug validation.** 38 tests pass; Qwen3-VL-4B completed 12 real debug requests. Full three-model benchmark results are pending. Replay results verify software only and are labelled synthetic. See the [project scope and roadmap](docs/roadmap.md), [evaluation protocol](docs/protocol.md), and [validation record](docs/validation.md).

Historical classifier files are archived locally and excluded from this repository. This repository contains only the evaluation project; datasets, weights, caches, and machine-local run directories are downloaded or generated separately.

## Research questions

1. How do object-existence errors differ from no-target coordinate errors on the same image/expression?
2. Can a model reliably withhold a box when no referent matches, while retaining localization capability when one does?
3. Do predictions remain correct when reviewed image-position changes alter the target's coordinates?

The intended deliverable is a reproducible comparison of Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct, and InternVL3.5-8B-HF, with per-sample evidence, confidence intervals, and a behavioral failure taxonomy. These are full-system comparisons; they do not isolate causal vision-versus-language mechanisms.

## What is measured

| Component | Measurements |
|---|---|
| Original COCO POPE | Accuracy, precision, recall, F1, yes ratio, invalid output; random/popular/adversarial separately |
| With-target grounding | Correct box, wrong box, empty, uncertain, invalid, inference error |
| No-target grounding | False-box rate, empty, uncertain, invalid, inference error |
| Reviewed position change | Original and transformed image both localized correctly |
| Independent existence question on grounding samples | `No + nonempty box` and `Yes + empty box` disagreements |

No combined hallucination score. Abstention does not establish calibrated honesty; transformation failure does not prove image neglect. No-target means no match to the **complete expression**, not necessarily an absent category.

## Quick verification: no GPU, weights, or external datasets

Python 3.10+; development host uses Python 3.11.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m vlm_eval demo --directory artifacts/eval-demo
python -m vlm_eval validate --config artifacts/eval-demo/config.yaml
python -m vlm_eval run --config artifacts/eval-demo/config.yaml
python -m vlm_eval run --config artifacts/eval-demo/config.yaml --resume
```

Open `artifacts/eval-demo/run/report.md`. Prerecorded fixture responses intentionally include wrong boxes, no-target false boxes, uncertainty, and invalid output. Every report is labelled **REPLAY FIXTURE — NOT model benchmark results**.

## Real inference

Install a CUDA-compatible PyTorch wheel, then `python -m pip install -e ".[models,test]"`. [requirements-eval.lock.txt](requirements-eval.lock.txt) records tested direct versions; it is not a full transitive or cross-platform CUDA lock. No vLLM, Linux, FlashAttention, or Accelerate required. The backend runs sequentially on one explicitly selected device.

Candidate configs:

- `configs/eval/qwen3vl_4b.yaml`: `Qwen/Qwen3-VL-4B-Instruct`
- `configs/eval/qwen3vl_8b.yaml`: `Qwen/Qwen3-VL-8B-Instruct`
- `configs/eval/internvl35_8b.yaml`: `OpenGVLab/InternVL3_5-8B-HF`, the official Transformers format

Check fit, processor behavior, and coordinate output on debug data before formal evaluation. Default: BF16, greedy decoding, batch size 1. Model-specific visual-token budgets differ and are recorded; this is not automatically an equal-compute comparison.

```bash
python -m vlm_eval pin --config configs/eval/qwen3vl_4b.yaml --output artifacts/frozen/qwen3vl_4b.yaml
python -m vlm_eval download --config artifacts/frozen/qwen3vl_4b.yaml
python -m vlm_eval validate --config artifacts/frozen/qwen3vl_4b.yaml
python -m vlm_eval run --config artifacts/frozen/qwen3vl_4b.yaml
```

`pin` records an immutable Hub SHA and machine-local resolved paths. `download --metadata-only` retrieves processor/config files without weights. Downloading into `model.cache_dir` is explicit; evaluation uses `local_files_only: true`. On managed Windows hosts with OS certificate stores, add `--system-trust` to `pin`/`download`; TLS verification stays enabled.

Grounding output contract:

```json
{"status":"found","boxes":[[100,200,500,600]]}
```

For no match: `{"status":"absent","boxes":[]}`. For uncertainty: `{"status":"uncertain","boxes":[]}`. Config declares `normalized`, `normalized_1000`, or `pixels`; units are never guessed. Predictions are converted to normalized xyxy. No grammar-constrained decoding, coordinate clipping, or repair is used. This evaluates prompt-based grounding under this contract, not a model's specialized detection API.

## Prepare real datasets

Download original project annotations and COCO images separately; do not commit data, weights, or checkpoints.

The CLI can fetch revision-pinned annotations directly from the projects' official repositories:

```bash
python -m vlm_eval fetch annotations --directory data/downloads
```

This writes source URLs/revisions/hashes. After preparing manifests, download only their selected COCO images with `python -m vlm_eval fetch images --manifests MANIFEST.jsonl`. Image downloads use the original COCO public S3 bucket with verified HTTPS. Use `prepare gref --defer-image-validation` to select samples before downloading images, then validate after fetching. Fetch commands require the model extras (their HTTP dependency).

### Original COCO POPE

Use published original files, not POPEv2 or newly sampled questions. Import each of the three variants:

```bash
python -m vlm_eval prepare pope --input data/downloads/pope/coco_pope_random.json --images data/raw/coco/val2014 --variant random --output data/eval/pope/random.jsonl
```

Repeat for `popular` and `adversarial`. Accepts official JSONL or JSON arrays. Source hashes are retained. Strict yes/no metrics and the original POPE heuristic are reported separately: the original heuristic defaults unrecognized generated answers to Yes.

### gRefCOCO

Use official JSON `grefs(unc).json`, `instances.json`, and COCO train2014 images. Downloaded pickle files are not loaded.

```bash
python -m vlm_eval prepare gref --refs "data/downloads/gref/grefs(unc).json" --instances data/downloads/gref/instances.json --images data/raw/coco/train2014 --output-dir data/eval/gref --debug-count 50 --eval-count 500 --defer-image-validation
```

Counts are per presence condition: 50 positive + 50 no-target debug, 500 positive + 500 no-target formal. Defaults: official train for debug, testA for formal. In the downloaded official revision, val contains no single-target positive refs, so it cannot support this balanced zero/single-target protocol. At most one query per image per condition; debug images excluded from formal. Validation also rejects identical image bytes across splits. `pool_audit.json` is written even if requested counts cannot be met; insufficient unique images fail rather than being repeated.

Imported negatives are `unspecified_no_match`. Category-absent versus attribute/relationship mismatch requires separate annotation review. Source ambiguity still needs human review.

### Reviewed position transformations

Create a JSON array of positive sample IDs after manually checking unique referents and expression validity under padded scene translation, e.g. `["gref:123:456"]`.

```bash
python -m vlm_eval prepare transforms --manifest data/eval/gref/eval.jsonl --approved-ids data/eval/gref/reviewed_ids.json --output data/eval/gref/transforms.jsonl
```

Add the transform manifest to every compared config alongside originals. All source pixels/context are retained; boxes update analytically. No cropping, flipping, or automatic query rewriting. Padding still changes context, so this is a specified robustness intervention. Pure resize under normalized coordinates is not a position-change test.

## Run artifacts

| File | Content |
|---|---|
| `config.resolved.json` | Complete prompts and resolved settings |
| `metadata.json` | Model SHA, template/hash, processor, generation settings, versions, fingerprints, status |
| `requests.jsonl` | Planned requests, image hashes, ground truth |
| `predictions.jsonl` | Raw answers, rendered prompts, units, time, errors |
| `scores.jsonl`, `failures.jsonl` | Every outcome and reviewable failures |
| `summary.json`, `report.md` | Counts, metrics, image-cluster intervals, readable tables |

Resume rejects changed data/images, prompts, implementation, model settings or scoring. It skips recorded requests and repairs an incomplete final JSONL line after a crash. Use `--resume --retry-errors` to retry recorded inference errors. Repeated errors stop the run; missing planned requests remain errors in report denominators. A completed resume does not reload weights.

```bash
python -m vlm_eval report artifacts/eval/qwen3vl-4b
python -m vlm_eval compare artifacts/eval/qwen3vl-4b artifacts/eval/qwen3vl-8b artifacts/eval/internvl35-8b --output artifacts/eval/comparison.md
```

Comparison requires identical sample/protocol fingerprints. Seeded bootstrap intervals resample original image IDs, keeping related queries together. They measure sample uncertainty, not GPU nondeterminism. Inspect model-specific processor/template settings before publishing comparisons.

## References

- [POPE official code and paper](https://github.com/RUCAIBox/POPE)
- [GREC/gRefCOCO official dataset](https://github.com/henghuiding/gRefCOCO)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [InternVL](https://github.com/OpenGVLab/InternVL)

This project does not claim to invent no-target grounding or infer causal vision/language failure mechanisms.
