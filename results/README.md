# Results

The full three-model comparison has not been completed. No leaderboard or formal aggregate hallucination rate is published yet.

Development evidence is documented in [the validation record](../docs/validation.md): 38 passing tests, audited official annotation pools, and real Qwen3-VL-4B debug inference. Debug examples validate the pipeline and protocol; they are not formal benchmark estimates.

To generate a clearly labelled replay report without model weights:

```bash
python -m vlm_eval demo --directory artifacts/eval-demo
python -m vlm_eval run --config artifacts/eval-demo/config.yaml
```

Formal publication will include model revisions, sample/protocol fingerprints, completion counts, POPE variant tables, grounding outcome breakdowns, image-cluster confidence intervals, and reviewed failure cases. Raw images and weights are not redistributed by this repository.
