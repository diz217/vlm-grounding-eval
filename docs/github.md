# GitHub repository

Repository name: `vlm-grounding-eval`

Description: `Reproducible POPE + coordinate-grounding evaluation for open-weight VLMs.`

Suggested topics: `vision-language-models`, `evaluation`, `hallucination`, `visual-grounding`, `pope`, `pytorch`, `transformers`.

The public repository contains the evaluation package, configs, tests, CI, protocol, roadmap, and result-status documentation. Local legacy files, original datasets, downloaded weights, caches, generated outputs and machine-specific frozen configs are ignored.

## Normal workflow after initial publication

```bash
git status
git add README.md docs results vlm_eval configs/eval tests .github pyproject.toml requirements-eval.lock.txt .gitignore .gitattributes
git diff --cached --stat
git commit -m "Describe the concrete change"
git push
```

Add an explicit code license before presenting the code as licensed open source; dataset and model licenses are separate. Do not describe the pending three-model study as a completed benchmark.
