"""Configuration is validated before model loading or result-directory mutation."""

from pathlib import Path

import yaml

DEFAULT_PROMPTS = {
    "pope": "{query}\nAnswer only Yes or No.",
    "existence": "Does any object in this image match the complete description: {query}?\nAnswer only Yes or No.",
    "grounding": (
        'Locate the object matching this complete description: {query}\n'
        'Return only JSON with exactly two keys: "status" and "boxes". '
        'status must be "found", "absent", or "uncertain". '
        'If exactly one matching object is visible, return its bounding box as [[x_min,y_min,x_max,y_max]]. '
        'If no object matches, use status "absent" and boxes []. '
        'If you cannot determine the answer, use status "uncertain" and boxes []. '
        'Use {coordinate_instruction}. Do not invent a box.'
    ),
}

COORDINATE_INSTRUCTIONS = {
    "normalized": "xyxy coordinates normalized to [0,1] relative to the supplied image",
    "normalized_1000": "xyxy coordinates normalized to [0,1000] relative to the supplied image",
    "pixels": "xyxy pixel coordinates relative to the supplied {width} by {height} image",
}


def _keys(value, allowed, name):
    if not isinstance(value, dict) or set(value)-set(allowed):
        raise ValueError(f"Invalid or unknown {name} keys: {set(value) - set(allowed) if isinstance(value, dict) else value}")


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _keys(raw, {"name", "model", "data", "output_dir", "paired_existence", "seed", "scoring", "prompts", "max_consecutive_errors"}, "config")
    for key in ("name", "model", "data", "output_dir"):
        if key not in raw:
            raise ValueError(f"Missing configuration field: {key}")
    model = raw["model"]
    _keys(model, {"backend", "id", "revision", "coordinate_format", "device", "dtype", "local_files_only",
                 "responses", "processor_kwargs", "max_new_tokens", "attention", "chat_template",
                 "chat_template_kwargs", "trust_remote_code", "cache_dir"}, "model")
    model = {"backend": "transformers", "revision": "main", "coordinate_format": "normalized_1000",
             "device": "cuda", "dtype": "bfloat16", "local_files_only": True, "max_new_tokens": 128,
             "attention": "sdpa", "processor_kwargs": {}, "chat_template_kwargs": {},
             "trust_remote_code": False, **model}
    if model["backend"] not in {"replay", "transformers"}:
        raise ValueError("backend must be replay or transformers")
    if model["coordinate_format"] not in COORDINATE_INSTRUCTIONS:
        raise ValueError("Invalid coordinate_format")
    if model["dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Invalid dtype")
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise ValueError("model.id is required")
    if type(model["max_new_tokens"]) is not int or model["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    data = raw["data"]
    _keys(data, {"manifests", "split"}, "data")
    if not isinstance(data.get("manifests"), list) or not data["manifests"]:
        raise ValueError("data.manifests must be a nonempty list")
    if data.get("split") not in {"debug", "eval"}:
        raise ValueError("data.split must be debug or eval")
    resolve = lambda p: str((path.parent / p).resolve())
    data = {"split": data["split"], "manifests": [resolve(p) for p in data["manifests"]]}
    for key in ("responses", "chat_template", "cache_dir"):
        if key in model:
            model[key] = resolve(model[key])
    for key in ("local_files_only", "trust_remote_code"):
        if type(model[key]) is not bool:
            raise ValueError(f"model.{key} must be boolean")
    if model["backend"] == "replay" and "responses" not in model:
        raise ValueError("Replay requires a response file")
    scoring = {"iou_threshold": 0.5, "bootstrap_samples": 1000, **raw.get("scoring", {})}
    _keys(scoring, {"iou_threshold", "bootstrap_samples"}, "scoring")
    if not 0 < scoring["iou_threshold"] <= 1 or type(scoring["bootstrap_samples"]) is not int or scoring["bootstrap_samples"] < 0:
        raise ValueError("Invalid scoring settings")
    prompts = {**DEFAULT_PROMPTS, **raw.get("prompts", {})}
    _keys(prompts, DEFAULT_PROMPTS, "prompts")
    for task, template in prompts.items():
        if not isinstance(template, str) or "{query}" not in template:
            raise ValueError(f"Prompt must contain {{query}}: {task}")
        template.format(query="test", coordinate_instruction="test", width=100, height=100)
    config = {"name": raw["name"], "model": model, "data": data, "output_dir": resolve(raw["output_dir"]),
              "paired_existence": raw.get("paired_existence", True), "seed": raw.get("seed", 42),
              "scoring": scoring, "prompts": prompts, "max_consecutive_errors": raw.get("max_consecutive_errors", 3)}
    if type(config["paired_existence"]) is not bool or type(config["seed"]) is not int:
        raise ValueError("paired_existence must be boolean and seed must be integer")
    if type(config["max_consecutive_errors"]) is not int or config["max_consecutive_errors"] < 1:
        raise ValueError("max_consecutive_errors must be positive")
    return config


def make_prompt(config, sample, task, size):
    instruction = COORDINATE_INSTRUCTIONS[config["model"]["coordinate_format"]].format(width=size[0], height=size[1])
    return config["prompts"][task].format(query=sample["query"], coordinate_instruction=instruction,
                                          width=size[0], height=size[1])
