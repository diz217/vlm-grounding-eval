"""Generate a tiny synthetic fixture; deliberately contains correct and failed answers."""

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .data import export_manifest, pad_boxes
from .io import write_json


def create_demo(directory):
    directory = Path(directory).resolve()
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows, responses = [], {}
    for index in range(6):
        image = Image.new("RGB", (160,120), "white")
        draw = ImageDraw.Draw(image)
        present = index not in {2,3}
        draw.rectangle((24,24,72,72), fill="red" if present else "blue")
        path = images / f"fixture_{index}.png"
        image.save(path)
        sample = {"id": f"fixture:{index}", "image_id": f"fixture:{index}", "image": str(path),
                  "task": "grounding", "query": "the red square", "target_present": present,
                  "boxes": [[0.15,0.2,0.45,0.6]] if present else [],
                  "group": "fixture", "split": "eval", "source": "synthetic_test_fixture",
                  "negative_type": None if present else "category_absent"}
        rows.append(sample)
        responses[f"fixture:{index}::existence"] = "Yes" if present else "No"
    correct = json.dumps({"status": "found", "boxes": [[0.15,0.2,0.45,0.6]]})
    responses.update({"fixture:0::grounding": correct,
                      "fixture:1::grounding": json.dumps({"status": "found", "boxes": [[0.6,0.2,0.9,0.6]]}),
                      "fixture:2::grounding": correct,
                      "fixture:3::grounding": json.dumps({"status": "absent", "boxes": []}),
                      "fixture:4::grounding": json.dumps({"status": "uncertain", "boxes": []}),
                      "fixture:5::grounding": "I would draw a box somewhere in the image."})
    original = rows[0]
    transform = {"kind": "pad_shift", "left": 40, "top": 30, "right": 10, "bottom": 10}
    transformed = dict(original, id="fixture:0:pad_shift", parent_id="fixture:0", transform=transform,
                       transform_reviewed=True, group="fixture/pad_shift",
                       boxes=pad_boxes(original["boxes"], (160,120), transform))
    rows.append(transformed)
    responses[transformed["id"]+"::grounding"] = json.dumps({"status": "found", "boxes": transformed["boxes"]})
    responses[transformed["id"]+"::existence"] = "Yes"
    for variant in ("random", "popular", "adversarial"):
        for index in (0,2):
            sample = dict(rows[index], id=f"fixture:pope:{variant}:{index}", task="pope",
                          group=variant, query="Is there a red square in the image?", boxes=[])
            rows.append(sample)
            responses[sample["id"]+"::pope"] = "Yes" if sample["target_present"] else "No"
    export_manifest(directory / "manifest.jsonl", rows)
    write_json(directory / "responses.json", responses)
    config = {"name": "REPLAY-fixture-not-a-benchmark", "seed": 42,
              "model": {"backend": "replay", "id": "recorded-test-fixture", "responses": "responses.json",
                        "coordinate_format": "normalized"},
              "data": {"manifests": ["manifest.jsonl"], "split": "eval"},
              "paired_existence": True, "output_dir": "run", "scoring": {"bootstrap_samples": 200}}
    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"config": str(directory / "config.yaml"), "synthetic": True, "samples": len(rows)}
