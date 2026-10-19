import json

import pytest
from PIL import Image

from vlm_eval.data import (balanced_subset, export_manifest, gref_candidates, import_pope,
                           load_image, load_manifest, load_manifests, prepare_gref, prepare_transforms)
from vlm_eval.io import write_json


def sample(tmp_path, sid="one", split="eval", color="red"):
    path = tmp_path / f"{sid}.png"
    Image.new("RGB", (100,80), color).save(path)
    return {"id": sid, "image_id": sid, "image": str(path), "split": split,
            "task": "grounding", "query": "the red square", "group": "gref",
            "target_present": True, "boxes": [[0.1,0.1,0.5,0.5]]}


def test_image_id_and_bytes_disjointness(tmp_path):
    a = sample(tmp_path, "a", "debug")
    b = dict(a, id="b", split="eval")
    export_manifest(tmp_path/"a.jsonl", [a,b])
    with pytest.raises(ValueError, match="image overlap"):
        load_manifests([tmp_path/"a.jsonl"])
    b = sample(tmp_path, "b", "eval")  # Different ID/path, identical bytes.
    export_manifest(tmp_path/"b.jsonl", [a,b])
    with pytest.raises(ValueError, match="duplicate image bytes"):
        load_manifests([tmp_path/"b.jsonl"])


def test_manifest_paths_are_relative_to_manifest(tmp_path):
    a = sample(tmp_path)
    path = tmp_path / "nested" / "samples.jsonl"
    export_manifest(path, [a])
    assert load_manifest(path)[0]["image"] == a["image"]


def test_transforms_need_explicit_review_and_correct_gt(tmp_path):
    a = sample(tmp_path)
    export_manifest(tmp_path/"original.jsonl", [a])
    write_json(tmp_path/"approvals.json", ["one"])
    prepare_transforms(tmp_path/"original.jsonl", tmp_path/"approvals.json", tmp_path/"transformed.jsonl")
    rows = load_manifests([tmp_path/"original.jsonl", tmp_path/"transformed.jsonl"])
    transformed = rows[1]
    assert transformed["parent_id"] == "one"
    assert transformed["boxes"] != a["boxes"]
    assert load_image(transformed).size == (141,106)
    transformed["boxes"] = a["boxes"]
    export_manifest(tmp_path/"bad.jsonl", [transformed])
    with pytest.raises(ValueError, match="Incorrect transformed"):
        load_manifests([tmp_path/"original.jsonl", tmp_path/"bad.jsonl"])
    transformed["transform_reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        export_manifest(tmp_path/"unreviewed.jsonl", [transformed])


def test_balanced_sampling_never_repeats_images_to_fill_quota():
    rows = [{"id": str(i), "image_id": "same", "target_present": present}
            for i, present in enumerate([True, True, False, False])]
    with pytest.raises(ValueError, match="Insufficient unique images"):
        balanced_subset(rows, 2, "eval", 42)
    chosen = balanced_subset(rows, 1, "debug", 42)
    assert len(chosen) == 2
    with pytest.raises(ValueError):
        balanced_subset(rows, 1, "eval", 42, {"same"})


def test_official_gref_zero_single_multi_and_audit(tmp_path):
    Image.new("RGB", (100,80), "red").save(tmp_path/"one.jpg")
    instances = {"images": [{"id": 1, "file_name": "one.jpg", "width":100, "height":80}],
                 "annotations": [{"id": 10, "image_id":1, "bbox":[10,8,40,32]}]}
    refs = [{"ref_id":1,"image_id":1,"ann_id":[10],"split":"val",
             "sentences":[{"sent_id":1,"sent":"the car"}]},
            {"ref_id":2,"image_id":1,"ann_id":[-1],"split":"val",
             "sentences":[{"sent_id":2,"sent":"the car right of the bus"}]},
            {"ref_id":3,"image_id":1,"ann_id":[10,10],"split":"val",
             "sentences":[{"sent_id":3,"sent":"the cars"}]}]
    write_json(tmp_path/"refs.json", refs)
    write_json(tmp_path/"instances.json", instances)
    rows, skipped = gref_candidates(tmp_path/"refs.json", tmp_path/"instances.json", tmp_path)
    assert len(rows) == 2 and skipped["multi_target_or_invalid"] == 1
    assert rows[1]["negative_type"] == "unspecified_no_match"
    assert rows[0]["boxes"][0] == pytest.approx([0.1,0.1,0.5,0.5])
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_gref(tmp_path/"refs.json", tmp_path/"instances.json", tmp_path, tmp_path/"prepared", 0, 500,
                     eval_source="val")
    audit = json.loads((tmp_path/"prepared/pool_audit.json").read_text())
    assert audit["eligible_pools"]["val/absent"]["unique_images"] == 1


def test_official_pope_import_retains_variant_and_ids(tmp_path):
    path = tmp_path/"pope.json"
    write_json(path, [{"question_id":1,"image":"COCO_val2014_000000000123.jpg",
                       "text":"Is there a car?","label":"no"}])
    import_pope(path, tmp_path, "adversarial", tmp_path/"manifest.jsonl")
    row = load_manifest(tmp_path/"manifest.jsonl")[0]
    assert row["image_id"] == "coco2014:123"
    assert row["group"] == "adversarial" and row["target_present"] is False
