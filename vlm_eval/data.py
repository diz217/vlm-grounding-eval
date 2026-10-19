"""Canonical manifests, official dataset import, and image-disjoint sampling."""

import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from .io import file_sha256, read_jsonl, read_records, write_json, write_jsonl

TASKS = {"pope", "existence", "grounding"}


def valid_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def validate_sample(sample):
    for key in ("id", "image_id", "image", "task", "query", "split", "group"):
        if not isinstance(sample.get(key), str) or not sample[key].strip():
            raise ValueError(f"Sample requires nonempty string {key}: {sample.get('id')}")
    if sample["task"] not in TASKS or sample["split"] not in {"debug", "eval"}:
        raise ValueError(f"Invalid task or split: {sample['id']}")
    if not isinstance(sample.get("target_present"), bool):
        raise ValueError(f"target_present must be boolean: {sample['id']}")
    boxes = sample.get("boxes", [])
    if not isinstance(boxes, list) or any(not valid_box(box) for box in boxes):
        raise ValueError(f"Boxes must be normalized xyxy within [0,1]: {sample['id']}")
    if sample["task"] == "grounding":
        if len(boxes) != int(sample["target_present"]):
            raise ValueError(f"Only zero/single-target grounding supported: {sample['id']}")
    if sample.get("transform"):
        if (sample["task"] != "grounding" or not sample["target_present"]
                or not sample.get("parent_id") or sample.get("transform_reviewed") is not True):
            raise ValueError("Transforms require a reviewed, positive grounding parent")
        t = sample["transform"]
        if t.get("kind") != "pad_shift" or set(t) != {"kind", "left", "top", "right", "bottom"}:
            raise ValueError("Only pad_shift with four integer margins is supported")
        if any(type(t[k]) is not int or t[k] < 0 for k in ("left", "top", "right", "bottom")):
            raise ValueError("Transform margins must be nonnegative integers")
    return sample


def load_manifest(path):
    path = Path(path).resolve()
    result = []
    for raw in read_jsonl(path):
        row = validate_sample(dict(raw))
        image = Path(row["image"])
        row["image"] = str((path.parent / image).resolve() if not image.is_absolute() else image)
        result.append(row)
    return result


def load_manifests(paths, verify_images=True):
    samples = [row for path in paths for row in load_manifest(path)]
    ids = [row["id"] for row in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs across manifests")
    image_splits, content_splits, image_hashes = {}, {}, {}
    image_sizes, identity_hashes = {}, {}
    for row in samples:
        prior = image_splits.setdefault(row["image_id"], row["split"])
        if prior != row["split"]:
            raise ValueError(f"Debug/eval image overlap: {row['image_id']}")
        if verify_images:
            path = row["image"]
            if path not in image_hashes:
                image_hashes[path] = file_sha256(path)
                with Image.open(path) as im:
                    image_sizes[path] = im.size
                    im.verify()
            row["image_sha256"] = image_hashes[path]
            if "width" in row and image_sizes[path] != (row["width"], row["height"]):
                raise ValueError(f"Image dimensions differ from annotations: {row['id']}")
            row["width"], row["height"] = image_sizes[path]
            identity = identity_hashes.setdefault(row["image_id"], row["image_sha256"])
            if identity != row["image_sha256"]:
                raise ValueError(f"Same image ID maps to different source bytes: {row['image_id']}")
            prior = content_splits.setdefault(row["image_sha256"], row["split"])
            if prior != row["split"]:
                raise ValueError("Debug/eval duplicate image bytes under different IDs")
    by_id = {row["id"]: row for row in samples}
    for row in samples:
        if row.get("transform"):
            parent = by_id.get(row["parent_id"])
            if not parent or parent.get("transform"):
                raise ValueError(f"Missing original parent for transform: {row['id']}")
            if any(parent[k] != row[k] for k in ("image_id", "image", "query", "split", "target_present")):
                raise ValueError(f"Transform changes parent identity/query: {row['id']}")
            if verify_images:
                expected = pad_boxes(parent["boxes"], image_sizes[row["image"]], row["transform"])
                if any(abs(a-b) > 1e-8 for a, b in zip(expected[0], row["boxes"][0])):
                    raise ValueError(f"Incorrect transformed ground truth: {row['id']}")
    return samples


def export_manifest(path, rows):
    path = Path(path).resolve()
    portable = []
    for row in rows:
        row = dict(row)
        row["image"] = os.path.relpath(row["image"], path.parent).replace("\\", "/")
        validate_sample(row)
        portable.append(row)
    write_jsonl(path, portable)


def coco_image_id(filename):
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(filename).name)
    if not match:
        raise ValueError(f"Cannot identify COCO image: {filename}")
    return f"coco2014:{int(match.group(1))}"


def import_pope(input_path, images, variant, output):
    if variant not in {"random", "popular", "adversarial"}:
        raise ValueError("Unknown POPE variant")
    rows = []
    source_hash = file_sha256(input_path)
    for raw in read_records(input_path):
        label = str(raw["label"]).lower()
        if label not in {"yes", "no"}:
            raise ValueError("POPE labels must be yes/no")
        rows.append({"id": f"pope:{variant}:{raw['question_id']}",
                     "image_id": coco_image_id(raw["image"]),
                     "image": str((Path(images) / raw["image"]).resolve()),
                     "task": "pope", "query": raw["text"], "split": "eval",
                     "group": variant, "target_present": label == "yes", "boxes": [],
                     "source": "POPE-COCO-original", "source_sha256": source_hash})
    export_manifest(output, rows)
    return {"samples": len(rows), "variant": variant}


def gref_candidates(refs_path, instances_path, images):
    refs = read_records(refs_path)
    instances = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    annotations = {ann["id"]: ann for ann in instances["annotations"]}
    image_index = {im["id"]: im for im in instances["images"]}
    source_hashes = {"refs": file_sha256(refs_path), "instances": file_sha256(instances_path)}
    rows, seen = [], set()
    skipped = Counter()
    for ref in refs:
        ann_ids = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
        no_target = ann_ids == [-1] or (not ann_ids and ref.get("no_target") is True)
        if "no_target" in ref and bool(ref["no_target"]) != no_target:
            raise ValueError(f"Inconsistent gRef no_target and annotation IDs: {ref['ref_id']}")
        if not no_target and (len(ann_ids) != 1 or ann_ids[0] == -1):
            skipped["multi_target_or_invalid"] += 1
            continue
        im = image_index[ref["image_id"]]
        boxes = []
        if not no_target:
            ann = annotations[ann_ids[0]]
            if ann["image_id"] != ref["image_id"]:
                raise ValueError("gRef annotation references another image")
            if ann.get("iscrowd"):
                skipped["crowd"] += 1
                continue
            x, y, w, h = ann["bbox"]
            box = [x/im["width"], y/im["height"], (x+w)/im["width"], (y+h)/im["height"]]
            if not valid_box(box):
                skipped["invalid_box"] += 1
                continue
            boxes = [box]
        for sent in ref["sentences"]:
            query = sent.get("sent", sent.get("raw", "")).strip()
            key = (ref["image_id"], " ".join(query.lower().split()))
            if not query or key in seen:
                continue
            seen.add(key)
            rows.append({"id": f"gref:{ref['ref_id']}:{sent['sent_id']}",
                         "image_id": f"coco2014:{im['id']}",
                         "image": str((Path(images) / im["file_name"]).resolve()),
                         "width": im["width"], "height": im["height"],
                         "task": "grounding", "query": query, "boxes": boxes,
                         "target_present": not no_target, "group": "gref",
                         "source": "gRefCOCO", "source_split": ref["split"],
                         "source_hashes": source_hashes,
                         "negative_type": "unspecified_no_match" if no_target else None})
    return rows, dict(skipped)


def balanced_subset(rows, count, split, seed, excluded_images=()):
    rng = random.Random(seed)
    chosen = []
    for present in (True, False):
        pool = sorted((row for row in rows if row["target_present"] == present
                       and row["image_id"] not in excluded_images), key=lambda row: row["id"])
        rng.shuffle(pool)
        images = set()
        selected = []
        for row in pool:
            if row["image_id"] not in images:
                selected.append(dict(row, split=split))
                images.add(row["image_id"])
            if len(selected) == count:
                break
        if count == 0:
            selected = []
        if len(selected) != count:
            raise ValueError(f"Insufficient unique images for {split}, target_present={present}: "
                             f"requested {count}, available {len(images)}; no repeat sampling performed")
        chosen.extend(selected)
    return sorted(chosen, key=lambda row: row["id"])


def prepare_gref(refs, instances, images, output_dir, debug_count=50, eval_count=500,
                 debug_source="train", eval_source="testA", seed=42, verify_images=True):
    if debug_count < 0 or eval_count < 0:
        raise ValueError("Sample counts must be nonnegative")
    rows, skipped = gref_candidates(refs, instances, images)
    audit = {"eligible_pools": {}, "skipped_refs": skipped, "seed": seed,
             "negative_label_meaning": "No object matches the complete expression, not necessarily absent category."}
    for source in sorted({row["source_split"] for row in rows}):
        for present in (True, False):
            pool = [row for row in rows if row["source_split"] == source and row["target_present"] == present]
            audit["eligible_pools"][f"{source}/{'present' if present else 'absent'}"] = {
                "image_query_pairs": len(pool), "unique_images": len({row["image_id"] for row in pool})}
    output_dir = Path(output_dir)
    # Write the pool audit even when the requested sample size is impossible.
    write_json(output_dir / "pool_audit.json", audit)
    debug = balanced_subset([r for r in rows if r["source_split"] == debug_source], debug_count, "debug", seed)
    excluded = {row["image_id"] for row in debug}
    formal = balanced_subset([r for r in rows if r["source_split"] == eval_source], eval_count, "eval", seed+1, excluded)
    export_manifest(output_dir / "debug.jsonl", debug)
    export_manifest(output_dir / "eval.jsonl", formal)
    load_manifests([output_dir / "debug.jsonl", output_dir / "eval.jsonl"], verify_images=verify_images)
    audit["selected"] = {"debug": len(debug), "eval": len(formal),
                         "debug_source": debug_source, "eval_source": eval_source,
                         "image_validation": "complete" if verify_images else "deferred"}
    write_json(output_dir / "pool_audit.json", audit)
    return audit


def pad_boxes(boxes, size, t):
    w, h = size
    nw, nh = w+t["left"]+t["right"], h+t["top"]+t["bottom"]
    return [[(b[0]*w+t["left"])/nw, (b[1]*h+t["top"])/nh,
             (b[2]*w+t["left"])/nw, (b[3]*h+t["top"])/nh] for b in boxes]


def load_image(sample):
    with Image.open(sample["image"]) as image:
        # Do not silently EXIF-transpose: annotations refer to the stored pixel grid.
        image = image.convert("RGB")
    if sample.get("transform"):
        t = sample["transform"]
        image = ImageOps.expand(image, border=(t["left"], t["top"], t["right"], t["bottom"]), fill=(127,127,127))
    return image


def prepare_transforms(manifest, approvals, output):
    """Approvals are a reviewed list of IDs, not an automatic keyword filter."""
    rows = load_manifest(manifest)
    approved = json.loads(Path(approvals).read_text(encoding="utf-8"))
    if not isinstance(approved, list) or any(not isinstance(x, str) for x in approved):
        raise ValueError("Approvals must be a JSON array of manually reviewed sample IDs")
    if len(set(approved)) != len(approved):
        raise ValueError("Duplicate approval IDs")
    indexed = {row["id"]: row for row in rows}
    result = []
    for sid in approved:
        if sid not in indexed:
            raise ValueError(f"Unknown approved sample ID: {sid}")
        row = indexed[sid]
        if row["task"] != "grounding" or not row["target_present"] or row.get("transform"):
            raise ValueError("Only original positive grounding samples can be transformed")
        with Image.open(row["image"]) as im:
            w, h = im.size
        # Preserve every source pixel and all scene context. Shift the scene in a larger canvas.
        t = {"kind": "pad_shift", "left": max(1, w//3), "top": max(1, h//4),
             "right": max(1, w//12), "bottom": max(1, h//12)}
        result.append(dict(row, id=sid+":pad_shift", parent_id=sid, transform=t,
                           group=row["group"]+"/pad_shift", approval_sha256=file_sha256(approvals),
                           transform_reviewed=True, boxes=pad_boxes(row["boxes"], (w,h), t)))
    export_manifest(output, result)
    return {"transformed": len(result), "approval_sha256": file_sha256(approvals)}
