"""Deterministic answer parsing; abstention is not treated as calibrated honesty."""

import json
import re

from .data import valid_box


def iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def final_answer(text):
    """Ignore a *closed* reasoning section, but never score unfinished reasoning."""
    if not isinstance(text, str):
        return ""
    if "<think>" in text:
        if "</think>" not in text:
            return ""
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_yes_no(text):
    answer = final_answer(text)
    match = re.match(r"^(yes|no)\b", answer, flags=re.IGNORECASE)
    if not match:
        return None
    # Bare yes/no or an explanation is accepted; conflicting explicit labels are not.
    if re.search(r"\b(yes|no)\b", answer[match.end():], re.IGNORECASE):
        return None
    return match.group(1).lower() == "yes"


def legacy_pope_label(text):
    """Original POPE first-sentence token heuristic, including its default-yes behavior.

    Kept as an explicitly named compatibility metric, never used for strict parsing.
    Reference: RUCAIBox/POPE evaluate.py.
    """
    words = set(text.partition(".")[0].replace(",", "").split(" "))
    return not bool(words.intersection({"No", "no", "not"}))


def parse_grounding(text, coordinate_format, image_size):
    answer = final_answer(text)
    if answer.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", answer)
        if not match:
            return {"kind": "invalid", "reason": "malformed_code_fence"}
        answer = match.group(1)
    try:
        result = json.loads(answer)
    except (ValueError, TypeError):
        return {"kind": "invalid", "reason": "not_json"}
    if not isinstance(result, dict) or set(result) != {"status", "boxes"}:
        return {"kind": "invalid", "reason": "wrong_schema"}
    status, boxes = result["status"], result["boxes"]
    if not isinstance(status, str) or status not in {"found", "absent", "uncertain"} or not isinstance(boxes, list):
        return {"kind": "invalid", "reason": "wrong_schema"}
    if (status == "found") != bool(boxes):
        return {"kind": "invalid", "reason": "status_box_contradiction"}
    if not boxes:
        return {"kind": "empty", "status": status, "boxes": []}
    w, h = image_size
    scales = {"normalized": [1,1,1,1], "normalized_1000": [1000]*4,
              "pixels": [w,h,w,h]}
    if coordinate_format not in scales:
        raise ValueError("Unknown coordinate format")
    normalized = []
    for box in boxes:
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in {int, float} for v in box)):
            return {"kind": "invalid", "reason": "non_numeric_box"}
        converted = [v/s for v, s in zip(box, scales[coordinate_format])]
        if not valid_box(converted):
            return {"kind": "invalid", "reason": "out_of_bounds_or_degenerate_box"}
        normalized.append(converted)
    return {"kind": "boxes", "status": status, "boxes": normalized}


def score_record(record, threshold=0.5):
    sample = record["sample"]
    base = {"request_id": record["request_id"], "sample_id": sample["id"],
            "image_id": sample["image_id"], "task": record["task"],
            "group": sample["group"], "split": sample["split"],
            "target_present": sample["target_present"], "parent_id": sample.get("parent_id"),
            "negative_type": sample.get("negative_type"), "success": False}
    if record.get("error"):
        return dict(base, outcome="inference_error", error=record["error"])
    if record["task"] in {"pope", "existence"}:
        prediction = parse_yes_no(record["raw_answer"])
        return dict(base, outcome="invalid" if prediction is None else "correct" if prediction == sample["target_present"] else "incorrect",
                    prediction=prediction, success=prediction is not None and prediction == sample["target_present"],
                    legacy_pope_prediction=legacy_pope_label(record["raw_answer"]) if record["task"] == "pope" else None)
    parsed = parse_grounding(record["raw_answer"], record["coordinate_format"], record["image_size"])
    if parsed["kind"] == "invalid":
        return dict(base, outcome="invalid", reason=parsed["reason"])
    if parsed["kind"] == "empty":
        return dict(base, outcome="uncertain" if parsed["status"] == "uncertain" else "empty",
                    success=not sample["target_present"] and parsed["status"] == "absent",
                    predicted_boxes=[])
    if not sample["target_present"]:
        return dict(base, outcome="false_box", predicted_boxes=parsed["boxes"])
    overlap = iou(parsed["boxes"][0], sample["boxes"][0]) if len(parsed["boxes"]) == 1 else 0.0
    return dict(base, outcome="correct_box" if overlap >= threshold else "wrong_box",
                success=overlap >= threshold, iou=overlap, predicted_boxes=parsed["boxes"])
