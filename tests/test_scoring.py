import json

import pytest

from vlm_eval.scoring import iou, legacy_pope_label, parse_grounding, parse_yes_no, score_record


@pytest.mark.parametrize("text,expected", [("Yes", True), ("No.", False),
    ("No, the target is absent.", False), ("I cannot tell", None), ("", None),
    ("Yes or no", None), ("<think>guess yes</think>\nNo", False),
    ("<think>Yes", None)])
def test_yes_no_does_not_coerce_unknown_to_yes(text, expected):
    assert parse_yes_no(text) is expected


def test_legacy_pope_kept_separate():
    assert legacy_pope_label("I cannot determine the answer") is True
    assert parse_yes_no("I cannot determine the answer") is None
    assert legacy_pope_label("No, there is a cat. It is not a dog.") is False
    assert legacy_pope_label("Yes. There is not a dog.") is True


@pytest.mark.parametrize("units,box", [
    ("normalized", [0.1,0.2,0.5,0.6]),
    ("normalized_1000", [100,200,500,600]),
    ("pixels", [20,20,100,60])])
def test_units_are_explicit_not_guessed(units, box):
    parsed = parse_grounding(json.dumps({"status": "found", "boxes": [box]}), units, (200,100))
    assert parsed["boxes"][0] == pytest.approx([0.1,0.2,0.5,0.6])


@pytest.mark.parametrize("text,reason", [
    ('{"status":"absent","boxes":[[0,0,1,1]]}', "status_box_contradiction"),
    ('{"status":"found","boxes":[]}', "status_box_contradiction"),
    ('{"status":"found","boxes":[[-0.1,0,1,1]]}', "out_of_bounds_or_degenerate_box"),
    ('{"status":"found","boxes":[[0,0,NaN,1]]}', "out_of_bounds_or_degenerate_box"),
    ('{"status":"found","boxes":[[true,0,1,1]]}', "non_numeric_box"),
    ('{"status":{},"boxes":[]}', "wrong_schema"),
    ("the object is not present", "not_json")])
def test_invalid_output_never_becomes_successful_abstention(text, reason):
    parsed = parse_grounding(text, "normalized", (200,100))
    assert parsed == {"kind": "invalid", "reason": reason}


def record(present, answer):
    return {"request_id": "a::grounding", "task": "grounding", "coordinate_format": "normalized",
            "image_size": [100,100], "raw_answer": answer,
            "sample": {"id": "a", "image_id": "im", "group": "gref", "split": "eval",
                       "target_present": present, "boxes": [[0.1,0.2,0.5,0.6]] if present else []}}


def test_with_target_empty_uncertain_wrong_and_invalid_are_distinct():
    answers = ['{"status":"absent","boxes":[]}', '{"status":"uncertain","boxes":[]}',
               '{"status":"found","boxes":[[0.6,0.6,0.9,0.9]]}', "garbage"]
    scores = [score_record(record(True, answer)) for answer in answers]
    assert [s["outcome"] for s in scores] == ["empty", "uncertain", "wrong_box", "invalid"]
    assert not any(s["success"] for s in scores)


def test_no_target_false_box_and_uncertainty():
    assert score_record(record(False, '{"status":"found","boxes":[[0,0,1,1]]}'))["outcome"] == "false_box"
    assert score_record(record(False, '{"status":"absent","boxes":[]}'))["success"]
    assert not score_record(record(False, '{"status":"uncertain","boxes":[]}'))["success"]


def test_inference_error_is_in_denominator():
    raw = record(True, "")
    raw["error"] = "CUDA out of memory"
    assert score_record(raw)["outcome"] == "inference_error"


def test_iou_geometry():
    assert iou([0,0,0.5,0.5], [0,0,0.5,0.5]) == 1
    assert iou([0,0,0.5,0.5], [0.5,0.5,1,1]) == 0
    assert iou([0,0,1,1], [0,0,0.5,0.5]) == 0.25
