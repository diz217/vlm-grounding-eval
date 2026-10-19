import json

import pytest
import yaml

from vlm_eval.cli import main
from vlm_eval.config import load_config
from vlm_eval.demo import create_demo
from vlm_eval.io import read_jsonl
from vlm_eval.report import compare_runs, make_report, rate
from vlm_eval.runner import recover_predictions, run


@pytest.fixture
def config(tmp_path):
    demo = create_demo(tmp_path/"demo")
    return load_config(demo["config"])


def test_end_to_end_resume_and_failure_breakdowns(config):
    result = run(config)
    assert result["status"] == "complete" and result["synthetic"]
    output = config["output_dir"]
    before = read_jsonl(f"{output}/predictions.jsonl")
    assert len(before) == 20
    assert run(config, resume=True)["recorded"] == len(before)
    assert read_jsonl(f"{output}/predictions.jsonl") == before
    summary = make_report(output)
    target = summary["groups"]["grounding/fixture"]["with_target"]
    assert target["n"] == 4
    for outcome in ("correct_box", "wrong_box", "uncertain", "invalid"):
        assert target["outcomes"][outcome] == 1
    absent = summary["groups"]["grounding/fixture"]["no_target"]
    assert absent["false_box_rate"]["value"] == 0.5
    assert summary["transform_joint_success"]["value"] == 1
    assert summary["paired_existence_grounding"]["counts"]["existence_no_but_nonempty_box"] == 1
    assert "NOT model benchmark results" in open(f"{output}/report.md", encoding="utf-8").read()


def test_resume_rejects_changed_protocol(config):
    run(config)
    config["prompts"]["pope"] += " Changed."
    with pytest.raises(ValueError, match="Resume rejected"):
        run(config, resume=True)


def test_run_never_silently_overwrites(config):
    run(config)
    with pytest.raises(ValueError, match="already exists"):
        run(config)


def test_resume_recovers_only_incomplete_final_line(tmp_path):
    path = tmp_path/"partial.jsonl"
    path.write_bytes(b'{"a":1}\n{"broken":')
    assert recover_predictions(path) == [{"a":1}]
    path.write_bytes(b'{"broken":\n{"a":1}\n')
    with pytest.raises(ValueError):
        recover_predictions(path)


def test_missing_requests_stay_in_report_denominators(config):
    run(config)
    path = __import__("pathlib").Path(config["output_dir"])/"predictions.jsonl"
    rows = read_jsonl(path)
    path.write_text(json.dumps(rows[0])+"\n", encoding="utf-8")
    summary = make_report(config["output_dir"])
    assert summary["recorded"] == 1 and summary["planned"] == 20
    assert sum(v["outcomes"].get("inference_error", 0) for v in summary["groups"].values()) == 19


def test_compare_rejects_different_samples_or_prompts(config, tmp_path):
    run(config)
    original = config["output_dir"]
    config["output_dir"] = str(tmp_path/"second")
    config["prompts"]["grounding"] += " Extra instruction."
    run(config)
    with pytest.raises(ValueError, match="protocols differ"):
        compare_runs([original, config["output_dir"]], tmp_path/"comparison.md")


def test_cluster_bootstrap_uses_images_not_questions():
    rows = [{"image_id":"a","success":True} for _ in range(10)] + [{"image_id":"b","success":False}]
    result = rate(rows, lambda r:r["success"], bootstrap=200, seed=1)
    assert result["unique_images"] == 2
    assert result["value"] == pytest.approx(10/11)
    assert result["ci95"] == [0,1]


def test_unknown_config_keys_fail_and_cli_propagates_error(config, tmp_path):
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump({"typo":True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_config(path)
    assert main(["validate", "--config", str(path)]) == 2


def test_report_rejects_modified_ground_truth(config):
    run(config)
    path = __import__("pathlib").Path(config["output_dir"])/"predictions.jsonl"
    rows = read_jsonl(path)
    rows[0]["sample"]["target_present"] = False
    path.write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ground-truth snapshot differs"):
        make_report(config["output_dir"])
