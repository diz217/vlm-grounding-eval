"""Reports preserve denominators, outcome breakdowns, and image-cluster uncertainty."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_jsonl, write_json, write_jsonl
from .scoring import score_record


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def rate(rows, numerator, denominator=lambda row: True, bootstrap=1000, seed=42):
    clusters = defaultdict(lambda: [0, 0])
    for row in rows:
        if denominator(row):
            cluster = clusters[row["image_id"]]
            cluster[0] += int(numerator(row))
            cluster[1] += 1
    blocks = list(clusters.values())
    n, d = sum(b[0] for b in blocks), sum(b[1] for b in blocks)
    result = {"value": ratio(n, d), "numerator": n, "denominator": d,
              "unique_images": len(blocks), "ci95": None}
    if len(blocks) >= 2 and bootstrap > 0:
        rng = random.Random(seed)
        values = []
        for _ in range(bootstrap):
            sample = rng.choices(blocks, k=len(blocks))
            values.append(sum(b[0] for b in sample) / sum(b[1] for b in sample))
        values.sort()
        result["ci95"] = [values[int((len(values)-1)*0.025)], values[int((len(values)-1)*0.975)]]
    return result


def classification(rows, legacy=False):
    key = "legacy_pope_prediction" if legacy else "prediction"
    tp = sum(r.get(key) is True and r["target_present"] for r in rows)
    fp = sum(r.get(key) is True and not r["target_present"] for r in rows)
    tn = sum(r.get(key) is False and not r["target_present"] for r in rows)
    fn = sum(r.get(key) is False and r["target_present"] for r in rows)
    positives = sum(r["target_present"] for r in rows)
    negatives = len(rows)-positives
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy_all_samples": ratio(tp+tn, len(rows)),
            "precision": ratio(tp, tp+fp), "recall_all_positive_samples": ratio(tp, positives),
            "f1": ratio(2*tp, 2*tp+fp+positives-tp),
            "false_positive_rate": ratio(fp, negatives), "yes_ratio": ratio(tp+fp, len(rows)),
            "valid_response_rate": ratio(tp+tn+fp+fn, len(rows))}


def summarize(scored, bootstrap=1000, seed=42):
    grouped = defaultdict(list)
    for row in scored:
        grouped[f"{row['task']}/{row['group']}"].append(row)
    groups = {}
    for name, rows in sorted(grouped.items()):
        item = {"n": len(rows), "unique_images": len({r["image_id"] for r in rows}),
                "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "success_rate": rate(rows, lambda r: r["success"], bootstrap=bootstrap, seed=seed),
                "invalid_rate": rate(rows, lambda r: r["outcome"] == "invalid", bootstrap=bootstrap, seed=seed),
                "inference_error_rate": rate(rows, lambda r: r["outcome"] == "inference_error", bootstrap=bootstrap, seed=seed)}
        if rows[0]["task"] in {"pope", "existence"}:
            item["strict_classification"] = classification(rows)
            if rows[0]["task"] == "pope":
                item["legacy_pope_compatibility"] = classification(rows, legacy=True)
        else:
            for present, prefix in ((True, "with_target"), (False, "no_target")):
                subset = [r for r in rows if r["target_present"] == present]
                item[prefix] = {"n": len(subset), "outcomes": dict(Counter(r["outcome"] for r in subset))}
                outcomes = ["correct_box", "wrong_box", "empty", "uncertain", "invalid", "inference_error"] if present else ["false_box", "empty", "uncertain", "invalid", "inference_error"]
                for outcome in outcomes:
                    item[prefix][outcome+"_rate"] = rate(subset, lambda r, o=outcome: r["outcome"] == o,
                                                          bootstrap=bootstrap, seed=seed)
            item["negative_type_counts"] = dict(Counter(r.get("negative_type") or "unspecified" for r in rows if not r["target_present"]))
        groups[name] = item
    index = {(r["sample_id"], r["task"]): r for r in scored}
    pairs, contradictions = [], Counter()
    for row in scored:
        if row["task"] != "grounding":
            continue
        if row.get("parent_id"):
            original = index.get((row["parent_id"], "grounding"))
            if original:
                pairs.append({"image_id": row["image_id"], "success": original["success"] and row["success"]})
        existence = index.get((row["sample_id"], "existence"))
        if existence:
            if existence.get("prediction") is False and row["outcome"] in {"correct_box", "wrong_box", "false_box"}:
                contradictions["existence_no_but_nonempty_box"] += 1
            if existence.get("prediction") is True and row["outcome"] == "empty":
                contradictions["existence_yes_but_empty_box"] += 1
    taxonomy = Counter()
    for row in scored:
        if row["task"] == "grounding":
            taxonomy[row["outcome"]] += 1
        elif row.get("prediction") is not None and not row["success"]:
            taxonomy["existence_false_positive" if row["prediction"] else "existence_false_negative"] += 1
    if pairs:
        taxonomy["transform_joint_failure"] = sum(not r["success"] for r in pairs)
    return {"groups": groups, "transform_joint_success": rate(pairs, lambda r: r["success"], bootstrap=bootstrap, seed=seed),
            "paired_existence_grounding": {"counts": dict(contradictions),
                                          "n_paired": sum((r["sample_id"], "existence") in index for r in scored if r["task"] == "grounding")},
            "behavioral_taxonomy": dict(taxonomy),
            "uncertainty_method": "percentile bootstrap, resampling original image IDs, seed="+str(seed)}


def percent(value):
    if isinstance(value, dict):
        value = value.get("value")
    return "—" if value is None else f"{value*100:.2f}%"


def make_report(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    plan = read_jsonl(run_dir / "requests.jsonl")
    actual = read_jsonl(run_dir / "predictions.jsonl") if (run_dir / "predictions.jsonl").exists() else []
    indexed = {}
    for record in actual:
        if record["request_id"] in indexed:
            raise ValueError("Duplicate prediction IDs; refuse ambiguous report")
        indexed[record["request_id"]] = record
    expected_ids = {r["request_id"] for r in plan}
    if set(indexed)-expected_ids:
        raise ValueError("Predictions contain unexpected requests")
    for expected in plan:
        recorded = indexed.get(expected["request_id"])
        if recorded is not None and any(recorded.get(k) != value for k,value in expected.items()):
            raise ValueError(f"Prediction request/ground-truth snapshot differs: {expected['request_id']}")
    records = [indexed.get(row["request_id"], dict(row, raw_answer="", error="not_run")) for row in plan]
    scored = [score_record(r, config["scoring"]["iou_threshold"]) for r in records]
    summary = summarize(scored, config["scoring"]["bootstrap_samples"], config["seed"])
    summary.update({"name": config["name"], "synthetic": metadata["backend"]["synthetic"],
                    "planned": len(plan), "recorded": len(actual),
                    "truncated": sum(bool(r.get("truncated")) for r in actual),
                    "run_fingerprint": metadata["run_fingerprint"],
                    "protocol_fingerprint": metadata["protocol_fingerprint"],
                    "runtime": {"total_request_seconds": sum(r.get("elapsed_seconds", 0) for r in actual),
                                "generated_tokens": sum(r.get("generated_tokens") or 0 for r in actual)}})
    write_jsonl(run_dir / "scores.jsonl", scored)
    failures = [dict(score, raw_answer=record.get("raw_answer"), query=record["sample"]["query"],
                     image=record["sample"]["image"], expected_boxes=record["sample"].get("boxes"))
                for score, record in zip(scored, records) if not score["success"]]
    write_jsonl(run_dir / "failures.jsonl", failures)
    write_json(run_dir / "summary.json", summary)
    warning = "**REPLAY FIXTURE — synthetic pipeline verification, NOT model benchmark results.**" if summary["synthetic"] else "Actual model inference; inspect completion and error counts before interpreting scores."
    lines = [f"# {config['name']}", "", warning, "",
             f"Recorded requests: {len(actual)}/{len(plan)}. Truncated responses: {summary['truncated']}.", "",
             "| Task / group | N | Success (all samples) | Invalid | Inference error |",
             "|---|---:|---:|---:|---:|"]
    for name, item in summary["groups"].items():
        lines.append(f"| {name} | {item['n']} | {percent(item['success_rate'])} | {percent(item['invalid_rate'])} | {percent(item['inference_error_rate'])} |")
    for name, item in summary["groups"].items():
        lines += ["", f"## {name}", ""]
        if "strict_classification" in item:
            lines += ["Strict yes/no metrics (invalid answers remain in task denominators):", "",
                      "```json", json.dumps(item["strict_classification"], indent=2), "```"]
            if "legacy_pope_compatibility" in item:
                lines += ["", "Original POPE heuristic, reported separately; unrecognized successful responses default to YES:", "",
                          "```json", json.dumps(item["legacy_pope_compatibility"], indent=2), "```"]
        else:
            lines += ["| Target condition | Outcome | Count | Rate | 95% image-cluster CI |",
                      "|---|---|---:|---:|---|"]
            for condition in ("with_target", "no_target"):
                part = item[condition]
                for key, value in part.items():
                    if key.endswith("_rate"):
                        outcome = key[:-5]
                        ci = value["ci95"]
                        ci_text = "—" if ci is None else f"[{percent(ci[0])}, {percent(ci[1])}]"
                        lines.append(f"| {condition} (n={part['n']}) | {outcome} | {value['numerator']} | {percent(value)} | {ci_text} |")
    lines += ["", "## Paired checks", "",
              f"Original + transformed both correct: {percent(summary['transform_joint_success'])} "
              f"({summary['transform_joint_success']['denominator']} pairs).", "",
              "```json", json.dumps(summary["paired_existence_grounding"], indent=2), "```", "",
              "## Interpretation", "",
              "- Empty/uncertain answers are distinct from wrong boxes and invalid output. Abstention is not evidence of calibrated honesty.",
              "- No-target means no match to the complete expression; it does not necessarily mean the category is absent.",
              "- Transformation failures describe behavior, not proof that a model ignored the image.",
              "- See summary.json for denominators and confidence intervals, scores.jsonl for every outcome, failures.jsonl for cases.",
              "- Standard POPE and the custom grounding/existence probe are separate tasks; no combined hallucination score is reported."]
    (run_dir / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def compare_runs(run_dirs, output):
    summaries = [make_report(p) for p in run_dirs]
    if len({s["protocol_fingerprint"] for s in summaries}) != 1:
        raise ValueError("Comparison rejected: datasets, split, prompts, or scoring protocols differ")
    if len({s["synthetic"] for s in summaries}) != 1:
        raise ValueError("Cannot compare replay fixtures with real model measurements")
    lines = ["# Model comparison", "",
             "REPLAY FIXTURES ONLY — NOT model benchmark results." if summaries[0]["synthetic"] else "Models evaluated under a matching dataset/protocol fingerprint.", "",
             "| Run | Task / group | N | Success | No-target false box | With-target correct | With-target empty | With-target uncertain |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        for name, item in summary["groups"].items():
            absent = item.get("no_target", {})
            present = item.get("with_target", {})
            lines.append(f"| {summary['name']} | {name} | {item['n']} | {percent(item['success_rate'])} | "
                         f"{percent(absent.get('false_box_rate'))} | {percent(present.get('correct_box_rate'))} | "
                         f"{percent(present.get('empty_rate'))} | {percent(present.get('uncertain_rate'))} |")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines)+"\n", encoding="utf-8")
    return {"runs": len(summaries), "output": str(output)}
