"""Sequential generation, immutable run snapshots, crash-safe resume, and reporting."""

import json
import os
import platform
import re
import time
from pathlib import Path

from .backends import create_backend
from .config import make_prompt
from .data import load_image, load_manifests
from .io import digest, file_sha256, read_jsonl, write_json, write_jsonl
from .report import make_report


def plan_requests(config):
    # Validate *all* manifests together before selecting the requested split.
    samples = load_manifests(config["data"]["manifests"])
    samples = [s for s in samples if s["split"] == config["data"]["split"]]
    if not samples:
        raise ValueError("No samples in requested split")
    requests = []
    for sample in samples:
        size = [sample["width"], sample["height"]]
        if sample.get("transform"):
            t = sample["transform"]
            size = [size[0]+t["left"]+t["right"], size[1]+t["top"]+t["bottom"]]
        tasks = [sample["task"]]
        if sample["task"] == "grounding" and config["paired_existence"]:
            tasks.append("existence")
        for task in tasks:
            requests.append({"request_id": f"{sample['id']}::{task}", "task": task, "sample": sample,
                             "prompt": make_prompt(config, sample, task, size), "image_size": size,
                             "coordinate_format": config["model"]["coordinate_format"]})
    return requests


def fingerprints(config, requests):
    # Model-specific coordinate units are excluded from the common protocol, but
    # remain in the full run fingerprint. Canonical GT is always normalized xyxy.
    canonical_samples = []
    for request in requests:
        sample = {k:v for k,v in request["sample"].items() if k != "image"}
        canonical_samples.append({"task": request["task"], "sample": sample})
    protocol = digest({"requests": canonical_samples, "prompts": config["prompts"],
                       "scoring": config["scoring"], "seed": config["seed"], "schema": 1,
                       "generation_budget": config["model"]["max_new_tokens"],
                       "dtype": config["model"]["dtype"], "decoding": "greedy"})
    code = {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
    model = dict(config["model"])
    for key in ("responses", "chat_template"):
        if model.get(key):
            model[key+"_sha256"] = file_sha256(model[key])
    full = digest({"protocol": protocol, "model": model, "code": code,
                   "prompts": [r["prompt"] for r in requests]})
    return full, protocol


def recover_predictions(path):
    """Only an incomplete final JSON line may be discarded on explicit resume."""
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        last = content.rfind(b"\n")
        tail = content[last+1:]
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            path.write_bytes(content[:last+1])
        else:
            path.write_bytes(content+b"\n")
    return read_jsonl(path)


def run(config, resume=False, retry_errors=False):
    if retry_errors and not resume:
        raise ValueError("--retry-errors requires --resume")
    if (config["model"]["backend"] == "transformers" and config["data"]["split"] == "eval"
            and not re.fullmatch(r"[0-9a-f]{40}", config["model"]["revision"])):
        raise ValueError("Formal evaluation requires an immutable 40-character revision; run 'pin' first")
    requests = plan_requests(config)
    fingerprint, protocol = fingerprints(config, requests)
    output = Path(config["output_dir"])
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if not resume:
            raise ValueError("Run already exists; choose a new output_dir or use --resume")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["run_fingerprint"] != fingerprint:
            raise ValueError("Resume rejected: model, inputs, code, prompts, or scoring changed")
        stored_requests = read_jsonl(output / "requests.jsonl")
        if stored_requests != requests:
            raise ValueError("Resume rejected: stored request snapshot differs")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is nonempty and has no valid run metadata")
        output.mkdir(parents=True, exist_ok=True)
        metadata = {"schema": 1, "run_fingerprint": fingerprint, "protocol_fingerprint": protocol,
                    "platform": platform.platform(), "python": platform.python_version(),
                    "status": "initializing", "backend": {"synthetic": config["model"]["backend"] == "replay"}}
        write_json(output / "config.resolved.json", config)
        write_jsonl(output / "requests.jsonl", requests)
        write_json(metadata_path, metadata)
    predictions_path = output / "predictions.jsonl"
    previous = recover_predictions(predictions_path) if resume else []
    expected_ids = {r["request_id"] for r in requests}
    if len({r["request_id"] for r in previous}) != len(previous):
        raise ValueError("Duplicate predictions; refuse resume")
    if any(r["request_id"] not in expected_ids for r in previous):
        raise ValueError("Unexpected predictions; refuse resume")
    if retry_errors:
        previous = [r for r in previous if not r.get("error")]
        write_jsonl(predictions_path, previous)
    done = {r["request_id"] for r in previous}
    pending = [r for r in requests if r["request_id"] not in done]
    backend = None
    if pending:
        try:
            backend = create_backend(config["model"], config["seed"])
            if "versions" in metadata.get("backend", {}) and metadata["backend"] != backend.metadata:
                raise ValueError("Resume rejected: resolved backend, template, processor, or environment changed")
            metadata["backend"] = backend.metadata
            metadata["status"] = "running"
            write_json(metadata_path, metadata)
        except Exception:
            metadata["status"] = "backend_initialization_failed"
            write_json(metadata_path, metadata)
            raise
    consecutive_errors = 0
    interrupted = False
    try:
        with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
            for request in pending:
                started = time.perf_counter()
                record = dict(request)
                try:
                    result = backend.generate(request["request_id"], load_image(request["sample"]), request["prompt"])
                    record.update(raw_answer=result["text"], generated_tokens=result["generated_tokens"],
                                  truncated=result["truncated"], error=None)
                    for key in ("rendered_chat", "input_tokens", "input_token_ids_sha256"):
                        if key in result:
                            record[key] = result[key]
                    consecutive_errors = 0
                except Exception as exc:
                    record.update(raw_answer="", error=f"{type(exc).__name__}: {exc}")
                    consecutive_errors += 1
                record["elapsed_seconds"] = time.perf_counter()-started
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(f"[{len(done)+1}/{len(requests)}] {request['request_id']} {'ERROR' if record.get('error') else 'OK'}", flush=True)
                done.add(request["request_id"])
                if consecutive_errors >= config["max_consecutive_errors"]:
                    break
    except KeyboardInterrupt:
        interrupted = True
    actual = read_jsonl(predictions_path)
    metadata["status"] = ("interrupted" if interrupted else "incomplete" if len(actual) < len(requests)
                          else "complete_with_errors" if any(r.get("error") for r in actual) else "complete")
    write_json(metadata_path, metadata)
    summary = make_report(output)
    return {"status": metadata["status"], "output_dir": str(output), "recorded": summary["recorded"],
            "planned": summary["planned"], "synthetic": summary["synthetic"]}
