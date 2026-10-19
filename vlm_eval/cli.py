import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .data import import_pope, prepare_gref, prepare_transforms
from .demo import create_demo
from .report import compare_runs, make_report
from .runner import fingerprints, plan_requests, run


def build_parser():
    parser = argparse.ArgumentParser(description="POPE + coordinate grounding evaluation (Transformers, no vLLM)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Import locally downloaded official datasets")
    formats = prepare.add_subparsers(dest="format", required=True)
    pope = formats.add_parser("pope")
    pope.add_argument("--input", required=True)
    pope.add_argument("--images", required=True)
    pope.add_argument("--variant", required=True, choices=["random", "popular", "adversarial"])
    pope.add_argument("--output", required=True)
    gref = formats.add_parser("gref")
    gref.add_argument("--refs", required=True, help="Official grefs(unc).json; pickle files are not loaded")
    gref.add_argument("--instances", required=True)
    gref.add_argument("--images", required=True)
    gref.add_argument("--output-dir", required=True)
    gref.add_argument("--debug-count", type=int, default=50, help="Samples per target-presence condition")
    gref.add_argument("--eval-count", type=int, default=500)
    gref.add_argument("--debug-source", default="train")
    gref.add_argument("--eval-source", default="testA")
    gref.add_argument("--seed", type=int, default=42)
    gref.add_argument("--defer-image-validation", action="store_true", help="Prepare manifests before downloading their images; run validate afterwards")
    transforms = formats.add_parser("transforms")
    transforms.add_argument("--manifest", required=True)
    transforms.add_argument("--approved-ids", required=True, help="JSON list of manually reviewed positive query IDs")
    transforms.add_argument("--output", required=True)
    demo = commands.add_parser("demo", help="Create clearly labelled replay fixtures, not model results")
    demo.add_argument("--directory", default="artifacts/eval-demo")
    for name in ("validate", "run", "pin", "download"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        if name == "run":
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--retry-errors", action="store_true")
        if name == "pin":
            sub.add_argument("--output", required=True, help="Write config with immutable model revision")
        if name in {"pin", "download"}:
            sub.add_argument("--system-trust", action="store_true", help="Use verified OS/Python TLS trust store for Hub requests")
        if name == "download":
            sub.add_argument("--metadata-only", action="store_true", help="Only processor/config files; do not download weights")
    report = commands.add_parser("report")
    report.add_argument("run_dir")
    compare = commands.add_parser("compare")
    compare.add_argument("run_dirs", nargs="+")
    compare.add_argument("--output", required=True)
    fetch = commands.add_parser("fetch", help="Explicitly download official annotations or manifest-selected COCO images")
    fetch_types = fetch.add_subparsers(dest="fetch_type", required=True)
    annotations = fetch_types.add_parser("annotations")
    annotations.add_argument("--directory", default="data/downloads")
    images = fetch_types.add_parser("images")
    images.add_argument("--manifests", nargs="+", required=True)
    images.add_argument("--workers", type=int, default=4)
    return parser


def pin_config(config, output):
    import yaml
    from huggingface_hub import HfApi
    if config["model"]["backend"] != "transformers":
        raise ValueError("Only Transformers checkpoints need pinning")
    info = HfApi().model_info(config["model"]["id"], revision=config["model"]["revision"])
    config["model"]["revision"] = info.sha
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {"config": str(output), "revision": info.sha}


def configure_system_trust():
    import ssl
    import httpx
    from huggingface_hub import set_client_factory
    # Verification remains enabled. This accommodates managed Windows trust stores
    # without installing certificates or disabling TLS verification.
    set_client_factory(lambda: httpx.Client(verify=ssl.create_default_context(), follow_redirects=True,
                                            timeout=httpx.Timeout(60, connect=15)))


def download_model(config, metadata_only=False):
    from huggingface_hub import snapshot_download
    model = config["model"]
    if not model.get("cache_dir"):
        raise ValueError("Set model.cache_dir explicitly before downloading")
    if model["revision"] == "main":
        raise ValueError("Pin the model revision before downloading")
    patterns = ["*.json", "*.jinja", "*.model", "*.txt", "merges.txt"]
    if not metadata_only:
        patterns.append("*.safetensors")
    snapshot = snapshot_download(model["id"], revision=model["revision"], cache_dir=model["cache_dir"],
                                 allow_patterns=patterns, max_workers=2)
    return {"snapshot": snapshot, "metadata_only": metadata_only, "revision": model["revision"]}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if args.format == "pope":
                result = import_pope(args.input, args.images, args.variant, args.output)
            elif args.format == "gref":
                result = prepare_gref(args.refs, args.instances, args.images, args.output_dir,
                                      args.debug_count, args.eval_count, args.debug_source, args.eval_source, args.seed,
                                      not args.defer_image_validation)
            else:
                result = prepare_transforms(args.manifest, args.approved_ids, args.output)
        elif args.command == "demo":
            result = create_demo(args.directory)
        elif args.command == "report":
            summary = make_report(args.run_dir)
            result = {"report": str(Path(args.run_dir)/"report.md"), "planned": summary["planned"], "recorded": summary["recorded"]}
        elif args.command == "compare":
            result = compare_runs(args.run_dirs, args.output)
        elif args.command == "fetch":
            from .downloads import fetch_annotations, fetch_images
            if args.fetch_type == "annotations":
                result = fetch_annotations(args.directory)
            else:
                if not 1 <= args.workers <= 16:
                    raise ValueError("workers must be between 1 and 16")
                result = fetch_images(args.manifests, args.workers)
        else:
            config = load_config(args.config)
            if args.command == "validate":
                requests = plan_requests(config)
                _, protocol = fingerprints(config, requests)
                from collections import Counter
                result = {"requests": len(requests), "tasks": dict(Counter(r["task"] for r in requests)),
                          "split": config["data"]["split"], "protocol_fingerprint": protocol}
            elif args.command == "pin":
                if args.system_trust:
                    configure_system_trust()
                result = pin_config(config, args.output)
            elif args.command == "download":
                if args.system_trust:
                    configure_system_trust()
                result = download_model(config, args.metadata_only)
            else:
                result = run(config, args.resume, args.retry_errors)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.command == "run" and result["status"] != "complete":
            return 2
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
