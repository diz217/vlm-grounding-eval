"""Explicit downloads from the datasets' official repositories; verified TLS only."""

import json
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from .data import load_manifests
from .io import file_sha256, write_json


def client():
    import httpx
    return httpx.Client(verify=ssl.create_default_context(), follow_redirects=True,
                        timeout=httpx.Timeout(120, connect=20))


def fetch(url, path):
    path = Path(path)
    if path.exists():
        return {"path": str(path), "sha256": file_sha256(path), "cached": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+".part")
    with client() as connection:
        with connection.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_bytes(1024*1024):
                    stream.write(chunk)
    partial.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "url": url, "cached": False}


def fetch_annotations(directory):
    directory = Path(directory).resolve()
    with client() as connection:
        response = connection.get("https://huggingface.co/api/datasets/FudanCVL/gRefCOCO")
        response.raise_for_status()
        gref_revision = response.json()["sha"]
        response = connection.get("https://api.github.com/repos/RUCAIBox/POPE/commits/main")
        response.raise_for_status()
        pope_revision = response.json()["sha"]
    jobs = []
    for filename in ("grefs(unc).json", "instances.json"):
        jobs.append((f"https://huggingface.co/datasets/FudanCVL/gRefCOCO/resolve/{gref_revision}/{quote(filename)}",
                     directory/"gref"/filename))
    for variant in ("random", "popular", "adversarial"):
        filename = f"coco_pope_{variant}.json"
        jobs.append((f"https://raw.githubusercontent.com/RUCAIBox/POPE/{pope_revision}/output/coco/{filename}",
                     directory/"pope"/filename))
    # Revision-specific local directories prevent reusing different upstream versions.
    provenance_path = directory/"sources.json"
    revisions = {"gref_revision": gref_revision, "pope_revision": pope_revision}
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if any(previous[k] != v for k,v in revisions.items()):
            raise ValueError("Upstream revision changed; use a new download directory")
        for item in previous.get("files", []):
            if Path(item["path"]).exists() and file_sha256(item["path"]) != item["sha256"]:
                raise ValueError("Cached annotation hash differs from source provenance")
    elif any(path.exists() for _,path in jobs):
        raise ValueError("Existing annotations have no download provenance; use a new fetch directory or import them explicitly")
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, url, path) for url,path in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(f"annotation: {result['path']}", flush=True)
            records.append(result)
    result = dict(revisions, files=sorted(records, key=lambda r:r["path"]))
    write_json(provenance_path, result)
    return result


def fetch_images(manifests, workers=4):
    rows = load_manifests(manifests, verify_images=False)
    images = {row["image"] for row in rows}
    jobs = []
    for path in sorted(images):
        filename = Path(path).name
        if filename.startswith("COCO_train2014_"):
            split = "train2014"
        elif filename.startswith("COCO_val2014_"):
            split = "val2014"
        else:
            raise ValueError(f"Cannot map to an official COCO URL: {filename}")
        # COCO's image hostname can present a mismatched HTTPS certificate.
        # The same public bucket via AWS's verified path-style endpoint avoids
        # disabling TLS verification or falling back to plaintext HTTP.
        jobs.append((f"https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}", path))
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, url, path) for url,path in jobs]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"images: {completed}/{len(jobs)}", flush=True)
    load_manifests(manifests)
    return {"images": completed, "validated": True}
