"""Step 6 - Independently verify the prepared YOLO dataset for data leakage.

Checks data/processed/fruitnet_cls/ against the audit (data/interim/fruitnet_audit/):
  1. every image in the class folders is listed in manifest.csv (and vice versa), and decodes
  2. no leakage group appears in more than one split
  3. no capture session (photo burst) appears in more than one split
  4. no known near-duplicate pair (pHash <= audit threshold) is split across sets
  5. no byte-identical image (SHA-256) appears in more than one split
  6. informational: test/val images that are visually similar to a train image
     at a looser pHash threshold (--loose, default 10)
  7. informational: test/val images captured within --time-window seconds of a
     train image (possible same apple that grouping could not prove)

Writes verification.json next to the dataset and exits with an error if 1-5 fail.

Usage:
    python scripts/verify_split.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

import cv2
import numpy as np

from dataset_common import AUDIT_DIR, IMAGES_CSV, PROCESSED_DIR, read_csv

SPLITS = ["train", "val", "test"]


def popcount64(arr: np.ndarray) -> np.ndarray:
    return np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loose", type=int, default=10, help="looser pHash distance for the similarity report")
    ap.add_argument("--time-window", type=float, default=60,
                    help="report val/test images captured within this many seconds of a train image")
    args = ap.parse_args()

    manifest = read_csv(PROCESSED_DIR / "manifest.csv")
    audit = {r["image_id"]: r for r in read_csv(IMAGES_CSV)}
    if not manifest:
        sys.exit("No manifest.csv - run scripts/build_yolo_dataset.py first.")

    problems: list[str] = []

    # 1. files on disk <-> manifest
    on_disk = {p.relative_to(PROCESSED_DIR).as_posix()
               for s in SPLITS for p in (PROCESSED_DIR / s).rglob("*") if p.is_file()}
    listed = {m["path"] for m in manifest}
    if on_disk != listed:
        problems.append(f"manifest/disk mismatch: {len(on_disk - listed)} unlisted, {len(listed - on_disk)} missing")
    unreadable = [p for p in listed if cv2.imread(str(PROCESSED_DIR / p)) is None]
    if unreadable:
        problems.append(f"{len(unreadable)} unreadable images, e.g. {unreadable[:3]}")
    for m in manifest:
        if m["path"].split("/")[:2] != [m["split"], m["label"]]:
            problems.append(f"folder does not match manifest label/split: {m['path']}")
            break

    split_of = {m["image_id"]: m["split"] for m in manifest}

    def overlap(key: str) -> dict:
        where = collections.defaultdict(set)
        for m in manifest:
            v = audit[m["image_id"]][key]
            if v != "":
                where[v].add(m["split"])
        return {k: sorted(v) for k, v in where.items() if len(v) > 1}

    # 2-3. groups and capture sessions
    group_overlap, session_overlap = overlap("group_id"), overlap("session_id")
    if group_overlap:
        problems.append(f"{len(group_overlap)} leakage groups span several splits")
    if session_overlap:
        problems.append(f"{len(session_overlap)} capture sessions span several splits")

    # 4. known near-duplicate pairs
    nd_cross = [p for p in read_csv(AUDIT_DIR / "near_duplicate_pairs.csv")
                if p["image_a"] in split_of and p["image_b"] in split_of
                and split_of[p["image_a"]] != split_of[p["image_b"]]]
    if nd_cross:
        problems.append(f"{len(nd_cross)} near-duplicate pairs split across sets")

    # 5. identical bytes
    sha_overlap = overlap("sha256")
    if sha_overlap:
        problems.append(f"{len(sha_overlap)} identical files in several splits")

    # 6. informational looser similarity: val/test vs train
    train = [m["image_id"] for m in manifest if m["split"] == "train"]
    th = np.array([int(audit[i]["phash"], 16) for i in train], dtype=np.uint64)
    similar = collections.Counter()
    closest = []
    for m in manifest:
        if m["split"] == "train":
            continue
        d = popcount64(th ^ np.uint64(int(audit[m["image_id"]]["phash"], 16)))
        closest.append(int(d.min()))
        if d.min() <= args.loose:
            similar[m["split"]] += 1

    # 7. informational: val/test images captured close in time to a train image
    from datetime import datetime
    train_times = np.array(sorted(datetime.fromisoformat(audit[i]["capture_time"]).timestamp()
                                  for i in train if audit[i]["capture_time"]))
    near_in_time = collections.Counter()
    for m in manifest:
        ct = audit[m["image_id"]]["capture_time"]
        if m["split"] == "train" or not ct:
            continue
        t = datetime.fromisoformat(ct).timestamp()
        if np.abs(train_times - t).min() <= args.time_window:
            near_in_time[m["split"]] += 1

    counts = collections.Counter((m["split"], m["label"]) for m in manifest)
    groups = collections.defaultdict(set)
    sessions = collections.defaultdict(set)
    for m in manifest:
        groups[(m["split"], m["label"])].add(audit[m["image_id"]]["group_id"])
        sessions[(m["split"], m["label"])].add(audit[m["image_id"]]["session_id"])
    labels = sorted({m["label"] for m in manifest})

    report = {
        "passed": not problems,
        "problems": problems,
        "images": {s: {c: counts[(s, c)] for c in labels} for s in SPLITS},
        "groups": {s: {c: len(groups[(s, c)]) for c in labels} for s in SPLITS},
        "capture_sessions": {s: {c: len(sessions[(s, c)]) for c in labels} for s in SPLITS},
        "groups_spanning_splits": len(group_overlap),
        "sessions_spanning_splits": len(session_overlap),
        "near_duplicate_pairs_across_splits": len(nd_cross),
        "identical_files_across_splits": len(sha_overlap),
        "loose_similarity_threshold": args.loose,
        "val_test_images_similar_to_a_train_image": dict(similar),
        "min_phash_distance_val_test_to_train": {"min": min(closest), "median": float(np.median(closest))},
        "time_window_seconds": args.time_window,
        "val_test_images_captured_near_a_train_image": dict(near_in_time),
    }
    (PROCESSED_DIR / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if problems:
        sys.exit("VERIFICATION FAILED")
    print("\nVERIFICATION PASSED - no known leakage between train, val and test.")


if __name__ == "__main__":
    main()
