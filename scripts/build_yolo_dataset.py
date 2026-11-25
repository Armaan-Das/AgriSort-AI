"""Step 5 - Build the Ultralytics YOLO classification dataset from human labels.

Input : data/interim/fruitnet_audit/images.csv   (from fruitnet_audit.py)
        data/labels/fruitnet_labels_*.csv        (from the labelling tool; merged here)
Output: data/processed/fruitnet_cls/
          train/{Good,Damaged,Rotten}/*.jpg
          val/{Good,Damaged,Rotten}/*.jpg
          test/{Good,Damaged,Rotten}/*.jpg
          manifest.csv     (every image: label, split, group, source)
          summary.json     (counts per split and class, settings used)

Rules:
  * Only images with a HUMAN label of Good / Damaged / Rotten are used by default.
    (--use-source-good additionally accepts FruitNet's own 'Good' folder label for
    unlabelled Good-folder images; these are counted separately as unverified.)
  * Exclude labels, unreadable files and redundant exact duplicates are dropped.
  * Near-duplicate groups are never split across train / val / test (no leakage).
  * Split is stratified by class: 70 / 15 / 15 by default.
  * --balance caps every class at (smallest class x --ratio) for a balanced set.

Usage:
    python scripts/build_yolo_dataset.py --balance
    python scripts/build_yolo_dataset.py --balance --ratio 1.5
    python scripts/build_yolo_dataset.py --max-per-class 300
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
from datetime import datetime

from dataset_common import (
    IMAGES_CSV,
    LABEL_FIELDS,
    MERGED_LABELS_CSV,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_CLASSES,
    read_csv,
    write_csv_atomic,
)
from merge_labels import merge

SPLITS = ["train", "val", "test"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--balance", action="store_true", help="cap each class at smallest class x ratio")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--use-source-good", action="store_true",
                    help="also use unlabelled images from FruitNet's Good folder as Good (unverified)")
    ap.add_argument("--min-per-class", type=int, default=20,
                    help="refuse to build if any class has fewer images than this")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if abs(sum(args.split) - 1) > 1e-6:
        raise SystemExit("--split values must add up to 1")

    images = {r["image_id"]: r for r in read_csv(IMAGES_CSV)}
    if not images:
        raise SystemExit("Run scripts/fruitnet_audit.py first.")

    merged, conflicts, info = merge()
    write_csv_atomic(MERGED_LABELS_CSV, merged, LABEL_FIELDS)
    human = {r["image_id"]: r["label"] for r in merged}

    # ---- select usable images --------------------------------------------
    selected = {}  # image_id -> (label, verified)
    for image_id, img in images.items():
        if img["readable"] != "True" or img["exact_dup_of"]:
            continue
        label = human.get(image_id)
        if label in TARGET_CLASSES:
            selected[image_id] = (label, True)
        elif label is None and args.use_source_good and img["source_quality"] == "Good":
            selected[image_id] = ("Good", False)

    # ---- groups (near-duplicates stay together) ----------------------------
    groups = collections.defaultdict(list)
    for image_id in selected:
        groups[images[image_id]["group_id"]].append(image_id)

    def group_class(gid: str) -> str:
        return collections.Counter(selected[i][0] for i in groups[gid]).most_common(1)[0][0]

    by_class = collections.defaultdict(list)
    for gid in groups:
        by_class[group_class(gid)].append(gid)

    available = {c: sum(len(groups[g]) for g in by_class[c]) for c in TARGET_CLASSES}
    print("Available human-verified (+ optional unverified Good) images:", available)
    too_small = [c for c in TARGET_CLASSES if available[c] < args.min_per_class]
    if too_small:
        raise SystemExit(f"Not enough images for {too_small} (minimum {args.min_per_class}). "
                         "Label more images first.")

    # ---- balancing (whole groups, random but reproducible) ------------------
    rng = random.Random(args.seed)
    cap = None
    if args.balance:
        cap = int(min(available.values()) * args.ratio)
    if args.max_per_class:
        cap = min(cap or args.max_per_class, args.max_per_class)
    for c in TARGET_CLASSES:
        gids = sorted(by_class[c], key=int)
        rng.shuffle(gids)
        # prefer verified images first when capping
        gids.sort(key=lambda g: not all(selected[i][1] for i in groups[g]))
        if cap is not None:
            kept, total = [], 0
            for g in gids:
                if total >= cap:
                    break
                kept.append(g)
                total += len(groups[g])
            gids = kept
        by_class[c] = gids

    # ---- stratified, group-aware split ------------------------------------
    assignment = {}
    for c in TARGET_CLASSES:
        gids = by_class[c][:]
        rng.shuffle(gids)
        gids.sort(key=lambda g: -len(groups[g]))  # place big groups first
        total = sum(len(groups[g]) for g in gids)
        target = [total * f for f in args.split]
        count = [0, 0, 0]
        for g in gids:
            k = max(range(3), key=lambda s: (target[s] - count[s], -s))
            count[k] += len(groups[g])
            for i in groups[g]:
                assignment[i] = SPLITS[k]

    # ---- write YOLO folder structure ------------------------------------
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    manifest = []
    for image_id, split in sorted(assignment.items()):
        label, verified = selected[image_id]
        dest = PROCESSED_DIR / split / label / image_id.replace("/", "__")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW_DIR / image_id, dest)
        manifest.append({"image_id": image_id, "label": label, "split": split,
                         "human_verified": verified, "group_id": images[image_id]["group_id"],
                         "source_quality": images[image_id]["source_quality"],
                         "path": dest.relative_to(PROCESSED_DIR).as_posix()})
    write_csv_atomic(PROCESSED_DIR / "manifest.csv", manifest, list(manifest[0].keys()))

    table = {s: {c: 0 for c in TARGET_CLASSES} for s in SPLITS}
    for m in manifest:
        table[m["split"]][m["label"]] += 1
    leak = collections.defaultdict(set)
    for m in manifest:
        leak[m["group_id"]].add(m["split"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "FruitNet (Meshram & Patil 2022), Mendeley Data DOI 10.17632/b6fftwbr2v.3, CC BY 4.0",
        "settings": vars(args),
        "label_files": info["files"],
        "label_conflicts_excluded": info["conflicting_images"],
        "counts": table,
        "totals_per_class": {c: sum(table[s][c] for s in SPLITS) for c in TARGET_CLASSES},
        "total_images": len(manifest),
        "unverified_images": sum(not m["human_verified"] for m in manifest),
        "groups_split_across_sets": sum(len(v) > 1 for v in leak.values()),
    }
    (PROCESSED_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nWrote {len(manifest)} images to {PROCESSED_DIR}")
    print(f"{'':8s}" + "".join(f"{c:>9s}" for c in TARGET_CLASSES) + f"{'Total':>9s}")
    for s in SPLITS:
        print(f"{s:8s}" + "".join(f"{table[s][c]:9d}" for c in TARGET_CLASSES)
              + f"{sum(table[s].values()):9d}")
    print(f"Unverified images: {summary['unverified_images']} · "
          f"groups split across sets: {summary['groups_split_across_sets']}")


if __name__ == "__main__":
    main()
