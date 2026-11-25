"""Step 2 - Audit the extracted FruitNet apple images.

Checks every image in data/raw/fruitnet/ for:
  * corrupted / unreadable files (OpenCV decode + Pillow verify)
  * exact duplicates            (identical file bytes, SHA-256)
  * near-duplicates             (perceptual-hash Hamming distance <= threshold)
  * label inconsistencies       (duplicates / near-duplicates found in different
                                 FruitNet quality folders, e.g. Good and Bad)

Images are joined into leakage *groups* when they are:
  * exact or near-duplicates, or
  * part of the same capture burst: taken within --session-gap seconds of each
    other (capture time from EXIF DateTime, else the timestamp in the file name).
    FruitNet photographs each apple 4-16 times in quick succession, so a burst
    is the best available proxy for "the same physical apple". Over-grouping
    is harmless; under-grouping would leak the same apple into the test set.
A group is always kept together in the same batch and the same split.

It also assigns every image to one of 6 fixed, non-overlapping labelling batches
(group-aware and balanced per source folder) and a random labelling order.

Outputs (data/interim/fruitnet_audit/):
  images.csv              one row per image (used by the labelling tool and build script)
  exact_duplicates.csv    sets of byte-identical files
  near_duplicate_pairs.csv pairs within the threshold
  label_conflicts.csv     duplicate / near-duplicate pairs across quality folders
  summary.json            headline numbers

Usage:
    python scripts/fruitnet_audit.py [--threshold 6] [--seed 42]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import warnings
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from dataset_common import (
    AUDIT_DIR,
    IMAGE_EXTENSIONS,
    IMAGES_CSV,
    NUM_BATCHES,
    RAW_DIR,
    perceptual_hash,
    sha256_file,
    write_csv_atomic,
)

IMAGE_FIELDS = ["image_id", "source_quality", "readable", "error", "width", "height",
                "sha256", "phash", "capture_time", "time_source", "camera", "session_id",
                "exact_dup_of", "group_id", "group_size",
                "cross_folder_conflict", "batch", "queue_order"]

_NAME_TIME = re.compile(r"(\d{8})_?(\d{6})")


def capture_info(path) -> tuple[datetime | None, str, str]:
    """(capture time, source of that time, camera model)."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
        raw, model = exif.get(306), str(exif.get(272) or "").strip("\x00 ")
        if raw:
            return datetime.strptime(str(raw).strip("\x00 "), "%Y:%m:%d %H:%M:%S"), "exif", model
    except Exception:  # noqa: BLE001
        model = ""
    m = _NAME_TIME.search(path.name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), "filename", model
        except ValueError:
            pass
    return None, "none", model


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def check_image(path) -> tuple[np.ndarray | None, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # treat truncated-image warnings as errors
            with Image.open(path) as im:
                im.verify()
    except Exception as exc:  # noqa: BLE001
        return None, f"Pillow: {exc}"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None, "OpenCV could not decode"
    return img, ""


def popcount64(arr: np.ndarray) -> np.ndarray:
    return np.unpackbits(arr.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=6,
                    help="max pHash Hamming distance (of 64 bits) to call two images near-duplicates")
    ap.add_argument("--session-gap", type=float, default=10,
                    help="photos taken within this many seconds form one capture burst (group)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(p for p in RAW_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise SystemExit(f"No images in {RAW_DIR}. Run scripts/fruitnet_extract.py first.")
    print(f"Auditing {len(files)} images ...")

    rows = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(RAW_DIR).as_posix()
        img, err = check_image(p)
        row = {"image_id": rel, "source_quality": rel.split("/")[0], "readable": img is not None,
               "error": err, "width": "", "height": "", "sha256": sha256_file(p), "phash": ""}
        if img is not None:
            row["height"], row["width"] = img.shape[:2]
            row["phash"] = f"{perceptual_hash(img):016x}"
        t, source, camera = capture_info(p)
        row["capture_time"] = t.isoformat() if t else ""
        row["time_source"], row["camera"] = source, camera
        rows.append(row)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}")

    n = len(rows)
    uf = UnionFind(n)

    # Exact duplicates (same bytes)
    by_hash = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_hash[r["sha256"]].append(i)
    exact_sets = [ids for ids in by_hash.values() if len(ids) > 1]
    for r in rows:
        r["exact_dup_of"] = ""
    for ids in exact_sets:
        for j in ids[1:]:
            rows[j]["exact_dup_of"] = rows[ids[0]]["image_id"]
            uf.union(ids[0], j)

    # Near-duplicates (perceptual hash)
    ok = [i for i, r in enumerate(rows) if r["phash"]]
    hashes = np.array([int(rows[i]["phash"], 16) for i in ok], dtype=np.uint64)
    near_pairs = []
    for a in range(len(ok)):
        d = popcount64(hashes[a + 1:] ^ hashes[a])
        for off in np.nonzero(d <= args.threshold)[0]:
            i, j = ok[a], ok[a + 1 + off]
            near_pairs.append((i, j, int(d[off])))
            uf.union(i, j)

    # Capture bursts (same physical apple, photographed seconds apart)
    timed = sorted((datetime.fromisoformat(r["capture_time"]), i)
                   for i, r in enumerate(rows) if r["capture_time"])
    session, prev = 0, None
    for t, i in timed:
        if prev is not None and (t - prev[0]).total_seconds() > args.session_gap:
            session += 1
        elif prev is not None:
            uf.union(prev[1], i)
        rows[i]["session_id"] = session
        prev = (t, i)
    for r in rows:
        r.setdefault("session_id", "")

    # Groups
    members = collections.defaultdict(list)
    for i in range(n):
        members[uf.find(i)].append(i)
    group_of = {}
    for gid, (_, ids) in enumerate(sorted(members.items())):
        for i in ids:
            group_of[i] = gid
    for i, r in enumerate(rows):
        r["group_id"] = group_of[i]
        r["group_size"] = len(members[uf.find(i)])

    # Label conflicts: duplicate / near-duplicate across different quality folders
    conflicts = []
    for i, j, d in near_pairs:
        if rows[i]["source_quality"] != rows[j]["source_quality"]:
            conflicts.append({"image_a": rows[i]["image_id"], "image_b": rows[j]["image_id"],
                              "folder_a": rows[i]["source_quality"],
                              "folder_b": rows[j]["source_quality"], "phash_distance": d})
    conflict_ids = {c["image_a"] for c in conflicts} | {c["image_b"] for c in conflicts}
    for r in rows:
        r["cross_folder_conflict"] = r["image_id"] in conflict_ids

    # Six non-overlapping batches: whole groups, balanced per source folder
    rng = random.Random(args.seed)
    groups_by_quality = collections.defaultdict(list)
    gid_members = collections.defaultdict(list)
    for i in range(n):
        gid_members[group_of[i]].append(i)
    for gid, ids in gid_members.items():
        q = collections.Counter(rows[i]["source_quality"] for i in ids).most_common(1)[0][0]
        groups_by_quality[q].append(gid)
    for q in sorted(groups_by_quality):
        gids = groups_by_quality[q]
        rng.shuffle(gids)
        load = [0] * NUM_BATCHES
        for gid in gids:
            b = min(range(NUM_BATCHES), key=lambda k: (load[k], k))
            load[b] += len(gid_members[gid])
            for i in gid_members[gid]:
                rows[i]["batch"] = b + 1

    # Random labelling order, so a partially labelled set is a fair random sample
    order = list(range(n))
    rng.shuffle(order)
    for pos, i in enumerate(order):
        rows[i]["queue_order"] = pos

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(IMAGES_CSV, rows, IMAGE_FIELDS)
    write_csv_atomic(AUDIT_DIR / "exact_duplicates.csv",
                     [{"set": k, "image_id": rows[i]["image_id"], "source_quality": rows[i]["source_quality"]}
                      for k, ids in enumerate(exact_sets) for i in ids],
                     ["set", "image_id", "source_quality"])
    write_csv_atomic(AUDIT_DIR / "near_duplicate_pairs.csv",
                     [{"image_a": rows[i]["image_id"], "image_b": rows[j]["image_id"], "phash_distance": d}
                      for i, j, d in near_pairs],
                     ["image_a", "image_b", "phash_distance"])
    write_csv_atomic(AUDIT_DIR / "label_conflicts.csv", conflicts,
                     ["image_a", "image_b", "folder_a", "folder_b", "phash_distance"])

    per_q = collections.Counter(r["source_quality"] for r in rows)
    sizes = collections.Counter(r["width"] and f'{r["width"]}x{r["height"]}' for r in rows)
    multi_groups = [g for g, ids in gid_members.items() if len(ids) > 1]
    summary = {
        "total_images": n,
        "per_source_folder": dict(per_q),
        "unreadable_images": sum(not r["readable"] for r in rows),
        "exact_duplicate_sets": len(exact_sets),
        "redundant_exact_copies": sum(len(s) - 1 for s in exact_sets),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_threshold": args.threshold,
        "capture_time_source": dict(collections.Counter(r["time_source"] for r in rows)),
        "cameras": dict(collections.Counter(r["camera"] or "unknown" for r in rows)),
        "capture_bursts": session + 1 if timed else 0,
        "session_gap_seconds": args.session_gap,
        "groups_total": len(gid_members),
        "groups_with_multiple_images": len(multi_groups),
        "images_in_multi_image_groups": sum(len(gid_members[g]) for g in multi_groups),
        "cross_folder_conflict_pairs": len(conflicts),
        "image_sizes": dict(sizes.most_common()),
        "batch_sizes": dict(sorted(collections.Counter(r["batch"] for r in rows).items())),
        "seed": args.seed,
    }
    (AUDIT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
